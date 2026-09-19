module LossFunctionsModule

using DispatchDoctor: @stable
using DynamicExpressions:
    AbstractExpression,
    AbstractExpressionNode,
    ArrayBuffer,
    EvalContext,
    get_tree,
    eval_tree_array
using DynamicExpressions.EvaluateModule: reset_index!
using LossFunctions: LossFunctions
using LossFunctions: SupervisedLoss
using SpecialFunctions: loggamma
using ..CoreModule:
    AbstractOptions,
    Dataset,
    create_expression,
    DATA_TYPE,
    LOSS_TYPE,
    is_weighted,
    get_indices,
    get_full_dataset,
    init_value
using ..ComplexityModule: compute_complexity
using ..InterfaceDynamicExpressionsModule:
    expected_array_type, takes_eval_context, _process_eval_options

function create_eval_context(dataset::Dataset, options::AbstractOptions, num_arrays::Int)
    if options.bumper isa Val{true} || !takes_eval_context(options.operators)
        return nothing
    end
    array = similar(dataset.X, axes(dataset.X, 2))
    arrays = [similar(array) for _ in 1:num_arrays]
    return EvalContext(; turbo=options.turbo, buffer=ArrayBuffer(arrays, Ref(0)))
end

function _loss(
    ::AbstractArray{T1}, ::AbstractArray{T2}, ::LT
) where {T1,T2,LT<:Union{Function,SupervisedLoss}}
    return error(
        "Element type of `x` is $(T1) is different from element type of `y` which is $(T2)."
    )
end
function _weighted_loss(
    ::AbstractArray{T1}, ::AbstractArray{T2}, ::AbstractArray{T3}, ::LT
) where {T1,T2,T3,LT<:Union{Function,SupervisedLoss}}
    return error(
        "Element type of `x` is $(T1), element type of `y` is $(T2), and element type of `w` is $(T3). " *
        "All element types must be the same.",
    )
end

# Measurement uncertainty is kept in Dataset.extra so the Dataset ABI remains
# compatible with existing serialized states and custom Dataset constructors.
# The public equation_search wrapper stores arrays with output rows first.
function _extra_array(dataset::Dataset, name::Symbol)
    extra = dataset.extra
    return hasproperty(extra, name) ? getproperty(extra, name) : nothing
end

function _dataset_output_array(dataset::Dataset, value)
    value === nothing && return nothing
    indices = get_indices(dataset)
    if value isa AbstractVector
        return indices === nothing ? value : view(value, indices)
    elseif value isa AbstractMatrix
        size(value, 1) == 1 &&
            return indices === nothing ? view(value, 1, :) : view(value, 1, indices)
        dataset.index <= size(value, 1) ||
            throw(DimensionMismatch("uncertainty array has too few output rows."))
        return indices === nothing ? view(value, dataset.index, :) : view(value, dataset.index, indices)
    else
        throw(ArgumentError("uncertainty values must be vectors or matrices."))
    end
end

function _measurement_uncertainty(dataset::Dataset, mode::Symbol)
    if mode == :symmetry
        sigma = _dataset_output_array(dataset, _extra_array(dataset, :sigma))
        sigma === nothing && return nothing
        length(sigma) == dataset.n ||
            throw(DimensionMismatch("sigma must have one entry per dataset sample."))
        all(v -> isfinite(v) && v > zero(v), sigma) ||
            throw(ArgumentError("sigma values must be finite and strictly positive."))
        return sigma
    elseif mode == :asymmetry
        sigma_minus = _dataset_output_array(dataset, _extra_array(dataset, :sigma_minus))
        sigma_plus = _dataset_output_array(dataset, _extra_array(dataset, :sigma_plus))
        if sigma_minus === nothing || sigma_plus === nothing
            throw(ArgumentError("asymmetry requires sigma_minus and sigma_plus."))
        end
        (length(sigma_minus) == dataset.n && length(sigma_plus) == dataset.n) ||
            throw(DimensionMismatch("sigma_minus and sigma_plus must match dataset samples."))
        all(v -> isfinite(v) && v > zero(v), sigma_minus) &&
            all(v -> isfinite(v) && v > zero(v), sigma_plus) ||
            throw(ArgumentError("sigma_minus and sigma_plus must be finite and strictly positive."))
        return sigma_minus, sigma_plus
    elseif mode == :none
        return nothing
    end
    throw(ArgumentError("uncertainty_mode must be :none, :symmetry, or :asymmetry."))
end

@inline function _huber_value(u, delta)
    au = abs(u)
    return au <= delta ? (u * u) / 2 : delta * (au - delta / 2)
end

@inline function _pseudo_huber_value(u, delta)
    return delta^2 * (sqrt(one(u) + (u / delta)^2) - one(u))
end

function _aggregate_values(values, weights)
    if weights === nothing
        return sum(values) / length(values)
    end
    return sum(values .* weights) / sum(weights)
end

function _preset_loss(prediction, target, dataset::Dataset, options::AbstractOptions)
    preset = options.loss_preset
    mode = options.uncertainty_mode
    weights = dataset.weights
    if mode != :none && weights !== nothing
        throw(ArgumentError(
            "Measurement uncertainty cannot be combined with observation weights."
        ))
    end
    if mode == :asymmetry
        sigma_minus, sigma_plus = _measurement_uncertainty(dataset, mode)
        preset = preset == :default ? :asymmetric_gaussian_nll : preset
        values = similar(prediction)
        for i in eachindex(prediction)
            residual = prediction[i] - target[i]
            sigma = residual < zero(residual) ? sigma_minus[i] : sigma_plus[i]
            standardized = residual / sigma
            values[i] = if preset == :asymmetric_gaussian_nll
                (standardized^2) / 2 + log(sigma_minus[i] + sigma_plus[i]) +
                log(pi / 2) / 2
            elseif preset == :asymmetric_huber
                _huber_value(standardized, options.robust_delta)
            elseif preset == :asymmetric_pseudo_huber
                _pseudo_huber_value(standardized, options.robust_delta)
            elseif preset == :asymmetric_student_t_nll
                nu = options.student_nu
                (nu + one(nu)) / 2 * log1p(standardized^2 / nu) +
                log(sigma_minus[i] + sigma_plus[i]) - log(2) +
                loggamma(nu / 2) - loggamma((nu + one(nu)) / 2) +
                log(nu * pi) / 2
            else
                throw(ArgumentError("Unsupported asymmetric loss preset: $(preset)."))
            end
        end
        # Uncertainty is a likelihood scale, not a residual-dependent sample
        # weight. Use a fixed N denominator so a sign change cannot alter it.
        return sum(values) / length(values)
    elseif mode == :symmetry
        sigma = _measurement_uncertainty(dataset, mode)
        sigma === nothing && throw(ArgumentError("symmetry requires sigma."))
        values = similar(prediction)
        for i in eachindex(prediction)
            residual = (prediction[i] - target[i]) / sigma[i]
            values[i] = if preset == :gaussian_nll
                residual^2 / 2 + log(sigma[i]) + log(2pi) / 2
            elseif preset in (:default, :l2)
                residual^2
            elseif preset == :l1
                abs(residual)
            elseif preset == :huber
                _huber_value(residual, options.robust_delta)
            elseif preset == :pseudo_huber
                _pseudo_huber_value(residual, options.robust_delta)
            elseif preset == :log_cosh
                logcosh(residual)
            else
                throw(ArgumentError("Loss preset $(preset) is incompatible with symmetric uncertainty."))
            end
        end
        return sum(values) / length(values)
    end

    values = similar(prediction)
    for i in eachindex(prediction)
        residual = prediction[i] - target[i]
        values[i] = if preset in (:default, :l2)
            residual^2
        elseif preset == :l1
            abs(residual)
        elseif preset == :huber
            _huber_value(residual, options.robust_delta)
        elseif preset == :pseudo_huber
            _pseudo_huber_value(residual, options.robust_delta)
        elseif preset == :log_cosh
            logcosh(residual)
        else
            throw(ArgumentError("Loss preset $(preset) requires a compatible uncertainty mode."))
        end
    end
    return _aggregate_values(values, weights)
end

function _loss(
    x::AbstractArray{T}, y::AbstractArray{T}, loss::LT
) where {T,LT<:Union{Function,SupervisedLoss}}
    if loss isa SupervisedLoss
        return LossFunctions.mean(loss, x, y)
    else
        l(i) = loss(x[i], y[i])
        return LossFunctions.mean(l, eachindex(x))
    end
end

function _weighted_loss(
    x::AbstractArray{T}, y::AbstractArray{T}, w::AbstractArray{T}, loss::LT
) where {T,LT<:Union{Function,SupervisedLoss}}
    if loss isa SupervisedLoss
        return sum(loss, x, y, w; normalize=true)
    else
        l(i) = loss(x[i], y[i], w[i])
        return sum(l, eachindex(x)) / sum(w)
    end
end

@stable(
    default_mode = "disable",
    default_union_limit = 2,
    begin
        function eval_tree_dispatch(
            tree::AbstractExpression,
            dataset::Dataset,
            options::AbstractOptions,
            eval_context,
        )
            A = expected_array_type(dataset.X, typeof(tree))
            out, complete = eval_tree_array(
                tree, dataset.X, options; eval_context=eval_context
            )
            if isnothing(out)
                return out, false
            else
                return out::A, complete::Bool
            end
        end
        function eval_tree_dispatch(
            tree::AbstractExpressionNode,
            dataset::Dataset,
            options::AbstractOptions,
            eval_context,
        )
            A = expected_array_type(dataset.X, typeof(tree))
            out, complete = eval_tree_array(
                tree, dataset.X, options; eval_context=eval_context
            )
            if isnothing(out)
                return out, false
            else
                return out::A, complete::Bool
            end
        end
    end
)

# Evaluate the loss of a particular expression on the input dataset.
function _eval_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
    regularization::Bool,
    eval_context,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    eval_context === nothing || reset_index!(eval_context.buffer)
    (prediction, completion) = eval_tree_dispatch(tree, dataset, options, eval_context)
    if !completion || isnothing(prediction)
        return L(Inf)
    end

    loss_val = if options.loss_preset != :default || options.uncertainty_mode != :none
        _preset_loss(prediction, dataset.y::AbstractArray, dataset, options)
    elseif is_weighted(dataset)
        _weighted_loss(
            prediction,
            dataset.y::AbstractArray,
            dataset.weights,
            options.elementwise_loss,
        )
    else
        _loss(prediction, dataset.y::AbstractArray, options.elementwise_loss)
    end

    return loss_val
end

# This evaluates function F:
function evaluator(
    f::F,
    tree::Union{AbstractExpressionNode{T},AbstractExpression{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
    idx,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE,F}
    full_dataset = get_full_dataset(dataset)
    idx = @something(idx, get_indices(dataset), Some(nothing))
    if hasmethod(f, typeof((tree, full_dataset, options, idx)))
        # If user defines method that accepts batching indices, we
        # can convert the SubDataset to the old version
        return f(tree, full_dataset, options, idx)
    else
        return f(tree, dataset, options)
    end
end

# Evaluate the loss of a particular expression on the input dataset.
function eval_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions;
    regularization::Bool=true,
    idx=nothing,
    eval_context=nothing,
    kws...,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    eval_context = _process_eval_options(eval_context, kws, :eval_loss)
    kws = Base.structdiff((; kws...), (; eval_options=nothing))
    isempty(kws) || Base.kwerr(kws, eval_loss, tree, dataset, options)
    loss_val = if !isnothing(options.loss_function)
        f = options.loss_function::Function
        inner_tree = tree isa AbstractExpression ? get_tree(tree) : tree
        evaluator(f, inner_tree, dataset, options, idx)
    elseif !isnothing(options.loss_function_expression)
        f = options.loss_function_expression::Function
        @assert tree isa AbstractExpression
        evaluator(f, tree, dataset, options, idx)
    else
        _eval_loss(tree, dataset, options, regularization, eval_context)
    end

    return loss_val
end

"""
    eval_case_losses(tree, dataset, options; eval_context=nothing)

Evaluate an expression's loss contribution for every observation.  This is
used by parent selectors that need the error vector rather than only the
aggregate scalar loss.  Custom aggregate objectives and custom elementwise
functions are deliberately unsupported here because their case semantics
cannot be inferred safely; callers should fall back to scalar-cost selection
in that situation.

For weighted data, the returned values include the per-observation weight.
The common normalization by `sum(weights)` is omitted because it is the same
positive constant for every candidate and therefore does not affect
epsilon-lexicase comparisons.
"""
function eval_case_losses(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions;
    eval_context=nothing,
)::Union{Nothing,Vector{L}} where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # A user-supplied aggregate objective may depend on correlations between
    # observations or on derivative information.  There is no correct generic
    # way to split such a scalar into case losses.
    isnothing(options.loss_function) || return nothing
    isnothing(options.loss_function_expression) || return nothing
    options.elementwise_loss isa SupervisedLoss || return nothing
    isnothing(dataset.y) && return nothing

    eval_context === nothing || reset_index!(eval_context.buffer)
    prediction, completion = eval_tree_dispatch(tree, dataset, options, eval_context)
    if !completion || isnothing(prediction)
        return fill(L(Inf), dataset.n)
    end

    errors = Vector{L}(undef, dataset.n)
    loss = options.elementwise_loss::SupervisedLoss
    if is_weighted(dataset)
        @inbounds for i in eachindex(errors, prediction, dataset.y, dataset.weights)
            errors[i] = L(dataset.weights[i]) * L(loss(prediction[i], dataset.y[i]))
        end
    else
        @inbounds for i in eachindex(errors, prediction, dataset.y)
            errors[i] = L(loss(prediction[i], dataset.y[i]))
        end
    end
    return errors
end

# Just so we can pass either PopMember or Node here:
get_tree_from_member(t::Union{AbstractExpression,AbstractExpressionNode}) = t
get_tree_from_member(m) = m.tree
# Beware: this is a circular dependency situation...
# PopMember is using losses, but then we also want
# losses to use the PopMember's cached complexity for trees.
# TODO!

# Compute a cost which includes a complexity penalty in the loss
function loss_to_cost(
    loss::L,
    use_baseline::Bool,
    baseline::L,
    member,
    options::AbstractOptions,
    complexity::Union{Int,Nothing}=nothing,
)::L where {L<:LOSS_TYPE}
    # TODO: Come up with a more general normalization scheme.
    normalization = if baseline >= L(0.01) && use_baseline
        baseline
    else
        L(0.01)
    end
    loss_val = loss / normalization
    size = @something(complexity, compute_complexity(member, options))
    parsimony_term = size * options.parsimony
    loss_val += L(parsimony_term)

    return loss_val
end

# Score an equation
function eval_cost(
    dataset::Dataset{T,L},
    member,
    options::AbstractOptions;
    complexity::Union{Int,Nothing}=nothing,
    eval_context=nothing,
    kws...,
)::Tuple{L,L} where {T<:DATA_TYPE,L<:LOSS_TYPE}
    eval_context = _process_eval_options(eval_context, kws, :eval_cost)
    kws = Base.structdiff((; kws...), (; eval_options=nothing))
    isempty(kws) || Base.kwerr(kws, eval_cost, dataset, member, options)
    result_loss = eval_loss(get_tree_from_member(member), dataset, options; eval_context)
    cost = loss_to_cost(
        result_loss,
        dataset.use_baseline,
        dataset.baseline_loss,
        member,
        options,
        complexity,
    )
    return cost, result_loss
end

# Deprecated form
function score_func end

"""
    update_baseline_loss!(dataset::Dataset{T,L}, options::AbstractOptions) where {T<:DATA_TYPE,L<:LOSS_TYPE}

Update the baseline loss of the dataset using the loss function specified in `options`.
"""
function update_baseline_loss!(
    dataset::Dataset{T,L}, options::AbstractOptions
) where {T<:DATA_TYPE,L<:LOSS_TYPE}
    example_tree = create_expression(init_value(T), options, dataset)
    # constructorof(options.node_type)(T; val=dataset.avg_y)
    # TODO: It could be that the loss function is not defined for this example type?
    baseline_loss = eval_loss(example_tree, dataset, options)
    if isfinite(baseline_loss)
        dataset.baseline_loss = baseline_loss
        dataset.use_baseline = true
    else
        dataset.baseline_loss = one(L)
        dataset.use_baseline = false
    end
    return nothing
end

end
