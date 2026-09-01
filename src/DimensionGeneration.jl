module DimensionGenerationModule

using Random: AbstractRNG, default_rng, rand, shuffle!
import DynamicExpressions
import DynamicQuantities
using DynamicExpressions: AbstractExpression, AbstractExpressionNode, get_op_name
using DynamicQuantities: DimensionError, dimension

using ..CoreModule: AbstractOptions, Dataset, DATA_TYPE, dimension_policy, sample_value

struct DimensionCandidate
    tree::Any
    output_dimension::Any
    node_count::Int
end

function _dimension_value(qtype, scalar_type::Type, output_dimension)
    return DynamicQuantities.constructorof(qtype)(one(scalar_type), output_dimension)
end

function _operator_name(op)
    try
        return lowercase(String(get_op_name(op)))
    catch
        return lowercase(string(op))
    end
end

function _known_transition(
    op, child_dimensions::Tuple, qtype, scalar_type::Type, dimensionless
)
    name = _operator_name(op)
    if length(child_dimensions) == 2
        left, right = child_dimensions
        if name in ("+", "-", "plus", "sub")
            return left == right ? left : nothing
        elseif name in ("*", "×", "mult", "square", "cube")
            return dimension(
                _dimension_value(qtype, scalar_type, left) *
                _dimension_value(qtype, scalar_type, right)
            )
        elseif name in ("/", "÷")
            return dimension(
                _dimension_value(qtype, scalar_type, left) /
                _dimension_value(qtype, scalar_type, right)
            )
        elseif name in ("^", "pow", "pow_abs")
            return (left == dimensionless && right == dimensionless) ? dimensionless : nothing
        elseif name == "mod"
            return left == right ? left : nothing
        else
            return nothing
        end
    elseif length(child_dimensions) == 1
        child = first(child_dimensions)
        if name in ("neg", "-", "abs", "relu", "round", "floor", "ceil")
            return child
        elseif name in ("square",)
            return dimension(
                _dimension_value(qtype, scalar_type, child) *
                _dimension_value(qtype, scalar_type, child)
            )
        elseif name in ("cube",)
            return dimension(
                _dimension_value(qtype, scalar_type, child) *
                _dimension_value(qtype, scalar_type, child) *
                _dimension_value(qtype, scalar_type, child)
            )
        elseif name in ("sqrt", "safe_sqrt")
            try
                return dimension(sqrt(_dimension_value(qtype, scalar_type, child)))
            catch e
                (e isa DimensionError || e isa MethodError) && return nothing
                return nothing
            end
        elseif name == "cbrt"
            try
                return dimension(cbrt(_dimension_value(qtype, scalar_type, child)))
            catch e
                (e isa DimensionError || e isa MethodError) && return nothing
                return nothing
            end
        elseif name == "inv"
            try
                return dimension(inv(_dimension_value(qtype, scalar_type, child)))
            catch e
                (e isa DimensionError || e isa MethodError) && return nothing
                return nothing
            end
        elseif name in (
            "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan",
            "asinh", "acosh", "atanh", "exp", "exp2", "exp10", "expm1", "log",
            "log2", "log10", "log1p", "erf", "erfc", "gamma", "sign",
        )
            return child == dimensionless ? dimensionless : nothing
        else
            return nothing
        end
    end
    return nothing
end

function _leaf_candidates(
    dataset::Dataset, options::AbstractOptions, nfeatures::Int, ::Type{T}, rng::AbstractRNG
) where {T}
    candidates = DimensionCandidate[]
    dimensions = dataset.X_dimensions
    for feature in 1:min(nfeatures, length(dimensions))
        push!(
            candidates,
            DimensionCandidate(
                DynamicExpressions.constructorof(options.node_type)(T; feature=feature),
                dimension(dimensions[feature]),
                1,
            ),
        )
    end
    dimensionless = dimension(dimensions[1] / dimensions[1])
    for _ in 1:4
        push!(
            candidates,
            DimensionCandidate(
                DynamicExpressions.constructorof(options.node_type)(
                    T; val=sample_value(rng, T, options)
                ),
                dimensionless,
                1,
            ),
        )
    end
    return candidates
end

function _random_composition(rng::AbstractRNG, total::Int, parts::Int)
    total < parts && return nothing
    parts == 1 && return Int[total]
    available = collect(1:(total - 1))
    shuffle!(rng, available)
    cuts = sort!(available[1:(parts - 1)])
    values = Int[]
    previous = 0
    for cut in cuts
        push!(values, cut - previous)
        previous = cut
    end
    push!(values, total - previous)
    any(≤(0), values) && return nothing
    return values
end

function _make_operator_candidate(
    op,
    op_index::Int,
    children::Vector{DimensionCandidate},
    qtype,
    scalar_type::Type,
    node_type,
    dimensionless,
)
    output_dimension = _known_transition(
        op,
        Tuple(candidate.output_dimension for candidate in children),
        qtype,
        scalar_type,
        dimensionless,
    )
    output_dimension === nothing && return nothing
    tree = DynamicExpressions.constructorof(node_type)(;
        op=op_index,
        children=Tuple(candidate.tree for candidate in children),
    )
    return DimensionCandidate(
        tree,
        output_dimension,
        1 + sum(candidate.node_count for candidate in children),
    )
end

function _target_closure(
    leaves,
    options::AbstractOptions,
    target,
    qtype,
    ::Type{T},
    max_nodes::Int,
    dimensionless,
) where {T}
    by_size = [DimensionCandidate[] for _ in 1:max_nodes]
    by_size[1] = leaves
    max_states = 64
    for candidate in leaves
        candidate.output_dimension == target && return candidate
    end
    for node_count in 2:max_nodes
        states = by_size[node_count]
        for degree in eachindex(options.nops)
            degree == 0 && continue
            for op_index in 1:options.nops[degree]
                op = options.operators.ops[degree][op_index]
                if degree == 1
                    child_sizes = (node_count - 1,)
                    for child in by_size[first(child_sizes)]
                        candidate = _make_operator_candidate(
                            op, op_index, [child], qtype, T, options.node_type, dimensionless
                        )
                        candidate === nothing && continue
                        candidate.output_dimension == target && return candidate
                        length(states) < max_states && push!(states, candidate)
                    end
                elseif degree == 2
                    for left_size in 1:(node_count - 2),
                        left in by_size[left_size],
                        right in by_size[(node_count - 1) - left_size]
                        candidate = _make_operator_candidate(
                            op, op_index, [left, right], qtype, T, options.node_type, dimensionless
                        )
                        candidate === nothing && continue
                        candidate.output_dimension == target && return candidate
                        length(states) < max_states && push!(states, candidate)
                    end
                end
                length(states) >= max_states && break
            end
            length(states) >= max_states && break
        end
    end
    return nothing
end

function gen_random_tree_dimensional(
    dataset::Dataset,
    options::AbstractOptions,
    nlength::Int,
    nfeatures::Int,
    ::Type{T},
    rng::AbstractRNG=default_rng();
    max_nodes::Union{Nothing,Int}=nothing,
) where {T<:DATA_TYPE}
    policy = dimension_policy(options)
    policy in (:compatible, :strict) || return nothing
    dataset.X_dimensions === nothing && return nothing
    policy === :strict && dataset.y_dimensions === nothing && return nothing
    dimensions = dataset.X_dimensions
    isempty(dimensions) && return nothing
    qtype = typeof(first(dimensions))
    dimensionless = dimension(dimensions[1] / dimensions[1])
    budget = isnothing(max_nodes) ?
        max(1, min(options.maxsize, 1 + length(options.nops) * max(nlength, 1))) :
        max(1, min(options.maxsize, max_nodes))
    leaves = _leaf_candidates(dataset, options, nfeatures, T, rng)

    if policy === :strict
        target = dimension(dataset.y_dimensions)
        direct = _target_closure(leaves, options, target, qtype, T, budget, dimensionless)
        direct !== nothing && return direct.tree
    end

    by_size = [DimensionCandidate[] for _ in 1:budget]
    by_size[1] = leaves
    max_states = 96
    max_trials = 128
    degree_choices = [d for d in eachindex(options.nops) if options.nops[d] > 0]
    isempty(degree_choices) && return nothing
    for node_count in 2:budget
        trials = 0
        while trials < max_trials && length(by_size[node_count]) < max_states
            trials += 1
            degree = rand(rng, degree_choices)
            parts = _random_composition(rng, node_count - 1, degree)
            parts === nothing && continue
            children = DimensionCandidate[]
            valid_parts = true
            for part in parts
                isempty(by_size[part]) && (valid_parts = false; break)
                push!(children, rand(rng, by_size[part]))
            end
            valid_parts || continue
            op_index = rand(rng, 1:options.nops[degree])
            candidate = _make_operator_candidate(
                options.operators.ops[degree][op_index],
                op_index,
                children,
                qtype,
                T,
                options.node_type,
                dimensionless,
            )
            candidate === nothing && continue
            push!(by_size[node_count], candidate)
            if policy === :strict && candidate.output_dimension == dimension(dataset.y_dimensions)
                return candidate.tree
            end
        end
    end

    if policy === :strict
        target = dimension(dataset.y_dimensions)
        for states in by_size, candidate in states
            candidate.output_dimension == target && return candidate.tree
        end
        return nothing
    end

    available = DimensionCandidate[candidate for states in by_size for candidate in states]
    isempty(available) && return nothing
    shuffle!(rng, available)
    return first(available).tree
end

end # module DimensionGenerationModule
