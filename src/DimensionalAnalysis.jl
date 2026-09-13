module DimensionalAnalysisModule

import DynamicQuantities
using DynamicExpressions:
    AbstractExpression,
    AbstractExpressionNode,
    constructorof,
    get_tree,
    get_child,
    with_contents
using DynamicQuantities: Quantity, DimensionError, AbstractQuantity

using ..CoreModule: AbstractOptions, Dataset, dimension_policy
import DynamicQuantities: dimension, ustrip
import ..CoreModule.OperatorsModule: safe_pow, safe_sqrt

function safe_sqrt(x::Q) where {T,Q<:AbstractQuantity{T}}
    ustrip(x) < 0 && return sqrt(abs(x)) * T(NaN)
    return sqrt(x)
end

"""Structured result of a static dimensional validation."""
struct DimensionCheckResult
    valid::Bool
    output_dimension
    internal_violation::Bool
    reason::Symbol
end

"""
    dimensional_scale_operator_index(options)

Return the binary multiplication operator index used by the semi-theoretical
outer scale. The scale is represented as a normal fitted scalar node at the
root, so it remains compatible with DynamicExpressions and existing constant
optimization/serialization code.
"""
function dimensional_scale_operator_index(options::AbstractOptions)
    idx = findfirst(
        op -> op === (*) || lowercase(string(op)) in ("*", "mult", "multiply"),
        options.operators.binops,
    )
    idx === nothing &&
        throw(ArgumentError(
            "formula_type=:semi_theoretical requires multiplication (*) in binary_operators " *
            "to represent the outer coefficient C_dim.",
        ))
    return idx
end

function is_dimensional_scale_wrapper(
    tree::AbstractExpressionNode, options::AbstractOptions
)
    policy = dimension_policy(options)
    policy === :compatible || return false
    tree.degree == 2 || return false
    tree.op == dimensional_scale_operator_index(options) || return false
    left, right = get_child(tree, 1), get_child(tree, 2)
    return left.degree == 0 && left.constant && right.degree >= 0
end

"""
    wrap_dimensional_scale(tree, options)

Wrap an internal semi-theoretical tree in one fitted scalar coefficient.
The coefficient is initialized to one and is optimized by the normal constant
optimization path. Existing wrappers are preserved, making the operation
idempotent across population, mutation and crossover boundaries.
"""
function dimensional_scale_coefficient(
    tree::AbstractExpressionNode, options::AbstractOptions
)
    is_dimensional_scale_wrapper(tree, options) || return nothing
    return get_child(tree, 1).val
end

function dimensional_scale_coefficient(ex::AbstractExpression, options::AbstractOptions)
    dimension_policy(options) === :compatible || return nothing
    return dimensional_scale_coefficient(get_tree(ex), options)
end

"""Return the multiplicative identity for a value type when it exists."""
dimensional_scale_identity(::Type{T}) where {T} = applicable(one, T) ? one(T) : nothing

function wrap_dimensional_scale(
    tree::AbstractExpressionNode{T}, options::AbstractOptions;
    coefficient=nothing,
) where {T}
    dimension_policy(options) === :compatible || return tree
    is_dimensional_scale_wrapper(tree, options) && return tree
    # TypeSpec searches may use non-numeric value types (for example strings).
    # A semi-theoretical scale is meaningful only when the value type supports
    # a multiplicative identity; leave such trees unchanged instead of raising
    # an opaque `one(::Type{T})` MethodError during population initialization.
    coefficient_value = if coefficient === nothing
        identity = dimensional_scale_identity(T)
        identity === nothing && return tree
        identity
    else
        convert(T, coefficient)
    end
    mult_idx = dimensional_scale_operator_index(options)
    coefficient_node = constructorof(typeof(tree))(; val=coefficient_value)
    return constructorof(typeof(tree))(;
        op=mult_idx,
        children=(coefficient_node, tree),
    )
end

function wrap_dimensional_scale(
    ex::AbstractExpression, options::AbstractOptions;
    coefficient=nothing,
)
    dimension_policy(options) === :compatible || return ex
    tree = get_tree(ex)
    wrapped = coefficient === nothing ?
        wrap_dimensional_scale(tree, options) :
        wrap_dimensional_scale(tree, options; coefficient=coefficient)
    return with_contents(ex, wrapped)
end

"""
    rewrap_dimensional_scale(tree, options; coefficient)

Restore the protected semi-theoretical outer coefficient after an operation
that may have returned an already wrapped expression. Mutation helpers from
DynamicExpressions can preserve the wrapper node when copying an expression,
so calling `wrap_dimensional_scale` directly is not sufficient: its idempotent
behavior would keep a mutated outer coefficient. This helper always removes
one existing wrapper and rebuilds it with the supplied coefficient.
"""
function rewrap_dimensional_scale(
    tree::AbstractExpressionNode,
    options::AbstractOptions;
    coefficient,
)
    return wrap_dimensional_scale(
        unwrap_dimensional_scale(tree, options),
        options;
        coefficient,
    )
end

function rewrap_dimensional_scale(
    ex::AbstractExpression,
    options::AbstractOptions;
    coefficient,
)
    return wrap_dimensional_scale(
        unwrap_dimensional_scale(ex, options),
        options;
        coefficient,
    )
end

"""Remove the protected outer C_dim node before internal tree operators run."""
function unwrap_dimensional_scale(
    tree::AbstractExpressionNode, options::AbstractOptions
)
    return is_dimensional_scale_wrapper(tree, options) ? get_child(tree, 2) : tree
end

function unwrap_dimensional_scale(ex::AbstractExpression, options::AbstractOptions)
    return dimension_policy(options) === :compatible ?
        with_contents(ex, unwrap_dimensional_scale(get_tree(ex), options)) : ex
end

"""Return the internal f(X;θ) tree represented by a semi-theoretical member."""
function internal_dimensional_tree(tree::AbstractExpressionNode, options::AbstractOptions)
    return unwrap_dimensional_scale(tree, options)
end

"""
    violates_dimensional_constraints(tree::AbstractExpressionNode, dataset::Dataset, options::AbstractOptions)

Checks whether an expression violates dimensional constraints.
"""
function violates_dimensional_constraints(
    tree::AbstractExpressionNode, dataset::Dataset, options::AbstractOptions
)
    return !infer_dimension_static(tree, dataset, options).valid
end
function violates_dimensional_constraints(
    tree::AbstractExpression, dataset::Dataset, options::AbstractOptions
)
    return violates_dimensional_constraints(get_tree(tree), dataset, options)
end
function violates_dimensional_constraints(
    tree::AbstractExpressionNode{T},
    X_dimensions::AbstractVector{<:Quantity},
    y_dimensions::Union{Quantity,Nothing},
    x::AbstractVector{T},
    options::AbstractOptions,
) where {T}
    policy = dimension_policy(options)
    policy === :ignore && return false
    input_dimensions = [dimension(item) for item in X_dimensions]
    output_dimension = _infer_dimension_static(tree, input_dimensions, options, T)
    output_dimension === nothing && return true
    return y_dimensions !== nothing && output_dimension != dimension(y_dimensions)
end
function violates_dimensional_constraints(
    ::AbstractExpressionNode{T},
    ::Nothing,
    ::Quantity,
    ::AbstractVector{T},
    ::AbstractOptions,
) where {T}
    return true
end

"""Infer an expression dimension using dimension-only quantity placeholders."""
function _transition_dimension(op, nodes, child_dimensions, ::Type{T}) where {T}
    name = lowercase(string(op))
    identity = dimensional_scale_identity(T)
    identity === nothing && return nothing
    # The common built-in operators only need dimension algebra.  Avoid
    # constructing temporary Quantity values here: this function is called for
    # every node during generation-time constraint checks.  Unknown/custom
    # operators still use the Quantity fallback below so their existing
    # semantics are unchanged.
    zero_dimension = dimension(1)
    if length(child_dimensions) == 2
        left, right = child_dimensions
        if name in ("+", "-", "plus", "sub", "mod")
            return left == right ? left : nothing
        elseif name in ("*", "×", "mult", "multiply")
            return left * right
        elseif name in ("/", "÷")
            return left / right
        elseif name in ("^", "pow", "safe_pow")
            # Only integer exponents are safe to apply directly to dimensions.
            # Other exponents continue through the Quantity fallback below.
            child = nodes[2]
            if child.constant && child_dimensions[2] == zero_dimension && child.val isa Integer
                return left ^ child.val
            end
        end
    elseif length(child_dimensions) == 1
        child = child_dimensions[1]
        if name in ("neg", "-", "abs", "relu", "round", "floor", "ceil")
            return child
        elseif name == "inv"
            return inv(child)
        elseif name in ("sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "exp", "log")
            return child == zero_dimension ? zero_dimension : nothing
        end
    end
    quantities = [DynamicQuantities.constructorof(
        Quantity{T,typeof(first(child_dimensions))}
    )(identity, d) for d in child_dimensions]
    try
        result = length(quantities) == 1 ? op(quantities[1]) : op(quantities...)
        return result isa AbstractQuantity ? dimension(result) : zero_dimension
    catch error
        error isa DimensionError || error isa MethodError || rethrow(error)
        return nothing
    end
end

function _infer_dimension_static(tree::AbstractExpressionNode, input_dimensions, options::AbstractOptions, ::Type{T}) where {T}
    if tree.degree == 0
        return tree.constant ? dimension(input_dimensions[1] / input_dimensions[1]) :
            (1 <= tree.feature <= length(input_dimensions) ? input_dimensions[tree.feature] : nothing)
    end
    children = [get_child(tree, i) for i in 1:tree.degree]
    child_dimensions = [_infer_dimension_static(child, input_dimensions, options, T) for child in children]
    any(isnothing, child_dimensions) && return nothing
    op = options.operators.ops[tree.degree][tree.op]
    return _transition_dimension(op, children, child_dimensions, T)
end
"""Infer dimensions without reading a data sample.

The previous implementation used the first numeric row while checking a tree.
That is unsuitable for generation-time rejection because a zero, negative, or
non-finite first sample can make a structurally valid expression look invalid.
This routine evaluates the dimension propagation with dimension-valued placeholders.
"""
function infer_dimension_static(
    tree::AbstractExpressionNode{T}, dataset::Dataset, options::AbstractOptions;
    scope::Symbol=:full,
) where {T}
    policy = dimension_policy(options)
    policy === :ignore &&
        return DimensionCheckResult(true, nothing, false, :ignored)
    if policy === :compatible && scope === :full
        # The public semi-theoretical tree is C_dim * f(X; θ). Validate the
        # internal expression and let the fitted outer coefficient supply the
        # remaining output dimension.
        tree = unwrap_dimensional_scale(tree, options)
        scope = :internal
    end
    X_dimensions = dataset.X_dimensions
    y_dimensions = dataset.y_dimensions
    X_dimensions === nothing &&
        return DimensionCheckResult(false, nothing, true, :missing_input_dimensions)
    if scope === :full && y_dimensions === nothing
        return DimensionCheckResult(false, nothing, false, :missing_output_dimension)
    end
    input_dimensions = [dimension(dimension_value) for dimension_value in X_dimensions]
    dimensional_output = _infer_dimension_static(tree, input_dimensions, options, T)
    dimensional_output === nothing &&
        return DimensionCheckResult(false, nothing, true, :dimension_constraint)
    if scope === :full && dimensional_output != dimension(y_dimensions)
        return DimensionCheckResult(
            false,
            dimensional_output,
            false,
            :output_dimension_mismatch,
        )
    end
    return DimensionCheckResult(
        true,
        dimensional_output,
        false,
        :ok,
    )
end

infer_dimension_static(tree::AbstractExpression, dataset::Dataset, options::AbstractOptions; kws...) =
    infer_dimension_static(get_tree(tree), dataset, options; kws...)

"""Validate a candidate according to the active formula type."""
function validate_search_candidate(
    tree::Union{AbstractExpression,AbstractExpressionNode},
    dataset::Dataset,
    options::AbstractOptions;
    scope::Symbol=:full,
)
    return infer_dimension_static(tree, dataset, options; scope).valid
end
end
