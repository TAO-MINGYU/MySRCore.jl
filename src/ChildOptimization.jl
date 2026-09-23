module ChildOptimizationModule

using Random: AbstractRNG, default_rng
using DynamicExpressions:
    AbstractExpression,
    AbstractExpressionNode,
    copy_node,
    get_tree,
    get_child,
    with_contents,
    count_scalar_constants,
    simplify_tree!,
    combine_operators,
    set_child!

using ..CoreModule:
    AbstractOptions,
    Dataset,
    DATA_TYPE,
    LOSS_TYPE,
    dataset_fraction,
    plus,
    sub,
    mult,
    neg
using ..CheckConstraintsModule: check_constraints
using ..ComplexityModule: compute_complexity
using ..ConstantOptimizationModule: bounded_optimizer_options, optimize_constants
using ..DimensionalAnalysisModule:
    dimensional_scale_coefficient,
    unwrap_dimensional_scale,
    wrap_dimensional_scale
using ..LossFunctionsModule: eval_cost
using ..PopMemberModule: AbstractPopMember, create_child

"""Return whether a node is a scalar constant leaf."""
@inline _is_constant(node::AbstractExpressionNode) = node.degree == 0 && node.constant

@inline function _is_zero_constant(node::AbstractExpressionNode)
    return _is_constant(node) && iszero(node.val)
end

@inline function _is_one_constant(node::AbstractExpressionNode)
    return _is_constant(node) && node.val == one(node.val)
end

@inline _is_plus(op) = op === (+) || op === plus
@inline _is_sub(op) = op === (-) || op === sub
@inline _is_mult(op) = op === (*) || op === mult
@inline _is_neg(op) = op === (-) || op === neg

@inline function _child_refinement_mode(options::AbstractOptions)
    try
        mode = getproperty(options, :child_refinement)
        return mode isa Symbol ? mode : :none
    catch
        return :none
    end
end

"""Apply a small set of exact, domain-safe neutral-element rewrites.

The rewrite deliberately excludes identities such as `x * 0` and `x ^ 0`.
Those identities are not safe for protected operators, NaNs, infinities, or
out-of-domain values.  The returned tree is a new subtree, so the caller can
compare it against the unmodified child and retain the original whenever the
finite-data objective does not improve.
"""
function _neutral_simplify(node::AbstractExpressionNode, operators)
    node.degree == 0 && return node
    for i in 1:node.degree
        set_child!(node, _neutral_simplify(get_child(node, i), operators), i)
    end

    if node.degree == 1
        op = operators.unaops[node.op]
        child = get_child(node, 1)
        if _is_neg(op) && child.degree == 1 && _is_neg(operators.unaops[child.op])
            return copy_node(get_child(child, 1))
        end
        return node
    end

    node.degree == 2 || return node
    op = operators.binops[node.op]
    left, right = get_child(node, 1), get_child(node, 2)
    if _is_plus(op)
        _is_zero_constant(right) && return copy_node(left)
        _is_zero_constant(left) && return copy_node(right)
    elseif _is_sub(op)
        _is_zero_constant(right) && return copy_node(left)
    elseif _is_mult(op)
        _is_one_constant(right) && return copy_node(left)
        _is_one_constant(left) && return copy_node(right)
    elseif op === (/)
        _is_one_constant(right) && return copy_node(left)
    end
    return node
end

function _simplified_tree(tree::AbstractExpression, options::AbstractOptions)
    coefficient = dimensional_scale_coefficient(tree, options)
    inner_expression = unwrap_dimensional_scale(tree, options)
    inner = copy_node(get_tree(inner_expression))
    inner = simplify_tree!(inner, options.operators)
    inner = combine_operators(inner, options.operators)
    inner = _neutral_simplify(inner, options.operators)
    wrapped = coefficient === nothing ?
        wrap_dimensional_scale(inner, options) :
        wrap_dimensional_scale(inner, options; coefficient=coefficient)
    return with_contents(tree, wrapped)
end

"""Return a copied child expression after exact algebraic simplification."""
simplify_child_tree(tree::AbstractExpression, options::AbstractOptions) =
    _simplified_tree(tree, options)

function _same_node(left::AbstractExpressionNode, right::AbstractExpressionNode)
    left.degree == right.degree || return false
    left.constant == right.constant || return false
    isequal(left.val, right.val) || return false
    left.feature == right.feature || return false
    left.degree == 0 && return true
    left.op == right.op || return false
    for i in 1:left.degree
        _same_node(get_child(left, i), get_child(right, i)) || return false
    end
    return true
end

function _same_expression(left::AbstractExpression, right::AbstractExpression)
    try
        return _same_node(get_tree(left), get_tree(right))
    catch
        return false
    end
end

function _candidate_is_better(candidate, best)
    candidate.cost isa AbstractFloat && !isfinite(candidate.cost) && return false
    best.cost isa AbstractFloat && isnan(best.cost) && return true
    candidate.cost < best.cost && return true
    candidate.cost == best.cost || return false
    return getfield(candidate, :complexity) < getfield(best, :complexity)
end

function _optimize_child_member(
    dataset::Dataset,
    member::P,
    options::AbstractOptions,
    rng::AbstractRNG,
    eval_context,
) where {T,L,N,P<:AbstractPopMember{T,L,N}}
    options.should_optimize_constants || return member, 0.0
    mode = _child_refinement_mode(options)
    mode === :none && return member, 0.0
    mode in (:safe, :thorough) || return member, 0.0
    dimensional_coefficient = dimensional_scale_coefficient(member.tree, options)
    if dimensional_coefficient !== nothing &&
       count_scalar_constants(unwrap_dimensional_scale(member.tree, options)) == 0
        return member, 0.0
    end
    original_tree = copy(member.tree)
    original_cost = member.cost
    original_loss = member.loss
    original_birth = getfield(member, :birth)
    original_complexity = getfield(member, :complexity)
    restore_original!() = begin
        setfield!(member, :tree, original_tree)
        setfield!(member, :cost, original_cost)
        setfield!(member, :loss, original_loss)
        setfield!(member, :birth, original_birth)
        setfield!(member, :complexity, original_complexity)
        member
    end
    try
        optimized, num_evals = if mode === :safe
            bounded = bounded_optimizer_options(options; iterations=4, f_calls_limit=64)
            optimize_constants(
                dataset,
                member,
                options;
                rng=rng,
                optimizer_options_override=bounded,
                optimizer_nrestarts_override=0,
            )
        else
            optimize_constants(dataset, member, options; rng=rng)
        end
        # Child refinement is part of evaluating one offspring, not a new
        # evolutionary generation.  The shared constant optimizer refreshes
        # `birth` when a fit improves the member; preserve the child lineage
        # timestamp here so age-aware survival remains deterministic and does
        # not reward an internal refinement step as an extra generation.
        setfield!(optimized, :birth, original_birth)
        dimensional_coefficient === nothing && return optimized, num_evals

        # Refit candidates may include the protected semi-theoretical outer
        # coefficient. Restore its lineage value and keep the fitted inner
        # constants only when the restored expression is still an improvement.
        restored_tree = wrap_dimensional_scale(
            unwrap_dimensional_scale(optimized.tree, options),
            options;
            coefficient=dimensional_coefficient,
        )
        restored_complexity = compute_complexity(restored_tree, options)
        restored_cost, restored_loss = eval_cost(
            dataset,
            restored_tree,
            options;
            complexity=restored_complexity,
            eval_context=eval_context,
        )
        num_evals += dataset_fraction(dataset)
        if isfinite(restored_cost) && restored_cost <= original_cost
            setfield!(optimized, :tree, restored_tree)
            setfield!(optimized, :cost, restored_cost)
            setfield!(optimized, :loss, restored_loss)
            setfield!(optimized, :complexity, restored_complexity)
            setfield!(optimized, :birth, original_birth)
            return optimized, num_evals
        end
        return restore_original!(), num_evals
    catch
        # A single invalid child must not abort an otherwise valid evolutionary
        # run.  The caller still has the evaluated, unoptimized member as a
        # safe fallback.
        return restore_original!(), 0.0
    end
end

"""
    refine_child(dataset, parent, tree, cost, loss, options; ...)

Refine one newly generated child before the evolutionary acceptance decision.
The raw child is always retained as a fallback.  A bounded constant fit is
performed first, then a copy receives exact constant folding, operator
combination, and neutral-element rewrites.  A simplification is accepted only
when its fully reevaluated cost is no worse, so bloat is removed without
discarding a structure merely because a rewrite changed the finite-sample
semantics.

The returned evaluation count includes the simplified candidate evaluation and
all optimizer calls.  The raw candidate evaluation is owned by the caller.
"""
function refine_child(
    dataset::Dataset{T,L},
    parent::P,
    tree::N,
    cost::L,
    loss::L,
    options::AbstractOptions;
    complexity::Int=compute_complexity(tree, options),
    curmaxsize::Int=options.maxsize,
    parent_ref::Int=parent.ref,
    eval_context=nothing,
    rng::AbstractRNG=default_rng(),
)::Tuple{P,Float64} where {
    T<:DATA_TYPE,
    L<:LOSS_TYPE,
    N<:AbstractExpression{T},
    P<:AbstractPopMember{T,L,N},
}
    raw_member = create_child(
        parent,
        copy(tree),
        cost,
        loss,
        options;
        complexity=complexity,
        parent_ref=parent_ref,
    )::P
    _child_refinement_mode(options) === :none && return raw_member, 0.0

    best_member, num_evals = _optimize_child_member(
        dataset, raw_member, options, rng, eval_context
    )
    best_complexity = compute_complexity(best_member, options)

    if options.should_simplify
        simplified_tree = try
            _simplified_tree(tree, options)
        catch
            nothing
        end
        if simplified_tree !== nothing && !_same_expression(simplified_tree, tree)
            simplified_complexity = compute_complexity(simplified_tree, options)
            if simplified_complexity <= curmaxsize &&
               check_constraints(
                   simplified_tree,
                   dataset,
                   options,
                   curmaxsize,
                   simplified_complexity,
               )
                simplified_cost, simplified_loss = eval_cost(
                    dataset,
                    simplified_tree,
                    options;
                    complexity=simplified_complexity,
                    eval_context=eval_context,
                )
                num_evals += dataset_fraction(dataset)
                simplified_member = create_child(
                    parent,
                    simplified_tree,
                    simplified_cost,
                    simplified_loss,
                    options;
                    complexity=simplified_complexity,
                    parent_ref=parent_ref,
                )::P
                simplified_member, optimizer_evals = _optimize_child_member(
                    dataset, simplified_member, options, rng, eval_context
                )
                num_evals += optimizer_evals
                if _candidate_is_better(simplified_member, best_member)
                    best_member = simplified_member
                    best_complexity = simplified_complexity
                end
            end
        end
    end

    # `best_member` may have been returned by a custom optimizer with a stale
    # cached complexity.  The tree itself is authoritative for the final child.
    getfield(best_member, :complexity) == best_complexity ||
        setfield!(best_member, :complexity, best_complexity)
    return best_member, num_evals
end

end
