module ParentSelectionModule

using Random: AbstractRNG, default_rng, rand, randperm
using Statistics: median
using DispatchDoctor: @unstable
using DynamicExpressions: AbstractExpression, AbstractExpressionNode, get_child, get_tree
using LossFunctions: SupervisedLoss

using ..CoreModule:
    AbstractOptions,
    Dataset,
    DATA_TYPE,
    LOSS_TYPE,
    get_full_dataset,
    is_weighted,
    tournament_cost_multiplier,
    use_batching
using ..LossFunctionsModule: eval_case_losses
using ..PopMemberModule: AbstractPopMember

"""State shared by all parent draws in one evolutionary cycle."""
mutable struct ParentSelectionContext{D,R}
    dataset::D
    rng::R
    case_losses::IdDict{Any,Any}
    fallback_reason::Union{Nothing,Symbol}
end

"""Explain whether the requested parent selector can run on a dataset."""
function parent_selection_diagnostic(
    options::AbstractOptions, dataset::Dataset
)
    requested = options.parent_selection
    requested === :epsilon_lexicase ||
        return (; requested, effective=:tournament, reason=:not_requested)
    use_batching(options, dataset) &&
        return (; requested, effective=:tournament, reason=:batching_enabled)
    !isnothing(options.loss_function) &&
        return (; requested, effective=:tournament, reason=:custom_aggregate_loss)
    !isnothing(options.loss_function_expression) &&
        return (; requested, effective=:tournament, reason=:custom_expression_loss)
    !(options.elementwise_loss isa SupervisedLoss) &&
        return (; requested, effective=:tournament, reason=:custom_elementwise_loss)
    return (; requested, effective=:epsilon_lexicase, reason=:supported)
end

"""Create the per-cycle cache used by epsilon-lexicase selection."""
function make_parent_selection_context(
    dataset::Dataset,
    options::AbstractOptions;
    rng::AbstractRNG=default_rng(),
)
    full_dataset = get_full_dataset(dataset)
    diagnostic = parent_selection_diagnostic(options, full_dataset)
    fallback_reason = diagnostic.effective === :tournament ? diagnostic.reason : nothing
    return ParentSelectionContext(full_dataset, rng, IdDict{Any,Any}(), fallback_reason)
end

@inline function _finite_or_inf(value)
    return isfinite(value) ? value : oftype(value, Inf)
end

function _case_losses!(context::ParentSelectionContext, member, options::AbstractOptions)
    return get!(context.case_losses, member) do
        eval_case_losses(member.tree, context.dataset, options)
    end
end

function _selection_multiplier(member, options::AbstractOptions, plugin_states::Tuple, ::Type{L}) where {L}
    multiplier = one(L)
    for (plugin, plugin_state) in zip(options.plugins, plugin_states)
        multiplier *= L(tournament_cost_multiplier(plugin_state, plugin, member, options))
    end
    return multiplier
end

@inline function _mad(values::AbstractVector{T}) where {T<:Real}
    finite_values = T[value for value in values if isfinite(value)]
    isempty(finite_values) && return T(Inf)
    center = median(finite_values)
    return max(zero(T), median(abs.(finite_values .- center)))
end

function _epsilon_lexicase_index(
    candidate_indices::Vector{Int},
    case_indices::Vector{Int},
    value_at::F,
    rng::AbstractRNG,
) where {F}
    isempty(candidate_indices) && throw(ArgumentError("epsilon-lexicase needs at least one candidate"))
    isempty(case_indices) && return rand(rng, candidate_indices)

    candidates = copy(candidate_indices)
    for case_index in case_indices[randperm(rng, length(case_indices))]
        values = [_finite_or_inf(value_at(candidate, case_index)) for candidate in candidates]
        finite_values = [value for value in values if isfinite(value)]
        isempty(finite_values) && continue
        best = minimum(finite_values)
        epsilon = _mad(values)
        threshold = best + epsilon
        kept = Int[
            candidate for (candidate, value) in zip(candidates, values) if value <= threshold
        ]
        isempty(kept) || (candidates = kept)
        length(candidates) == 1 && break
    end
    return rand(rng, candidates)
end

"""Pure epsilon-lexicase selector used by tests and algorithm diagnostics."""
function epsilon_lexicase_index(
    errors::AbstractMatrix{T}; rng::AbstractRNG=default_rng()
) where {T<:Real}
    n_candidates, n_cases = size(errors)
    n_candidates > 0 || throw(ArgumentError("errors must contain at least one candidate"))
    n_cases > 0 || return rand(rng, 1:n_candidates)
    candidates = collect(1:n_candidates)
    cases = collect(1:n_cases)
    return _epsilon_lexicase_index(
        candidates, cases, (candidate, case_index) -> errors[candidate, case_index], rng
    )
end

"""Select one parent from the full population using epsilon-lexicase."""
function epsilon_lexicase_parent(
    members::AbstractVector{P},
    dataset::Dataset{T,L},
    options::AbstractOptions;
    plugin_states::Tuple,
    context::ParentSelectionContext,
) where {T,L,N,P<:AbstractPopMember{T,L,N}}
    context.fallback_reason === nothing || return nothing
    isempty(members) && throw(ArgumentError("cannot select a parent from an empty population"))

    first_losses = _case_losses!(context, first(members), options)
    first_losses === nothing && return nothing
    n_cases = length(first_losses)
    case_indices = if is_weighted(dataset)
        Int[i for i in 1:n_cases if dataset.weights[i] > 0]
    else
        collect(1:n_cases)
    end
    isempty(case_indices) && return nothing

    candidate_indices = collect(eachindex(members))
    value_at = function (candidate, case_index)
        errors = _case_losses!(context, members[candidate], options)
        errors === nothing && return Inf
        multiplier = _selection_multiplier(
            members[candidate], options, plugin_states, eltype(errors)
        )
        return errors[case_index] * multiplier
    end
    selected_index = _epsilon_lexicase_index(
        candidate_indices, case_indices, value_at, context.rng
    )
    return members[selected_index]
end

@inline function _afp_cost(member)
    value = member.cost
    return isfinite(value) ? value : oftype(value, Inf)
end

@inline function _dominates(a, b)
    cost_a, cost_b = _afp_cost(a), _afp_cost(b)
    age_a, age_b = a.birth, b.birth
    return (cost_a <= cost_b && age_a >= age_b) &&
           (cost_a < cost_b || age_a > age_b)
end

function _afp_worse_index(
    members,
    active::Vector{Int};
    prefer_simple::Bool=false,
)
    worst = first(active)
    worst_dominance = -1
    for candidate in active
        dominance_count = count(
            other -> other != candidate && _dominates(members[other], members[candidate]),
            active,
        )
        candidate_cost = _afp_cost(members[candidate])
        worst_cost = _afp_cost(members[worst])
        same_rank = dominance_count == worst_dominance
        same_cost = candidate_cost == worst_cost
        same_age = members[candidate].birth == members[worst].birth
        candidate_complexity = _member_complexity(members[candidate])
        worst_complexity = _member_complexity(members[worst])
        complexity_worse = candidate_complexity > worst_complexity
        complexity_equal = candidate_complexity == worst_complexity
        should_replace = dominance_count > worst_dominance ||
            (same_rank && candidate_cost > worst_cost) ||
            (same_rank && same_cost && members[candidate].birth < members[worst].birth) ||
            (same_rank && same_cost && same_age &&
             ((prefer_simple && complexity_worse) ||
              ((!prefer_simple || complexity_equal) && candidate > worst)))
        if should_replace
            worst = candidate
            worst_dominance = dominance_count
        end
    end
    return worst
end

"""Return survivors from a parent-plus-offspring pool using AFP pressure.

`birth` is MySRCore's monotonic creation order.  Larger values are younger,
so the two Pareto objectives are lower scalar cost and greater recency.
"""
function age_fitness_pareto_survivor_indices(
    members::AbstractVector,
    capacity::Int;
    prefer_simple::Bool=false,
)
    0 <= capacity <= length(members) ||
        throw(ArgumentError("AFP capacity must be between zero and pool size"))
    active = collect(eachindex(members))
    while length(active) > capacity
        deleteat!(
            active,
            findfirst(
                ==(_afp_worse_index(members, active; prefer_simple=prefer_simple)),
                active,
            ),
        )
    end
    return active
end

@inline function _member_complexity(member)
    try
        return getfield(member, :complexity)
    catch
        return typemax(Int)
    end
end

@inline function _member_ref(member)
    try
        return getfield(member, :ref)
    catch
        return nothing
    end
end

"""Return a hash for the operator/feature shape of an expression tree."""
function _survival_structural_hash(tree::AbstractExpressionNode)
    value = hash(tree.degree)
    if tree.degree == 0
        value = hash(tree.constant, value)
        value = hash(tree.constant ? :constant : tree.feature, value)
        return value
    end
    value = hash(tree.op, value)
    for child_index in 1:tree.degree
        value = hash(_survival_structural_hash(get_child(tree, child_index)), value)
    end
    return value
end

_survival_structural_hash(expression::AbstractExpression) =
    _survival_structural_hash(get_tree(expression))

@inline function _survival_structural_hash(member)
    return _survival_structural_hash(member.tree)
end

@inline function _candidate_preferred(a, b, index_a::Int, index_b::Int)
    cost_a, cost_b = _afp_cost(a), _afp_cost(b)
    cost_a < cost_b && return true
    cost_a > cost_b && return false

    complexity_a = _member_complexity(a)
    complexity_b = _member_complexity(b)
    complexity_a < complexity_b && return true
    complexity_a > complexity_b && return false

    # Keep the existing member for an exact structural/cost tie.  This avoids
    # replacing a parent by a newly-created copy when a mutation failed.
    return index_a < index_b
end

function _child_wins_parent(child, parent)
    child_cost, parent_cost = _afp_cost(child), _afp_cost(parent)
    child_cost < parent_cost && return true
    child_cost > parent_cost && return false

    child_complexity = _member_complexity(child)
    parent_complexity = _member_complexity(parent)
    child_complexity < parent_complexity && return true
    child_complexity > parent_complexity && return false

    # An equal-cost copy of the parent is not an evolutionary improvement.
    return _survival_structural_hash(child) != _survival_structural_hash(parent)
end

"""
    competitive_survivor_indices(old_members, babies, parent_refs, capacity)

Build the parent-plus-eligible-offspring pool used by the competitive age/
fitness survival strategy.  Each child must first beat the parent identified by
its reference.  The eligible pool is then de-duplicated by operator/feature
shape and reduced with age-fitness Pareto survival.  `parent_refs === nothing`
disables the local gate and is useful for callers that do not retain lineage.

The returned indices address `vcat(old_members, babies)` and contain
`capacity` entries when the old population itself has at least that many
members.
"""
function competitive_survivor_indices(
    old_members::AbstractVector,
    babies::AbstractVector,
    parent_refs,
    capacity::Int,
)
    n_old = length(old_members)
    n_babies = length(babies)
    0 <= capacity <= n_old + n_babies ||
        throw(ArgumentError("competitive survival capacity must be between zero and pool size"))
    parent_refs === nothing || length(parent_refs) == n_babies ||
        throw(ArgumentError("parent_refs must contain one entry per baby"))

    candidates = vcat(old_members, collect(babies))
    eligible = collect(1:n_old)
    for baby_index in 1:n_babies
        include_baby = true
        if parent_refs !== nothing
            parent_ref = parent_refs[baby_index]
            parent_slot = findfirst(
                member -> _member_ref(member) == parent_ref,
                old_members,
            )
            if parent_slot !== nothing
                include_baby = _child_wins_parent(
                    candidates[n_old + baby_index], old_members[parent_slot]
                )
            end
        end
        include_baby && push!(eligible, n_old + baby_index)
    end

    capacity == 0 && return Int[]

    # Keep only the best representative of each structural shape before AFP.
    representatives = Dict{UInt,Int}()
    duplicate_indices = Int[]
    for candidate_index in eligible
        signature = _survival_structural_hash(candidates[candidate_index])
        representative = get(representatives, signature, 0)
        if representative == 0
            representatives[signature] = candidate_index
        elseif _candidate_preferred(
            candidates[candidate_index], candidates[representative], candidate_index, representative
        )
            push!(duplicate_indices, representative)
            representatives[signature] = candidate_index
        else
            push!(duplicate_indices, candidate_index)
        end
    end

    # Hash-table iteration order is not a search policy.  Restore candidate
    # order before AFP so deterministic searches do not depend on hash layout.
    representative_indices = sort!(collect(values(representatives)))
    if length(representative_indices) >= capacity
        representative_members = candidates[representative_indices]
        survivor_local = age_fitness_pareto_survivor_indices(
            representative_members, capacity; prefer_simple=true
        )
        return representative_indices[survivor_local]
    end

    # A small population may not contain enough unique shapes.  Fill the
    # remaining slots with the best duplicate representatives so population
    # size remains invariant.
    sort!(duplicate_indices; lt=(a, b) -> _candidate_preferred(
        candidates[a], candidates[b], a, b
    ))
    survivors = copy(representative_indices)
    for candidate_index in duplicate_indices
        length(survivors) >= capacity && break
        push!(survivors, candidate_index)
    end
    return survivors
end

end
