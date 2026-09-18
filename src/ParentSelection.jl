module ParentSelectionModule

using Random: AbstractRNG, default_rng, rand, randperm
using Statistics: median
using DispatchDoctor: @unstable
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

function _afp_worse_index(members, active::Vector{Int})
    worst = first(active)
    worst_dominance = -1
    for candidate in active
        dominance_count = count(
            other -> other != candidate && _dominates(members[other], members[candidate]),
            active,
        )
        candidate_cost = _afp_cost(members[candidate])
        worst_cost = _afp_cost(members[worst])
        should_replace = dominance_count > worst_dominance ||
            (dominance_count == worst_dominance && candidate_cost > worst_cost) ||
            (dominance_count == worst_dominance && candidate_cost == worst_cost &&
             members[candidate].birth < members[worst].birth) ||
            (dominance_count == worst_dominance && candidate_cost == worst_cost &&
             members[candidate].birth == members[worst].birth && candidate > worst)
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
function age_fitness_pareto_survivor_indices(members::AbstractVector, capacity::Int)
    0 <= capacity <= length(members) ||
        throw(ArgumentError("AFP capacity must be between zero and pool size"))
    active = collect(eachindex(members))
    while length(active) > capacity
        deleteat!(active, findfirst(==(_afp_worse_index(members, active)), active))
    end
    return active
end

end
