module RegularizedEvolutionModule

using Random: AbstractRNG, default_rng, rand
using ..CoreModule:
    AbstractOptions,
    Dataset,
    MaybeTrace,
    DATA_TYPE,
    LOSS_TYPE,
    MutationStepResult,
    wrap_mutation_step
using ..PopulationModule: Population, best_of_sample
using ..ParentSelectionModule:
    age_fitness_pareto_survivor_indices,
    competitive_survivor_indices,
    make_parent_selection_context
using ..HallOfFameModule: HallOfFame, update_hall_of_fame!, _update_hall_of_fame_unchecked!
using ..ComplexityModule: compute_complexity
using ..MutateModule: next_generation
using ..CrossoverModule: crossover_generation
using ..SurrogateModule: SurrogateState
using ..TracingModule:
    new_trace,
    new_step_trace,
    new_traced_steps,
    reset_traced_steps!,
    trace_crossover!,
    trace_mutation_attempts!,
    trace_mutation_step!
using ..UtilsModule: argmin_fast, strictmap

"""
One precomposed mutation-middleware layer.
"""
struct MutationStepLayer{W,F}
    wrapper::W
    next_step::F
end
@inline function (layer::MutationStepLayer)(parent)
    return layer.wrapper(parent, layer.next_step)
end

build_mutation_step(::Tuple{}, base_step) = base_step
function build_mutation_step(wrappers::Tuple, base_step::F) where {F}
    inner = build_mutation_step(Base.tail(wrappers), base_step)
    return _add_mutation_step_layer(first(wrappers), inner)
end
_add_mutation_step_layer(::Nothing, inner) = inner
function _add_mutation_step_layer(wrapper, inner)  # COV_EXCL_LINE
    return MutationStepLayer(wrapper, inner)
end

"""
Engine-owned state for one mutation step. Mutable contents accumulate every
middleware attempt so evaluation counts, Hall-of-Fame updates, and tracing
stay under engine control.
"""
struct MutationStep{D,P,O,S,E,H,A,M,R,G}
    dataset::D
    population::P
    curmaxsize::Int
    options::O
    plugin_states::S
    eval_context::E
    best_seen::H
    attempted_results::A
    attempted_members::M
    traced_steps::R
    surrogate_state::Union{Nothing,SurrogateState}
    rng::G
end

function (step::MutationStep)(parent)
    step_trace = new_step_trace(step.traced_steps)
    member, accepted, num_evals = next_generation(
        step.dataset,
        parent,
        step.curmaxsize,
        step.options;
        tmp_trace=step_trace,
        plugin_states=step.plugin_states,
        eval_context=step.eval_context,
        population_for_backsolve=step.population,
        surrogate_state=step.surrogate_state,
        rng=step.rng,
    )
    attempt_id = isnothing(step.attempted_results) ? 1 : length(step.attempted_results) + 1
    result = MutationStepResult(member, accepted, attempt_id, num_evals)
    !isnothing(step.attempted_results) && push!(step.attempted_results, result)
    !isnothing(step.attempted_members) && push!(step.attempted_members, copy(member))
    trace_mutation_step!(step.traced_steps, parent, member, step_trace)
    accepted &&
        !isnothing(step.attempted_members) &&
        update_hall_of_fame!(step.best_seen, member, step.dataset, step.options)
    return result
end

function reset!(step::MutationStep)
    !isnothing(step.attempted_results) && empty!(step.attempted_results)
    !isnothing(step.attempted_members) && empty!(step.attempted_members)
    reset_traced_steps!(step.traced_steps)
    return nothing
end

function _replace_regularized!(pop, babies)
    n_babies = length(babies)
    n_babies == 0 && return Int[]
    n_babies <= pop.n || throw(ArgumentError("cannot insert more babies than population capacity"))
    birth_order = [pop.members[member].birth for member in 1:pop.n]
    slots = Int[]
    for _ in 1:n_babies
        slot = argmin_fast([
            i in slots ? typemax(eltype(birth_order)) : birth_order[i] for i in 1:pop.n
        ])
        push!(slots, slot)
    end
    for (slot, baby) in zip(slots, babies)
        pop.members[slot] = baby
    end
    return slots
end

function _replace_age_fitness_pareto!(pop, babies)
    n_babies = length(babies)
    n_babies == 0 && return Int[]
    n_babies <= pop.n || throw(ArgumentError("cannot insert more babies than population capacity"))

    old_members = copy(pop.members)
    candidates = vcat(old_members, collect(babies))
    survivors = age_fitness_pareto_survivor_indices(candidates, pop.n)
    survivor_set = Set(survivors)
    removed_slots = Int[i for i in 1:pop.n if !(i in survivor_set)]
    surviving_babies = Int[
        j for j in 1:n_babies if pop.n + j in survivor_set
    ]
    replacement_slots = fill(0, n_babies)
    for (slot, baby_index) in zip(removed_slots, surviving_babies)
        replacement_slots[baby_index] = slot
    end
    survivor_members = candidates[survivors]
    for i in 1:pop.n
        pop.members[i] = survivor_members[i]
    end
    return replacement_slots
end

function _replace_competitive!(pop, babies, parent_refs)
    n_babies = length(babies)
    n_babies == 0 && return Int[]
    n_babies <= pop.n || throw(ArgumentError("cannot insert more babies than population capacity"))

    old_members = copy(pop.members)
    candidates = vcat(old_members, collect(babies))
    survivors = competitive_survivor_indices(old_members, babies, parent_refs, pop.n)
    survivor_set = Set(survivors)
    removed_slots = Int[i for i in 1:pop.n if !(i in survivor_set)]
    surviving_babies = Int[
        j for j in 1:n_babies if pop.n + j in survivor_set
    ]
    replacement_slots = fill(0, n_babies)
    for (slot, baby_index) in zip(removed_slots, surviving_babies)
        replacement_slots[baby_index] = slot
    end

    survivor_members = candidates[survivors]
    for i in 1:pop.n
        pop.members[i] = survivor_members[i]
    end
    return replacement_slots
end

function _replace_with_survival!(pop, babies, options::AbstractOptions; parent_refs=nothing)
    if options.survival_strategy === :competitive_age_fitness
        return _replace_competitive!(pop, babies, parent_refs)
    end
    if options.survival_strategy === :age_fitness_pareto
        return _replace_age_fitness_pareto!(pop, babies)
    end
    return _replace_regularized!(pop, babies)
end

# Pass through the population several times, replacing the oldest
# with the fittest of a small subsample
function reg_evol_cycle(
    dataset::Dataset{T,L},
    pop::P,
    curmaxsize::Int,
    options::AbstractOptions,
    trace::MaybeTrace;
    plugin_states::Tuple,
    best_seen::HallOfFame,
    eval_context=nothing,
    surrogate_state::Union{Nothing,SurrogateState}=nothing,
    rng::AbstractRNG=default_rng(),
)::Tuple{P,Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:Population{T,L}}
    num_evals = 0.0
    n_evol_cycles = ceil(Int, pop.n / options.tournament_selection_n)
    selection_context = if options.parent_selection === :epsilon_lexicase
        make_parent_selection_context(dataset, options; rng)
    else
        nothing
    end
    mutation_wrappers = strictmap(wrap_mutation_step, plugin_states, options.plugins)
    traced_steps = new_traced_steps(trace, eltype(pop.members))
    has_mutation_wrappers = any(!isnothing, mutation_wrappers)
    attempted_results =
        has_mutation_wrappers ? MutationStepResult{eltype(pop.members)}[] : nothing
    attempted_members = has_mutation_wrappers ? eltype(pop.members)[] : nothing
    base_step = MutationStep(
        dataset,
        pop,
        curmaxsize,
        options,
        plugin_states,
        eval_context,
        best_seen,
        attempted_results,
        attempted_members,
        traced_steps,
        surrogate_state,
        rng,
    )
    wrapped_step = build_mutation_step(mutation_wrappers, base_step)

    for i in 1:n_evol_cycles
        if rand(rng) > options.crossover_probability
            allstar = best_of_sample(
                pop,
                options;
                plugin_states,
                dataset,
                selection_context,
                rng,
            )
            reset!(base_step)
            result = wrapped_step(allstar)
            selected_attempt_idx = result.attempt_id
            selected_result = if isnothing(base_step.attempted_results)
                num_evals += result.num_evals
                result
            else
                checkbounds(Bool, base_step.attempted_results, selected_attempt_idx) ||
                    throw(
                        ArgumentError(
                            "Mutation middleware must return a result from `next_step`."
                        ),
                    )
                num_evals += sum(attempt -> attempt.num_evals, base_step.attempted_results)
                base_step.attempted_results[selected_attempt_idx]
            end
            baby = if isnothing(base_step.attempted_members)
                selected_result.member
            else
                base_step.attempted_members[selected_attempt_idx]
            end
            mutation_accepted = selected_result.accepted

            should_replace = mutation_accepted || !options.skip_mutation_failures
            old_members = should_replace ? copy(pop.members) : nothing
            replacement_slots = should_replace ?
                _replace_with_survival!(
                    pop, [baby], options; parent_refs=[allstar.ref]
                ) : Int[]
            replacement_slot = isempty(replacement_slots) ? 0 : first(replacement_slots)

            trace_mutation_attempts!(
                trace,
                traced_steps,
                pop,
                replacement_slot,
                replacement_slot != 0,
                selected_attempt_idx,
                options
                ; oldest_member=(replacement_slot > 0 ? old_members[replacement_slot] : nothing),
            )

            should_replace || continue

        else # Crossover
            allstar1 = best_of_sample(
                pop,
                options;
                plugin_states,
                dataset,
                selection_context,
                rng,
            )
            allstar2 = best_of_sample(
                pop,
                options;
                plugin_states,
                dataset,
                selection_context,
                rng,
            )

            crossover_trace = new_trace(trace)
            baby1, baby2, crossover_accepted, tmp_num_evals = crossover_generation(
                allstar1,
                allstar2,
                dataset,
                curmaxsize,
                options;
                trace=crossover_trace,
                plugin_states,
                eval_context,
                surrogate_state=surrogate_state,
                rng,
            )
            num_evals += tmp_num_evals
            if crossover_accepted
                update_hall_of_fame!(best_seen, baby1, dataset, options)
                update_hall_of_fame!(best_seen, baby2, dataset, options)
            end

            if !crossover_accepted && options.skip_mutation_failures
                continue
            end

            old_members = copy(pop.members)
            replacement_slots = _replace_with_survival!(
                pop,
                [baby1, baby2],
                options;
                parent_refs=[allstar1.ref, allstar2.ref],
            )
            oldest1 = replacement_slots[1]
            oldest2 = replacement_slots[2]

            trace_crossover!(
                trace,
                allstar1,
                allstar2,
                baby1,
                baby2,
                pop,
                oldest1,
                oldest2,
                crossover_trace,
                options
                ; oldest_member1=(oldest1 > 0 ? old_members[oldest1] : nothing),
                oldest_member2=(oldest2 > 0 ? old_members[oldest2] : nothing),
            )
        end
    end

    return (pop, num_evals)
end

end
