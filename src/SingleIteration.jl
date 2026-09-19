module SingleIterationModule

using ADTypes: AutoEnzyme
using DynamicExpressions: AbstractExpression, simplify_tree!, combine_operators
using ..UtilsModule: @threads_if, strictmap
using ..CoreModule:
    AbstractOptions,
    Dataset,
    MaybeTrace,
    create_expression,
    batch,
    get_batch_size,
    batching_required,
    on_cycle_start!,
    on_cycle_end!
using ..PopMemberModule: generate_reference
using ..PopulationModule: Population, finalize_costs
using ..HallOfFameModule: HallOfFame, update_hall_of_fame!
using ..RegularizedEvolutionModule: reg_evol_cycle
using ..SurrogateModule: create_surrogate_state, observe_surrogate_member!
using ..LossFunctionsModule: create_eval_context, eval_cost
using ..ConstantOptimizationModule: optimize_constants
using ..DimensionalAnalysisModule:
    unwrap_dimensional_scale,
    wrap_dimensional_scale,
    dimensional_scale_coefficient
using ..TracingModule: trace_optimization!

# Cycle through regularized evolution many times,
# printing the fittest equation every 10% through
function s_r_cycle(
    dataset::D,
    pop::P,
    ncycles::Int,
    curmaxsize::Int;
    verbosity::Int=0,
    options::AbstractOptions,
    trace::MaybeTrace,
    plugin_states::Tuple,
    surrogate_snapshot=nothing,
    return_surrogate_state::Bool=false,
) where {T,L,D<:Dataset{T,L},N<:AbstractExpression{T},P<:Population{T,L,N}}
    best_examples_seen = HallOfFame(options, dataset)
    num_evals = 0.0

    batched_dataset = if batching_required(options, dataset)
        batch(dataset, get_batch_size(options, dataset.n))
    else
        dataset
    end
    eval_context = create_eval_context(batched_dataset, options, curmaxsize)
    surrogate_state = create_surrogate_state(
        batched_dataset,
        options;
        snapshot=surrogate_snapshot,
    )
    if surrogate_state !== nothing
        for member in pop.members
            observe_surrogate_member!(surrogate_state, member, batched_dataset, options)
        end
    end

    for cycle_idx in 1:ncycles
        strictmap(options.plugins, plugin_states) do plugin, pstate
            return on_cycle_start!(pstate, plugin, cycle_idx, ncycles, options)
        end
        pop, tmp_num_evals = reg_evol_cycle(
            batched_dataset,
            pop,
            curmaxsize,
            options,
            trace;
            plugin_states,
            best_seen=best_examples_seen,
            eval_context,
            surrogate_state=surrogate_state,
        )
        num_evals += tmp_num_evals
        update_hall_of_fame!(best_examples_seen, pop.members, batched_dataset, options)
        strictmap(options.plugins, plugin_states) do plugin, pstate
            return on_cycle_end!(
                pstate, plugin, pop, batched_dataset, best_examples_seen, options
            )
        end
    end

    if return_surrogate_state
        return (pop, best_examples_seen, num_evals, surrogate_state)
    end
    return (pop, best_examples_seen, num_evals)
end

function optimize_and_simplify_population(
    dataset::D, pop::P, options::AbstractOptions, curmaxsize::Int, trace::MaybeTrace
)::Tuple{P,Float64} where {T,L,D<:Dataset{T,L},P<:Population{T,L}}
    array_num_evals = zeros(Float64, pop.n)
    do_optimization = rand(pop.n) .< options.optimizer_probability
    # Note: we have to turn off this threading loop due to Enzyme, since we need
    # to manually allocate a new task with a larger stack for Enzyme.
    should_thread = !(options.deterministic) && !(isa(options.autodiff_backend, AutoEnzyme))

    batched_dataset = if batching_required(options, dataset)
        batch(dataset, get_batch_size(options, dataset.n))
    else
        dataset
    end

    @threads_if should_thread for j in 1:(pop.n)
        if options.should_simplify
            member_tree = pop.members[j].tree
            coefficient = dimensional_scale_coefficient(member_tree, options)
            tree = unwrap_dimensional_scale(member_tree, options)
            tree = simplify_tree!(tree, options.operators)
            tree = combine_operators(tree, options.operators)
            pop.members[j].tree = coefficient === nothing ?
                wrap_dimensional_scale(tree, options) :
                wrap_dimensional_scale(tree, options; coefficient=coefficient)
        end
        if options.should_optimize_constants && do_optimization[j]
            # TODO: Might want to do full batch optimization here?
            pop.members[j], array_num_evals[j] = optimize_constants(
                batched_dataset, pop.members[j], options
            )
        end
    end
    num_evals = sum(array_num_evals)
    pop, tmp_num_evals = finalize_costs(dataset, pop, options)
    num_evals += tmp_num_evals

    # Now, we create new references for every member.
    for j in 1:(pop.n)
        old_ref = pop.members[j].ref
        new_ref = generate_reference()
        pop.members[j].parent = old_ref
        pop.members[j].ref = new_ref

        trace_optimization!(
            trace, pop.members[j], old_ref, new_ref, do_optimization[j], options
        )
    end
    return (pop, num_evals)
end

end
