module PopulationSeedingModule

using Random: AbstractRNG, MersenneTwister, default_rng
using Statistics: median
using DynamicExpressions:
    AbstractExpression, AbstractExpressionNode, constructorof, get_tree, string_tree

using ..CoreModule:
    AbstractOptions,
    Dataset,
    max_features,
    sample_value,
    dimension_policy,
    fork_plugin_state
using ..CheckConstraintsModule: check_constraints
using ..MutationFunctionsModule: gen_random_tree_fixed_size
using ..DimensionGenerationModule: gen_random_tree_dimensional
using ..DimensionalAnalysisModule: unwrap_dimensional_scale
using ..PopMemberModule: AbstractPopMember, reset_birth!
using ..PopulationModule: Population
using ..SingleIterationModule: s_r_cycle
using ..TracingModule: new_trace

function _expression_tokens(expression, options::AbstractOptions, nfeatures::Int)
    tree = expression isa AbstractExpression ? get_tree(expression) : expression
    tokens = Int[]
    for node in tree
        token = if node.degree == 0
            node.constant ? 1 : 1 + node.feature
        else
            # A unary node has no lower-arity operator block.  Julia's
            # `sum(::Tuple{})` throws, so use an explicit zero for degree 1.
            lower_arity_offset = node.degree == 1 ? 0 :
                sum(options.nops[1:(node.degree - 1)])
            1 + nfeatures + lower_arity_offset + node.op
        end
        push!(tokens, token)
    end
    return tokens
end

function _tree_tokens(tree, options::AbstractOptions, nfeatures::Int)
    internal_tree = dimension_policy(options) === :compatible ?
        unwrap_dimensional_scale(tree, options) : tree
    return _expression_tokens(internal_tree, options, nfeatures)
end

_rnn_dimension_scope(options::AbstractOptions) =
    dimension_policy(options) === :compatible ? :internal : :full

function _token_arities(options::AbstractOptions, nfeatures::Int)
    arities = zeros(Int, 1 + nfeatures + sum(options.nops))
    offset = 1 + nfeatures
    for degree in eachindex(options.nops)
        operator_count = options.nops[degree]
        if operator_count > 0
            arities[(offset + 1):(offset + operator_count)] .= degree
        end
        offset += operator_count
    end
    return arities
end

function _decode_expression_tokens(
    tokens::AbstractVector{Int},
    ::Type{T},
    options::AbstractOptions,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T}
    isempty(tokens) && throw(ArgumentError("Generated token sequence is empty."))
    position = Ref(1)

    function parse_node()
        position[] <= length(tokens) ||
            throw(ArgumentError("Generated token sequence ended before the tree was complete."))
        token = tokens[position[]]
        position[] += 1
        if token == 1
            return constructorof(options.node_type)(
                T; val=sample_value(rng, T, options)
            )
        elseif 2 <= token <= 1 + nfeatures
            return constructorof(options.node_type)(T; feature=token - 1)
        end

        relative_operator = token - (1 + nfeatures)
        relative_operator >= 1 ||
            throw(ArgumentError("Generated token $(token) is outside the vocabulary."))
        offset = 0
        for degree in eachindex(options.nops)
            operator_count = options.nops[degree]
            if relative_operator <= offset + operator_count
                operator_index = relative_operator - offset
                children = ntuple(_ -> parse_node(), degree)
                return constructorof(options.node_type)(;
                    op=operator_index, children=children
                )
            end
            offset += operator_count
        end
        throw(ArgumentError("Generated token $(token) is outside the vocabulary."))
    end

    tree = parse_node()
    position[] == length(tokens) + 1 ||
        throw(ArgumentError("Generated token sequence contains trailing tokens."))
    return tree
end

function _generate_proposal_trees(
    rnn_generator,
    training_sequences::AbstractVector{<:AbstractVector{<:Integer}},
    training_costs::AbstractVector{<:Real},
    dataset::Dataset,
    ::Type{T},
    options::AbstractOptions,
    nfeatures::Int,
    maxsize::Int,
    proposal_count::Int,
    seed::Int,
    rng::AbstractRNG,
    feedback_round::Int=1,
    training_source::Symbol=:bootstrap_structural,
    backend_costs_used::Bool=false,
) where {T}
    isnothing(rnn_generator) && throw(
        ArgumentError(
            "RNN-GPSR seeding requires an `rnn_generator` callback. " *
            "MySR supplies a PyTorch generator; direct Julia callers may provide " *
            "a function with arguments training_sequences, costs, token_arities, " *
            "proposal_count, max_length, seed, formula_type returning token sequences.",
        ),
    )
    token_arities = _token_arities(options, nfeatures)
    raw_sequences = if applicable(
        rnn_generator,
        training_sequences,
        training_costs,
        token_arities,
        proposal_count,
        maxsize,
        seed,
        options.formula_type,
        feedback_round,
        training_source,
        backend_costs_used,
    )
        Base.invokelatest(
            rnn_generator,
            training_sequences,
            training_costs,
            token_arities,
            proposal_count,
            maxsize,
            seed,
            options.formula_type,
            feedback_round,
            training_source,
            backend_costs_used,
        )
    elseif applicable(
        rnn_generator,
        training_sequences,
        training_costs,
        token_arities,
        proposal_count,
        maxsize,
        seed,
        options.formula_type,
    )
        Base.invokelatest(
            rnn_generator,
            training_sequences,
            training_costs,
            token_arities,
            proposal_count,
            maxsize,
            seed,
            options.formula_type,
        )
    else
        # Keep direct Julia callbacks written against the pre-dimension
        # six-argument prototype working while MySR uses the new contract.
        Base.invokelatest(
            rnn_generator,
            training_sequences,
            training_costs,
            token_arities,
            proposal_count,
            maxsize,
            seed,
        )
    end
    # A policy callback may have no valid proposal in a round (for example,
    # after grammar/dimension filtering).  Treat an explicit `nothing` as an
    # empty proposal batch so the bounded random fallback can complete the
    # requested seed pool instead of crashing after a successful search setup.
    raw_sequences = something(raw_sequences, Vector{Vector{Int}}())
    trees = Any[]
    seen = Set{Any}()
    for raw_sequence in raw_sequences
        tokens = Int[token for token in raw_sequence]
        key = Tuple(tokens)
        key in seen && continue
        push!(seen, key)
        length(tokens) <= maxsize || continue
        tree = try
            _decode_expression_tokens(tokens, T, options, nfeatures, rng)
        catch error
            error isa ArgumentError || rethrow()
            nothing
        end
        isnothing(tree) && continue
        check_constraints(
            tree,
            dataset,
            options,
            maxsize;
            scope=_rnn_dimension_scope(options),
        ) || continue
        push!(trees, tree)
        length(trees) >= proposal_count && break
    end
    while length(trees) < proposal_count
        push!(trees, _valid_random_tree(dataset, T, options, nfeatures, maxsize, rng))
    end
    return trees
end

"""Generate a grammar/dimension-aware data-scored bootstrap corpus for RNN training.

Each candidate is evaluated against the current dataset before the first RNN
round.  This supplies a weak but target-aware signal instead of teaching the
policy only expression complexity.  Formal population-member/HOF results are
still excluded; later rounds replace this prior with real GPSR feedback."""
function _independent_training_corpus(
    dataset::Dataset{T},
    options::AbstractOptions,
    nfeatures::Int,
    maxsize::Int,
    rng::AbstractRNG,
    ;
    return_evaluations::Bool=false,
) where {T}
    requested = max(8, options.rnn_gpsr_candidate_count)
    sequences = Vector{Vector{Int}}()
    structural_costs = Float64[]
    max_attempts = max(100, requested * 20)
    for _ in 1:max_attempts
        length(sequences) >= requested && break
        tree = _valid_random_tree(dataset, T, options, nfeatures, maxsize, rng)
        tokens = _tree_tokens(tree, options, nfeatures)
        push!(sequences, tokens)
        # Constructing a temporary member uses the same eval_loss →
        # loss_to_cost path as ordinary populations and later GPSR feedback.
        # This keeps the RNN bootstrap target aligned with the backend's actual
        # selection objective, including uncertainty-aware losses and parsimony.
        candidate_cost = try
            candidate_member = constructorof(options.popmember_type)(
                dataset,
                tree,
                options;
                parent=-1,
                deterministic=options.deterministic,
            )
            _member_cost(candidate_member)
        catch
            Inf
        end
        push!(structural_costs, candidate_cost)
    end
    length(sequences) >= 8 ||
        throw(ArgumentError("Unable to construct the RNN-GPSR bootstrap training corpus."))
    return return_evaluations ? (sequences, structural_costs, length(sequences)) :
        (sequences, structural_costs)
end

_member_cost(member) = isfinite(member.cost) ? Float64(member.cost) : Inf

# The Python RNN-GPSR policy needs at least eight examples to form a stable
# training batch.  Keep this invariant at the Julia/Python boundary so a small
# feedback fraction cannot erase the structural bootstrap corpus.
const MIN_RNN_GPSR_TRAINING_EXAMPLES = 8

function _population_quality(members::AbstractVector{<:AbstractPopMember})
    costs = _member_cost.(members)
    return (median(costs), minimum(costs))
end

function _append_feedback_examples!(
    training_sequences::Vector{Vector{Int}},
    training_costs::Vector{Float64},
    members::AbstractVector{<:AbstractPopMember},
    options::AbstractOptions,
    nfeatures::Int,
    ;
    replace_bootstrap::Bool=false,
)
    ordered = sort(collect(members); by=_member_cost)
    keep = clamp(
        ceil(Int, length(ordered) * options.rnn_gpsr_feedback_fraction),
        0,
        length(ordered),
    )
    finite_members = [member for member in Iterators.take(ordered, keep) if isfinite(_member_cost(member))]
    isempty(finite_members) && return 0
    # A first feedback round may contain fewer elites than the policy's minimum
    # (for example ceil(27 * 0.2) == 6).  In that case retain the bootstrap
    # corpus and append the scored feedback instead of handing an undersized
    # corpus to Python.  Replacement is only safe once enough finite examples
    # are available on its own.
    if replace_bootstrap && length(finite_members) >= MIN_RNN_GPSR_TRAINING_EXAMPLES
        empty!(training_sequences)
        empty!(training_costs)
    end
    used = 0
    for member in finite_members
        cost = _member_cost(member)
        push!(training_sequences, _tree_tokens(member.tree, options, nfeatures))
        push!(training_costs, cost)
        used += 1
    end
    return used
end

function _valid_random_tree(
    dataset::Dataset{T},
    ::Type{T},
    options::AbstractOptions,
    nfeatures::Int,
    maxsize::Int,
    rng::AbstractRNG,
) where {T}
    for _ in 1:100
        requested_size = rand(rng, 1:maxsize)
        typed_tree = gen_random_tree_dimensional(
            dataset,
            options,
            requested_size,
            nfeatures,
            T,
            rng;
            max_nodes=requested_size,
        )
        tree = something(
            typed_tree,
            gen_random_tree_fixed_size(requested_size, options, nfeatures, T, rng),
        )
        check_constraints(
            tree,
            dataset,
            options,
            maxsize;
            scope=_rnn_dimension_scope(options),
        ) && return tree
    end
    throw(ArgumentError("Unable to generate a valid RNN-GPSR seed expression."))
end

function _valid_random_member(
    ::Type{PM},
    dataset::Dataset{T},
    options::AbstractOptions,
    nfeatures::Int,
    maxsize::Int,
    rng::AbstractRNG,
) where {T,PM}
    tree = _valid_random_tree(dataset, T, options, nfeatures, maxsize, rng)
    return constructorof(PM)(
        dataset,
        tree,
        options;
        parent=-1,
        deterministic=options.deterministic,
    )
end

function _fork_seed_plugin_states(
    options::AbstractOptions,
    plugin_states::Tuple,
    dataset::Dataset,
)
    return Tuple(
        fork_plugin_state(state, plugin, dataset) for
        (plugin, state) in zip(options.plugins, plugin_states)
    )
end

function _trim_seed_pool!(
    seed_pool::Vector{AbstractPopMember},
    options::AbstractOptions,
    limit::Int,
)
    isempty(seed_pool) && return seed_pool
    sort!(seed_pool; by=_member_cost)
    seen = Set{Any}()
    unique_members = AbstractPopMember[]
    for member in seed_pool
        # Token sequences intentionally collapse all constants to one token for
        # RNN training.  Use the rendered tree for seed-pool de-duplication so
        # distinct optimized constants are not discarded as duplicates.
        key = string_tree(member.tree, options; pretty=false)
        key in seen && continue
        push!(seen, key)
        push!(unique_members, member)
        length(unique_members) >= limit && break
    end
    empty!(seed_pool)
    append!(seed_pool, unique_members)
    return seed_pool
end

"""
    build_rnn_gpsr_seed_pool(dataset, options, plugin_states; rnn_generator, seed_offset=0)

Build an alternating RNN-to-lightweight-GPSR seed pool in bounded stages:

1. generate a grammar/dimension-aware structural bootstrap corpus;
2. ask an external trainable recurrent policy to generate grammar-complete proposal
   sequences, parse and validate them, then compare their best real-loss members with
   a random control group evaluated under the same candidate budget;
3. evolve each accepted lightweight population with the existing regularized
   GP-SR cycle for the configured lightweight iterations and cycles;
4. append the best post-GPSR members and their real costs to the next RNN training
   round, and repeat for `rnn_gpsr_rounds` feedback rounds;
5. return post-GPSR members for formal population injection.

Returns `(members, evaluations)`. Bootstrap candidate cost evaluations are counted
in the returned total, but bootstrap candidates are never inserted as formal
population members; proposal and lightweight-GPSR evaluations are counted too.
"""
function build_rnn_gpsr_seed_pool(
    dataset::Dataset{T},
    options::AbstractOptions,
    plugin_states::Tuple;
    rnn_generator=nothing,
    seed_offset::Int=0,
) where {T}
    options.rnn_gpsr_seeding || return (AbstractPopMember[], 0.0)
    PM = options.popmember_type
    nfeatures = max_features(dataset, options)
    maxsize = min(options.maxsize, options.rnn_gpsr_maxsize)
    base_seed = isnothing(options.seed) ? rand(default_rng(), 0:(typemax(Int32))) : options.seed
    rng = MersenneTwister(base_seed + seed_offset)

    training_sequences, training_costs, bootstrap_evaluations = _independent_training_corpus(
        dataset, options, nfeatures, maxsize, rng; return_evaluations=true
    )
    evaluations = Float64(bootstrap_evaluations)
    seed_pool = AbstractPopMember[]
    backend_feedback_count = 0
    seed_pool_limit = max(
        options.population_size * options.populations,
        options.rnn_gpsr_population_size * options.rnn_gpsr_populations,
    )

    for round_index in 1:(options.rnn_gpsr_rounds)
        has_backend_feedback = backend_feedback_count > 0
        round_feedback_members = AbstractPopMember[]
        for lightweight_population_index in 1:(options.rnn_gpsr_populations)
            proposal_count = options.rnn_gpsr_proposal_count
            lightweight_population_size = options.rnn_gpsr_population_size
            neural_keep = min(lightweight_population_size, proposal_count)
            population_seed = base_seed +
                seed_offset +
                1_000_003 * round_index +
                10_007 * lightweight_population_index
            population_rng = MersenneTwister(population_seed)
            proposal_trees = _generate_proposal_trees(
                rnn_generator,
                training_sequences,
                training_costs,
                dataset,
                T,
                options,
                nfeatures,
                maxsize,
                proposal_count,
                population_seed,
                population_rng,
                round_index,
                has_backend_feedback ? :backend_gpsr_feedback : :bootstrap_structural,
                has_backend_feedback,
            )
            neural_members = PM[
                constructorof(PM)(
                    dataset,
                    tree,
                    options;
                    parent=-1,
                    deterministic=options.deterministic,
                ) for tree in proposal_trees
            ]
            evaluations += length(neural_members)
            sort!(neural_members; by=_member_cost)
            resize!(neural_members, neural_keep)

            accepted_members = neural_members
            if options.rnn_gpsr_quality_gate
                control_members = PM[
                    _valid_random_member(
                        PM, dataset, options, nfeatures, maxsize, population_rng
                    ) for _ in 1:proposal_count
                ]
                evaluations += length(control_members)
                sort!(control_members; by=_member_cost)
                resize!(control_members, neural_keep)
                if _population_quality(control_members) < _population_quality(neural_members)
                    accepted_members = control_members
                end
            end

            while length(accepted_members) < lightweight_population_size
                push!(
                    accepted_members,
                    _valid_random_member(
                        PM, dataset, options, nfeatures, maxsize, population_rng
                    ),
                )
                evaluations += 1
            end
            sort!(accepted_members; by=_member_cost)
            population = Population(copy.(accepted_members[1:lightweight_population_size]))
            evolved_members = copy.(population.members)
            population_plugin_states = _fork_seed_plugin_states(
                options, plugin_states, dataset
            )
            for iteration_index in 1:(options.rnn_gpsr_niterations)
                options.rnn_gpsr_ncycles_per_iteration == 0 && break
                iteration_seed = population_seed + 97 * iteration_index
                evolved_population, _, gpsr_evaluations = s_r_cycle(
                    dataset,
                    population,
                    options.rnn_gpsr_ncycles_per_iteration,
                    maxsize;
                    verbosity=0,
                    options,
                    trace=new_trace(options),
                    plugin_states=population_plugin_states,
                    rng=MersenneTwister(iteration_seed),
                )
                evaluations += gpsr_evaluations
                previous_members = population.members
                valid_members = [
                    copy(member) for member in evolved_population.members if
                    check_constraints(
                        member.tree,
                        dataset,
                        options,
                        maxsize;
                        scope=_rnn_dimension_scope(options),
                    )
                ]
                while length(valid_members) < lightweight_population_size
                    push!(
                        valid_members,
                        copy(
                            previous_members[
                                mod1(length(valid_members) + 1, length(previous_members))
                            ],
                        ),
                    )
                end
                resize!(valid_members, lightweight_population_size)
                population = Population(copy.(valid_members))
                evolved_members = copy.(population.members)
            end
            append!(round_feedback_members, evolved_members)
            append!(seed_pool, copy.(evolved_members))
        end
        backend_feedback_count += _append_feedback_examples!(
            training_sequences,
            training_costs,
            round_feedback_members,
            options,
            nfeatures,
            replace_bootstrap=backend_feedback_count == 0,
        )
        _trim_seed_pool!(seed_pool, options, seed_pool_limit)
    end
    _trim_seed_pool!(seed_pool, options, seed_pool_limit)
    return (seed_pool, evaluations)
end

"""Inject user guesses first, then RNN-GPSR seeds, retaining random members."""
function inject_initial_seeds!(
    population::Population,
    user_seed_members::AbstractVector,
    rnn_seed_members::AbstractVector,
    options::AbstractOptions;
    population_index::Int=1,
)
    user_members = if length(user_seed_members) <= population.n
        user_seed_members
    else
        start = (population_index - 1) * population.n + 1
        stop = min(population_index * population.n, length(user_seed_members))
        start <= stop ? user_seed_members[start:stop] : user_seed_members[1:0]
    end

    location = 1
    for source in user_members
        population.members[location] = copy(source)
        reset_birth!(population.members[location]; deterministic=options.deterministic)
        location += 1
    end

    remaining = population.n - length(user_members)
    rnn_count = min(
        remaining,
        clamp(round(Int, population.n * options.rnn_gpsr_seed_fraction), 0, population.n),
    )
    if rnn_count > 0 && !isempty(rnn_seed_members)
        offset = (population_index - 1) * max(rnn_count, 1)
        for index in 1:rnn_count
            source = rnn_seed_members[mod1(offset + index, length(rnn_seed_members))]
            population.members[location] = copy(source)
            reset_birth!(population.members[location]; deterministic=options.deterministic)
            location += 1
        end
    end
    return population
end

"""Backward-compatible RNN-only injection helper."""
function inject_rnn_gpsr_seeds!(
    population::Population,
    seed_members::AbstractVector,
    options::AbstractOptions;
    population_index::Int=1,
)
    return inject_initial_seeds!(
        population,
        eltype(seed_members)[],
        seed_members,
        options;
        population_index,
    )
end

end
