module PopulationSeedingModule

using Random: AbstractRNG, MersenneTwister, default_rng
using Statistics: median
using DynamicExpressions:
    AbstractExpression, AbstractExpressionNode, constructorof, get_tree

using ..CoreModule: AbstractOptions, Dataset, max_features, sample_value
using ..CheckConstraintsModule: check_constraints
using ..MutationFunctionsModule: gen_random_tree_fixed_size
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
            1 + nfeatures + sum(options.nops[1:(node.degree - 1)]) + node.op
        end
        push!(tokens, token)
    end
    return tokens
end

_member_tokens(member::AbstractPopMember, options::AbstractOptions, nfeatures::Int) =
    _expression_tokens(member.tree, options, nfeatures)

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
    training_members::AbstractVector{<:AbstractPopMember},
    ::Type{T},
    options::AbstractOptions,
    nfeatures::Int,
    maxsize::Int,
    proposal_count::Int,
    seed::Int,
    rng::AbstractRNG,
) where {T}
    isnothing(rnn_generator) && throw(
        ArgumentError(
            "RNN-GPSR seeding requires an `rnn_generator` callback. " *
            "MySR supplies a PyTorch generator; direct Julia callers may provide " *
            "a function `(training_sequences, costs, token_arities, " *
            "proposal_count, max_length, seed) -> token_sequences`.",
        ),
    )
    training_sequences = [
        _member_tokens(member, options, nfeatures) for member in training_members
    ]
    training_costs = Float64[
        isfinite(member.cost) ? member.cost : Inf for member in training_members
    ]
    raw_sequences = Base.invokelatest(
        rnn_generator,
        training_sequences,
        training_costs,
        _token_arities(options, nfeatures),
        proposal_count,
        maxsize,
        seed,
    )
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
        check_constraints(tree, options, maxsize) || continue
        push!(trees, tree)
        length(trees) >= proposal_count && break
    end
    while length(trees) < proposal_count
        push!(trees, _valid_random_tree(T, options, nfeatures, maxsize, rng))
    end
    return trees
end

_member_cost(member) = isfinite(member.cost) ? Float64(member.cost) : Inf

function _population_quality(members::AbstractVector{<:AbstractPopMember})
    costs = _member_cost.(members)
    return (median(costs), minimum(costs))
end

function _valid_random_tree(
    ::Type{T},
    options::AbstractOptions,
    nfeatures::Int,
    maxsize::Int,
    rng::AbstractRNG,
) where {T}
    for _ in 1:100
        requested_size = rand(rng, 1:maxsize)
        tree = gen_random_tree_fixed_size(requested_size, options, nfeatures, T, rng)
        check_constraints(tree, options, maxsize) && return tree
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
    tree = _valid_random_tree(T, options, nfeatures, maxsize, rng)
    return constructorof(PM)(
        dataset,
        tree,
        options;
        parent=-1,
        deterministic=options.deterministic,
    )
end

"""
    build_rnn_gpsr_seed_pool(dataset, options, plugin_states; rnn_generator, seed_offset=0)

Build a data-informed seed pool in three bounded stages:

1. generate and evaluate random expression individuals using an RNG derived from
   `options.seed`;
2. ask an external trainable recurrent policy to generate grammar-complete proposal
   sequences, parse and validate them, then compare their best real-loss members with
   a random control group evaluated under the same candidate budget;
3. evolve the better group with the existing regularized GP-SR cycle and feed its
   elites into the next recurrent-policy round. Pre-evolution elites are retained so
   the bounded GP phase cannot discard the best members already found in that round.

Returns `(members, evaluations)`. This is an initialization budget and is kept
separate from the later formal SR cycles.
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

    training_count = max(options.population_size, options.rnn_gpsr_candidate_count)
    training_members = [
        _valid_random_member(PM, dataset, options, nfeatures, maxsize, rng) for
        _ in 1:training_count
    ]
    evaluations = Float64(training_count)

    sort!(training_members; by=_member_cost)
    population = Population(copy.(training_members[1:(options.population_size)]))

    for round_index in 1:(options.rnn_gpsr_rounds)
        neural_keep = options.population_size
        proposal_count = max(options.rnn_gpsr_proposal_count, neural_keep)
        generation_seed = base_seed + seed_offset + 1_000_003 * round_index
        proposal_trees = _generate_proposal_trees(
            rnn_generator,
            training_members,
            T,
            options,
            nfeatures,
            maxsize,
            proposal_count,
            generation_seed,
            rng,
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
                _valid_random_member(PM, dataset, options, nfeatures, maxsize, rng) for
                _ in 1:proposal_count
            ]
            evaluations += length(control_members)
            sort!(control_members; by=_member_cost)
            resize!(control_members, neural_keep)
            if _population_quality(control_members) < _population_quality(neural_members)
                accepted_members = control_members
            end
        end

        combined = vcat(training_members, population.members, accepted_members)
        sort!(combined; by=_member_cost)
        population = Population(copy.(combined[1:(options.population_size)]))
        if options.rnn_gpsr_cycles > 0
            pre_evolution_members = copy.(population.members)
            evolved_population, _, gpsr_evaluations = s_r_cycle(
                dataset,
                population,
                options.rnn_gpsr_cycles,
                maxsize;
                verbosity=0,
                options,
                trace=new_trace(options),
                plugin_states,
            )
            evaluations += gpsr_evaluations
            evolution_candidates = vcat(
                pre_evolution_members, evolved_population.members
            )
            sort!(evolution_candidates; by=_member_cost)
            population = Population(
                copy.(evolution_candidates[1:(options.population_size)])
            )
        end
        append!(training_members, copy.(population.members))
    end
    sort!(population.members; by=member -> isfinite(member.cost) ? member.cost : Inf)
    return (population.members, evaluations)
end

"""Replace an exact fraction of a formal initial population with seed-pool members."""
function inject_rnn_gpsr_seeds!(
    population::Population,
    seed_members::AbstractVector,
    options::AbstractOptions;
    population_index::Int=1,
)
    isempty(seed_members) && return population
    count = clamp(
        round(Int, population.n * options.rnn_gpsr_seed_fraction), 0, population.n
    )
    offset = (population_index - 1) * max(count, 1)
    for location in 1:count
        source = seed_members[mod1(offset + location, length(seed_members))]
        population.members[location] = copy(source)
        reset_birth!(population.members[location]; deterministic=options.deterministic)
    end
    return population
end

end
