# MySRCore package-contract tests added to the SymbolicRegression.jl baseline.
using MySRCore
using Test
using DynamicQuantities: dimension
using Random: MersenneTwister

@testset "MySRCore package identity" begin
    @test nameof(MySRCore) == :MySRCore
    @test isdefined(MySRCore, :Options)
    @test isdefined(MySRCore, :equation_search)
end

@testset "Size-matched crossover" begin
    SR = MySRCore.SymbolicRegression
    MutationFunctions = SR.MutationFunctionsModule
    @test SR.default_crossovers() == [SR.SubtreeCrossover() => 1.0]
    @test SR.SizeMatchedCrossover(; size_tolerance=0).size_tolerance == 0.0
    @test_throws ArgumentError SR.SizeMatchedCrossover(; size_tolerance=-0.1)
    @test_throws ArgumentError SR.SizeMatchedCrossover(; size_tolerance=NaN)

    options = SR.Options(
        binary_operators=(+, -, *, /),
        unary_operators=(sin,),
        default_plugins=(),
    )
    parent1 = SR.parse_expression(
        "x1 + x2";
        operators=options.operators,
        variable_names=["x1", "x2"],
        node_type=SR.Node{Float64,2},
    )
    parent2 = SR.parse_expression(
        "sin(x1 + x2)";
        operators=options.operators,
        variable_names=["x1", "x2"],
        node_type=SR.Node{Float64,2},
    )
    @test_throws ArgumentError MutationFunctions.size_matched_crossover_trees(
        parent1, parent2, -0.1, MersenneTwister(1)
    )
    before1, before2 = SR.string_tree(parent1), SR.string_tree(parent2)
    for seed in 1:12
        child1, child2 = MutationFunctions.size_matched_crossover_trees(
            parent1, parent2, 0.0, MersenneTwister(seed)
        )
        @test SR.count_nodes(SR.get_tree(child1)) == SR.count_nodes(SR.get_tree(parent1))
        @test SR.count_nodes(SR.get_tree(child2)) == SR.count_nodes(SR.get_tree(parent2))
        @test SR.string_tree(parent1) == before1
        @test SR.string_tree(parent2) == before2
    end
end

@testset "Dimension-aware mutation affinity" begin
    MutationFunctions = MySRCore.SymbolicRegression.MutationFunctionsModule
    X = Float64[1 2 3; 1 2 3]
    y = Float64[2, 4, 6]
    dimensions = [[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]]
    dataset = Dataset(
        X,
        y;
        variable_names=["x1", "x2"],
        X_dimensions=dimensions,
        y_dimensions=[1, 0, 0, 0, 0, 0, 0],
    )
    options = Options(
        binary_operators=(+, -, *, /),
        unary_operators=(),
        formula_type=:theoretical,
        default_plugins=(),
    )
    expr = parse_expression(
        "x1 + x2";
        operators=options.operators,
        variable_names=["x1", "x2"],
        node_type=Node{Float64,2},
    )
    tree = get_tree(expr)
    # Strict mode allows subtraction (same dimensions), but rejects product and quotient.
    @test MutationFunctions._mutation_operator_targets(
        tree, tree, options; dataset, scope=:full
    ) == [2]
    @test options.operator_affinity[2][1, 2] == 4.0
    @test options.operator_affinity[2][1, 3] == 1.0

    semi = Options(
        binary_operators=(+, -, *, /),
        unary_operators=(),
        formula_type=:semi_theoretical,
        default_plugins=(),
    )
    semi_expr = parse_expression(
        "x1 + x2";
        operators=semi.operators,
        variable_names=["x1", "x2"],
        node_type=Node{Float64,2},
    )
    # Semi-theoretical mutation checks the internal f(X), not the fitted outer scale.
    @test MutationFunctions._mutation_operator_targets(
        get_tree(semi_expr), get_tree(semi_expr), semi; dataset, scope=:internal
    ) == [2, 3, 4]

    @test_throws ArgumentError Options(mutation_affinity=:unknown)
    @test_throws ArgumentError Options(mutation_affinity_strength=0.0)
    @test_throws ArgumentError Options(mutation_affinity_exploration=1.1)
    @test_throws ArgumentError Options(
        binary_operators=(+, -),
        unary_operators=(),
        operator_affinity=Dict(2 => ones(1, 1)),
    )

    # A feature with an incompatible dimension is not a legal strict mutation.
    incompatible = Dataset(
        X,
        y;
        variable_names=["x1", "x2"],
        X_dimensions=[[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
        y_dimensions=[1, 0, 0, 0, 0, 0, 0],
    )
    feature_options = Options(
        binary_operators=(+, -),
        unary_operators=(),
        formula_type=:theoretical,
        default_plugins=(),
    )
    feature_expr = parse_expression(
        "x1 + x1";
        operators=feature_options.operators,
        variable_names=["x1", "x2"],
        node_type=Node{Float64,2},
    )
    before = [node.feature for node in get_tree(feature_expr) if node.degree == 0]
    mutated = MutationFunctions.mutate_feature(
        feature_expr, 2, MersenneTwister(7); dataset=incompatible, options=feature_options
    )
    after = [node.feature for node in get_tree(mutated) if node.degree == 0]
    @test after == before
end

@testset "RNN-GPSR population seeding" begin
    @test !Options().rnn_gpsr_seeding
    @test_throws ArgumentError Options(rnn_gpsr_seed_fraction=1.1)

    X = reshape(collect(Float64, -2:0.2:2), 1, :)
    y = 2 .* vec(X) .+ 1

    generator(training_sequences, costs, token_arities, count, max_length, seed) =
        [Int[2], Int[1]]

    function seeded_frontier(seed)
        options = Options(
            default_plugins=(),
            populations=1,
            population_size=6,
            tournament_selection_n=2,
            ncycles_per_iteration=1,
            maxsize=7,
            seed=seed,
            deterministic=true,
            rnn_gpsr_seeding=true,
            rnn_gpsr_seed_fraction=0.5,
            rnn_gpsr_candidate_count=8,
            rnn_gpsr_proposal_count=12,
            rnn_gpsr_cycles=1,
            rnn_gpsr_rounds=1,
            rnn_gpsr_maxsize=5,
            save_to_file=false,
        )
        hall = equation_search(
            X,
            y;
            niterations=0,
            options,
            rnn_generator=generator,
            parallelism=:serial,
            progress=false,
            verbosity=0,
        )
        return [
            string_tree(hall.members[index].tree, options) for
            index in eachindex(hall.exists) if hall.exists[index]
        ]
    end

    @test seeded_frontier(2026) == seeded_frontier(2026)
end

@testset "RNN-GPSR training corpus is independent of backend member results" begin
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = copy(vec(X))
    options = Options(
        default_plugins=(),
        populations=1,
        population_size=4,
        tournament_selection_n=2,
        rnn_gpsr_candidate_count=8,
        rnn_gpsr_maxsize=5,
        maxsize=7,
        seed=2026,
        deterministic=true,
        save_to_file=false,
    )
    dataset = Dataset(
        X,
        y;
        variable_names=["x1"],
    )
    sequences, structural_costs =
        MySRCore.SymbolicRegression.PopulationSeedingModule._independent_training_corpus(
            dataset,
            options,
            1,
            5,
            MersenneTwister(7),
        )
    @test length(sequences) == 8
    @test length(structural_costs) == 8
    @test all(isfinite, structural_costs)
end

@testset "RNN-GPSR feeds lightweight GPSR elites into later RNN rounds" begin
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = copy(vec(X))
    observed_training_counts = Int[]
    observed_feedback_rounds = Int[]
    observed_backend_cost_flags = Bool[]
    generator(
        training_sequences,
        costs,
        token_arities,
        count,
        max_length,
        seed,
        formula_type,
        feedback_round,
        training_source,
        backend_costs_used,
    ) = begin
        push!(observed_training_counts, length(training_sequences))
        push!(observed_feedback_rounds, feedback_round)
        push!(observed_backend_cost_flags, backend_costs_used)
        [Int[2]]
    end
    options = Options(
        default_plugins=(),
        populations=1,
        population_size=4,
        tournament_selection_n=2,
        rnn_gpsr_seeding=true,
        rnn_gpsr_candidate_count=8,
        rnn_gpsr_proposal_count=4,
        rnn_gpsr_cycles=0,
        rnn_gpsr_rounds=2,
        rnn_gpsr_feedback_fraction=0.5,
        rnn_gpsr_quality_gate=false,
        rnn_gpsr_maxsize=5,
        maxsize=7,
        seed=2026,
        deterministic=true,
        save_to_file=false,
    )
    equation_search(
        X,
        y;
        niterations=0,
        options,
        rnn_generator=generator,
        parallelism=:serial,
        progress=false,
        verbosity=0,
    )
    @test observed_training_counts == [8, 10]
    @test observed_feedback_rounds == [1, 2]
    @test observed_backend_cost_flags == [false, true]
end

@testset "RNN-GPSR reports backend feedback only when examples were appended" begin
    observed_sources = Symbol[]
    observed_backend_cost_flags = Bool[]
    generator(
        training_sequences,
        costs,
        token_arities,
        count,
        max_length,
        seed,
        formula_type,
        feedback_round,
        training_source,
        backend_costs_used,
    ) = begin
        push!(observed_sources, training_source)
        push!(observed_backend_cost_flags, backend_costs_used)
        [Int[2]]
    end
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = copy(vec(X))
    options = Options(
        default_plugins=(),
        populations=1,
        population_size=4,
        tournament_selection_n=2,
        rnn_gpsr_seeding=true,
        rnn_gpsr_candidate_count=8,
        rnn_gpsr_proposal_count=4,
        rnn_gpsr_cycles=0,
        rnn_gpsr_rounds=2,
        rnn_gpsr_feedback_fraction=0.0,
        rnn_gpsr_quality_gate=false,
        rnn_gpsr_maxsize=5,
        maxsize=7,
        seed=2026,
        deterministic=true,
        save_to_file=false,
    )
    equation_search(
        X,
        y;
        niterations=0,
        options,
        rnn_generator=generator,
        parallelism=:serial,
        progress=false,
        verbosity=0,
    )
    @test observed_sources == [:bootstrap_structural, :bootstrap_structural]
    @test observed_backend_cost_flags == [false, false]
end

@testset "RNN-GPSR respects the configured proposal budget" begin
    observed_counts = Int[]
    generator(training_sequences, costs, token_arities, count, max_length, seed, args...) = begin
        push!(observed_counts, count)
        [Int[2]]
    end
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = copy(vec(X))
    options = Options(
        default_plugins=(),
        populations=1,
        population_size=20,
        tournament_selection_n=2,
        rnn_gpsr_seeding=true,
        rnn_gpsr_candidate_count=8,
        rnn_gpsr_proposal_count=4,
        rnn_gpsr_cycles=0,
        rnn_gpsr_rounds=1,
        rnn_gpsr_quality_gate=false,
        rnn_gpsr_maxsize=5,
        maxsize=7,
        seed=2026,
        deterministic=true,
        save_to_file=false,
    )
    equation_search(
        X,
        y;
        niterations=0,
        options,
        rnn_generator=generator,
        parallelism=:serial,
        progress=false,
        verbosity=0,
    )
    @test observed_counts == [4]

    empty!(observed_counts)
    dataset = Dataset(X, y; variable_names=["x1"])
    _, evaluations =
        MySRCore.SymbolicRegression.PopulationSeedingModule.build_rnn_gpsr_seed_pool(
            dataset,
            options,
            ();
            rnn_generator=generator,
        )
    @test observed_counts == [4]
    @test evaluations == options.population_size
end

@testset "RNN-GPSR feedback count excludes nonfinite members" begin
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = copy(vec(X))
    options = Options(default_plugins=(), maxsize=7, save_to_file=false)
    dataset = Dataset(X, y; variable_names=["x1"])
    tree = parse_expression(
        "x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    member = PopMember(dataset, tree, options; deterministic=true)
    member.cost = Inf
    training_sequences = [Int[2]]
    training_costs = [1.0]
    used = MySRCore.SymbolicRegression.PopulationSeedingModule._append_feedback_examples!(
        training_sequences,
        training_costs,
        [member],
        options,
        1,
    )
    @test used == 0
    @test training_sequences == [Int[2]]
    @test training_costs == [1.0]
end

@testset "User guesses have priority over RNN-GPSR seeds" begin
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = copy(vec(X))
    options = Options(
        default_plugins=(),
        populations=1,
        population_size=4,
        tournament_selection_n=2,
        rnn_gpsr_seeding=true,
        rnn_gpsr_seed_fraction=1.0,
        maxsize=7,
        deterministic=true,
        save_to_file=false,
    )
    dataset = Dataset(X, y; variable_names=["x1"])
    user_tree = parse_expression(
        "x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    rnn_tree = parse_expression(
        "x1 + x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    user_member = PopMember(dataset, user_tree, options; deterministic=true)
    rnn_member = PopMember(dataset, rnn_tree, options; deterministic=true)
    population = Population([copy(user_member) for _ in 1:options.population_size])

    MySRCore.SymbolicRegression.PopulationSeedingModule.inject_initial_seeds!(
        population,
        [user_member],
        [rnn_member],
        options;
        population_index=1,
    )
    @test string_tree(population.members[1].tree, options) == "x1"
    @test string_tree(population.members[2].tree, options) == "x1 + x1"
end

@testset "User guesses beyond initial population capacity remain accepted" begin
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = copy(vec(X))
    options = Options(
        default_plugins=(),
        populations=2,
        population_size=2,
        tournament_selection_n=1,
        maxsize=7,
        deterministic=true,
        save_to_file=false,
    )
    guesses = ["x1", "x1 + x1", "x1 * x1", "x1 + x1 + x1", "x1 * x1 + x1"]
    hall = equation_search(
        X,
        y;
        niterations=0,
        options,
        guesses=guesses,
        parallelism=:serial,
        progress=false,
        verbosity=0,
    )
    @test hall isa HallOfFame
end

@testset "Dimension generator accepts constant powers of dimensional inputs" begin
    X = reshape(Float64[1, 2, 3], 1, :)
    y = copy(vec(X) .^ 2)
    options = Options(
        formula_type=:theoretical,
        binary_operators=(^,),
        default_plugins=(),
        maxsize=7,
    )
    dataset = Dataset(
        X,
        y;
        variable_names=["x1"],
        X_dimensions=[[1, 0, 0, 0, 0, 0, 0]],
        y_dimensions=[2, 0, 0, 0, 0, 0, 0],
    )
    generation = MySRCore.SymbolicRegression.DimensionGenerationModule
    qtype = typeof(dataset.X_dimensions[1])
    dimensionless = dimension(dataset.X_dimensions[1] / dataset.X_dimensions[1])
    base = generation.DimensionCandidate(
        parse_expression(
            "x1";
            operators=options.operators,
            variable_names=["x1"],
            node_type=Node{Float64,2},
        ),
        dimension(dataset.X_dimensions[1]),
        1,
    )
    exponent = generation.DimensionCandidate(
        parse_expression(
            "2.0";
            operators=options.operators,
            variable_names=["x1"],
            node_type=Node{Float64,2},
        ),
        dimensionless,
        1,
    )
    candidate = generation._make_operator_candidate(
        options.operators.ops[2][1],
        1,
        [base, exponent],
        qtype,
        Float64,
        options.node_type,
        dimensionless,
    )
    @test candidate !== nothing
    @test candidate.output_dimension == dimension(dataset.y_dimensions)
end

@testset "RNN-GPSR follows formula_type dimensional gate" begin
    X = reshape(Float64[1, 2, 3], 1, :)
    y = copy(vec(X))
    options = Options(
        formula_type=:theoretical,
        default_plugins=(),
        populations=1,
        population_size=4,
        tournament_selection_n=2,
        maxsize=7,
        deterministic=true,
        save_to_file=false,
    )
    dataset = Dataset(
        X,
        y;
        variable_names=["x1"],
        X_dimensions=[[1, 0, 0, 0, 0, 0, 0]],
        y_dimensions=[1, 0, 0, 0, 0, 0, 0],
    )
    valid_tree = parse_expression(
        "x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    invalid_tree = parse_expression(
        "x1 * x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    training_sequences = [Int[2] for _ in 1:8]
    training_costs = ones(Float64, 8)
    invalid_tokens = MySRCore.SymbolicRegression.PopulationSeedingModule._expression_tokens(
        invalid_tree, options, 1
    )
    observed_formula_type = Ref{Any}(nothing)
    generator(training_sequences, costs, token_arities, count, max_length, seed, formula_type) = begin
        observed_formula_type[] = formula_type
        [invalid_tokens]
    end
    trees = MySRCore.SymbolicRegression.PopulationSeedingModule._generate_proposal_trees(
        generator,
        training_sequences,
        training_costs,
        dataset,
        Float64,
        options,
        1,
        5,
        1,
        2026,
        MersenneTwister(9),
    )
    @test observed_formula_type[] == :theoretical
    @test length(trees) == 1
    @test MySRCore.SymbolicRegression.infer_dimension_static(trees[1], dataset, options).valid
end

@testset "Formula type dimensional contract" begin
    @test MySRCore.SymbolicRegression.dimension_policy(Options()) == :ignore
    @test MySRCore.SymbolicRegression.dimension_policy(Options(formula_type=:empirical)) == :ignore
    @test MySRCore.SymbolicRegression.dimension_policy(Options(formula_type=:theoretical)) == :strict
    X_semi = reshape(Float64[1, 2, 3, 4], 1, :)
    y_semi = 3 .* vec(X_semi) .^ 2
    semi_options = Options(
        formula_type=:semi_theoretical,
        default_plugins=(),
        populations=1,
        population_size=4,
        tournament_selection_n=2,
        ncycles_per_iteration=1,
        maxsize=9,
        seed=1,
        deterministic=true,
        optimizer_probability=1.0,
        save_to_file=false,
    )
    @test_nowarn equation_search(
        X_semi,
        y_semi;
        options=semi_options,
        niterations=1,
        X_dimensions=[[1, 0, 0, 0, 0, 0, 0]],
        y_dimensions=[2, 0, 0, 0, 0, 0, 0],
        parallelism=:serial,
        progress=false,
        verbosity=0,
    )
    @test_throws ArgumentError equation_search(
        reshape([1.0, 2.0], 1, :),
        [1.0, 2.0];
        options=Options(formula_type=:semi_theoretical, default_plugins=()),
        niterations=0,
        parallelism=:serial,
        progress=false,
        verbosity=0,
    )

    X = reshape(Float64[1, 2, 3], 1, :)
    y = copy(vec(X))
    options = Options(formula_type=:theoretical, default_plugins=())
    dataset = Dataset(
        X,
        y;
        variable_names=["x1"],
        X_dimensions=[[1, 0, 0, 0, 0, 0, 0]],
        y_dimensions=[1, 0, 0, 0, 0, 0, 0],
    )
    valid_tree = parse_expression(
        "x1 + x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    invalid_tree = parse_expression(
        "x1 * x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    @test infer_dimension_static(valid_tree, dataset, options).valid
    @test !infer_dimension_static(invalid_tree, dataset, options).valid
    @test MySRCore.SymbolicRegression.check_constraints(valid_tree, dataset, options, options.maxsize)
    @test !MySRCore.SymbolicRegression.check_constraints(invalid_tree, dataset, options, options.maxsize)

    @test_throws ArgumentError Dataset(
        X,
        y;
        X_units=["m"],
        y_units="m",
    )
    @test_throws ArgumentError Dataset(
        X,
        y;
        dimensional_constraint_penalty=1000,
    )
    @test_throws ArgumentError Dataset(
        X,
        y;
        dimensionless_constants_only=true,
    )
    @test_throws Exception Options(dimensional_constraint_penalty=1000)
    @test_throws Exception Options(dimensionless_constants_only=true)
end

@testset "Semi-theoretical C_dim boundary" begin
    X = reshape(Float64[1, 2, 3], 1, :)
    y = copy(vec(X))
    options = Options(formula_type=:semi_theoretical, default_plugins=(), maxsize=9)
    dataset = Dataset(X, y; variable_names=["x1"], X_dimensions=[[1, 0, 0, 0, 0, 0, 0]], y_dimensions=[2, 0, 0, 0, 0, 0, 0])
    tree = parse_expression(
        "x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    member = PopMember(dataset, tree, options; deterministic=true)
    @test string_tree(member.tree, options) == "1.0 * x1"
    @test infer_dimension_static(member.tree, dataset, options).valid
    @test infer_dimension_static(member.tree, dataset, options; scope=:internal).valid
    @test MySRCore.SymbolicRegression.DimensionalAnalysisModule.is_dimensional_scale_wrapper(
        get_tree(member.tree), options
    )
    @test MySRCore.SymbolicRegression.DimensionalAnalysisModule.wrap_dimensional_scale(
        member.tree, options
    ) === member.tree
    @test_throws ArgumentError equation_search(
        X,
        y;
        options=Options(
            formula_type=:semi_theoretical,
            should_optimize_constants=false,
            default_plugins=(),
        ),
        niterations=0,
        X_dimensions=[[1, 0, 0, 0, 0, 0, 0]],
        y_dimensions=[2, 0, 0, 0, 0, 0, 0],
        parallelism=:serial,
        progress=false,
        verbosity=0,
    )
    no_scale_operator_options = Options(
        formula_type=:semi_theoretical,
        binary_operators=(+,),
        default_plugins=(),
    )
    @test_throws ArgumentError equation_search(
        X,
        y;
        options=no_scale_operator_options,
        niterations=0,
        X_dimensions=[[1, 0, 0, 0, 0, 0, 0]],
        y_dimensions=[2, 0, 0, 0, 0, 0, 0],
        parallelism=:serial,
        progress=false,
        verbosity=0,
    )
end

@testset "Semi-theoretical C_dim survives crossover and mutation" begin
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = 3 .* vec(X)
    options = Options(
        formula_type=:semi_theoretical,
        default_plugins=(),
        mutations=[DoNothingMutation() => 1.0],
        populations=1,
        population_size=4,
        tournament_selection_n=2,
        ncycles_per_iteration=1,
        maxsize=12,
        deterministic=true,
        save_to_file=false,
    )
    dataset = Dataset(
        X,
        y;
        variable_names=["x1"],
        X_dimensions=[[1, 0, 0, 0, 0, 0, 0]],
        y_dimensions=[2, 0, 0, 0, 0, 0, 0],
    )
    t1 = parse_expression(
        "x1 + x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    t2 = parse_expression(
        "x1 * x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    wrapped1 = MySRCore.SymbolicRegression.wrap_dimensional_scale(
        t1, options; coefficient=2.5
    )
    wrapped2 = MySRCore.SymbolicRegression.wrap_dimensional_scale(
        t2, options; coefficient=4.0
    )
    member1 = PopMember(dataset, wrapped1, options; deterministic=true)
    member2 = PopMember(dataset, wrapped2, options; deterministic=true)

    crossover_result = MySRCore.SymbolicRegression.crossover(
        member1, member2, SubtreeCrossover(), options; trace=nothing
    )
    @test MySRCore.SymbolicRegression.dimensional_scale_coefficient(
        crossover_result.child1, options
    ) == 2.5
    @test MySRCore.SymbolicRegression.dimensional_scale_coefficient(
        crossover_result.child2, options
    ) == 4.0
    @test MySRCore.SymbolicRegression.DimensionalAnalysisModule.is_dimensional_scale_wrapper(
        get_tree(crossover_result.child1), options
    )
    @test MySRCore.SymbolicRegression.DimensionalAnalysisModule.is_dimensional_scale_wrapper(
        get_tree(crossover_result.child2), options
    )
    matched_result = MySRCore.SymbolicRegression.crossover(
        member1, member2, SizeMatchedCrossover(; size_tolerance=0.0), options;
        trace=nothing
    )
    @test MySRCore.SymbolicRegression.dimensional_scale_coefficient(
        matched_result.child1, options
    ) == 2.5
    @test MySRCore.SymbolicRegression.dimensional_scale_coefficient(
        matched_result.child2, options
    ) == 4.0

    mutated, accepted, _ = MySRCore.SymbolicRegression.MutateModule.next_generation(
        dataset,
        member1,
        options.maxsize,
        options;
        tmp_trace=nothing,
        plugin_states=(),
    )
    @test accepted
    @test MySRCore.SymbolicRegression.dimensional_scale_coefficient(
        mutated.tree, options
    ) == 2.5
    @test infer_dimension_static(mutated.tree, dataset, options).valid

    fit_options = Options(
        formula_type=:semi_theoretical,
        default_plugins=(),
        populations=1,
        population_size=4,
        tournament_selection_n=2,
        ncycles_per_iteration=1,
        maxsize=9,
        deterministic=true,
        optimizer_probability=1.0,
        save_to_file=false,
    )
    fit_dataset = Dataset(
        X,
        y;
        variable_names=["x1"],
        X_dimensions=[[1, 0, 0, 0, 0, 0, 0]],
        y_dimensions=[2, 0, 0, 0, 0, 0, 0],
    )
    fit_tree = parse_expression(
        "x1";
        operators=fit_options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    fit_member = PopMember(
        fit_dataset,
        MySRCore.SymbolicRegression.wrap_dimensional_scale(fit_tree, fit_options),
        fit_options;
        deterministic=true,
    )
    optimized_member, _ = MySRCore.SymbolicRegression.optimize_constants(
        fit_dataset, fit_member, fit_options
    )
    @test MySRCore.SymbolicRegression.dimensional_scale_coefficient(
        optimized_member.tree, fit_options
    ) ≈ 3.0 atol=1e-8
end

@testset "Formula type dimensional initial generation" begin
    X = reshape(Float64[1, 2, 3], 1, :)
    y_length = copy(vec(X))
    dataset_length = Dataset(X, y_length; X_dimensions=[[1, 0, 0, 0, 0, 0, 0]], y_dimensions=[1, 0, 0, 0, 0, 0, 0])
    strict_length = Options(
        formula_type=:theoretical,
        default_plugins=(),
        maxsize=7,
        seed=11,
    )
    for seed in 1:16
        tree = gen_random_tree_dimensional(
            dataset_length,
            strict_length,
            3,
            1,
            Float64,
            MersenneTwister(seed),
        )
        @test tree !== nothing
        @test infer_dimension_static(tree, dataset_length, strict_length).valid
    end

    y_area = Float64[1, 4, 9]
    dataset_area = Dataset(X, y_area; X_dimensions=[[1, 0, 0, 0, 0, 0, 0]], y_dimensions=[2, 0, 0, 0, 0, 0, 0])
    strict_area = Options(
        formula_type=:theoretical,
        default_plugins=(),
        maxsize=7,
        populations=1,
        population_size=4,
        tournament_selection_n=2,
        ncycles_per_iteration=1,
        seed=12,
        save_to_file=false,
    )
    area_tree = gen_random_tree_dimensional(
        dataset_area,
        strict_area,
        3,
        1,
        Float64,
        MersenneTwister(12),
    )
    @test area_tree !== nothing
    @test infer_dimension_static(area_tree, dataset_area, strict_area).valid
    @test_nowarn equation_search(
        X,
        y_area;
        niterations=0,
        options=strict_area,
        X_dimensions=[[1, 0, 0, 0, 0, 0, 0]],
        y_dimensions=[2, 0, 0, 0, 0, 0, 0],
        parallelism=:serial,
        progress=false,
        verbosity=0,
    )

    semi = Options(
        formula_type=:semi_theoretical,
        default_plugins=(),
        maxsize=7,
        seed=13,
    )
    for seed in 1:16
        tree = gen_random_tree_dimensional(
            dataset_area,
            semi,
            3,
            1,
            Float64,
            MersenneTwister(seed),
        )
        @test tree !== nothing
        @test infer_dimension_static(tree, dataset_area, semi; scope=:internal).valid
    end

    empirical = Options(formula_type=:empirical, default_plugins=())
    @test gen_random_tree_dimensional(
        dataset_length, empirical, 3, 1, Float64, MersenneTwister(1)
    ) === nothing
end
