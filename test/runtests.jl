# MySRCore package-contract tests added to the SymbolicRegression.jl baseline.
using MySRCore
using Test
using Random: MersenneTwister

@testset "MySRCore package identity" begin
    @test nameof(MySRCore) == :MySRCore
    @test isdefined(MySRCore, :Options)
    @test isdefined(MySRCore, :equation_search)
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
