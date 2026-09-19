# MySRCore package-contract tests added to the SymbolicRegression.jl baseline.
using MySRCore
using Test
using DynamicExpressions: get_child, get_metadata
using DynamicQuantities: dimension
using Random: MersenneTwister

@testset "Input validation guards" begin
    @test_throws ArgumentError TemplateStructure{(:f,)}(
        x -> x;
        num_features=(; f=-1),
    )
    @test_throws ArgumentError TemplateStructure{(:f,), (:p,)}(
        ((; f), (; p), x) -> f(x) + p[1];
        num_features=(; f=1),
        num_parameters=(; p=0),
    )
    X = reshape(Float64[1, 2], 1, :)
    @test_throws DimensionMismatch Dataset(X, [1.0, 2.0]; weights=[1.0])
    @test_throws ArgumentError Dataset(X, [1.0, 2.0]; weights=[1.0, -1.0])
    @test_throws ArgumentError Dataset(X, [1.0, 2.0]; weights=[0.0, 0.0])
end

@testset "Uncertainty-aware loss presets" begin
    SR = MySRCore.SymbolicRegression
    X = reshape(Float64[1.0, -1.0], 1, :)
    y = zeros(2)
    expr = SR.parse_expression(
        "x1";
        operators=SR.Options(binary_operators=(+,), unary_operators=()).operators,
        variable_names=["x1"],
        node_type=SR.Node{Float64,2},
    )
    asym_dataset = SR.Dataset(
        X,
        y;
        extra=(sigma_minus=Float64[2.0, 2.0], sigma_plus=Float64[4.0, 4.0]),
    )
    asym_options = SR.Options(
        binary_operators=(+,),
        unary_operators=(),
        uncertainty_mode=:asymmetry,
        loss_preset=:asymmetric_gaussian_nll,
        loss_scale=:linear,
        default_plugins=(),
    )
    asym_loss = SR.LossFunctionsModule.eval_loss(expr, asym_dataset, asym_options)
    @test isfinite(asym_loss)
    @test asym_loss > 0
    asym_case_losses = SR.LossFunctionsModule.eval_case_losses(expr, asym_dataset, asym_options)
    @test asym_loss ≈ sum(asym_case_losses) / length(asym_case_losses)
    batched_asym_dataset = SR.batch(asym_dataset, [2])
    @test isfinite(SR.LossFunctionsModule.eval_loss(expr, batched_asym_dataset, asym_options))
    robust_options = SR.Options(
        binary_operators=(+,),
        unary_operators=(),
        uncertainty_mode=:asymmetry,
        loss_preset=:asymmetric_huber,
        robust_delta=1.0,
        default_plugins=(),
    )
    @test SR.LossFunctionsModule.eval_loss(expr, asym_dataset, robust_options) ≈ 0.078125
    robust_case_losses = SR.LossFunctionsModule.eval_case_losses(expr, asym_dataset, robust_options)
    @test SR.LossFunctionsModule.eval_loss(expr, asym_dataset, robust_options) ≈
        sum(robust_case_losses) / length(robust_case_losses)
    student_options = SR.Options(
        binary_operators=(+,),
        unary_operators=(),
        uncertainty_mode=:asymmetry,
        loss_preset=:asymmetric_student_t_nll,
        loss_scale=:linear,
        student_nu=4.0,
        default_plugins=(),
    )
    @test isfinite(SR.LossFunctionsModule.eval_loss(expr, asym_dataset, student_options))
    student_case_losses = SR.LossFunctionsModule.eval_case_losses(expr, asym_dataset, student_options)
    @test SR.LossFunctionsModule.eval_loss(expr, asym_dataset, student_options) ≈
        sum(student_case_losses) / length(student_case_losses)
    negative_dataset = SR.Dataset(
        zeros(1, 2),
        zeros(2);
        extra=(sigma_minus=Float64[0.1, 0.1], sigma_plus=Float64[0.1, 0.1]),
    )
    negative_loss = SR.LossFunctionsModule.eval_loss(expr, negative_dataset, asym_options)
    @test negative_loss < 0
    negative_cost, returned_loss = SR.LossFunctionsModule.eval_cost(
        negative_dataset,
        expr,
        asym_options,
    )
    @test returned_loss == negative_loss
    @test isfinite(negative_cost)
    symmetric_dataset = SR.Dataset(
        X,
        y;
        extra=(sigma=Float64[2.0, 2.0],),
    )
    symmetric_options = SR.Options(
        binary_operators=(+,),
        unary_operators=(),
        uncertainty_mode=:symmetry,
        loss_preset=:gaussian_nll,
        loss_scale=:linear,
        default_plugins=(),
    )
    @test isfinite(SR.LossFunctionsModule.eval_loss(expr, symmetric_dataset, symmetric_options))
    symmetric_case_losses = SR.LossFunctionsModule.eval_case_losses(
        expr,
        symmetric_dataset,
        symmetric_options,
    )
    @test SR.LossFunctionsModule.eval_loss(expr, symmetric_dataset, symmetric_options) ≈
        sum(symmetric_case_losses) / length(symmetric_case_losses)
    no_uncertainty_options = SR.Options(
        binary_operators=(+,),
        unary_operators=(),
        loss_preset=:l1,
        default_plugins=(),
    )
    @test SR.LossFunctionsModule.eval_loss(expr, SR.Dataset(X, y), no_uncertainty_options) ≈ 1.0
    @test_throws ArgumentError SR.LossFunctionsModule.eval_loss(
        expr,
        SR.Dataset(X, y; extra=NamedTuple()),
        symmetric_options,
    )
    @test_throws ArgumentError SR.LossFunctionsModule.eval_loss(
        expr,
        SR.Dataset(X, y; weights=[1.0, 1.0], extra=(sigma=[1.0, 1.0],)),
        symmetric_options,
    )
    @test_throws ArgumentError SR.Options(
        uncertainty_mode=:asymmetry,
        loss_preset=:huber,
        default_plugins=(),
    )
    @test_throws ArgumentError SR.Options(
        uncertainty_mode=:asymmetry,
        loss_preset=:asymmetric_gaussian_nll,
        default_plugins=(),
    )
    @test_throws ArgumentError SR.equation_search(
        X,
        y;
        sigma_minus=[1.0, -1.0],
        sigma_plus=[1.0, 1.0],
        options=asym_options,
        niterations=0,
        parallelism=:serial,
    )
end

@testset "MySRCore package identity" begin
    @test nameof(MySRCore) == :MySRCore
    @test isdefined(MySRCore, :Options)
    @test isdefined(MySRCore, :equation_search)
end

@testset "Parent selection policies" begin
    SR = MySRCore.SymbolicRegression
    @test SR.Options().parent_selection == :tournament
    @test SR.Options().survival_strategy == :regularized_evolution
    @test SR.Options(
        parent_selection=:epsilon_lexicase,
        survival_strategy=:age_fitness_pareto,
        default_plugins=(),
    ).parent_selection == :epsilon_lexicase
    @test_throws ArgumentError SR.Options(parent_selection=:unknown)
    @test_throws ArgumentError SR.Options(survival_strategy=:unknown)

    errors = [0.0 1.0; 2.0 3.0]
    @test SR.epsilon_lexicase_index(errors; rng=MersenneTwister(1)) == 1
    @test SR.epsilon_lexicase_index(errors; rng=MersenneTwister(2)) == 1

    pool = [
        (cost=1.0, birth=1),
        (cost=2.0, birth=2),
        (cost=0.5, birth=3),
    ]
    survivors = SR.age_fitness_pareto_survivor_indices(pool, 2)
    @test length(survivors) == 2
    @test 3 in survivors
    @test !(2 in survivors)

    dataset = SR.Dataset(reshape(Float64[1, 2, 3], 1, :), [1.0, 2.0, 3.0])
    options = SR.Options(
        parent_selection=:epsilon_lexicase,
        batching=false,
        default_plugins=(),
    )
    @test SR.parent_selection_diagnostic(options, dataset).reason == :supported
    batched_options = SR.Options(
        parent_selection=:epsilon_lexicase,
        batching=true,
        default_plugins=(),
    )
    @test SR.parent_selection_diagnostic(batched_options, dataset).reason == :batching_enabled
    uncertainty_options = SR.Options(
        parent_selection=:epsilon_lexicase,
        uncertainty_mode=:symmetry,
        batching=false,
        default_plugins=(),
    )
    uncertainty_diagnostic = SR.parent_selection_diagnostic(uncertainty_options, dataset)
    @test uncertainty_diagnostic.effective == :epsilon_lexicase
    @test uncertainty_diagnostic.reason == :supported
    expr = SR.parse_expression(
        "x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=SR.Node{Float64,1},
    )
    case_losses = SR.LossFunctionsModule.eval_case_losses(expr, dataset, options)
    @test case_losses isa Vector{Float64}
    @test length(case_losses) == dataset.n

    uncertainty_dataset = SR.Dataset(
        reshape(Float64[1, 2, 3], 1, :),
        [0.0, 2.0, 1.0];
        extra=(sigma=Float64[1.0, 2.0, 4.0],),
    )
    uncertainty_case_losses = SR.LossFunctionsModule.eval_case_losses(
        expr,
        uncertainty_dataset,
        uncertainty_options,
    )
    uncertainty_loss = SR.LossFunctionsModule.eval_loss(
        expr,
        uncertainty_dataset,
        uncertainty_options,
    )
    @test uncertainty_case_losses ≈ [1.0, 0.0, 0.25]
    @test uncertainty_loss ≈ sum(uncertainty_case_losses) / length(uncertainty_case_losses)

    preset_options = SR.Options(
        parent_selection=:epsilon_lexicase,
        loss_preset=:l1,
        batching=false,
        default_plugins=(),
    )
    preset_diagnostic = SR.parent_selection_diagnostic(preset_options, dataset)
    @test preset_diagnostic.effective == :epsilon_lexicase
    @test preset_diagnostic.reason == :supported
    preset_case_losses = SR.LossFunctionsModule.eval_case_losses(expr, dataset, preset_options)
    preset_loss = SR.LossFunctionsModule.eval_loss(expr, dataset, preset_options)
    @test preset_loss ≈ sum(preset_case_losses) / length(preset_case_losses)

    weighted_dataset = SR.Dataset(
        reshape(Float64[1, 2, 3], 1, :),
        [0.0, 2.0, 1.0];
        weights=[2.0, 1.0, 3.0],
    )
    weighted_case_losses = SR.LossFunctionsModule.eval_case_losses(
        expr,
        weighted_dataset,
        preset_options,
    )
    weighted_loss = SR.LossFunctionsModule.eval_loss(expr, weighted_dataset, preset_options)
    @test weighted_case_losses ≈ [2.0, 0.0, 6.0]
    @test weighted_loss ≈ sum(weighted_case_losses) / sum(weighted_dataset.weights)

    custom_options = SR.Options(
        parent_selection=:epsilon_lexicase,
        loss_function=(tree, dataset, options) -> 0.0,
        batching=false,
        default_plugins=(),
    )
    @test SR.parent_selection_diagnostic(custom_options, dataset).reason ==
        :custom_aggregate_loss
end

@testset "Opt-in surrogate gate" begin
    SR = MySRCore.SymbolicRegression
    @test !SR.Options(default_plugins=()).surrogate_enabled
    @test SR.surrogate_stats(nothing).samples == 0
    options = SR.Options(
        binary_operators=(+,),
        unary_operators=(),
        default_plugins=(),
        surrogate_enabled=true,
        surrogate_warmup_evals=2,
        surrogate_probe_size=3,
        surrogate_neighbors=2,
        surrogate_max_samples=8,
    )
    @test options.surrogate_enabled
    @test options.surrogate_model == :knn
    @test_throws ArgumentError SR.Options(surrogate_neighbors=0)
    @test_throws ArgumentError SR.Options(surrogate_model=:invalid)

    X = reshape(Float64[1.0, 2.0, 3.0, 4.0], 1, :)
    y = copy(vec(X))
    dataset = SR.Dataset(X, y; variable_names=["x1"])
    tree = SR.parse_expression(
        "x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=SR.Node{Float64,2},
    )
    state = SR.SurrogateModule.create_surrogate_state(dataset, options)
    member = SR.PopMember(dataset, tree, options; deterministic=true)
    SR.SurrogateModule.observe_surrogate_member!(state, member, dataset, options)
    @test SR.surrogate_stats(state).true_evaluations == 1
    @test SR.surrogate_stats(state).samples == 1
    decision = SR.SurrogateModule.consider_surrogate!(
        state,
        tree,
        dataset,
        options,
        SR.compute_complexity(tree, options),
        member.cost,
    )
    @test decision isa SR.SurrogateDecision
    @test SR.surrogate_stats(state).proposals == 1
end

@testset "Surrogate search integration" begin
    SR = MySRCore.SymbolicRegression
    X = reshape(Float64[1, 2, 3, 4, 5, 6], 1, :)
    y = 2 .* vec(X) .+ 1
    options = SR.Options(
        binary_operators=(+, -, *),
        unary_operators=(),
        default_plugins=(),
        surrogate_enabled=true,
        surrogate_warmup_evals=2,
        surrogate_probe_size=4,
        surrogate_neighbors=2,
        surrogate_max_samples=16,
        surrogate_true_eval_fraction=0.5,
        surrogate_exploration_fraction=0.0,
        maxsize=8,
        population_size=8,
        populations=1,
        tournament_selection_n=2,
        ncycles_per_iteration=2,
        crossover_probability=0.5,
        should_optimize_constants=false,
        save_to_file=false,
        seed=1,
    )
    hall = SR.equation_search(
        X,
        y;
        options=options,
        niterations=1,
        parallelism=:serial,
        verbosity=0,
    )
    @test hall isa SR.HallOfFame
    @test !isempty(hall.members)
end

@testset "Surrogate snapshot synchronization" begin
    SR = MySRCore.SymbolicRegression
    options = SR.Options(
        binary_operators=(+,),
        unary_operators=(),
        default_plugins=(),
        surrogate_enabled=true,
        surrogate_probe_size=3,
        surrogate_max_samples=4,
    )
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    dataset = SR.Dataset(X, copy(vec(X)); variable_names=["x1"])
    tree = SR.parse_expression(
        "x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=SR.Node{Float64,2},
    )
    state = SR.SurrogateModule.create_surrogate_state(dataset, options)
    member = SR.PopMember(dataset, tree, options; deterministic=true)
    SR.SurrogateModule.observe_surrogate_member!(state, member, dataset, options)
    report = SR.SurrogateModule.surrogate_report(
        state;
        output=1,
        population=1,
        iteration=0,
    )
    snapshot = SR.SurrogateModule.merge_surrogate_reports(
        nothing,
        [report, report];
        generation=1,
    )
    @test snapshot.generation == 1
    @test length(snapshot.features) == 1
    seeded = SR.SurrogateModule.create_surrogate_state(
        dataset,
        options;
        snapshot=snapshot,
    )
    @test SR.surrogate_stats(seeded).samples == 1
    @test seeded.base_generation == 1
end

@testset "Surrogate shared snapshot search integration" begin
    SR = MySRCore.SymbolicRegression
    X = reshape(Float64[1, 2, 3, 4, 5, 6], 1, :)
    y = 2 .* vec(X) .+ 1
    options = SR.Options(
        binary_operators=(+, -, *),
        unary_operators=(),
        default_plugins=(),
        surrogate_enabled=true,
        surrogate_warmup_evals=2,
        surrogate_probe_size=4,
        surrogate_neighbors=2,
        surrogate_max_samples=16,
        surrogate_true_eval_fraction=0.5,
        surrogate_exploration_fraction=0.0,
        maxsize=8,
        population_size=6,
        populations=2,
        tournament_selection_n=2,
        ncycles_per_iteration=1,
        crossover_probability=0.5,
        should_optimize_constants=false,
        save_to_file=false,
        seed=2,
    )
    hall = SR.equation_search(
        X,
        y;
        options=options,
        niterations=1,
        parallelism=:serial,
        verbosity=0,
    )
    @test hall isa SR.HallOfFame
    @test !isempty(hall.members)
end

@testset "Size-matched crossover" begin
    SR = MySRCore.SymbolicRegression
    MutationFunctions = SR.MutationFunctionsModule
    @test SR.default_crossovers() == [SR.SubtreeCrossover() => 1.0]
    @test SR.SizeMatchedCrossover(; size_tolerance=0).size_tolerance == 0.0
    @test_throws ArgumentError SR.SizeMatchedCrossover(; size_tolerance=-0.1)
    @test_throws ArgumentError SR.SizeMatchedCrossover(; size_tolerance=NaN)
    @test_throws ArgumentError SR.SizeMatchedCrossover(; size_tolerance=Inf)

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
    @test MutationFunctions._select_size_matched_index(
        [1, 2, 3], 2, 0.0, MersenneTwister(1)
    ) == 2
    @test_throws ArgumentError MutationFunctions._select_size_matched_index(
        Int[], 1, 0.0, MersenneTwister(1)
    )
    @test_throws ArgumentError MutationFunctions._select_size_matched_index(
        [1], 0, 0.0, MersenneTwister(1)
    )
    @test_throws ArgumentError MutationFunctions._select_size_matched_index(
        [1], 1, -0.1, MersenneTwister(1)
    )
    @test_throws ArgumentError MutationFunctions._select_size_matched_index(
        [1], 1, NaN, MersenneTwister(1)
    )
    for seed in 1:12
        @test MutationFunctions._select_size_matched_index(
            [1, 2, 4], 3, 0.0, MersenneTwister(seed)
        ) in (2, 3)
    end
    # Exercise the public dispatch entry, not only the tree-level helper.
    dispatch_dataset = SR.Dataset(
        [1.0 2.0; 2.0 3.0],
        [3.0, 5.0];
        variable_names=["x1", "x2"],
        X_dimensions=[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
        y_dimensions=[0, 0, 0, 0, 0, 0, 0],
    )
    dispatch_member1 = SR.PopMember(dispatch_dataset, parent1, options; deterministic=true)
    dispatch_member2 = SR.PopMember(dispatch_dataset, parent2, options; deterministic=true)
    dispatched = SR.crossover(
        dispatch_member1,
        dispatch_member2,
        SR.SizeMatchedCrossover(; size_tolerance=0.0),
        options;
        trace=nothing,
    )
    @test dispatched isa SR.CrossoverResult
    @test SR.count_nodes(SR.get_tree(dispatched.child1)) == SR.count_nodes(SR.get_tree(parent1))
    @test SR.count_nodes(SR.get_tree(dispatched.child2)) == SR.count_nodes(SR.get_tree(parent2))
    before1, before2 = SR.string_tree(parent1), SR.string_tree(parent2)
    for seed in 1:12
        child1, child2 = MutationFunctions.size_matched_crossover_trees(
            parent1, parent2, 0.0, MersenneTwister(seed)
        )
        parent_node_ids = Set(objectid(node) for node in SR.get_tree(parent1))
        parent_node_ids = union(
            parent_node_ids, Set(objectid(node) for node in SR.get_tree(parent2))
        )
        child_node_ids = Set(objectid(node) for node in SR.get_tree(child1))
        child2_node_ids = Set(objectid(node) for node in SR.get_tree(child2))
        child_node_ids = union(child_node_ids, child2_node_ids)
        @test SR.count_nodes(SR.get_tree(child1)) == SR.count_nodes(SR.get_tree(parent1))
        @test SR.count_nodes(SR.get_tree(child2)) == SR.count_nodes(SR.get_tree(parent2))
        @test isempty(intersect(parent_node_ids, child_node_ids))
        @test isempty(intersect(
            Set(objectid(node) for node in SR.get_tree(child1)), child2_node_ids
        ))
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

@testset "Population-specific search profiles" begin
    base = Options(
        binary_operators=(+, -, *, /),
        unary_operators=(sin, cos),
        populations=2,
        default_plugins=(),
    )
    affinity = [ones(size(matrix)) for matrix in base.operator_affinity]
    affinity[2][1, 2] = 7.0
    mutation_multipliers = ones(length(base.mutations))
    mutation_multipliers[1] = 0.0
    profile = IslandProfile(
        id=:algebraic,
        role=:algebraic,
        operator_affinity=affinity,
        mutation_weights=mutation_multipliers,
        crossover_weights=ones(length(base.crossovers)),
        exploration_floor=0.35,
    )
    generalist = IslandProfile(id=:generalist, role=:generalist)
    options = Options(
        binary_operators=(+, -, *, /),
        unary_operators=(sin, cos),
        populations=2,
        population_profiles=[profile, generalist],
        default_plugins=(),
    )
    local_options = profiled_options(options, 1)
    @test local_options.profile.id == :algebraic
    @test local_options.profile.role == :algebraic
    @test local_options.operator_affinity[2][1, 2] == 7.0
    @test local_options.mutations[1].second == 0.0
    @test local_options.mutation_affinity_exploration == 0.35
    @test profiled_options(options, 2).profile.id == :generalist
    @test profiled_options(options, 2).operator_affinity == options.operator_affinity

    trigonometric = profiled_options(
        base, IslandProfile(role=:trigonometric, operator_preference_strength=3.0)
    )
    algebraic = profiled_options(
        base, IslandProfile(role=:algebraic, operator_preference_strength=3.0)
    )
    @test trigonometric.operator_affinity[1][1, 1] > algebraic.operator_affinity[1][1, 1]
    no_global_affinity = Options(
        binary_operators=(+, -, *, /),
        unary_operators=(sin, cos),
        mutation_affinity=:none,
        populations=1,
        default_plugins=(),
    )
    custom_view = profiled_options(
        no_global_affinity,
        IslandProfile(operator_affinity=[ones(size(m)) for m in no_global_affinity.operator_affinity]),
    )
    @test custom_view.mutation_affinity == :family

    @test_throws ArgumentError Options(
        populations=2,
        population_profiles=[generalist],
        default_plugins=(),
    )
    @test_throws ArgumentError IslandProfile(exploration_floor=1.1)
end

@testset "Migration topology candidate pools" begin
    Migration = MySRCore.SymbolicRegression.MigrationModule
    best_sub_pops = [(members=[1, 2],), (members=[3, 4],), (members=[5, 6],)]
    @test Migration.migration_candidates(best_sub_pops, 1, :ring) == [5, 6]
    @test Migration.migration_candidates(best_sub_pops, 2, :ring) == [1, 2]
    @test Migration.migration_candidates(best_sub_pops, 3, :ring) == [3, 4]
    @test Migration.migration_candidates(best_sub_pops, 2, :pooled) == [1, 2, 3, 4, 5, 6]
    @test_throws ArgumentError Migration.migration_candidates(best_sub_pops, 1, :star)
    @test Options(migration_policy=:best_plus_novelty).migration_policy == :best_plus_novelty
    @test_throws ArgumentError Options(migration_policy=:unknown)
end

@testset "Migration novelty and profile compatibility" begin
    Migration = MySRCore.SymbolicRegression.MigrationModule
    X = reshape(Float64[1, 2, 3], 1, :)
    y = vec(X)
    dataset = Dataset(X, y; variable_names=["x1"])
    options = Options(
        binary_operators=(+, -),
        unary_operators=(),
        populations=2,
        default_plugins=(),
        deterministic=true,
    )
    plus_tree = parse_expression(
        "x1 + x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    minus_tree = parse_expression(
        "x1 - x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    plus_member = PopMember(dataset, plus_tree, options; deterministic=true)
    minus_member = PopMember(dataset, minus_tree, options; deterministic=true)
    source = [(members=[plus_member],), (members=[minus_member],)]
    destination = Population([plus_member])
    # Structural novelty removes the duplicate plus tree from the destination;
    # ring destination 1 receives only the predecessor (population 2).
    @test Migration.migration_candidates(
        source,
        1,
        :ring;
        policy=:best_plus_novelty,
        destination_pop=destination,
    ) == [minus_member]
    # A profile matrix with a zero destination column acts as an explicit
    # migration compatibility mask for that operator.
    compatible_plus_only = IslandProfile(
        operator_affinity=[zeros(0, 0), Float64[1 0; 1 0]]
    )
    @test isempty(Migration.migration_candidates(
        [(members=[plus_member],), (members=[minus_member],)],
        1,
        :ring;
        policy=:best_plus_novelty,
        destination_pop=destination,
        profile=compatible_plus_only,
    ))
    @test_throws ArgumentError Migration.migration_candidates(
        source, 1, :ring; policy=:unknown
    )
end

@testset "Population profiles run through the search loop" begin
    X = reshape(Float64[-2, -1, 0, 1, 2], 1, :)
    y = 2 .* vec(X) .+ 1
    base = Options(
        binary_operators=(+, -, *, /),
        unary_operators=(sin, cos),
        populations=2,
        population_size=6,
        tournament_selection_n=2,
        ncycles_per_iteration=1,
        maxsize=7,
        default_plugins=(),
    )
    matrices = [ones(size(matrix)) for matrix in base.operator_affinity]
    profile_a = IslandProfile(id=:algebraic, operator_affinity=matrices)
    profile_b = IslandProfile(id=:trigonometric, operator_affinity=matrices)
    options = Options(
        binary_operators=(+, -, *, /),
        unary_operators=(sin, cos),
        populations=2,
        population_size=6,
        tournament_selection_n=2,
        ncycles_per_iteration=1,
        maxsize=7,
        population_profiles=[profile_a, profile_b],
        migration_topology=:ring,
        migration_policy=:best_plus_novelty,
        default_plugins=(),
        save_to_file=false,
        deterministic=true,
        seed=2026,
    )
    hall = equation_search(
        X,
        y;
        niterations=1,
        options,
        parallelism=:serial,
        progress=false,
        verbosity=0,
    )
    @test length(hall.members) == options.maxsize
    @test profiled_options(options, 1).profile.id == :algebraic
    @test profiled_options(options, 2).profile.id == :trigonometric
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

@testset "RNN-GPSR tokenization handles unary roots" begin
    PopulationSeeding = MySRCore.SymbolicRegression.PopulationSeedingModule
    options = Options(
        binary_operators=(+,),
        unary_operators=(sin,),
        default_plugins=(),
        maxsize=7,
    )
    expression = parse_expression(
        "sin(x1)";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    tokens = PopulationSeeding._expression_tokens(get_tree(expression), options, 1)
    @test length(tokens) == 2
    @test tokens[1] > 1 + 1
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
        population_size=27,
        tournament_selection_n=2,
        rnn_gpsr_seeding=true,
        rnn_gpsr_candidate_count=27,
        rnn_gpsr_proposal_count=27,
        rnn_gpsr_cycles=0,
        rnn_gpsr_rounds=2,
        rnn_gpsr_feedback_fraction=0.2,
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
    # A small first feedback fraction is appended to (rather than replacing)
    # the bootstrap corpus.  With candidate_count=27, six feedback examples
    # extend the initial 27-example corpus to 33 and preserve the minimum.
    @test observed_training_counts == [27, 33]
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
    @test evaluations == options.population_size + options.rnn_gpsr_candidate_count
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

@testset "RNN-GPSR real feedback can replace structural bootstrap" begin
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = copy(vec(X))
    options = Options(
        default_plugins=(),
        maxsize=7,
        save_to_file=false,
        rnn_gpsr_feedback_fraction=1.0,
    )
    dataset = Dataset(X, y; variable_names=["x1"])
    tree = parse_expression(
        "x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=Node{Float64,2},
    )
    member = PopMember(dataset, tree, options; deterministic=true)
    member.cost = 0.25
    feedback_members = [copy(member) for _ in 1:8]
    training_sequences = [Int[2], Int[2, 2]]
    training_costs = [9.0, 8.0]
    used = MySRCore.SymbolicRegression.PopulationSeedingModule._append_feedback_examples!(
        training_sequences,
        training_costs,
        feedback_members,
        options,
        1;
        replace_bootstrap=true,
    )
    @test used == 8
    @test training_sequences == [Int[2] for _ in 1:8]
    @test training_costs == [0.25 for _ in 1:8]
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

@testset "Hall of fame frontier preserves nonfinite-loss semantics" begin
    SR = MySRCore.SymbolicRegression
    options = SR.Options(
        default_plugins=(),
        binary_operators=(+,),
        maxsize=5,
        save_to_file=false,
    )
    dataset = SR.Dataset(reshape(Float64[1, 2, 3], 1, :), Float64[1, 2, 3]; variable_names=["x1"])
    hall = SR.HallOfFame(options, dataset)
    HOFModule = SR.HallOfFameModule
    losses = [10.0, 5.0, NaN, Inf, -Inf]
    for (complexity, loss) in enumerate(losses)
        # Reuse the HOF's expression metadata so the concrete PopMember type
        # matches the preallocated member slots.
        tree = copy(hall.members[1].tree)
        member = SR.PopMember(
            tree,
            loss,
            loss,
            options,
            complexity;
            deterministic=true,
        )
        hall.members[complexity] = member
        hall.exists[complexity] = true
    end
    frontier = HOFModule.calculate_pareto_frontier(hall)
    @test isequal([member.loss for member in frontier], [10.0, 5.0, NaN, -Inf])
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

@testset "RNN-GPSR empty callback falls back to valid random trees" begin
    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = copy(vec(X))
    options = Options(
        default_plugins=(),
        maxsize=5,
        deterministic=true,
        save_to_file=false,
    )
    dataset = Dataset(X, y; variable_names=["x1"])
    generator(args...) = nothing

    trees = MySRCore.SymbolicRegression.PopulationSeedingModule._generate_proposal_trees(
        generator,
        [Int[2] for _ in 1:8],
        ones(Float64, 8),
        dataset,
        Float64,
        options,
        1,
        5,
        3,
        2026,
        MersenneTwister(9),
    )

    @test length(trees) == 3
    @test all(
        tree -> MySRCore.SymbolicRegression.check_constraints(
            tree,
            dataset,
            options,
            5,
        ),
        trees,
    )
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

@testset "Dimension-only operator fast paths" begin
    # These expressions exercise the common arithmetic/unary branches in
    # `_transition_dimension`; the matching output dimensions also verify that
    # bypassing temporary Quantity values preserves the public contract.
    options = Options(
        formula_type=:theoretical,
        unary_operators=(sqrt, sin),
        default_plugins=(),
    )
    X = [1.0 2.0; 2.0 4.0]
    dims = [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]]
    cases = (
        ("x1 * x2", [1, 1, 0, 0, 0, 0, 0]),
        ("x1 / x2", [1, -1, 0, 0, 0, 0, 0]),
        ("sin(x1 / x1)", [0, 0, 0, 0, 0, 0, 0]),
    )
    for (formula, y_dimension) in cases
        dataset = Dataset(X, [1.0, 2.0];
            variable_names=["x1", "x2"],
            X_dimensions=dims,
            y_dimensions=y_dimension,
        )
        tree = parse_expression(formula;
            operators=options.operators,
            variable_names=["x1", "x2"],
            node_type=Node{Float64,2},
        )
        result = infer_dimension_static(tree, dataset, options)
        @test result.valid
        @test result.output_dimension == dimension(dataset.y_dimensions)
    end
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

@testset "Dimensional randomize mutation preserves expression type" begin
    X = Float64[1 2 3; 1 2 3]
    y = Float64[2, 4, 6]
    length_dim = [1, 0, 0, 0, 0, 0, 0]
    dataset = Dataset(
        X,
        y;
        variable_names=["x1", "x2"],
        X_dimensions=[length_dim, length_dim],
        y_dimensions=length_dim,
    )
    options = Options(
        binary_operators=(+, -, *, /),
        unary_operators=(),
        formula_type=:theoretical,
        default_plugins=(),
        maxsize=7,
    )
    expr = parse_expression(
        "x1";
        operators=options.operators,
        variable_names=["x1", "x2"],
        node_type=Node{Float64,2},
    )
    member = PopMember(dataset, expr, options; deterministic=true)
    result = MySRCore.SymbolicRegression.MutateModule.mutate!(
        copy(member.tree),
        member,
        RandomizeMutation(),
        options;
        trace=nothing,
        dataset=dataset,
        curmaxsize=options.maxsize,
        nfeatures=2,
    )
    @test result.tree isa typeof(member.tree)
    @test infer_dimension_static(result.tree, dataset, options).valid

    semi_dataset = Dataset(
        X,
        y;
        variable_names=["x1", "x2"],
        X_dimensions=[length_dim, length_dim],
        y_dimensions=[2, 0, 0, 0, 0, 0, 0],
    )
    semi_options = Options(
        binary_operators=(+, -, *, /),
        unary_operators=(),
        formula_type=:semi_theoretical,
        default_plugins=(),
        maxsize=7,
    )
    semi_expr = parse_expression(
        "x1";
        operators=semi_options.operators,
        variable_names=["x1", "x2"],
        node_type=Node{Float64,2},
    )
    semi_tree = MySRCore.SymbolicRegression.wrap_dimensional_scale(
        semi_expr, semi_options; coefficient=2.5
    )
    semi_member = PopMember(semi_dataset, semi_tree, semi_options; deterministic=true)
    semi_result = MySRCore.SymbolicRegression.MutateModule.mutate!(
        copy(semi_member.tree),
        semi_member,
        RandomizeMutation(),
        semi_options;
        trace=nothing,
        dataset=semi_dataset,
        curmaxsize=semi_options.maxsize,
        nfeatures=2,
    )
    @test semi_result.tree isa typeof(semi_member.tree)
    @test MySRCore.SymbolicRegression.dimensional_scale_coefficient(
        semi_result.tree, semi_options
    ) == 2.5
    @test infer_dimension_static(semi_result.tree, semi_dataset, semi_options).valid
end

@testset "TemplateExpression get_tree preserves declared feature arity" begin
    MutationFunctions = MySRCore.SymbolicRegression.MutationFunctionsModule
    options = Options(
        binary_operators=(+, -, *, /),
        unary_operators=(),
        formula_type=:empirical,
        default_plugins=(),
    )
    inner = ComposableExpression(
        Node{Float64,2}(; feature=1);
        operators=options.operators,
        variable_names=["x", "y"],
    )
    combine_fn = ((; f), (x, y)) -> f(x, y)
    structure = TemplateStructure{(:f,)}(
        combine_fn;
        num_features=(; f=2),
    )
    template = TemplateExpression(
        (; f=inner);
        structure,
        operators=options.operators,
        variable_names=["x", "y"],
    )

    # `f(x, y)` must receive both declared variables even though the template
    # contains only one named inner expression.
    tree = get_tree(template)
    @test tree isa AbstractExpressionNode
    @test tree.degree == 0
    @test tree.feature == 1

    dataset = Dataset(
        Float64[1 2; 2 3],
        Float64[1, 2];
        variable_names=["x", "y"],
    )
    @test MySRCore.SymbolicRegression.CheckConstraintsModule.check_constraints(
        template, dataset, options, options.maxsize
    )

    # A fixed combiner operation is available to the template expression even
    # when it is intentionally absent from the evolutionary search vocabulary.
    fixed_template = TemplateExpression(
        (; f=inner);
        structure=TemplateStructure{(:f,)}(
            ((; f), (x, y)) -> sin(f(x, y));
            num_features=(; f=2),
        ),
        operators=options.operators,
        variable_names=["x", "y"],
    )
    @test sin ∉ options.operators.unaops
    @test sin ∉ get_metadata(fixed_template).operators.unaops
    @test fixed_template(Float64[1 2; 2 3]) ≈ sin.([1.0, 2.0])
    fixed_tree = get_tree(fixed_template)
    @test fixed_tree.degree == 1

    # For multiple inner expressions, feature arity is the sum of each
    # expression's declared inputs, rather than the maximum arity.
    multi_structure = TemplateStructure{(:f, :g)}(
        ((; f, g), (x1, x2, x3)) -> f(x1, x2) + g(x3);
        num_features=(; f=2, g=1),
    )
    multi_template = TemplateExpression(
        (; f=inner, g=inner);
        structure=multi_structure,
        operators=options.operators,
        variable_names=["x", "y", "z"],
    )
    multi_tree = get_tree(multi_template)
    @test multi_tree.degree == 2
    @test get_child(multi_tree, 1).feature == 1
    @test get_child(multi_tree, 2).feature == 3
    @test_throws ArgumentError MutationFunctions.size_matched_crossover_trees(
        fixed_template, multi_template, -0.1, MersenneTwister(3)
    )
end

@testset "TemplateExpression records custom combiner operators" begin
    struct CustomTemplateValue
        data::Float64
    end
    add_custom(x::CustomTemplateValue, y::CustomTemplateValue) =
        CustomTemplateValue(x.data + y.data)
    function add_custom(x::ValidVector, y::ValidVector)
        return ValidVector(map(add_custom, x.x, y.x), x.valid && y.valid)
    end

    operators = OperatorEnum(2 => (add_custom,))
    spec = @template_spec(expressions=(f, g), prototype=CustomTemplateValue(1.0)) do x1, x2
        add_custom(f(x1), g(x2))
    end
    inner = ComposableExpression(
        Node{CustomTemplateValue,2}(; feature=1);
        operators,
        variable_names=["x1", "x2"],
    )
    template = TemplateExpression(
        (; f=inner, g=inner);
        structure=spec.structure,
        operators,
        variable_names=["x1", "x2"],
    )

    tree = get_tree(template)
    @test tree.degree == 2
    @test tree.op == 1
    @test spec.structure.num_features == (; f=1, g=1)
end
