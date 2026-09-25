@testset "Final semantic subtree refinement" begin
    SR = MySRCore.SymbolicRegression
    @test SR.FinalRefinementOptions().max_evals == 500
    @test_throws ArgumentError SR.FinalRefinementOptions(max_evals=0)
    @test_throws ArgumentError SR.FinalRefinementOptions(proposal_temperature=0)

    X = reshape(Float64[1, 2, 3, 4], 1, :)
    y = 2 .* vec(X) .+ 1
    options = SR.Options(
        binary_operators=(+,),
        unary_operators=(),
        default_plugins=(),
        maxsize=8,
        should_optimize_constants=false,
        save_to_file=false,
        seed=11,
    )
    dataset = SR.Dataset(X, y; variable_names=["x1"])
    parsed = SR.parse_expression(
        "x1 + x1";
        operators=options.operators,
        variable_names=["x1"],
        node_type=SR.Node{Float64,2},
    )
    hof = SR.HallOfFame(options, dataset)
    member = SR.PopMember(dataset, SR.get_tree(parsed), options; deterministic=true)
    SR.HallOfFameModule.update_hall_of_fame!(hof, member, dataset, options)

    library = SR.FinalRefinementLibrary(["x1", "not_a_valid_expression"])
    original_hof = copy(hof)
    result = SR.finalize_search(
        hof,
        dataset;
        options,
        library,
        refinement=SR.FinalRefinementOptions(
            max_evals=5,
            max_rounds=1,
            beam_width=2,
            elite_count=1,
            max_subtrees_per_member=2,
            max_replacements_per_subtree=2,
        ),
        rng=MersenneTwister(7),
    )
    @test result isa SR.FinalRefinementResult
    @test result.original_hall_of_fame.members[3].tree == original_hof.members[3].tree
    @test result.report.evaluations <= 5
    @test result.report.library_counts.static >= 2
    @test !isempty(result.report.invalid_terms)
    @test isempty(library.static_terms)
    @test isempty(library.dynamic_terms)

    repeat_result = SR.finalize_search(
        hof,
        dataset;
        options,
        refinement=SR.FinalRefinementOptions(
            max_evals=5,
            max_rounds=1,
            beam_width=2,
            elite_count=1,
            max_subtrees_per_member=2,
            max_replacements_per_subtree=2,
        ),
        rng=MersenneTwister(7),
    )
    @test SR.string_tree(
        SR.get_tree(result.hall_of_fame.members[3].tree), options; pretty=false
    ) == SR.string_tree(
        SR.get_tree(repeat_result.hall_of_fame.members[3].tree), options; pretty=false
    )

    budget_options = SR.Options(
        binary_operators=(+,),
        unary_operators=(),
        default_plugins=(),
        maxsize=8,
        should_optimize_constants=true,
        save_to_file=false,
        seed=11,
    )
    constant_tree = SR.parse_expression(
        "x1 + 1.0";
        operators=budget_options.operators,
        variable_names=["x1"],
        node_type=SR.Node{Float64,2},
    )
    budget_hof = SR.HallOfFame(budget_options, dataset)
    budget_member = SR.PopMember(dataset, SR.get_tree(constant_tree), budget_options; deterministic=true)
    SR.HallOfFameModule.update_hall_of_fame!(budget_hof, budget_member, dataset, budget_options)
    budget_result = SR.finalize_search(
        budget_hof,
        dataset;
        options=budget_options,
        refinement=SR.FinalRefinementOptions(
            max_evals=1,
            max_rounds=1,
            beam_width=2,
            elite_count=1,
            max_subtrees_per_member=4,
            max_replacements_per_subtree=4,
        ),
        rng=MersenneTwister(13),
    )
    @test budget_result.report.evaluations <= 1

    validation_dataset = SR.Dataset(
        reshape(Float64[1.5, 2.5, 3.5, 4.5], 1, :),
        2 .* Float64[1.5, 2.5, 3.5, 4.5] .+ 1;
        variable_names=["x1"],
    )
    validation_result = SR.finalize_search(
        hof,
        dataset;
        options,
        validation_dataset,
        refinement=SR.FinalRefinementOptions(
            max_evals=3,
            max_rounds=1,
            beam_width=2,
            elite_count=1,
            max_subtrees_per_member=2,
            max_replacements_per_subtree=2,
        ),
        rng=MersenneTwister(17),
    )
    @test validation_result.report.validation_evaluations <=
        validation_result.report.attempted_replacements + 1
    @test length(validation_result.report.validation_costs) <=
        validation_result.report.validation_evaluations
    @test_throws DimensionMismatch SR.finalize_search(
        hof,
        dataset;
        options,
        validation_dataset=SR.Dataset(reshape(Float64[1, 2, 3, 4], 2, 2), [1.0, 2.0]),
        refinement=SR.FinalRefinementOptions(max_evals=1, max_rounds=1),
    )

    reused_result = SR.finalize_search(
        hof,
        validation_dataset;
        options,
        library=result.library,
        refinement=SR.FinalRefinementOptions(
            max_evals=1,
            max_rounds=1,
            beam_width=1,
            elite_count=1,
            max_subtrees_per_member=1,
            max_replacements_per_subtree=1,
        ),
        rng=MersenneTwister(19),
    )
    @test result.library.static_terms[1].dataset_signature !=
        reused_result.library.static_terms[1].dataset_signature
    @test all(entry.input_features isa Vector{Int} for entry in reused_result.library.static_terms)
end
