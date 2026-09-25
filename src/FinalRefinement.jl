"""Bounded, opt-in semantic subtree refinement performed after a search."""
module FinalRefinementModule

using Random: AbstractRNG, default_rng, shuffle!
using DynamicExpressions:
    AbstractExpression,
    AbstractExpressionNode,
    constructorof,
    copy_node,
    eval_tree_array,
    get_child,
    get_tree,
    has_constants,
    set_child!,
    with_contents,
    string_tree

using ..CoreModule: AbstractOptions, Dataset, Options
using ..PopMemberModule: AbstractPopMember, PopMember
using ..PopulationModule: Population
using ..HallOfFameModule: HallOfFame, calculate_pareto_frontier, update_hall_of_fame!
using ..ComplexityModule: compute_complexity
using ..CheckConstraintsModule: check_constraints
using ..LossFunctionsModule: eval_cost
using ..ConstantOptimizationModule: bounded_optimizer_options, optimize_constants
using ..MutationFunctionsModule: get_contents_for_mutation, with_contents_for_mutation
using ..SearchUtilsModule: AbstractSearchState
using DynamicExpressions: parse_expression, with_type_parameters

export FinalRefinementOptions,
    FinalRefinementLibrary,
    FinalRefinementLibraryEntry,
    FinalRefinementReport,
    FinalRefinementResult,
    finalize_search,
    add_final_refinement_term!,
    add_final_refinement_string!

"""Configuration for the bounded, post-search refinement pass."""
struct FinalRefinementOptions
    max_evals::Int
    max_rounds::Int
    beam_width::Int
    elite_count::Int
    max_subtrees_per_member::Int
    max_replacements_per_subtree::Int
    max_dynamic_terms::Int
    probe_size::Int
    proposal_temperature::Float64
    allow_root_replacement::Bool
end

function FinalRefinementOptions(;
    max_evals::Integer=500,
    max_rounds::Integer=3,
    beam_width::Integer=3,
    elite_count::Integer=8,
    max_subtrees_per_member::Integer=12,
    max_replacements_per_subtree::Integer=4,
    max_dynamic_terms::Integer=128,
    probe_size::Integer=128,
    proposal_temperature::Real=1.0,
    allow_root_replacement::Bool=false,
)
    for (name, value) in (
        (:max_evals, max_evals),
        (:max_rounds, max_rounds),
        (:beam_width, beam_width),
        (:elite_count, elite_count),
        (:max_subtrees_per_member, max_subtrees_per_member),
        (:max_replacements_per_subtree, max_replacements_per_subtree),
        (:max_dynamic_terms, max_dynamic_terms),
        (:probe_size, probe_size),
    )
        value > 0 || throw(ArgumentError("$name must be positive."))
    end
    isfinite(proposal_temperature) && proposal_temperature > 0 ||
        throw(ArgumentError("proposal_temperature must be finite and positive."))
    return FinalRefinementOptions(
        Int(max_evals),
        Int(max_rounds),
        Int(beam_width),
        Int(elite_count),
        Int(max_subtrees_per_member),
        Int(max_replacements_per_subtree),
        Int(max_dynamic_terms),
        Int(probe_size),
        Float64(proposal_temperature),
        allow_root_replacement,
    )
end

"""A library entry and its run-local replacement statistics."""
mutable struct FinalRefinementLibraryEntry
    expression::Any
    source::Symbol
    complexity::Int
    signature::Any
    weight::Float64
    attempts::Int
    accepted::Int
    gain::Float64
    rank::Int
end

"""Static and run-local dynamic terms used by final refinement.

`pending_terms` contains user supplied expressions and strings until a dataset
and the search options are available.  It is deliberately private to the
current run and is never written to global state.
"""
mutable struct FinalRefinementLibrary
    static_terms::Vector{FinalRefinementLibraryEntry}
    dynamic_terms::Vector{FinalRefinementLibraryEntry}
    pending_terms::Vector{Any}
    semantic_signatures::Dict{String,Any}
    replacement_gains::Dict{String,Float64}
end

function FinalRefinementLibrary(
    terms::AbstractVector=Any[];
    static_terms::AbstractVector=Any[],
    dynamic_terms::AbstractVector=Any[],
    dataset=nothing,
    options=nothing,
)
    static_entries = FinalRefinementLibraryEntry[]
    dynamic_entries = FinalRefinementLibraryEntry[]
    pending = Any[]
    for term in terms
        if term isa FinalRefinementLibraryEntry
            target = term.source in (:dynamic_hof, :dynamic_population) ? dynamic_entries : static_entries
            push!(target, _copy_entry(term))
        else
            push!(pending, term)
        end
    end
    for term in static_terms
        term isa FinalRefinementLibraryEntry ? push!(static_entries, _copy_entry(term)) : push!(pending, term)
    end
    for term in dynamic_terms
        term isa FinalRefinementLibraryEntry ? push!(dynamic_entries, _copy_entry(term)) : push!(pending, term)
    end
    return FinalRefinementLibrary(
        static_entries,
        dynamic_entries,
        pending,
        Dict{String,Any}(),
        Dict{String,Float64}(),
    )
end

"""Counters and diagnostics emitted by one refinement pass."""
mutable struct FinalRefinementReport
    evaluations::Int
    validation_evaluations::Int
    attempted_replacements::Int
    accepted_replacements::Int
    rejected_replacements::Int
    rounds_completed::Int
    stopped_reason::Symbol
    library_counts::NamedTuple{(:static, :dynamic),Tuple{Int,Int}}
    invalid_terms::Vector{Any}
    validation_costs::Vector{Float64}
    replacement_gains::Vector{Float64}
end

FinalRefinementReport() = FinalRefinementReport(
    0,
    0,
    0,
    0,
    0,
    0,
    :completed,
    (static=0, dynamic=0),
    Any[],
    Float64[],
    Float64[],
)

"""Non-destructive result of `finalize_search`."""
struct FinalRefinementResult{H,OH,L,R}
    hall_of_fame::H
    original_hall_of_fame::OH
    library::L
    report::R
end

_copy_entry(entry::FinalRefinementLibraryEntry) = FinalRefinementLibraryEntry(
    entry.expression isa AbstractExpressionNode ? copy_node(entry.expression) :
    entry.expression isa AbstractExpression ? with_contents(entry.expression, copy_node(get_tree(entry.expression))) :
    entry.expression,
    entry.source,
    entry.complexity,
    entry.signature,
    entry.weight,
    entry.attempts,
    entry.accepted,
    entry.gain,
    entry.rank,
)

function _copy_library(library::FinalRefinementLibrary)
    return FinalRefinementLibrary(
        [_copy_entry(entry) for entry in library.static_terms],
        [_copy_entry(entry) for entry in library.dynamic_terms],
        copy(library.pending_terms),
        copy(library.semantic_signatures),
        copy(library.replacement_gains),
    )
end

function add_final_refinement_term!(library::FinalRefinementLibrary, term; source::Symbol=:user)
    source in (:user, :static_user) || throw(ArgumentError("user terms must use source=:user."))
    push!(library.pending_terms, term)
    return library
end

function add_final_refinement_string!(library::FinalRefinementLibrary, formula::AbstractString)
    return add_final_refinement_term!(library, String(formula))
end

_term_key(tree, options) = try
    string_tree(tree, options; pretty=false)
catch
    sprint(show, tree)
end

function _node_term(term)
    if term isa AbstractExpression
        return copy_node(get_tree(term))
    elseif term isa AbstractExpressionNode
        return copy_node(term)
    end
    return nothing
end

function _parse_term(term, dataset::Dataset{T,L}, options::AbstractOptions) where {T,L}
    if term isa AbstractExpression || term isa AbstractExpressionNode
        return _node_term(term)
    elseif term isa AbstractString
        parsed = parse_expression(
            term;
            expression_type=options.expression_type,
            operators=options.operators,
            variable_names=dataset.variable_names,
            node_type=with_type_parameters(options.node_type, T),
        )
        return _node_term(parsed)
    end
    return nothing
end

function _signature(tree, dataset, options, probe_size)
    key = _term_key(tree, options)
    return key, get!(dataset === nothing ? Dict{String,Any}() : Dict{String,Any}(), key) do
        n = min(probe_size, dataset.n)
        n > 0 || return nothing
        X = dataset.X[:, 1:n]
        out, complete = try
            eval_tree_array(tree, X, options)
        catch
            nothing, false
        end
        complete && out !== nothing || return nothing
        values = try
            [isfinite(v) ? round(Float64(v), digits=8) : Float64(Inf) for v in out]
        catch
            return nothing
        end
        return hash(values)
    end
end

function _signature_cached!(library, tree, dataset, options, probe_size)
    key = _term_key(tree, options)
    if haskey(library.semantic_signatures, key)
        return library.semantic_signatures[key]
    end
    n = min(probe_size, dataset.n)
    signature = if n == 0
        nothing
    else
        out, complete = try
            eval_tree_array(tree, dataset.X[:, 1:n], options)
        catch
            nothing, false
        end
        if complete && out !== nothing
            try
                hash([isfinite(v) ? round(Float64(v), digits=8) : Float64(Inf) for v in out])
            catch
                nothing
            end
        else
            nothing
        end
    end
    library.semantic_signatures[key] = signature
    return signature
end

function _add_entry!(
    library,
    tree,
    source,
    dataset,
    options,
    report;
    rank=0,
    weight=1.0,
    probe_size=128,
)
    complexity = try
        compute_complexity(tree, options)
    catch
        0
    end
    complexity > 0 || return false
    complexity <= options.maxsize || return false
    expression = try
        tree
    catch
        nothing
    end
    signature = _signature_cached!(library, expression, dataset, options, probe_size)
    key = _term_key(expression, options)
    all_entries = (library.static_terms, library.dynamic_terms)
    any(entry -> _term_key(entry.expression, options) == key, Iterators.flatten(all_entries)) &&
        return false
    entry = FinalRefinementLibraryEntry(
        copy_node(expression), source, complexity, signature, Float64(weight), 0, 0, 0.0, rank
    )
    if source in (:dynamic_hof, :dynamic_population)
        push!(library.dynamic_terms, entry)
    else
        push!(library.static_terms, entry)
    return true
end

end

function _feature_leaf(::Type{N}, ::Type{T}, feature::Int) where {N<:AbstractExpressionNode,T}
    return constructorof(N)(T; feature=feature)
end

function _constant_leaf(::Type{N}, ::Type{T}) where {N<:AbstractExpressionNode,T}
    return constructorof(N)(T; val=one(T))
end

function _walk_nodes(node::AbstractExpressionNode; include_root::Bool=false)
    found = Tuple{Vector{Int},Any}[]
    function visit(current, path)
        (include_root || !isempty(path)) && push!(found, (copy(path), current))
        for index in 1:current.degree
            visit(get_child(current, index), [path... , index])
        end
    end
    visit(node, Int[])
    return found
end

function _static_library!(library, base_tree, dataset, options, refinement, report)
    raw_tree = base_tree isa AbstractExpression ? get_tree(base_tree) : base_tree
    node_type = typeof(raw_tree)
    T = eltype(dataset.X)
    _add_entry!(library, _constant_leaf(node_type, T), :static_builtin, dataset, options, report; weight=1.0, probe_size=refinement.probe_size)
    for feature in 1:size(dataset.X, 1)
        _add_entry!(library, _feature_leaf(node_type, T, feature), :static_builtin, dataset, options, report; weight=1.0, probe_size=refinement.probe_size)
    end
    for term in library.pending_terms
        parsed = try
            _parse_term(term, dataset, options)
        catch err
            push!(report.invalid_terms, (term=term, reason=sprint(showerror, err)))
            nothing
        end
        parsed === nothing && begin
            parsed === nothing && !(term isa AbstractExpression || term isa AbstractExpressionNode) &&
                push!(report.invalid_terms, (term=term, reason=:unparseable))
            continue
        end
        complexity = try compute_complexity(parsed, options) catch; 0 end
        valid = complexity > 0 && complexity <= options.maxsize &&
            check_constraints(parsed, dataset, options, options.maxsize, complexity)
        if valid
            _add_entry!(library, parsed, :static_user, dataset, options, report; weight=1.0, probe_size=refinement.probe_size)
        else
            push!(report.invalid_terms, (term=term, reason=:constraint_or_complexity))
        end
    end
    empty!(library.pending_terms)
    return library
end

function _candidate_members(populations)
    populations === nothing && return Any[]
    raw = if populations isa Population
        Any[populations.members...]
    elseif populations isa AbstractVector
        collected = Any[]
        for item in populations
            if item isa Population
                append!(collected, item.members)
            elseif item isa AbstractPopMember
                push!(collected, item)
            end
        end
        collected
    else
        Any[]
    end
    return sort([m for m in raw if m isa AbstractPopMember], by=m -> (isfinite(m.cost) ? m.cost : Inf, m.ref))
end

function _add_dynamic_terms!(library, hof, populations, dataset, options, refinement, report)
    if length(library.dynamic_terms) > refinement.max_dynamic_terms
        resize!(library.dynamic_terms, refinement.max_dynamic_terms)
    end
    sources = Tuple{Any,Symbol}[]
    for member in calculate_pareto_frontier(hof)
        push!(sources, (member, :dynamic_hof))
    end
    members = _candidate_members(populations)
    for member in Iterators.take(members, refinement.elite_count)
        push!(sources, (member, :dynamic_population))
    end
    seen = Set{String}()
    for (member, source) in sources
        nodes = try
            _walk_nodes(get_tree(member.tree); include_root=false)
        catch
            Tuple{Vector{Int},Any}[]
        end
        for (_, node) in nodes
            key = _term_key(node, options)
            key in seen && continue
            push!(seen, key)
            length(library.dynamic_terms) >= refinement.max_dynamic_terms && return library
            _add_entry!(library, node, source, dataset, options, report; rank=length(seen), weight=source == :dynamic_hof ? 1.5 : 1.0, probe_size=refinement.probe_size)
        end
    end
    return library
end

function _replace_path(tree, path, replacement)
    root = copy_node(tree)
    isempty(path) && return copy_node(replacement)
    parent = root
    for index in path[1:(end - 1)]
        parent = get_child(parent, index)
    end
    set_child!(parent, copy_node(replacement), last(path))
    return root
end

function _selection_cost(member, validation_dataset, options, report)
    validation_dataset === nothing && return Float64(member.cost)
    cost = try
        first(eval_cost(validation_dataset, member.tree, options; complexity=compute_complexity(member, options)))
    catch
        Inf
    end
    report.validation_evaluations += 1
    isfinite(cost) || return Inf
    push!(report.validation_costs, Float64(cost))
    return Float64(cost)
end

function _better_selection(candidate, parent, candidate_score, parent_score, options)
    isfinite(candidate_score) || return false
    isfinite(candidate.cost) || return false
    candidate_score < parent_score && return true
    candidate_score == parent_score && compute_complexity(candidate, options) < compute_complexity(parent, options)
end

function _proposal_entries(library, rng, refinement)
    entries = vcat(library.static_terms, library.dynamic_terms)
    entries = [entry for entry in entries if entry.expression !== nothing]
    isempty(entries) && return entries
    if !isnothing(rng) && !refinement.allow_root_replacement
        shuffle!(rng, entries)
    end
    sort!(entries, by=e -> (e.source in (:static_builtin, :static_user) ? 0 : 1, -e.weight, e.rank, sprint(show, e.expression)))
    return entries
end

@noinline function _refine_member(
    member,
    dataset,
    validation_dataset,
    options,
    refinement,
    library,
    report,
    rng,
)
    beam = Any[copy(member)]
    for round in 1:refinement.max_rounds
        report.rounds_completed = max(report.rounds_completed, round)
        next_beam = Any[copy(member)]
        for parent in beam
            report.evaluations >= refinement.max_evals && (report.stopped_reason = :max_evals; return beam)
            contents, context = try
                get_contents_for_mutation(parent.tree, rng)
            catch
                continue
            end
            paths = try
                _walk_nodes(contents; include_root=refinement.allow_root_replacement)
            catch
                Tuple{Vector{Int},Any}[]
            end
            length(paths) > refinement.max_subtrees_per_member && (paths = paths[1:refinement.max_subtrees_per_member])
            proposals = _proposal_entries(library, rng, refinement)
            for (path, _) in paths
                replacements = 0
                for entry in proposals
                    replacements >= refinement.max_replacements_per_subtree && break
                    report.evaluations >= refinement.max_evals && begin
                        report.stopped_reason = :max_evals
                        return next_beam
                    end
                    # Avoid replacing a subtree with itself and preserve the
                    # expression/context contract for TemplateExpression.
                    new_contents = try
                        _replace_path(contents, path, entry.expression)
                    catch
                        continue
                    end
                    expr = try
                        with_contents_for_mutation(parent.tree, new_contents, context)
                    catch
                        continue
                    end
                    complexity = try compute_complexity(expr, options) catch; typemax(Int) end
                    complexity <= options.maxsize || continue
                    check_constraints(expr, dataset, options, options.maxsize, complexity) || continue
                    candidate = try
                        PopMember(dataset, expr, options; deterministic=options.deterministic)
                    catch
                        continue
                    end
                    report.evaluations += 1
                    if options.should_optimize_constants && has_constants(get_tree(candidate.tree))
                        remaining = max(1, refinement.max_evals - report.evaluations)
                        bounded = bounded_optimizer_options(options; iterations=4, f_calls_limit=min(64, remaining))
                        try
                            candidate, optimizer_evals = optimize_constants(
                                dataset,
                                candidate,
                                options;
                                rng,
                                optimizer_options_override=bounded,
                                optimizer_nrestarts_override=0,
                            )
                            report.evaluations += min(remaining, max(0, Int(ceil(optimizer_evals))))
                        catch
                            # The unevaluated candidate remains a valid fallback.
                        end
                    end
                    parent_score = _selection_cost(parent, validation_dataset, options, report)
                    candidate_score = _selection_cost(candidate, validation_dataset, options, report)
                    entry.attempts += 1
                    report.attempted_replacements += 1
                    gain = parent_score - candidate_score
                    entry.gain += isfinite(gain) ? max(0.0, gain) : 0.0
                    library.replacement_gains[_term_key(entry.expression, options)] = get(library.replacement_gains, _term_key(entry.expression, options), 0.0) + (isfinite(gain) ? max(0.0, gain) : 0.0)
                    if _better_selection(candidate, parent, candidate_score, parent_score, options)
                        entry.accepted += 1
                        entry.weight = max(entry.weight, 1.0 + entry.gain / max(entry.attempts, 1))
                        push!(report.replacement_gains, gain)
                        report.accepted_replacements += 1
                        push!(next_beam, candidate)
                    else
                        report.rejected_replacements += 1
                    end
                    replacements += 1
                end
            end
        end
        sort!(next_beam, by=m -> (isfinite(_selection_cost(m, validation_dataset, options, report)) ? _selection_cost(m, validation_dataset, options, report) : Inf, compute_complexity(m, options)))
        beam = next_beam[1:min(refinement.beam_width, length(next_beam))]
        report.evaluations >= refinement.max_evals && (report.stopped_reason = :max_evals; return beam)
    end
    report.stopped_reason == :completed && (report.stopped_reason = :max_rounds)
    return beam
end

@noinline function _finalize_one(
    hall_of_fame::HallOfFame,
    dataset::Dataset,
    options::AbstractOptions;
    populations=nothing,
    refinement::FinalRefinementOptions=FinalRefinementOptions(),
    library::FinalRefinementLibrary=FinalRefinementLibrary(),
    validation_dataset=nothing,
    rng::AbstractRNG=default_rng(),
)
    original = copy(hall_of_fame)
    working_library = _copy_library(library)
    report = FinalRefinementReport()
    frontier = calculate_pareto_frontier(original)
    base = isempty(frontier) ? nothing : first(frontier)
    base === nothing && begin
        report.stopped_reason = :no_elites
        report.library_counts = (static=0, dynamic=0)
        return FinalRefinementResult(copy(original), original, working_library, report)
    end
    _static_library!(working_library, base.tree, dataset, options, refinement, report)
    _add_dynamic_terms!(working_library, original, populations, dataset, options, refinement, report)
    report.library_counts = (static=length(working_library.static_terms), dynamic=length(working_library.dynamic_terms))
    elites = frontier[1:min(refinement.elite_count, length(frontier))]
    refined = copy(original)
    for elite in elites
        report.evaluations >= refinement.max_evals && (report.stopped_reason = :max_evals; break)
        beams = _refine_member(elite, dataset, validation_dataset, options, refinement, working_library, report, rng)
        for candidate in beams
            update_hall_of_fame!(refined, candidate, dataset, options)
        end
    end
    report.stopped_reason == :completed || nothing
    return FinalRefinementResult(refined, original, working_library, report)
end

"""Finalize a completed single-output search without mutating its inputs."""
@noinline function finalize_search(
    hall_of_fame::HallOfFame,
    dataset::Dataset;
    options::AbstractOptions=Options(),
    populations=nothing,
    refinement::FinalRefinementOptions=FinalRefinementOptions(),
    library::FinalRefinementLibrary=FinalRefinementLibrary(),
    validation_dataset=nothing,
    rng::AbstractRNG=default_rng(),
)
    return _finalize_one(
        hall_of_fame,
        dataset,
        options;
        populations,
        refinement,
        library,
        validation_dataset,
        rng,
    )
end

"""Finalize one output from a completed `SearchState`."""
@noinline function finalize_search(
    state::AbstractSearchState,
    dataset::Dataset;
    options::AbstractOptions=Options(),
    refinement::FinalRefinementOptions=FinalRefinementOptions(),
    library::FinalRefinementLibrary=FinalRefinementLibrary(),
    validation_dataset=nothing,
    rng::AbstractRNG=default_rng(),
    output::Integer=1,
)
    1 <= output <= length(state.halls_of_fame) || throw(BoundsError(state.halls_of_fame, output))
    return finalize_search(
        state.halls_of_fame[output],
        dataset;
        options,
        populations=state.last_pops[output],
        refinement,
        library,
        validation_dataset,
        rng,
    )
end

"""Finalize all outputs from a completed `SearchState`."""
@noinline function finalize_search(
    state::AbstractSearchState,
    datasets::AbstractVector{<:Dataset};
    options::AbstractOptions=Options(),
    refinement::FinalRefinementOptions=FinalRefinementOptions(),
    library=FinalRefinementLibrary(),
    validation_dataset=nothing,
    rng::AbstractRNG=default_rng(),
)
    length(datasets) == length(state.halls_of_fame) ||
        throw(DimensionMismatch("datasets must have one entry per search output."))
    return [
        finalize_search(
            state,
            datasets[i];
            options,
            refinement,
            library=library isa AbstractVector ? library[i] : library,
            validation_dataset=validation_dataset isa AbstractVector ? validation_dataset[i] : validation_dataset,
            rng=rng,
            output=i,
        ) for i in eachindex(datasets)
    ]
end

end
