module SurrogateModule

using Random: rand
using DynamicExpressions: AbstractExpression, AbstractExpressionNode, eval_tree_array

using ..CoreModule: AbstractOptions, Dataset

"""Runtime data and immutable synchronization records for surrogate evaluation.

Workers receive a read-only `SurrogateSnapshot`, create a local mutable state,
and return only observations from real evaluations.  The head process merges
those observations after a population round and sends the next snapshot to
future dispatches.  Surrogate predictions therefore never enter either the
training set or the Hall of Fame.
"""
struct SurrogateSnapshot
    probe_indices::Vector{Int}
    features::Vector{Vector{Float64}}
    costs::Vector{Float64}
    losses::Vector{Float64}
    max_samples::Int
    generation::Int
end

"""True-evaluation observations produced by one worker dispatch."""
struct SurrogateReport
    output::Int
    population::Int
    iteration::Int
    base_generation::Int
    probe_indices::Vector{Int}
    max_samples::Int
    features::Vector{Vector{Float64}}
    costs::Vector{Float64}
    losses::Vector{Float64}
    proposals::Int
    true_evaluations::Int
    predictions::Int
    rejected::Int
    model_failures::Int
end

mutable struct SurrogateState
    probe_indices::Vector{Int}
    features::Vector{Vector{Float64}}
    costs::Vector{Float64}
    losses::Vector{Float64}
    proposals::Int
    true_evaluations::Int
    predictions::Int
    rejected::Int
    model_failures::Int
    max_samples::Int
    base_generation::Int
    seed_sample_count::Int
end

"""The result of a cheap surrogate gate before an expensive evaluation."""
struct SurrogateDecision
    evaluate::Bool
    features::Union{Nothing,Vector{Float64}}
    predicted_cost::Union{Nothing,Float64}
    uncertainty::Union{Nothing,Float64}
end

@inline function _option(options, name::Symbol, default)
    # Some option wrappers implement `propertynames` with a keyword-only
    # signature.  A guarded property access keeps surrogate defaults compatible
    # with those wrappers without requiring them to be updated.
    try
        return getproperty(options, name)
    catch
        return default
    end
end

"""Return whether the supplied options opt in to surrogate evaluation."""
@inline surrogate_enabled(options::AbstractOptions) =
    _option(options, :surrogate_enabled, false)::Bool

function create_surrogate_state(
    dataset::Dataset,
    options::AbstractOptions;
    snapshot::Union{Nothing,SurrogateSnapshot}=nothing,
)
    surrogate_enabled(options) || return nothing
    dataset.n > 0 || return nothing

    max_samples = Int(_option(options, :surrogate_max_samples, 2048))
    max_samples >= 1 || throw(ArgumentError("`surrogate_max_samples` must be positive."))
    if snapshot === nothing
        requested = Int(_option(options, :surrogate_probe_size, 64))
        requested >= 1 || throw(ArgumentError("`surrogate_probe_size` must be positive."))
        probe_size = min(requested, dataset.n)
        # Equally spaced probes are deterministic and cover the complete data
        # domain without consuming the search RNG stream.
        probe_indices = unique(round.(Int, range(1, dataset.n; length=probe_size)))
        features = Vector{Float64}[]
        costs = Float64[]
        losses = Float64[]
        generation = 0
    else
        all(1 <= index <= dataset.n for index in snapshot.probe_indices) ||
            throw(DimensionMismatch("surrogate snapshot probe indices do not match dataset"))
        probe_indices = copy(snapshot.probe_indices)
        features = [copy(entry) for entry in snapshot.features]
        costs = copy(snapshot.costs)
        losses = copy(snapshot.losses)
        generation = snapshot.generation
        if length(features) > max_samples
            first_index = length(features) - max_samples + 1
            features = features[first_index:end]
            costs = costs[first_index:end]
            losses = losses[first_index:end]
        end
    end
    seed_sample_count = length(features)
    return SurrogateState(
        probe_indices,
        features,
        costs,
        losses,
        0,
        seed_sample_count,
        0,
        0,
        0,
        max_samples,
        generation,
        seed_sample_count,
    )
end

@inline function _finite_feature_value(value)
    if value isa Real && isfinite(value)
        return Float64(value)
    end
    # Keep a fixed-dimensional feature vector even when a safe operator reports
    # an invalid value.  The large sentinel makes such expressions distant from
    # ordinary finite observations without throwing away the candidate.
    if value isa Real && value < 0
        return -1.0e6
    end
    return 1.0e6
end

"""Compute the phenotype vector used by the local surrogate model."""
function surrogate_features(
    tree::Union{AbstractExpression,AbstractExpressionNode},
    dataset::Dataset,
    options::AbstractOptions,
    state::SurrogateState,
    complexity::Integer,
)
    X_probe = view(dataset.X, :, state.probe_indices)
    prediction, complete = eval_tree_array(tree, X_probe, options)
    (!complete || isnothing(prediction)) && return nothing
    features = Float64[_finite_feature_value(value) for value in prediction]
    # Complexity is part of the phenotype because cost includes parsimony.
    push!(features, Float64(complexity))
    return features
end

function _distance(a::Vector{Float64}, b::Vector{Float64})
    length(a) == length(b) || return Inf
    total = 0.0
    @inbounds for i in eachindex(a, b)
        delta = a[i] - b[i]
        total += delta * delta
    end
    return sqrt(total / max(1, length(a)))
end

"""Predict cost and local uncertainty with distance-weighted KNN."""
function surrogate_predict(
    state::SurrogateState, features::Vector{Float64}, options::AbstractOptions
)
    warmup = Int(_option(options, :surrogate_warmup_evals, 32))
    neighbors = Int(_option(options, :surrogate_neighbors, 8))
    warmup >= 1 || throw(ArgumentError("`surrogate_warmup_evals` must be positive."))
    neighbors >= 1 || throw(ArgumentError("`surrogate_neighbors` must be positive."))
    length(state.costs) >= warmup || return nothing
    isempty(state.features) && return nothing

    distances = [_distance(features, entry) for entry in state.features]
    all(isfinite, distances) || return nothing
    order = sortperm(distances)
    k = min(neighbors, length(order))
    selected = order[1:k]

    zero_idx = findfirst(i -> distances[i] == 0.0, selected)
    if zero_idx !== nothing
        exact = state.costs[selected[zero_idx]]
        return (exact, 0.0)
    end

    weights = [1.0 / max(distances[i], eps(Float64)) for i in selected]
    weight_sum = sum(weights)
    weight_sum > 0.0 || return nothing
    prediction = sum(weights[j] * state.costs[selected[j]] for j in eachindex(selected)) /
                 weight_sum
    uncertainty = sum(
        weights[j] * abs(state.costs[selected[j]] - prediction) for j in eachindex(selected)
    ) / weight_sum
    return (prediction, uncertainty)
end

"""Add a fully evaluated candidate to the bounded local training set."""
observe_surrogate!(::Nothing, ::Nothing, ::Real, ::Real) = nothing
observe_surrogate!(::Nothing, ::Vector{Float64}, ::Real, ::Real) = nothing
function observe_surrogate!(
    state::SurrogateState,
    features::Union{Nothing,Vector{Float64}},
    cost::Real,
    loss::Real,
)
    state.true_evaluations += 1
    (features === nothing || !isfinite(cost) || !isfinite(loss)) && return nothing
    if length(state.features) >= state.max_samples
        popfirst!(state.features)
        popfirst!(state.costs)
        popfirst!(state.losses)
    end
    push!(state.features, features)
    push!(state.costs, Float64(cost))
    push!(state.losses, Float64(loss))
    return nothing
end

function observe_surrogate_member!(state::SurrogateState, member, dataset, options)
    complexity = getfield(member, :complexity)
    complexity < 0 && return nothing
    features = surrogate_features(member.tree, dataset, options, state, complexity)
    observe_surrogate!(state, features, member.cost, member.loss)
    return nothing
end

"""Create a report containing only samples observed after snapshot seeding."""
function surrogate_report(
    state::Nothing;
    output::Int,
    population::Int,
    iteration::Int,
)
    return nothing
end
function surrogate_report(
    state::SurrogateState;
    output::Int,
    population::Int,
    iteration::Int,
)
    first_new = state.seed_sample_count + 1
    features = first_new <= length(state.features) ?
        [copy(entry) for entry in state.features[first_new:end]] : Vector{Float64}[]
    costs = first_new <= length(state.costs) ? copy(state.costs[first_new:end]) : Float64[]
    losses = first_new <= length(state.losses) ? copy(state.losses[first_new:end]) : Float64[]
    return SurrogateReport(
        output,
        population,
        iteration,
        state.base_generation,
        copy(state.probe_indices),
        state.max_samples,
        features,
        costs,
        losses,
        state.proposals,
        state.true_evaluations - state.seed_sample_count,
        state.predictions,
        state.rejected,
        state.model_failures,
    )
end

"""Merge worker reports into a new immutable head-owned snapshot."""
function merge_surrogate_reports(
    snapshot::Union{Nothing,SurrogateSnapshot},
    reports::AbstractVector{<:SurrogateReport};
    generation::Int,
)
    isempty(reports) && return snapshot
    first_report = first(reports)
    probe_indices = snapshot === nothing ?
        copy(first_report.probe_indices) : copy(snapshot.probe_indices)
    max_samples = snapshot === nothing ?
        first_report.max_samples : snapshot.max_samples
    features = snapshot === nothing ? Vector{Float64}[] : [
        copy(entry) for entry in snapshot.features
    ]
    costs = snapshot === nothing ? Float64[] : copy(snapshot.costs)
    losses = snapshot === nothing ? Float64[] : copy(snapshot.losses)

    # Reports can arrive one dispatch late when populations finish at different
    # times.  De-duplicate exact observations before applying the FIFO bound.
    for report in reports
        length(report.features) == length(report.costs) == length(report.losses) ||
            throw(DimensionMismatch("surrogate report sample arrays have different lengths"))
        for index in eachindex(report.features, report.costs, report.losses)
            feature = report.features[index]
            cost = report.costs[index]
            loss = report.losses[index]
            duplicate = any(
                existing_features == feature &&
                existing_cost == cost &&
                existing_loss == loss for
                (existing_features, existing_cost, existing_loss) in
                zip(features, costs, losses)
            )
            duplicate && continue
            push!(features, copy(feature))
            push!(costs, cost)
            push!(losses, loss)
        end
    end
    if length(features) > max_samples
        first_index = length(features) - max_samples + 1
        features = features[first_index:end]
        costs = costs[first_index:end]
        losses = losses[first_index:end]
    end
    return SurrogateSnapshot(
        probe_indices,
        features,
        costs,
        losses,
        max_samples,
        generation,
    )
end

"""Decide whether a candidate receives the full cost evaluation."""
function consider_surrogate!(
    state::Union{Nothing,SurrogateState},
    tree,
    dataset::Dataset,
    options::AbstractOptions,
    complexity::Integer,
    parent_cost::Real,
)
    state === nothing && return SurrogateDecision(true, nothing, nothing, nothing)
    state.proposals += 1
    features = try
        surrogate_features(tree, dataset, options, state, complexity)
    catch
        state.model_failures += 1
        nothing
    end
    features === nothing && return SurrogateDecision(true, nothing, nothing, nothing)

    prediction = try
        surrogate_predict(state, features, options)
    catch
        state.model_failures += 1
        nothing
    end
    prediction === nothing && return SurrogateDecision(true, features, nothing, nothing)
    state.predictions += 1
    predicted_cost, uncertainty = prediction

    # The first part of the policy guarantees a configurable true-evaluation
    # budget.  The exploration branch keeps distant/unfamiliar phenotypes alive.
    true_fraction = Float64(_option(options, :surrogate_true_eval_fraction, 0.25))
    exploration = Float64(_option(options, :surrogate_exploration_fraction, 0.15))
    uncertainty_scale = Float64(_option(options, :surrogate_uncertainty_scale, 0.25))
    reject_margin = Float64(_option(options, :surrogate_reject_margin, 0.05))
    required_true = ceil(Int, true_fraction * state.proposals)
    if state.true_evaluations < required_true || rand() < exploration
        return SurrogateDecision(true, features, predicted_cost, uncertainty)
    end

    scale = max(1.0, abs(predicted_cost))
    if uncertainty > uncertainty_scale * scale ||
       predicted_cost - uncertainty <= Float64(parent_cost) * (1.0 + reject_margin)
        return SurrogateDecision(true, features, predicted_cost, uncertainty)
    end

    state.rejected += 1
    return SurrogateDecision(false, features, predicted_cost, uncertainty)
end

function surrogate_stats(state::Nothing)
    return (
        proposals=0,
        true_evaluations=0,
        predictions=0,
        rejected=0,
        model_failures=0,
        samples=0,
    )
end
function surrogate_stats(state::SurrogateState)
    return (
        proposals=state.proposals,
        true_evaluations=state.true_evaluations,
        predictions=state.predictions,
        rejected=state.rejected,
        model_failures=state.model_failures,
        samples=length(state.features),
    )
end

end
