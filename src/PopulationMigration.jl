module PopulationMigrationModule

using Random: default_rng, rand
using ..OptionsStructModule: AbstractOptions, IslandProfile
import ..OptionsStructModule: specialized_options
using ..OperatorsModule:
    safe_pow,
    safe_log,
    safe_log2,
    safe_log10,
    safe_log1p
using ..MutationsModule: AbstractMutation
using ..CrossoversModule: AbstractCrossover

"""
    ProfiledOptions

An `AbstractOptions` view that applies one `IslandProfile` to a population.
The base options remain shared, while soft preference fields are materialized
once so mutation and crossover hot paths do not repeatedly rebuild vectors.
"""
struct ProfiledOptions{O<:AbstractOptions} <: AbstractOptions
    base::O
    profile::IslandProfile
    operator_affinity::Vector{Matrix{Float64}}
    mutation_affinity::Symbol
    mutations::Vector{Pair{AbstractMutation,Float64}}
    crossovers::Vector{Pair{AbstractCrossover,Float64}}
    mutation_affinity_exploration::Float64
end

function _role_bias(operator, degree::Integer, role::Symbol)
    algebraic = degree == 2 && (operator === (+) || operator === (-) ||
        operator === (*) || operator === (/) || operator === safe_pow)
    rational = degree == 2 && (operator === (*) || operator === (/) || operator === safe_pow)
    trigonometric = degree == 1 && operator in
        (sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh)
    transcendental = degree == 1 && operator in
        (exp, log, safe_log, log2, safe_log2, log10, safe_log10, log1p, safe_log1p)
    if role === :algebraic
        return algebraic ? :preferred : (degree == 1 ? :discouraged : :neutral)
    elseif role === :rational
        return rational ? :preferred : (degree == 2 ? :discouraged : :neutral)
    elseif role === :trigonometric || role === :periodic
        return trigonometric ? :preferred : (degree == 1 || degree == 2 ? :discouraged : :neutral)
    elseif role === :transcendental || role === :log_exp
        return transcendental ? :preferred : (degree == 1 || degree == 2 ? :discouraged : :neutral)
    end
    return :neutral
end

function _role_operator_affinity(options::AbstractOptions, profile::IslandProfile)
    role = profile.role
    role === :generalist && return options.operator_affinity
    strength = profile.operator_preference_strength
    matrices = Matrix{Float64}[]
    for (degree, operators) in enumerate(options.operators.ops)
        base = options.operator_affinity[degree]
        matrix = similar(base)
        for row in axes(base, 1), column in axes(base, 2)
            role_weight = _role_bias(operators[column], degree, role)
            factor = role_weight === :preferred ? strength :
                role_weight === :discouraged ? inv(strength) : 1.0
            matrix[row, column] = base[row, column] * factor
        end
        push!(matrices, matrix)
    end
    return matrices
end

function Base.getproperty(options::ProfiledOptions, key::Symbol)
    key === :base && return getfield(options, :base)
    key === :profile && return getfield(options, :profile)
    key === :operator_affinity && return getfield(options, :operator_affinity)
    key === :mutation_affinity && return getfield(options, :mutation_affinity)
    key === :mutations && return getfield(options, :mutations)
    key === :crossovers && return getfield(options, :crossovers)
    key === :mutation_affinity_exploration &&
        return getfield(options, :mutation_affinity_exploration)
    return getproperty(getfield(options, :base), key)
end

function Base.propertynames(options::ProfiledOptions; private::Bool=false)
    base_names = propertynames(getfield(options, :base); private=private)
    derived = (:base, :profile, :operator_affinity, :mutation_affinity, :mutations, :crossovers,
        :mutation_affinity_exploration)
    return (derived..., base_names...)
end

function _validate_profile(profile::IslandProfile, options::AbstractOptions)
    operators = options.operators.ops
    if profile.operator_affinity !== nothing
        length(profile.operator_affinity) == length(operators) ||
            throw(ArgumentError("IslandProfile operator_affinity must contain one matrix per arity."))
        for (degree, matrix) in enumerate(profile.operator_affinity)
            expected = length(operators[degree])
            size(matrix) == (expected, expected) ||
                throw(ArgumentError(
                    "IslandProfile operator_affinity[$degree] must have size " *
                    "($expected, $expected).",
                ))
            all(isfinite, matrix) && all(>=(0), matrix) ||
                throw(ArgumentError("IslandProfile operator affinity entries must be finite and nonnegative."))
        end
    end
    profile.mutation_weights === nothing ||
        length(profile.mutation_weights) == length(options.mutations) ||
        throw(ArgumentError("IslandProfile mutation_weights must align with options.mutations."))
    profile.crossover_weights === nothing ||
        length(profile.crossover_weights) == length(options.crossovers) ||
        throw(ArgumentError("IslandProfile crossover_weights must align with options.crossovers."))
    return nothing
end

function profiled_options(options::AbstractOptions, profile::IslandProfile)
    _validate_profile(profile, options)
    operator_affinity = profile.operator_affinity === nothing ?
        _role_operator_affinity(options, profile) : profile.operator_affinity
    mutation_multipliers = profile.mutation_weights
    mutations = if mutation_multipliers === nothing
        options.mutations
    else
        Pair{AbstractMutation,Float64}[
            mutation => Float64(weight * multiplier) for
            ((mutation, weight), multiplier) in zip(options.mutations, mutation_multipliers)
        ]
    end
    crossover_multipliers = profile.crossover_weights
    crossovers = if crossover_multipliers === nothing
        options.crossovers
    else
        Pair{AbstractCrossover,Float64}[
            crossover => Float64(weight * multiplier) for
            ((crossover, weight), multiplier) in zip(options.crossovers, crossover_multipliers)
        ]
    end
    if !isempty(mutations) && !any(>(0), last.(mutations))
        throw(ArgumentError("IslandProfile must leave at least one mutation with positive weight."))
    end
    if !isempty(crossovers) && !any(>(0), last.(crossovers))
        throw(ArgumentError("IslandProfile must leave at least one crossover with positive weight."))
    end
    exploration = max(options.mutation_affinity_exploration, profile.exploration_floor)
    return ProfiledOptions(
        options,
        profile,
        operator_affinity,
        profile.operator_affinity === nothing ? options.mutation_affinity : :family,
        mutations,
        crossovers,
        exploration,
    )
end

function profile_for_population(options::AbstractOptions, population::Integer)
    hasproperty(options, :population_profiles) || return nothing
    profiles = options.population_profiles
    profiles === nothing && return nothing
    1 <= population <= length(profiles) ||
        throw(BoundsError(profiles, population))
    return profiles[population]
end

"""Return the population indices that share the destination's profile."""
function population_profile_indices(options::AbstractOptions, population::Integer)
    1 <= population <= options.populations ||
        throw(BoundsError(1:options.populations, population))
    profiles = hasproperty(options, :population_profiles) ?
        options.population_profiles : nothing
    profiles === nothing && return collect(1:options.populations)
    profile = profiles[population]
    return findall(==(profile.id), getfield.(profiles, :id))
end

"""Choose a random source population from the destination's profile group."""
function random_migration_source(
    options::AbstractOptions,
    destination::Integer;
    rng=default_rng(),
)
    candidates = filter(!=(destination), population_profile_indices(options, destination))
    isempty(candidates) && return nothing
    return rand(rng, candidates)
end

function profiled_options(options::AbstractOptions, population::Integer)
    profile = profile_for_population(options, population)
    return profile === nothing ? options : profiled_options(options, profile)
end

function specialized_options(options::ProfiledOptions)
    specialized_base = specialized_options(options.base)
    return profiled_options(specialized_base, options.profile)
end

end
