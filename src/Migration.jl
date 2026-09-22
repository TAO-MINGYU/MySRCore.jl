module MigrationModule

using Random: AbstractRNG, default_rng
using ..CoreModule: AbstractOptions
using ..PopulationModule: Population
using ..PopMemberModule: AbstractPopMember, PopMember, reset_birth!
using ..UtilsModule: poisson_sample
using DynamicExpressions: AbstractExpression, AbstractExpressionNode, get_child, get_tree

"""Return a hash for an expression's operator/feature tree shape."""
function _structural_hash(tree::AbstractExpressionNode)
    value = hash(tree.degree)
    if tree.degree == 0
        value = hash(tree.constant, value)
        value = hash(tree.constant ? :constant : tree.feature, value)
        return value
    end
    value = hash(tree.op, value)
    for child_index in 1:tree.degree
        value = hash(_structural_hash(get_child(tree, child_index)), value)
    end
    return value
end

_structural_hash(expression::AbstractExpression) = _structural_hash(get_tree(expression))

function _structurally_equal(a::AbstractExpressionNode, b::AbstractExpressionNode)
    a.degree == b.degree || return false
    if a.degree == 0
        a.constant == b.constant || return false
        return a.constant || a.feature == b.feature
    end
    a.op == b.op || return false
    for child_index in 1:a.degree
        _structurally_equal(get_child(a, child_index), get_child(b, child_index)) ||
            return false
    end
    return true
end

_structurally_equal(a::AbstractExpression, b::AbstractExpression) =
    _structurally_equal(get_tree(a), get_tree(b))
_structurally_equal(a, b) = _structurally_equal(a.tree, b.tree)

function _profile_compatible(member::AbstractPopMember, profile)
    profile === nothing && return true
    affinity = profile.operator_affinity
    affinity === nothing && return true
    for node in get_tree(member.tree)
        node.degree == 0 && continue
        node.degree <= length(affinity) || return false
        matrix = affinity[node.degree]
        node.op <= size(matrix, 2) || return false
        any(>(0), @view matrix[:, node.op]) || return false
    end
    return true
end

function _novel_migration_candidates(candidates, destination, profile)
    isempty(candidates) && return candidates
    ordered = sort(candidates; by=member -> member.cost)
    # Hashes are only a fast index. Store representatives in buckets and
    # confirm structural equality before suppressing a candidate.
    seen = Dict{UInt,Vector{eltype(candidates)}}()
    if destination !== nothing
        for member in destination.members
            push!(get!(seen, _structural_hash(member.tree)) do
                eltype(candidates)[]
            end, member)
        end
    end
    compatible = eltype(candidates)[]
    novel = eltype(candidates)[]
    for member in ordered
        _profile_compatible(member, profile) || continue
        push!(compatible, member)
        signature = _structural_hash(member.tree)
        bucket = get!(seen, signature) do
            eltype(candidates)[]
        end
        any(existing -> _structurally_equal(existing, member), bucket) && continue
        push!(novel, member)
        push!(bucket, member)
    end
    # A specialised profile must not silently receive an incompatible member.
    # If every legal candidate is already represented locally, retain the best
    # compatible set rather than disabling migration altogether.
    return isempty(novel) ? compatible : novel
end

"""Filter HOF or other global candidates to a target profile's legal members."""
function compatible_migration_candidates(candidates, profile)
    profile === nothing && return candidates
    return [member for member in candidates if _profile_compatible(member, profile)]
end

"""
    migration_candidates(best_sub_pops, source; kwargs...)

Resolve the candidate pool from one source population.  The caller chooses the
source from the destination's profile group; this function only applies the
configured group-local candidate policy.
"""
function migration_candidates(
    best_sub_pops::AbstractVector,
    source::Integer;
    policy::Symbol=:best_only,
    destination_pop=nothing,
    profile=nothing,
)
    isempty(best_sub_pops) && return eltype(best_sub_pops)[]
    1 <= source <= length(best_sub_pops) ||
        throw(BoundsError(best_sub_pops, source))
    candidates = best_sub_pops[source].members
    policy === :best_only && return candidates
    policy === :best_plus_novelty ||
        throw(ArgumentError("Unsupported migration policy: $policy"))
    return _novel_migration_candidates(candidates, destination_pop, profile)
end

"""
    migrate!(migration::Pair{Population{T,L},Population{T,L}}, options::AbstractOptions; frac::AbstractFloat)

Migrate a fraction of the population from one population to the other, creating copies
to do so. The original migrant population is not modified. Pass with, e.g.,
`migrate!(migration_candidates => destination, options; frac=0.1)`
"""
function migrate!(
    migration::Pair{Vector{PM},P}, options::AbstractOptions;
    frac::AbstractFloat,
    rng::AbstractRNG=default_rng(),
) where {T,L,N,PM<:AbstractPopMember{T,L,N},P<:Population{T,L,N,PM}}
    isfinite(frac) && 0 <= frac <= 1 ||
        throw(ArgumentError("migration fraction must be finite and in [0, 1]."))
    base_pop = migration.second
    population_size = length(base_pop.members)
    population_size == 0 && return nothing
    mean_number_replaced = population_size * frac
    num_replace = poisson_sample(rng, mean_number_replaced)

    migrant_candidates = migration.first
    isempty(migrant_candidates) && return nothing

    # Ensure `replace=true` is a valid setting:
    num_replace = min(num_replace, length(migrant_candidates))
    num_replace = min(num_replace, population_size)

    locations = rand(rng, 1:population_size, num_replace)
    migrants = rand(rng, migrant_candidates, num_replace)

    for (i, migrant) in zip(locations, migrants)
        base_pop.members[i] = copy(migrant)
        reset_birth!(base_pop.members[i]; options.deterministic)
    end
    return nothing
end

end
