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
    seen = Set{UInt}()
    if destination !== nothing
        for member in destination.members
            push!(seen, _structural_hash(member.tree))
        end
    end
    compatible = eltype(candidates)[]
    novel = eltype(candidates)[]
    for member in ordered
        _profile_compatible(member, profile) || continue
        push!(compatible, member)
        signature = _structural_hash(member.tree)
        signature in seen && continue
        push!(novel, member)
        push!(seen, signature)
    end
    # A specialised profile must not silently receive an incompatible member.
    # If every legal candidate is already represented locally, retain the best
    # compatible set rather than disabling migration altogether.
    return isempty(novel) ? compatible : novel
end

"""
    migration_candidates(best_sub_pops, destination, topology; kwargs...)

Resolve the source candidate pool for one destination population.  `:pooled`
preserves the historical all-island pool; `:ring` uses the predecessor island
in a directed ring.  `:best_plus_novelty` additionally applies cost-first
duplicate retention, structural novelty filtering, and the explicit profile
compatibility mask.
"""
function migration_candidates(
    best_sub_pops::AbstractVector,
    destination::Integer,
    topology::Symbol;
    policy::Symbol=:best_only,
    destination_pop=nothing,
    profile=nothing,
)
    isempty(best_sub_pops) && return eltype(best_sub_pops)[]
    1 <= destination <= length(best_sub_pops) ||
        throw(BoundsError(best_sub_pops, destination))
    candidates = if topology === :ring
        source = destination == 1 ? length(best_sub_pops) : destination - 1
        best_sub_pops[source].members
    elseif topology === :pooled
        [member for pop in best_sub_pops for member in pop.members]
    else
        throw(ArgumentError("Unsupported migration topology: $topology"))
    end
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
    mean_number_replaced = population_size * frac
    num_replace = poisson_sample(rng, mean_number_replaced)

    migrant_candidates = migration.first

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
