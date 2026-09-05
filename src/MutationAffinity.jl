module MutationAffinityModule

using Random: AbstractRNG
using ..OperatorsModule: plus, sub, mult

# Match actual functions, not names: a custom function named `sin` is not Base.sin.
function same_operator_family(a, b, degree::Integer)
    if degree == 2
        additive = (a === (+) || a === plus || a === (-) || a === sub) &&
            (b === (+) || b === plus || b === (-) || b === sub)
        multiplicative = (a === (*) || a === mult || a === (/)) &&
            (b === (*) || b === mult || b === (/))
        return additive || multiplicative
    elseif degree == 1
        return (a === sin && (b === sin || b === cos)) ||
            (a === cos && (b === sin || b === cos)) ||
            (a === sinh && (b === sinh || b === cosh)) ||
            (a === cosh && (b === sinh || b === cosh))
    end
    return false
end

function affinity_matrix(value, name::AbstractString)
    value isa AbstractMatrix{<:Real} ||
        throw(ArgumentError("`$name` must be a real matrix."))
    matrix = Matrix{Float64}(value)
    all(w -> isfinite(w) && w >= 0, matrix) ||
        throw(ArgumentError("`$name` weights must be finite and nonnegative."))
    return matrix
end

function build_operator_affinity(operators, strength::Float64, overrides)
    matrices = [
        [same_operator_family(a, b, degree) ? strength : 1.0 for a in ops, b in ops]
        for (degree, ops) in enumerate(operators.ops)
    ]
    overrides === nothing && return matrices
    overrides isa AbstractDict ||
        throw(ArgumentError("`operator_affinity` must map operator arities to matrices."))
    for (degree, value) in overrides
        degree isa Integer && 1 <= degree <= length(matrices) ||
            throw(ArgumentError("`operator_affinity` contains an unconfigured arity: $degree."))
        matrix = affinity_matrix(value, "operator_affinity[$degree]")
        size(matrix) == size(matrices[degree]) ||
            throw(ArgumentError("`operator_affinity[$degree]` must have size $(size(matrices[degree]))."))
        matrices[degree] = matrix
    end
    return matrices
end

function build_feature_affinity(value)
    value === nothing && return nothing
    matrix = affinity_matrix(value, "feature_affinity")
    size(matrix, 1) == size(matrix, 2) && size(matrix, 1) > 0 ||
        throw(ArgumentError("`feature_affinity` must be a nonempty square matrix."))
    return matrix
end

"""Mix a static prior with uniform exploration over legal targets only."""
function sample_affinity_target(rng::AbstractRNG, targets, weights, exploration::Float64)
    length(targets) == 1 && return only(targets)
    maxweight = maximum(weights)
    maxweight <= 0 && return targets[rand(rng, eachindex(targets))]

    # Keep the original max-scaling for overflow safety, but accumulate in place
    # to avoid allocating scaled/probability vectors on every mutation.
    scaled_total = zero(Float64)
    @inbounds for weight in weights
        scaled_total += weight / maxweight
    end
    if rand(rng) < exploration
        return targets[rand(rng, eachindex(targets))]
    end
    threshold = rand(rng) * scaled_total
    @inbounds for i in eachindex(targets, weights)
        threshold -= weights[i] / maxweight
        threshold < 0 && return targets[i]
    end
    return last(targets)
end

end
