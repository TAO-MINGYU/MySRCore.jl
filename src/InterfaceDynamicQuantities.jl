module InterfaceDynamicQuantitiesModule

using DispatchDoctor: @unstable
using DynamicQuantities:
    UnionAbstractQuantity,
    AbstractDimensions,
    AbstractQuantity,
    Dimensions,
    Quantity,
    dimension,
    dim_type,
    DEFAULT_DIM_BASE_TYPE

"""
    get_dimensions(T, dimensions)

Normalize user-provided dimension metadata to dimension-only DynamicQuantities
quantities. Public callers may pass an `AbstractDimensions`/`AbstractQuantity`,
a seven-component exponent vector in the order
length, mass, time, current, temperature, luminosity, amount, or a mapping with
those names. Unit strings are intentionally rejected.
"""
function _dimension_from_spec(spec::AbstractDimensions)
    return spec
end
function _dimension_from_spec(spec::AbstractQuantity)
    return dimension(spec)
end
function _dimension_from_spec(spec::AbstractString)
    throw(ArgumentError("dimension specifications must be exponent vectors or mappings; unit strings are not accepted"))
end
function _dimension_from_spec(spec::AbstractDict)
    values = ntuple(
        name -> get(spec, name, get(spec, Symbol(name), 0.0)),
        7,
    )
    return Dimensions(values...)
end
function _dimension_from_spec(spec::AbstractVector)
    length(spec) == 7 ||
        throw(ArgumentError("dimension vectors must contain exactly 7 exponents"))
    return Dimensions(spec...)
end
function _dimension_from_spec(spec::Tuple)
    length(spec) == 7 ||
        throw(ArgumentError("dimension vectors must contain exactly 7 exponents"))
    return Dimensions(spec...)
end
function _dimension_from_spec(spec)
    throw(ArgumentError("dimension specifications must be mappings, 7-element vectors, or AbstractDimensions"))
end

function _dimension_quantity(::Type{T}, spec) where {T}
    return Quantity(one(T), _dimension_from_spec(spec))
end

function get_dimensions(::Type{T}, ::Nothing) where {T}
    return nothing
end
function get_dimensions(::Type{T}, dimensions::AbstractVector) where {T}
    # A single seven-vector is one dimension; otherwise this is one spec per feature.
    if length(dimensions) == 7 && all(x -> x isa Real, dimensions)
        return _dimension_quantity(T, dimensions)
    end
    return Quantity{T}[_dimension_quantity(T, spec) for spec in dimensions]
end
function get_dimensions(::Type{T}, dimensions) where {T}
    return _dimension_quantity(T, dimensions)
end

"""Return the same dimension-only quantities for symbolic display metadata."""
function get_symbolic_dimensions(::Type{T}, dimensions) where {T}
    return get_dimensions(T, dimensions)
end

"""
    get_dimensions_type(A, default_dimensions)

Recursively finds the dimension type from an array, or,
if no quantity is found, returns the default type.
"""
@unstable function get_dimensions_type(A::AbstractArray, default::Type{D}) where {D}
    i = findfirst(a -> isa(a, UnionAbstractQuantity), A)
    if i === nothing
        return D
    else
        return typeof(dimension(A[i]))
    end
end
function get_dimensions_type(
    ::AbstractArray{Q}, default::Type
) where {Q<:UnionAbstractQuantity}
    return dim_type(Q)
end
function get_dimensions_type(_, default::Type{D}) where {D}
    return D
end

# Shortcut for basic numeric types
function get_dimensions_type(
    ::AbstractArray{
        <:Union{
            Bool,
            Int8,
            UInt8,
            Int16,
            UInt16,
            Int32,
            UInt32,
            Int64,
            UInt64,
            Int128,
            UInt128,
            Float16,
            Float32,
            Float64,
            BigFloat,
            BigInt,
            ComplexF16,
            ComplexF32,
            ComplexF64,
            Complex{BigFloat},
            Rational{Int8},
            Rational{UInt8},
            Rational{Int16},
            Rational{UInt16},
            Rational{Int32},
            Rational{UInt32},
            Rational{Int64},
            Rational{UInt64},
            Rational{Int128},
            Rational{UInt128},
            Rational{BigInt},
        },
    },
    default::Type{D},
) where {D}
    return D
end

end
