# # LOCAL DEFINITION OF SCITYPE

# This is to avoid making ScientificTypes a dependency.

function scitype(c::CategoricalValue)
    nc = length(levels(c.pool))
    return ifelse(c.pool.ordered, OrderedFactor{nc}, Multiclass{nc})
end

# # LEVELS FOR ARRAYS OF ARRAYS

const ERR_EMPTY_UNIVARIATE_FINITE = ArgumentError("Only missings found. ")

"""
    CategoricalDistributions.element_levels(vs)

*Private method.*

Return `levels(element)` for the first non-missing `element` of `vs`.

"""
function element_levels(vs::AbstractArray)
    i = findfirst(!ismissing, vs)
    i === nothing && throw(ERR_EMPTY_UNIVARIATE_FINITE)
    return levels(vs[i])
end


# # CATEGORICAL VALUES TO INTEGERS

"""
   int(x)

The positional integer of the `CategoricalValue` `x`, in the ordering
defined by the pool of `x`. The type of `int(x)` is the reference type
of `x` (which differentiates this method from
`CategoricalArrays.levelcode`).

    int(X::CategoricalArray)
    int(W::AbstractArray{<:CategoricalValue})

Broadcasted versions of `int`.

```julia-repl
julia> v = categorical(['c', 'b', 'c', 'a'])
julia> levels(v)
4-element CategoricalArray{Char,1,UInt32}:
 'c'
 'b'
 'c'
 'a'
julia> int(v)
4-element Array{UInt32,1}:
 0x00000003
 0x00000002
 0x00000003
 0x00000001
```

See  [`decoder`](@ref) on how to invert the `int` operation.
"""
int(x) = throw(
    DomainError(x, "Can only convert categorical elements to integers. "))

int(x::Missing)       = missing
int(x::AbstractArray) = int.(x)

# first line is no good because it promotes type to larger integer type:
# int(x::CategoricalValue) = CategoricalArrays.levelcode(x)
int(x::CategoricalValue) = CategoricalArrays.refcode(x)


# # INTEGERS BACK TO CATEGORICAL VALUES

struct CategoricalDecoder{V,R}
    classes::CategoricalVector{V, R, V, CategoricalValue{V,R}, Union{}}
end

"""
    d = decoder(x)

A callable object for decoding the integer representation of a
`CategoricalValue` sharing the same pool as the `CategoricalValue`
`x`. Specifically, one has `d(int(y)) == y` for all `y` in the same
pool as `x`. One can also call `d` on integer arrays, in which case
`d` is broadcast over all elements.

    julia> v = categorical([:c, :b, :c, :a])
    julia> int(v)
    4-element Array{UInt32,1}:
     0x00000003
     0x00000002
     0x00000003
     0x00000001
    julia> d = decoder(v[3])
    julia> d(int(v)) == v
    true

*Warning:* There is no guarantee that `int(d(u)) == u` will always holds.

See also: [`int`](@ref).

"""
decoder(x) = CategoricalDecoder(levels(x))

(d::CategoricalDecoder{V,R})(i::Integer) where {V,R} =
    CategoricalValue{V,R}(d.classes[i])
(d::CategoricalDecoder)(a::AbstractArray{<:Integer}) = d.(a)


## TRANSFORMING BETWEEN CATEGORICAL ELEMENTS AND RAW VALUES

err_missing_class(c) =  DomainError("Value `$c` not in pool")

function _transform(pool, x)
    ismissing(x) && return missing
    x in levels(pool) || throw(err_missing_class(x))
    return pool[get(pool, x)]
end

_transform(pool, X::AbstractArray) = broadcast(x -> _transform(pool, x), X)

"""
    transform(e::Union{CategoricalElement,CategoricalArray,CategoricalPool},  X)

**Private method.**

Transform the specified object `X` into a categorical version, using
the pool contained in `e`. Here `X` is a raw value (an element of
`unwrap.(levels(e))`) or an `AbstractArray` of such values.

```julia
v = categorical(["x", "y", "y", "x", "x"])
julia> transform(v, "x")
CategoricalValue{String,UInt32} "x"

julia> transform(v[1], ["x" "x"; missing "y"])
2×2 CategoricalArray{Union{Missing, Symbol},2,UInt32}:
 "x"       "x"
 missing   "y"


"""
transform(e::Union{CategoricalArray, CategoricalValue},
                            arg) = _transform(CategoricalArrays.pool(e), arg)
transform(e::CategoricalPool, arg) =
    _transform(e, arg)
