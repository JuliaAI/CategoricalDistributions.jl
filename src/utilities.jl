# # LOCAL DEFINITION OF SCITYPE

# This is to avoid making ScientificTypes a dependency.

function scitype(c::CategoricalValue)
    nc = length(levels(c.pool))
    return ifelse(c.pool.ordered, OrderedFactor{nc}, Multiclass{nc})
end



# # CLASSES

"""
    classes(x)

Return, as a `CategoricalVector`, all the categorical elements with
the same pool as `CategoricalValue` `x` (including `x`), with an
ordering consistent with the pool. Note that `x in classes(x)` is
always true.

Not to be confused with `levels(x.pool)`. See the example below.

Also, overloaded for `x` a `CategoricalArray`, `CategoricalPool`, and
for views of `CategoricalArray`.

**Private method.*

    julia>  v = categorical([:c, :b, :c, :a])
    4-element CategoricalArrays.CategoricalArray{Symbol,1,UInt32}:
     :c
     :b
     :c
     :a

    julia> levels(v)
    3-element Array{Symbol,1}:
     :a
     :b
     :c

    julia> x = v[4]
    CategoricalArrays.CategoricalValue{Symbol,UInt32} :a

    julia> classes(x)
    3-element CategoricalArrays.CategoricalArray{Symbol,1,UInt32}:
     :a
     :b
     :c

    julia> levels(x.pool)
    3-element Array{Symbol,1}:
     :a
     :b
     :c

"""
classes(p::CategoricalPool) = [p[i] for i in 1:length(p)]
classes(x::CategoricalValue) = classes(CategoricalArrays.pool(x))
classes(v::CategoricalArray) = classes(CategoricalArrays.pool(v))
classes(v::SubArray{<:Any, <:Any, <:CategoricalArray}) = classes(parent(v))

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

    julia> v = categorical([:c, :b, :c, :a])
    julia> levels(v)
    3-element Array{Symbol,1}:
     :a
     :b
     :c
    julia> int(v)
    4-element Array{UInt32,1}:
     0x00000003
     0x00000002
     0x00000003
     0x00000001

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
decoder(x) = CategoricalDecoder(classes(x))

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

Transform the specified object `X` into a categorical version, using
the pool contained in `e`. Here `X` is a raw value (an element of
`levels(e)`) or an `AbstractArray` of such values.

```julia
v = categorical(["x", "y", "y", "x", "x"])
julia> transform(v, "x")
CategoricalValue{String,UInt32} "x"

julia> transform(v[1], ["x" "x"; missing "y"])
2×2 CategoricalArray{Union{Missing, Symbol},2,UInt32}:
 "x"       "x"
 missing   "y"

**Private method.**

"""
transform(e::Union{CategoricalArray, CategoricalValue},
                            arg) = _transform(CategoricalArrays.pool(e), arg)
transform(e::CategoricalPool, arg) =
    _transform(e, arg)
