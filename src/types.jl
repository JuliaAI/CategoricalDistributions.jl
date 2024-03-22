const DOC_CONSTRUCTOR =
"""
    UnivariateFinite(support,
                     probs;
                     pool=nothing,
                     augmented=false,
                     ordered=false)

Construct a discrete univariate distribution whose finite support is
the elements of the vector `support`, and whose corresponding
probabilities are elements of the vector `probs`. Alternatively,
construct an abstract *array* of `UnivariateFinite` distributions by
choosing `probs` to be an array of one higher dimension than the array
generated.

Here the word "probabilities" is an abuse of terminology as there is
no requirement that the that probabilities actually sum to one. The
only requirement is that the probabilities have a common type `T` for
which `zero(T)` is defined. In particular, `UnivariateFinite` objects
implement arbitrary non-negative, signed, or complex measures over
finite sets of labelled points. A `UnivariateDistribution` will be a
bona fide probability measure when constructed using the
`augment=true` option (see below) or when `fit` to data. And the
probabilities of a `UnivariateFinite` object `d` must be non-negative,
with a non-zero sum, for `rand(d)` to be defined and interpretable.

Unless `pool` is specified, `support` should have type
 `AbstractVector{<:CategoricalValue}` and all elements are assumed to
 share the same categorical pool, which may be larger than `support`.

*Important.* All levels of the common pool have associated
probabilities, not just those in the specified `support`. However,
these probabilities are always zero (see example below).

If `probs` is a matrix, it should have a column for each class in
`support` (or one less, if `augment=true`). More generally, `probs`
will be an array whose size is of the form `(n1, n2, ..., nk, c)`,
where `c = length(support)` (or one less, if `augment=true`) and the
constructor then returns an array of `UnivariateFinite` distributions
of size `(n1, n2, ..., nk)`.

```
using CategoricalDistributions, CategoricalArrays, Distributions
samples = categorical(['x', 'x', 'y', 'x', 'z'])
julia> Distributions.fit(UnivariateFinite, samples)
           UnivariateFinite{Multiclass{3}}
     ┌                                        ┐
   x ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.6
   y ┤■■■■■■■■■■■■ 0.2
   z ┤■■■■■■■■■■■■ 0.2
     └                                        ┘

julia> d = UnivariateFinite([samples[1], samples[end]], [0.1, 0.9])
UnivariateFinite{Multiclass{3}(x=>0.1, z=>0.9)
           UnivariateFinite{Multiclass{3}}
     ┌                                        ┐
   x ┤■■■■ 0.1
   z ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.9
     └                                        ┘

julia> rand(d, 3)
3-element Array{Any,1}:
 CategoricalValue{Symbol,UInt32} 'z'
 CategoricalValue{Symbol,UInt32} 'z'
 CategoricalValue{Symbol,UInt32} 'z'

julia> levels(samples)
3-element Array{Symbol,1}:
 'x'
 'y'
 'z'

julia> pdf(d, 'y')
0.0
```

### Specifying a pool

Alternatively, `support` may be a list of raw (non-categorical)
elements if `pool` is:

- some `CategoricalArray`, `CategoricalValue` or `CategoricalPool`,
  such that `support` is a subset of `levels(pool)`

- `missing`, in which case a new categorical pool is created which has
  `support` as its only levels.

In the last case, specify `ordered=true` if the pool is to be
considered ordered.

```
julia> UnivariateFinite(['x', 'z'], [0.1, 0.9], pool=missing, ordered=true)
         UnivariateFinite{OrderedFactor{2}}
     ┌                                        ┐
   x ┤■■■■ 0.1
   z ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.9
     └                                        ┘

samples = categorical(['x', 'x', 'y', 'x', 'z'])
julia> d = UnivariateFinite(['x', 'z'], [0.1, 0.9], pool=samples)
     ┌                                        ┐
   x ┤■■■■ 0.1
   z ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.9
     └                                        ┘

julia> pdf(d, 'y') # allowed as `'y' in levels(samples)`
0.0

v = categorical(['x', 'x', 'y', 'x', 'z', 'w'])
probs = rand(100, 3)
probs = probs ./ sum(probs, dims=2)
julia> d1 = UnivariateFinite(['x', 'y', 'z'], probs, pool=v)
100-element UnivariateFiniteVector{Multiclass{4},Symbol,UInt32,Float64}:
 UnivariateFinite{Multiclass{4}}(x=>0.194, y=>0.3, z=>0.505)
 UnivariateFinite{Multiclass{4}}(x=>0.727, y=>0.234, z=>0.0391)
 UnivariateFinite{Multiclass{4}}(x=>0.674, y=>0.00535, z=>0.321)
   ⋮
 UnivariateFinite{Multiclass{4}}(x=>0.292, y=>0.339, z=>0.369)
```

### Probability augmentation

If `augment=true` the provided array is augmented by inserting
appropriate elements *ahead* of those provided, along the last
dimension of the array. This means the user only provides probabilities
for the classes `c2, c3, ..., cn`. The class `c1` probabilities are
chosen so that each `UnivariateFinite` distribution in the returned
array is a bona fide probability distribution.

```julia
julia> UnivariateFinite([0.1, 0.2, 0.3], augment=true, pool=missing)
3-element UnivariateFiniteArray{Multiclass{2}, String, UInt8, Float64, 1}:
 UnivariateFinite{Multiclass{2}}(class_1=>0.9, class_2=>0.1)
 UnivariateFinite{Multiclass{2}}(class_1=>0.8, class_2=>0.2)
 UnivariateFinite{Multiclass{2}}(class_1=>0.7, class_2=>0.3)

d2 = UnivariateFinite(['x', 'y', 'z'], probs[:, 2:end], augment=true, pool=v)
julia> pdf(d1, levels(v)) ≈ pdf(d2, levels(v))
true
```

---

    UnivariateFinite(prob_given_class; pool=nothing, ordered=false)

Construct a discrete univariate distribution whose finite support is
the set of keys of the provided dictionary, `prob_given_class`, and
whose values specify the corresponding probabilities.

The type requirements on the keys of the dictionary are the same as
the elements of `support` given above with this exception: if
non-categorical elements (raw labels) are used as keys, then
`pool=...` must be specified and cannot be `missing`.

If the values (probabilities) are arrays instead of scalars, then an
abstract array of `UnivariateFinite` elements is created, with the
same size as the array.

"""


# # TYPES - PLAIN AND ARRAY

# R - reference type <: Unsigned
# V - type of class labels (eg, Char in `categorical(['a', 'b'])`)
# P - raw probability type
# S - scitype of samples

# Note that the keys of `prob_given_ref` need not exhaust all the
# refs of all classes but will be ordered (LittleDicts preserve order)
struct UnivariateFinite{S,V,R,P}
    scitype::Type{S}
    decoder::CategoricalDecoder{V,R}
    prob_given_ref::LittleDict{R,P,Vector{R}, Vector{P}}
end

@doc DOC_CONSTRUCTOR UnivariateFinite

"""
    UnivariateFiniteArray

Array type whose elements are `UnivariateFinite` distributions sharing
a common sample space (`CategoricalArrays` pool).

See [`UnivariateFinite`](@ref) for constructor.

"""
struct UnivariateFiniteArray{S,V,R,P,N} <:
    AbstractArray{UnivariateFinite{S,V,R,P},N}
    scitype::Type{S}
    decoder::CategoricalDecoder{V,R}
    prob_given_ref::LittleDict{R,Array{P,N},Vector{R}, Vector{Array{P,N}}}
end

const UnivariateFiniteVector{S,V,R,P} = UnivariateFiniteArray{S,V,R,P,1}

# private:
const SingletonOrArray{S,V,R,P} = Union{UnivariateFinite{S,V,R,P},
                                        UnivariateFiniteArray{S,V,R,P}}


# # CHECKS AND ERROR MESSAGES

# checks that scalar probabilities lie in [0, 1] and checks that
# vector probabilities sum to one have now been dropped, except where
# `augment=true` is specified.

# not exported:
const Prob{P} = Union{P, AbstractArray{P}} where P

# TODO: have error functions return exceptions, not throw them
# TODO: are some of these now obsolete?

const ERR_01 = DomainError("Probabilities must be in [0,1].")
err_dim(support, probs) = DimensionMismatch(
    "Probability array is incompatible "*
    "with the number of classes, $(length(support)), which should "*
    "be equal to `$(size(probs)[end])`, the last dimension "*
    "of the probability array. Perhaps you meant to set `augment=true`? ")
err_dim_augmented(support, probs) = DimensionMismatch(
    "Probability array to be augmented is incompatible "*
    "with the number of classes, $(length(support)), which should "*
    "be one more than `$(size(probs)[end])`, the last dimension "*
    "of the probability array. ")
const ERR_AUG = ArgumentError(
    "Array cannot be augmented. There are "*
    "sums along the last axis exceeding one. ")

function _check_pool(pool)
    ismissing(pool) || pool == nothing ||
        @warn "Specified pool ignored, as class labels being "*
    "generated automatically. "
    return nothing
end
_check_probs_01(probs) =
    all(0 .<= probs .<= 1) || throw(ERR_01)
_check_probs(probs) = (_check_probs_01(probs); _check_probs_sum(probs))
_check_augmentable(support, probs) = _check_probs_01(probs) &&
    size(probs)[end] + 1 == length(support) ||
    throw(err_dim_augmented(support, probs))

err_incompatible_pool(support, classes) = ArgumentError(
    "Specified support, $support, not contained in "*
        "specified pool, $(levels(classes)). "
)


## AUGMENTING ARRAYS TO MAKE THEM PROBABILITY ARRAYS

_unwrap(A::AbstractArray) = A
_unwrap(A::AbstractVector) = first(A)

isbinary(support) = length(support) == 2

# augmentation inserts the sum-subarray *before* the array:
_augment_probs(support, probs) =
    _augment_probs(Val(isbinary(support)), support, probs,)
function _augment_probs(::Val{false},
                        support,
                        probs::AbstractArray{P,N}) where {P,N}
    _check_augmentable(support, probs)
    aug_size = size(probs) |> collect
    aug_size[end] += 1
    augmentation = _unwrap(one(P) .- sum(probs, dims=N))
    all(0 .<= augmentation .<= 1) || throw(ERR_AUG)
    aug_probs = Array{P}(undef, aug_size...)
    aug_probs[fill(:, N - 1)..., 2:end] = probs
    aug_probs[fill(:, N - 1)..., 1] = augmentation
    return aug_probs
end
function _augment_probs(::Val{true},
                        support,
                        probs::AbstractArray{P,N}) where {P,N}
    _check_probs_01(probs)
    aug_size = [size(probs)..., 2]
    augmentation = one(P) .- probs
    all(0 .<= augmentation .<= 1) || throw(ERR_AUG)
    aug_probs = Array{P}(undef, aug_size...)
    aug_probs[fill(:, N)..., 2] = probs
    aug_probs[fill(:, N)..., 1] = augmentation
    return aug_probs
end

_array_or_scalar(x::Array) = x
_array_or_scalar(x::AbstractArray) = copyto!(similar(Array{eltype(x)}, axes(x)), x)
_array_or_scalar(x) = x

## CONSTRUCTORS - FROM DICTIONARY

# The following constructor will get called by all the others. It
# returns a UnivariateFinite object *or* a UnivariateFiniteArray,
# depending on the values of the dictionary - scalar or array - which
# represent the probabilities, one for each class in the support.
function UnivariateFinite(
    prob_given_class::AbstractDict{<:CategoricalValue, <:Prob};
    kwargs...)

    # this constructor ignores kwargs

    probs = values(prob_given_class) |> collect
#    _check_probs_01.(probs)
#    _check_probs_sum(probs)

    # `LittleDict`s preserve order of keys, which we need for rand():
    _support  = keys(prob_given_class) |> collect |> sort
    isempty(_support) && throw(
        ArgumentError(
            "Cannot create `UnivariateFinite` and `UnivariateFininiteArray`"*
            " objects with no classes/support."
        )
    )
    # retrieve decoder and classes from first key:
    class1         = first(_support)
    parent_decoder = decoder(class1)
    parent_classes = classes(class1)

    # TODO: throw pre-defined exception below

    issubset(_support, parent_classes) ||
        error("Categorical elements are not from the same pool. ")

    pairs = [int(c) => _array_or_scalar(prob_given_class[c])
                for c in _support]

    probs1 = first(values(prob_given_class))
    S = scitype(class1) # this `scitype` defined in utilities.jl
    if  probs1 isa AbstractArray
        return UnivariateFiniteArray(S, parent_decoder, LittleDict(pairs...))
    else
        return UnivariateFinite(S, parent_decoder, LittleDict(pairs...))
    end
end

# TODO: throw and test pre-defined exceptions in this method:
function UnivariateFinite(d::AbstractDict{V,<:Prob};
                          pool=nothing,
                          ordered=false) where V

    ismissing(pool) &&
        throw(ArgumentError(
            "You cannot specify `pool=missing` "*
            "if passing `UnivariateFinite` a dictionary"))

    pool === nothing && throw(ArgumentError(
        "You must specify `pool=c` "*
        "where `c` is a "*
        "`CategoricalArray`, `CategoricalArray` or "*
        "CategoricalPool`"))

    ordered && @warn "Ignoring `ordered` key-word argument as using "*
    "specified pool to order. "

    raw_support = keys(d) |> collect
    _classes = classes(pool)
    issubset(raw_support, _classes) ||
        error("Specified support, $raw_support, not contained in "*
              "specified pool, $(levels(_classes)). ")
    support = filter(_classes) do c
        c in raw_support
    end

    prob_given_class =
        LittleDict([c=>d[CategoricalArrays.DataAPI.unwrap(c)] for c in support])

    return UnivariateFinite(prob_given_class)
end


## CONSTRUCTORS - FROM ARRAYS

# example: _get_on_last(A, 4) = A[:, :, 4] if A has 3 dims:
_get_on_last(probs::AbstractArray{<:Any,N}, i) where N =
    probs[fill(:,N-1)..., i]

# 1. Univariate Finite from a vector of classes or raw labels and
# array of probs; first, a dispatcher:
function UnivariateFinite(
    support::AbstractVector,
    probs;
    kwargs...)

    if support isa AbstractArray{<:CategoricalValue}
        if :pool in keys(kwargs)
            @warn "Ignoring value of `pool` as the specified "*
            "support defines one already. "
        end
        if :ordered in keys(kwargs)
            @warn "Ignoring value of `ordered` as the "*
            "specified support defines an order already. "
        end
    end

    return _UnivariateFinite(Val(isbinary(support)),
                             support,
                             probs;
                             kwargs...)
end

# The core method, ultimately called by 1.0, 1.1, 1.2, 1.3 below.
function _UnivariateFinite(support::AbstractVector{CategoricalValue{V,R}},
                           probs::AbstractArray{P},
                           N;
                           augment=false,
                           kwargs...) where {V,R,P}

    unique(support) == support ||
        error("Non-unique vector of classes specified")

    _probs = augment ? _augment_probs(support, probs) : probs

    augment || length(support) == size(_probs) |> last ||
        throw(err_dim(support, _probs))

    # it's necessary to force the typing of the LittleDict otherwise it
    # flips to Any type (unlike regular Dict):

    if N == 0
        prob_given_class = LittleDict{CategoricalValue{V,R},P}()
    else
        prob_given_class =
            LittleDict{CategoricalValue{V,R}, AbstractArray{P,N}}()
    end
    for i in eachindex(support)
        prob_given_class[support[i]] = _get_on_last(_probs, i)
    end

    # calls dictionary constructor above:
    return UnivariateFinite(prob_given_class; kwargs...)
end

# 1.0 support does not consist of categorical elements:
function _UnivariateFinite(support,
                           probs::AbstractArray,
                           N;
                           augment=false,
                           pool=nothing,
                           ordered=false)

    # If we got here, then the vector `support` is not
    # `AbstractVector{<:CategoricalValue}`

    if pool === nothing || ismissing(pool)
        if pool === nothing
            @warn "No `CategoricalValue` found from which to extract a "*
            "complete pool of classes. "*
            "Creating a new pool (ordered=$ordered). "*
            "You can:\n"*
            " (i) specify `pool=missing` to suppress this warning; or\n"*
            " (ii) use an existing pool by specifying `pool=c` "*
            "where `c` is a "*
            "`CategoricalArray`, `CategoricalValue` or "*
            "CategoricalPool`.\n"*
            "In case (i) "*
            "specify `ordered=true` if samples are to be `OrderedFactor`. "
        end
        v = categorical(support, ordered=ordered, compress=true)
        levels!(v, support)
        _support = classes(v)
    else
        _classes = classes(pool)
        issubset(support, _classes) || throw(err_incompatible_pool(support, _classes))
        idxs = getindex.(
            Ref(CategoricalArrays.DataAPI.invrefpool(_classes)),
            support
        )
        _support = _classes[idxs]
    end

    # calls core method:
    return _UnivariateFinite(_support, probs, N;
                             augment=augment, pool=pool, ordered=ordered)
end

# 1.1 generic (non-binary) case:
_UnivariateFinite(::Val{false},
                  support::AbstractVector,
                  probs::AbstractArray{<:Any,M};
                  augment=false,
                  kwargs...) where M =
                      _UnivariateFinite(support,
                                        probs,
                                        M - 1;
                                        augment=augment,
                                        kwargs...)

# 1.2 degenerate (binary) case:
_UnivariateFinite(::Val{true},
                  support::AbstractVector,
                  probs::AbstractArray{<:Any,M};
                  augment=false,
                  kwargs...) where M =
                      _UnivariateFinite(support,
                                        probs,
                                        augment ? M : M - 1;
                                        augment=augment,
                                        kwargs...)

# 1.3 corner case, probs a scalar:
_UnivariateFinite(::Val{true},
                  support::AbstractVector,
                  probs;
                  kwargs...) =
                      UnivariateFinite(support, [probs,]; kwargs...)[1]

# 2. probablity only; unspecified support:
function UnivariateFinite(probs::AbstractArray{<:Any,N};
                          pool=nothing,
                          augment=false,
                          kwargs...) where N
    _check_pool(pool)

    # try to infer number of classes:
    if N == 1
        if augment
            c = 2
        else
            c = length(probs)
        end
    elseif N == 2
        if augment
            c = size(probs, 2) + 1
        else
            c = size(probs, 2)
        end
    else
        throw(ArgumentError(
            "You need to explicitly specify a support for "*
            "probablility arrays of three "*
            "or more dimensions. "))
    end

    support = ["class_$i" for i in 1:c]

    UnivariateFinite(support,
                     probs;
                     pool=pool,
                     augment=augment,
                     kwargs...)
end

UnivariateFiniteArray(args...; kwargs...) = UnivariateFinite(args...; kwargs...)
