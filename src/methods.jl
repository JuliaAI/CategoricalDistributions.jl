# not for export:
const UnivariateFiniteUnion{S,V,R,P} =
    Union{UnivariateFinite{S,V,R,P}, UnivariateFiniteArray{S,V,R,P}}

"""
    classes(d::UnivariateFinite)
    classes(d::UnivariateFiniteArray)

A list of categorial elements in the common pool of classes used to
construct `d`.

    v = categorical(["yes", "maybe", "no", "yes"])
    d = UnivariateFinite(v[1:2], [0.3, 0.7])
    classes(d) # CategoricalArray{String,1,UInt32}["maybe", "no", "yes"]

"""
classes(d::UnivariateFiniteUnion) = d.decoder.classes

"""
    levels(d::UnivariateFinite)

A list of the raw levels in the common pool of classes used to
construct `d`, equal to
`CategoricalArrays.DataAPI.unwrap.(classes(d))`.

    v = categorical(["yes", "maybe", "no", "yes"])
    d = UnivariateFinite(v[1:2], [0.3, 0.7])
    levels(d) # Array{String, 1}["maybe", "no", "yes"]

"""
Missings.levels(d::UnivariateFinite)  =
    CategoricalArrays.DataAPI.unwrap.(classes(d))

function Dist.params(d::UnivariateFinite)
    raw = raw_support(d) # reflects order of pool at instantiation of d
    pairs = tuple([unwrap.(d.decoder(r))=>d.prob_given_ref[r] for r in raw]...)
    levs = unwrap.(classes(d))
    return (levels=levs, probs=pairs)
end

# get the internal integer representations of the support
raw_support(d::UnivariateFiniteUnion) = collect(keys(d.prob_given_ref))

"""
    Distributions.support(d::UnivariateFinite)
    Distributions.support(d::UnivariateFiniteArray)

Ordered list of classes associated with non-zero probabilities.

    v = categorical(["yes", "maybe", "no", "yes"])
    d = UnivariateFinite(v[1:2], [0.3, 0.7])
    Distributions.support(d) # CategoricalArray{String,1,UInt32}["maybe", "yes"]

"""
Distributions.support(d::UnivariateFiniteUnion) = classes(d)[raw_support(d)]

"""
    fast_support(d::UnivariateFinite)

Same as `Distributions.support(d)` except it returns a vector of `CategoricalValue`s,
rather than a `CategoricalVector`. It executes faster, about five times faster for a
three-class `UnivariateFinite` distribution.
"""
function fast_support(d::UnivariateFiniteUnion{S,V,R}) where {S,V,R}
    raw_support = keys(d.prob_given_ref)
    n = length(raw_support)
    ret = Vector{CategoricalValue{V,R}}(undef, n)
    for (i, ref) in enumerate(raw_support)
        ret[i] = d.decoder(ref)
    end
    ret
end

# TODO: If I manually give a class zero probability, it will appear in
# support, which is probably confusing. We may need two versions of
# support - one which are the labels with corresponding keys in the
# dictionary, and one for the true mathematical support (which is the
# one exported)

# not exported:
sample_scitype(d::UnivariateFiniteUnion) = d.scitype

CategoricalArrays.isordered(d::UnivariateFiniteUnion) = isordered(classes(d))


## DISPLAY

_round_prob(p) = p
_round_prob(p::Union{Float32,Float64}) = round(p, sigdigits=3)

function Base.show(stream::IO, d::UnivariateFinite)
    pairs = Dist.params(d).probs
    arg_str = join(["$(pr[1])=>$(_round_prob(pr[2]))" for pr in pairs], ", ")
    print(stream, "UnivariateFinite{$(d.scitype)}($arg_str)")
end

Base.show(io::IO, mime::MIME"text/plain",
          d::UnivariateFinite) = show(io, d)

show_prefix(u::UnivariateFiniteArray{S,V,R,P,1}) where {S,V,R,P} =
    "$(length(u))-element"
show_prefix(u::UnivariateFiniteArray) = join(size(u),'x')

"""
    isapprox(d1::UnivariateFinite, d2::UnivariateFinite; kwargs...)

Returns `true` if and only if `d1` and `d2` have the same support and
the corresponding probabilities are approximately equal. The key-word
arguments `kwargs` are passed through to each call of `isapprox` on
probability pairs.  Returns `false` otherwise.

"""
function Base.isapprox(d1::UnivariateFinite, d2::UnivariateFinite; kwargs...)
    support1 = fast_support(d1)
    support2 = fast_support(d2)
    for c in support1
        c in support2 || return false
        isapprox(pdf(d1, c), pdf(d2, c); kwargs...) ||
            return false # pdf defined below
    end
    return true
end
function Base.isapprox(d1::UnivariateFiniteArray,
                       d2::UnivariateFiniteArray; kwargs...)
    support1 = fast_support(d1)
    support2 = fast_support(d2)
    for c in support1
        c in support2 || return false
        isapprox(pdf.(d1, c), pdf.(d2, c); kwargs...) ||
            return false
    end
    return true
end

# TODO: It would be useful to define == as well.

"""
    Distributions.pdf(d::UnivariateFinite, x)

Probability of `d` at `x`.

    v = categorical(["yes", "maybe", "no", "yes"])
    d = UnivariateFinite(v[1:2], [0.3, 0.7])
    pdf(d, "yes")     # 0.3
    pdf(d, v[1])      # 0.3
    pdf(d, "no")      # 0.0
    pdf(d, "house")   # throws error

Other similar methods are available too:

    mode(d)    # CategoricalValue{String, UInt32} "maybe"
    rand(d, 5) # CategoricalArray{String,1,UInt32}["maybe", "no", "maybe", "maybe", "no"] or similar
    d = fit(UnivariateFinite, v)
    pdf(d, "maybe") # 0.25
    logpdf(d, "maybe") # log(0.25)

One can also do weighted fits:

    w = [1, 4, 5, 1] # some weights
    d = fit(UnivariateFinite, v, w)
    pdf(d, "maybe") ≈ 4/11 # true

See also `classes`, `support`.
"""
Dist.pdf(::UnivariateFinite, ::Missing) = missing

function Dist.pdf(d::UnivariateFinite{S,V,R,P}, c) where {S,V,R,P}
    _classes = classes(d)
    c in _classes || throw(DomainError("Value $c not in pool. "))
    pool = CategoricalArrays.pool(_classes)
    return get(d.prob_given_ref, get(pool, c), zero(P))
end

Dist.logpdf(d::UnivariateFinite, c) = log(pdf(d, c))

function Dist.mode(d::UnivariateFinite)
    dic = d.prob_given_ref
    p = values(dic)
    max_prob = maximum(p)
    m = first(first(dic)) # mode, just some ref for now

    # `maximum` of any iterable containing `NaN` would return `NaN`
    # For this case the index `m` won't be updated in the loop below as relations
    # involving NaN as one of it's arguments always returns false
    # (e.g `==(NaN, NaN)` returns false)
    throw_nan_error_if_needed(max_prob)
    for (x, prob) in dic
        if prob == max_prob
            m = x
            break
        end
    end
    return d.decoder(m)
end

function Dist.modes(d::UnivariateFinite{S,V,R,P}) where {S,V,R,P}
    dic = d.prob_given_ref
    p = values(dic)
    max_prob = maximum(p)
    M = R[] # modes

    # see comment in `mode` above
    throw_nan_error_if_needed(max_prob)
    for (x, prob) in dic
        if prob == max_prob
            push!(M, x)
        end
    end
    return d.decoder(M)
end

const ERR_NAN_FOUND = DomainError(
    NaN,
    "`mode(s)` is invalid for a `UnivariateFinite` distribution "*
    "with `pdf` containing `NaN`s"
)

function throw_nan_error_if_needed(x)
    if isnan(x)
        throw(ERR_NAN_FOUND)
    end
end


# # HELPERS FOR RAND

"""
    _cumulative(d::UnivariateFinite)

**Private method.**

Return the cumulative probability vector `C` for the distribution `d`, using only classes
in `Distributions.support(d)`, ordered according to the categorical elements used at
instantiation of `d`. Used only to implement random sampling from `d`. We have `C[1] == 0`
and `C[end] == 1`, assuming the probabilities have been normalized.

"""
function _cumulative(d::UnivariateFinite{S,V,R,P}) where {S,V,R,P<:Real}

    # the keys of `d` are in order; see constructor
    p = collect(values(d.prob_given_ref))
    K = length(p)
    p_cumulative = Array{P}(undef, K + 1)
    p_cumulative[1] = zero(P)
    for i in 2:K + 1
        p_cumulative[i] = p_cumulative[i-1] + p[i-1]
    end
    return p_cumulative
end

"""
_rand(rng, p_cumulative, R)

**Private method.**

Randomly sample the distribution with discrete support `R(1):R(n)`
which has cumulative probability vector `p_cumulative` (see
[`_cummulative`](@ref)).

"""
function _rand(rng, p_cumulative, R)
    real_sample = rand(rng)*p_cumulative[end]
    K = R(length(p_cumulative))
    index = K
    for i in R(2):R(K)
        if real_sample < p_cumulative[i]
            index = i - R(1)
            break
        end
    end
    return index
end


# # RAND

Random.eltype(::Type{<:UnivariateFinite{S,V,R}}) where {S,V,R} =
    CategoricalArrays.CategoricalValue{V,R}

# The Sampler hook into Random's API is discussed in the Julia documentation, in the
# Standard Library section on Random.


## Single samples

Random.Sampler(::AbstractRNG, d::UnivariateFinite, ::Val{1}) = Random.SamplerTrivial(d)

function Base.rand(
    rng::AbstractRNG,
    sampler::Random.SamplerTrivial{<:UnivariateFinite{<:Any,<:Any,V,P}},
    ) where {V, P}

    d = sampler[]
    u = rand(rng)

    total = zero(P)
    
    # For type stability we assign `zero(V)`` as the default ref
    # This isn't a problem since we know that `rand` is always defined 
    # as UnivariateFinite objects have non-negative probabilities,
    # summing up to a non-negative value.
    rng_key = zero(V)
    for (ref, prob) in pairs(d.prob_given_ref)
        total += prob
        u <= total && begin
            rng_key = ref
            break
        end
    end
    return d.decoder(rng_key)
end


## Multiple samples

function Random.Sampler(
    ::AbstractRNG,
    d::UnivariateFinite,
    ::Random.Repetition,
    )
    data = (_cumulative(d), fast_support(d))
    Random.SamplerSimple(d, data)
end

function Base.rand(
    rng::AbstractRNG,
    sampler::Random.SamplerSimple{<:UnivariateFinite{<:Any,<:Any,R}},
    ) where R
    p_cumulative, support = sampler.data
    return support[_rand(rng, p_cumulative, R)]
end


## FIT

function Dist.fit(d::Type{<:UnivariateFinite},
                           v::AbstractVector{C}) where C
    C <: CategoricalValue ||
        error("Can only fit a UnivariateFinite distribution to samples of "*
              "`CategoricalValue type`. ")
    y = skipmissing(v) |> collect
    isempty(y) && error("No non-missing data to fit. ")
    N = length(y)
    count_given_class = Dist.countmap(y)
    classes = Tuple(keys(count_given_class))
    probs = values(count_given_class)./N
    prob_given_class = LittleDict(classes, probs)
    return UnivariateFinite(prob_given_class)
end

# TODO: Implement an MLJ style fit/transform with an option to specify
# Laplacian smoothing.

function Dist.fit(d::Type{<:UnivariateFinite},
                           v::AbstractVector{C},
                           weights::Nothing) where C
    return Dist.fit(d, v)
end

function Dist.fit(d::Type{<:UnivariateFinite},
                           v::AbstractVector{C},
                           weights::AbstractVector{<:Real}) where C
    C <: CategoricalValue ||
        error("Can only fit a UnivariateFinite distribution to samples of "*
              "`CategoricalValue` type. ")
    y = broadcast(identity, skipmissing(v))
    isempty(y) && error("No non-missing data to fit. ")
    classes_seen = filter(in(unique(y)), classes(y[1]))

    # instantiate and initialize prob dictionary:
    prob_given_class = LittleDict{C,Float64}()
    for c in classes_seen
        prob_given_class[c] = 0
    end

    # compute unnormalized  probablilities:
    for i in eachindex(y)
        prob_given_class[y[i]] += weights[i]
    end

    # normalize the probabilities:
    S = sum(values(prob_given_class))
    for c in keys(prob_given_class)
        prob_given_class[c] /=S
    end

    return UnivariateFinite(prob_given_class)
end


# # BROADCASTING OVER SINGLE UNIVARIATE FINITE

# This mirrors behaviour assigned Distributions.Distribution objects,
# which allows `pdf.(d::UnivariateFinite, support(d))` to work.

Broadcast.broadcastable(d::UnivariateFinite) = Ref(d)
