const UniFinArr = UnivariateFiniteArray

# TODO: make sure approx methods work for arrays

Base.size(u::UniFinArr, args...) =
    size(first(values(u.prob_given_ref)), args...)

function Base.getindex(u::UniFinArr{<:Any,<:Any,R,P,N},
                       i::Integer...) where {R,P,N}
    prob_given_ref = LittleDict{R,P}()
    for ref in keys(u.prob_given_ref)
        prob_given_ref[ref] = getindex(u.prob_given_ref[ref], i...)
    end
    return UnivariateFinite(u.scitype, u.decoder, prob_given_ref)
end

Base.getindex(u::UniFinArr,
              idx::CartesianIndex) = u[Tuple(idx)...]

function Base.getindex(u::UniFinArr{<:Any,<:Any,R,P,N},
                       I...) where {R,P,N}
    prob_given_ref = LittleDict{R,Array{P,N}}()
    for ref in keys(u.prob_given_ref)
        prob_given_ref[ref] = getindex(u.prob_given_ref[ref], I...)
    end
    return UniFinArr(u.scitype, u.decoder, prob_given_ref)
end

function Base.setindex!(u::UniFinArr{S,V,R,P,N},
                        v::UnivariateFinite{S,V,R,P},
                        i::Integer...) where {S,V,R,P,N}
    for ref in keys(u.prob_given_ref)
       setindex!(u.prob_given_ref[ref], v.prob_given_ref[ref], i...)
    end
    return u
end

# TODO: return an exception without throwing it:

_err_incompatible_levels() = throw(DomainError(
    "Cannot concatenate `UnivariateFiniteArray`s with "*
    "different categorical levels (classes), "*
    "or whose levels, when ordered, are not  "*
    "consistently ordered. "))

# terminology:

# "classes"  - full pool of `CategoricalElement`s, even "unseen" ones (those
#             missing from support)
# "levels"   - same thing but in raw form (eg, `Symbol`s) aka "labels"
# "suppport" - those classes with a corresponding probability (the ones
#              named at time of construction of the `UnivariateFiniteArray`)

function Base.cat(us::UniFinArr{S,V,R,P,N}...;
                  dims::Integer) where {S,V,R,P,N}

    isempty(us) && return []

    # build combined raw_support and check compatibility of levels:
    u1 = first(us)
    ordered = isordered(classes(u1))
    support_with_duplicates = Dist.support(u1)
    _classes = classes(u1)
    for i in 2:length(us)
        isordered(us[i]) == ordered || _err_incompatible_levels()
        if ordered
            classes(us[i]) ==
                _classes|| _err_incompatible_levels()
        else
            Set(classes(us[i])) ==
                Set(_classes) || _err_incompatible_levels()
        end
        support_with_duplicates =
            vcat(support_with_duplicates, Dist.support(us[i]))
    end
    _support = unique(support_with_duplicates) # no-longer categorical!

    # build the combined `prob_given_class` dictionary:
    pairs = (class => cat((pdf.(u, class) for u in us)..., dims=dims)
             for class in _support)
    prob_given_class = Dict(pairs)

    return UnivariateFinite(prob_given_class, pool=_classes)
end

Base.vcat(us::UniFinArr...) = cat(us..., dims=1)
Base.hcat(us::UniFinArr...) = cat(us..., dims=2)


## CONVENIENCE METHODS pdf(array_of_univariate_finite, labels)
## AND logpdf(array_of_univariate_finite, labels)

# Next bit is not specific to `UnivariateFiniteArray` but is for any
# abstract array with eltype `UnivariateFinite`.

# This is type piracy that has been adopted only after much
# agonizing over alternatives. Note that `pdf.(u, labels)` must
# necessarily have a different meaning (and only makes sense if `u` and
# labels have the same length or `labels` is a scalar)

for func in [:pdf, :logpdf]
    eval(quote
        function Distributions.$func(
            u::AbstractArray{UnivariateFinite{S,V,R,P}, N},
            C::AbstractVector) where {S,V,R,P,N}

            ret = zeros(P, size(u)..., length(C))
            # note that we do not require C to use 1-base indexing
            for (i, c) in enumerate(C)
                ret[fill(:,N)..., i] .= broadcast($func, u, c)
            end
            return ret
        end
    end)
end


##
## PERFORMANT BROADCASTING OF pdf and logpdf
##

# u - a UnivariateFiniteArray
# cv - a CategoricalValue
# v - a vector of CategoricalArrays

# dummy function
# returns `x[i]` for `Array` inputs `x`
# For non-Array inputs returns `zero(dtype)`
#This avoids using an if statement
_getindex(x::Array, i, dtype)=x[i]
_getindex(::Nothing, i, dtype) = zero(dtype)

# pdf.(u, cv)
function Base.Broadcast.broadcasted(
    ::typeof(pdf),
    u::UniFinArr{S,V,R,P,N},
    cv::CategoricalValue) where {S,V,R,P,N}

    # we assume that we compare categorical values by their unwrapped value
    # and pick the index of this value from classes(u)
    cv_loc = findfirst(==(cv), classes(u))
    cv_loc == 0 && throw(err_missing_class(cv))

    f() = zeros(P, size(u)) #default caller function

    return Base.Broadcast.Broadcasted(
        identity,
        (get(f, u.prob_given_ref, cv_loc),)
        )
end

Base.Broadcast.broadcasted(
    ::typeof(pdf),
    u::UniFinArr{S,V,R,P,N},
    ::Missing) where {S,V,R,P,N} = Missings.missings(P, length(u))

# pdf.(u, v)
function Base.Broadcast.broadcasted(
    ::typeof(pdf),
    u::UniFinArr{S,V,R,P,N},
    v::AbstractArray{<:Union{
        Missing,
        CategoricalValue{V,R}},N}) where {S,V,R,P,N}

    length(u) == length(v) ||throw(DimensionMismatch(
        "Arrays could not be broadcast to a common size; "*
        "got a dimension with lengths $(length(u)) and $(length(v))"))

    v_loc_flat = [(ismissing(x) ? missing : findfirst(x, classes(u)), i)
                  for (i, x) in enumerate(v)]
    any(isequal(0), v_loc_flat) && throw(err_missing_class(cv))

    getter((cv_loc, i), dtype) =
        _getindex(get(u.prob_given_ref, cv_loc, nothing), i, dtype)
    getter(::Tuple{Missing,Any}, dtype) = missing
    ret_flat = getter.(v_flat, P)
    return reshape(ret_flat, size(u))
end

# pdf.(u, raw) where raw is scalar or vec
function Base.Broadcast.broadcasted(
    ::typeof(pdf),
    u::UniFinArr{S,V,R,P,N},
    raw::Union{V,AbstractArray{<:Union{Missing,V},N}}) where {S,V,R,P,N}

    cat = transform(classes(u), raw)
    return Base.Broadcast.broadcasted(pdf, u, cat)
end

# logpdf.(u::UniFinArr{S,V,R,P,N}, cv::CategoricalValue)
# logpdf.(u::UniFinArr{S,V,R,P,N},
#         v::AbstractArray{<:Union{Missing,CategoricalValue{V,R}},N})
# logpdf.(u::UniFinArr{S,V,R,P,N}, raw::AbstractArray{V,N})
# logpdf.(u::UniFinArr{S,V,R,P,N}, raw::V)
for typ in (:CategoricalValue,
            :(AbstractArray{<:Union{Missing,CategoricalValue{V,R}},N}),
            :V,
            :(AbstractArray{V,N}))
   if typ == :CategoricalValue || typ == :V
        eval(quote
        function Base.Broadcast.broadcasted(
                                ::typeof(logpdf),
                                u::UniFinArr{S,V,R,P,N},
                                c::$typ) where {S,V,R,P,N}

            # Start with the pdf array
            # take advantage of loop fusion
            result = log.(pdf.(u, c))
            return result
        end
        end)

  else
        eval(quote
        function Base.Broadcast.broadcasted(
                                ::typeof(logpdf),
                                u::UniFinArr{S,V,R,P,N},
                                c::$typ) where {S,V,R,P,N}

            # Start with the pdf array
            result = pdf.(u, c)

            # Take the log of each entry in-place
            @simd for j in eachindex(result)
                @inbounds result[j] = log(result[j])
            end
            return result
        end
        end)
  end

end
Base.Broadcast.broadcasted(
    ::typeof(logpdf),
    u::UniFinArr{S,V,R,P,N},
    c::Missing) where {S,V,R,P,N} = Missings.missings(P, length(u))


## PERFORMANT BROADCASTING OF mode:

function Base.Broadcast.broadcasted(::typeof(mode),
                                    u::UniFinArr{S,V,R,P,N}) where {S,V,R,P,N}
    dic = u.prob_given_ref

    # using linear indexing:
    mode_flat = map(1:length(u)) do i
        max_prob = maximum(dic[ref][i] for ref in keys(dic))
        m = zero(R)

        # `maximum` of any iterable containing `NaN` would return `NaN`
        # For this case the index `m` won't be updated in the loop as relations
        # involving NaN as one of it's argument always returns false
        # (e.g `==(NaN, NaN)` returns false)
        throw_nan_error_if_needed(max_prob)
        for ref in keys(dic)
            if dic[ref][i] == max_prob
                m = ref
                break
            end
        end
        return u.decoder(m)
    end
    return reshape(mode_flat, size(u))
end


## EXTENSION OF CLASSES TO ARRAYS OF UNIVARIATE FINITE

# We already have `classes(::UnivariateFininiteArray)

const ERR_EMPTY_UNIVARIATE_FINITE = ArgumentError(
    "No `UnivariateFinite` object found from which to extract classes. ")

function classes(yhat::AbstractArray{<:Union{Missing,UnivariateFinite}})
    i = findfirst(x->!ismissing(x), yhat)
    i === nothing && throw(ERR_EMPTY_UNIVARIATE_FINITE)
    return classes(yhat[i])
end
