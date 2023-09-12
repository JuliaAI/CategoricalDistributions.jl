const UniFinArr = UnivariateFiniteArray

# TODO: make sure approx methods work for arrays

Base.size(u::UniFinArr, args...) =
    size(first(values(u.prob_given_ref)), args...)
    
function Base.getindex(u::UniFinArr{<:Any,<:Any,R, P}, i...) where {R, P}
    # It's faster to generate `Array`s of `refs` and indexed `ref_probs`
    # and pass them to the `LittleDict` constructor.
    # The first element of `u.prob_given_ref` is used to get the dimensions
    # for allocating these arrays.
    u_dict = u.prob_given_ref
    a, rest = Iterators.peel(u_dict)
    # `a` is of the form `key => value`. 
    a_ref, a_prob = first(a), getindex(last(a), i...)
    
    # Preallocate Arrays using the key and value of the first 
    # element (i.e `a`) of `u_dict`. 
    n_refs = length(u_dict)
    refs = Vector{R}(undef, n_refs)
    if a_prob isa AbstractArray
        ref_probs = Vector{Array{P, ndims(a_prob)}}(undef, n_refs)
        unf_constructor = UniFinArr
    else
        ref_probs = Vector{P}(undef, n_refs)
        unf_constructor = UnivariateFinite
    end
    
    # Fill in the first elements
    # Both `refs` and `ref_probs` are both of type `Vector` and hence support 
    # linear indexing with index starting at `1`
    refs[1] = a_ref
    ref_probs[1] = a_prob

    # Fill in the rest
    iter = 2
    for (ref, ref_prob) in rest
        refs[iter] = ref
        ref_probs[iter] = getindex(ref_prob, i...) 
        iter += 1
    end

    # `keytype(prob_given_ref)` is always same as `keytype(u_dict)`. 
    # But `ndims(valtype(prob_given_ref))` might not be the same 
    # as `ndims(valtype(u_dict))`.
    prob_given_ref = LittleDict{R, eltype(ref_probs)}(refs, ref_probs)
    
    return unf_constructor(u.scitype, u.decoder, prob_given_ref)
end

function Base.getindex(u::UniFinArr, idx::CartesianIndex)
    checkbounds(u, idx)
    return u[Tuple(idx)...]
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
    "Cannot concatenate `UnivariateFiniteArray`s with " *
    "different categorical levels (classes), " *
    "or whose levels, when ordered, are not  " *
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
            classes(us[i]) == _classes || _err_incompatible_levels()
        else
            Set(classes(us[i])) == Set(_classes) || _err_incompatible_levels()
        end
        support_with_duplicates = vcat(support_with_duplicates,
                                       Dist.support(us[i]))
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
            u::AbstractArray{UnivariateFinite{S,V,R,P},N},
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
# For non-Array inputs returns the input
#This avoids using an if statement
_getindex(x::Array, i) = x[i]
_getindex(x, i) = x

# pdf.(u, cv)
function Base.Broadcast.broadcasted(
    ::typeof(pdf),
    u::UniFinArr{S,V,R,P,N},
    cv::CategoricalValue) where {S,V,R,P,N}

    # we assume that we compare categorical values by their unwrapped value
    # and pick the index of this value from classes(u)
    _classes = classes(u)
    cv_loc = get(CategoricalArrays.pool(_classes), cv, zero(R))
    isequal(cv_loc, 0) && throw(err_missing_class(cv))

    f() = zeros(P, size(u)) #default caller function

    return Base.Broadcast.Broadcasted(
        identity,
        (get(f, u.prob_given_ref, cv_loc),)
    )
end

Base.Broadcast.broadcasted(
    ::typeof(pdf),
    u::UniFinArr{S,V,R,P,N},
    ::Missing) where {S,V,R,P,N} = Missings.missings(P, size(u))

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
    
    _classes = classes(u)
    _classes_pool = CategoricalArrays.pool(_classes)
    T = eltype(v) >: Missing ? Missing : Union{}
    v_loc_flat = Vector{Tuple{Union{R, T}, Int}}(undef, length(v))
    
    
    for (i, x) in enumerate(v)
        cv_ref = ismissing(x) ? missing : get(_classes_pool, x, zero(R))
        isequal(cv_ref, 0) && throw(err_missing_class(x))
        v_loc_flat[i] = (cv_ref, i)
    end

    getter((cv_ref, i)) =
        _getindex(get(u.prob_given_ref, cv_ref, zero(P)), i)
    getter(::Tuple{Missing,Any}) = missing
    ret_flat = getter.(v_loc_flat)
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


## PERFORMANT BROADCASTING OF mode(s):

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

function Base.Broadcast.broadcasted(::typeof(modes),
                                    u::UniFinArr{S,V,R,P,N}) where {S,V,R,P,N}
    dic = u.prob_given_ref

    # using linear indexing:
    mode_flat = map(1:length(u)) do i
        max_prob = maximum(dic[ref][i] for ref in keys(dic))
        M = R[]

        # see comment for in broadcasted(::mode) above
        throw_nan_error_if_needed(max_prob)
        for ref in keys(dic)
            if dic[ref][i] == max_prob
                push!(M, ref)
            end
        end
        return u.decoder(M)
    end
    return reshape(mode_flat, size(u))
end

## EXTENSION OF CLASSES TO ARRAYS OF UNIVARIATE FINITE

# We already have `classes(::UnivariateFininiteArray)

const ERR_EMPTY_UNIVARIATE_FINITE = ArgumentError(
    "No `UnivariateFinite` object found from which to extract classes. ")

function classes(yhat::AbstractArray{<:Union{Missing,UnivariateFinite}})
    i = findfirst(!ismissing, yhat)
    i === nothing && throw(ERR_EMPTY_UNIVARIATE_FINITE)
    return classes(yhat[i])
end
