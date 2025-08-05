# These deprecations can be removed in CategoricalDistributions 0.3.0

# # CLASSES

@deprecate classes(x::CategoricalValue) levels(x)
@deprecate classes(v::CategoricalArray) levels(v)
@deprecate classes(v::SubArray{<:Any, <:Any, <:CategoricalArray}) levels(v)
@deprecate classes(d::UnivariateFinite) levels(d)

const ERR_EMPTY_UNIVARIATE_FINITE = ArgumentError(
    "No `UnivariateFinite` object found from which to extract classes. ")

function classes(ds::AbstractArray{<:Union{Missing,UnivariateFinite}})
    Base.depwarn(
        "`classes(v::AbstractArray{<:Union{Missing,UnivariateFinite}})` "*
            "is deprecated. Use `levels(first(v))` instead. ",
        :classes,
    )
    return levels(first(ds))
end
