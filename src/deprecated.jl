# These deprecations can be removed in CategoricalDistributions 0.3.0

# # CLASSES

@deprecate classes(x::CategoricalValue) levels(x)
@deprecate classes(v::CategoricalArray) levels(v)
@deprecate classes(v::SubArray{<:Any, <:Any, <:CategoricalArray}) levels(v)
@deprecate classes(d::UnivariateFinite) levels(d)

function classes(ds::AbstractArray{<:Union{Missing,UnivariateFinite}})
    Base.depwarn(
        "`classes(v::AbstractArray{<:Union{Missing,UnivariateFinite}})` "*
            "is deprecated. Assuming no missings, Use `levels(first(v))` instead. ",
        :classes,
    )
    return CategoricalDistributions.element_levels(ds)
end
