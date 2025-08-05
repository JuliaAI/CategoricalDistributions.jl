# These deprecations can be removed in CategoricalDistributions 0.3.0

# # CLASSES

@deprecate classes(x::CategoricalValue) levels(x)
@deprecate classes(v::CategoricalArray) levels(v)
@deprecate classes(v::SubArray{<:Any, <:Any, <:CategoricalArray}) levels(v)
@deprecate classes(d::UnivariateFinite) levels(d)
