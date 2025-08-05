# These deprecations can be removed in CategoricalDistributions 0.3.0

# # CLASSES

@deprecated classes(x::CategoricalValue) levels(x)
@deprecated classes(v::CategoricalArray) levels(v)
@deprecated classes(v::SubArray{<:Any, <:Any, <:CategoricalArray}) levels(v)
@deprecated classes(d::UnivariateFinite) = levels(d)
