const STB = ScientificTypesBase

STB.scitype(::UnivariateFinite{S}, ::DefaultConvention) where S = Density{S}
STB.scitype(
    ::UnivariateFiniteArray{S,V,R,P,N},
    ::DefaultConvention
) where {S,V,R,P,N} = AbstractArray{Density{S},N}

# For performance for ordinary arrays of `UnivariateFinite` elements:
STB.Scitype(::Type{<:UnivariateFinite{S}}, ::DefaultConvention) where S =
    Density{S}
