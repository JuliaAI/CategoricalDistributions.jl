# ## ARITHMETIC

const ERR_DIFFERENT_SAMPLE_SPACES = ArgumentError(
    "Adding two `UnivariateFinite` objects whose " *
    "sample spaces have different labellings is not allowed. ")

import Base: +, *, /, -

pdf_matrix(d::UnivariateFinite, L) = pdf.(d, L)
pdf_matrix(d::AbstractArray{<:UnivariateFinite}, L) = pdf(d, L)

function +(d1::U, d2::U) where U <: SingletonOrArray
    L = classes(d1)
    L == classes(d2) || throw(ERR_DIFFERENT_SAMPLE_SPACES)
    return UnivariateFinite(L, pdf_matrix(d1, L) + pdf_matrix(d2, L))
end

function _minus(d, T)
    S = d.scitype
    decoder = d.decoder
    prob_given_ref = copy(d.prob_given_ref)
    for ref in keys(prob_given_ref)
        prob_given_ref[ref] = -prob_given_ref[ref]
    end
    return T(S, decoder, prob_given_ref)
end
-(d::UnivariateFinite) = _minus(d, UnivariateFinite)
-(d::UnivariateFiniteArray) = _minus(d, UnivariateFiniteArray)

function -(d1::U, d2::U) where U <: SingletonOrArray
    L = classes(d1)
    L == classes(d2) || throw(ERR_DIFFERENT_SAMPLE_SPACES)
    return UnivariateFinite(L, pdf_matrix(d1, L) - pdf_matrix(d2, L))
end

# It seems that the restriction `x::Number` below (applying only to the
# array case) is unavoidable because of a method ambiguity with
# `Base.*(::AbstractArray, ::Number)`.

function _times(d, x, T)
    S = d.scitype
    decoder = d.decoder
    prob_given_ref = copy(d.prob_given_ref)
    for ref in keys(prob_given_ref)
        prob_given_ref[ref] *= x
    end
    return T(d.scitype, decoder, prob_given_ref)
end
*(d::UnivariateFinite, x) = _times(d, x, UnivariateFinite)
*(d::UnivariateFiniteArray, x::Number) = _times(d, x, UnivariateFiniteArray)

*(x, d::UnivariateFinite) = d*x
*(x::Number, d::UnivariateFiniteArray) = d*x
/(d::UnivariateFinite, x) = d*inv(x)
/(d::UnivariateFiniteArray, x::Number) = d*inv(x)
