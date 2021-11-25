# ## ARITHMETIC

const ERR_DIFFERENT_SAMPLE_SPACES = ArgumentError(
    "Adding two `UnivariateFinite` objects whose "*
    "sample spaces have different labellings is not allowed. ")

import Base: +, *, /, -

function _plus(d1, d2, T)
    classes(d1) == classes(d2) || throw(ERR_DIFFERENT_SAMPLE_SPACES)
    S = d1.scitype
    decoder = d1.decoder
    prob_given_ref = copy(d1.prob_given_ref)
    for ref in keys(prob_given_ref)
        prob_given_ref[ref] += d2.prob_given_ref[ref]
    end
    return T(S, decoder, prob_given_ref)
end
+(d1::U, d2::U) where U <: UnivariateFinite = _plus(d1, d2, UnivariateFinite)
+(d1::U, d2::U) where U <: UnivariateFiniteArray =
    _plus(d1, d2, UnivariateFiniteArray)

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

function _minus(d1, d2, T)
    classes(d1) == classes(d2) || throw(ERR_DIFFERENT_SAMPLE_SPACES)
    S = d1.scitype
    decoder = d1.decoder
    prob_given_ref = copy(d1.prob_given_ref)
    for ref in keys(prob_given_ref)
        prob_given_ref[ref] -= d2.prob_given_ref[ref]
    end
    return T(S, decoder, prob_given_ref)
end
-(d1::U, d2::U) where U <: UnivariateFinite = _minus(d1, d2, UnivariateFinite)
-(d1::U, d2::U) where U <: UnivariateFiniteArray =
    _minus(d1, d2, UnivariateFiniteArray)

# TODO: remove type restrction on `x` in the following methods if
# https://github.com/JuliaStats/Distributions.jl/issues/1438 is
# resolved. Currently we'd have a method ambiguity

function _times(d, x, T)
    S = d.scitype
    decoder = d.decoder
    prob_given_ref = copy(d.prob_given_ref)
    for ref in keys(prob_given_ref)
        prob_given_ref[ref] *= x
    end
    return T(d.scitype, decoder, prob_given_ref)
end
*(d::UnivariateFinite, x::Real) = _times(d, x, UnivariateFinite)
*(d::UnivariateFiniteArray, x::Real) = _times(d, x, UnivariateFiniteArray)

*(x::Real, d::SingletonOrArray) = d*x
/(d::SingletonOrArray, x::Real) = d*inv(x)
