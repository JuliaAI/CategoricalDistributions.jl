module UnivariateFiniteDisplayExt

const MAX_NUM_LEVELS_TO_SHOW_BARS = 12

using CategoricalDistributions
import CategoricalArrays
import UnicodePlots
import ScientificTypes.Finite

# The following is a specialization of a `show` method already in /src/ for the common
# case of `Real` probabilities.
function Base.show(io::IO, mime::MIME"text/plain",
                   d::UnivariateFinite{<:Finite{K},V,R,P}) where {K,V,R,P<:Real}
    show_bars = false
    if K <= MAX_NUM_LEVELS_TO_SHOW_BARS &&
        all(>=(0), values(d.prob_given_ref))
        show_bars = true
    end
    show_bars || return show(io, d)
    s = support(d)
    x = string.(CategoricalArrays.DataAPI.unwrap.(s))
    y = pdf.(d, s)
    S = d.scitype
    plt = UnicodePlots.barplot(x, y, title="UnivariateFinite{$S}")
    show(io, mime, plt)
end

end
