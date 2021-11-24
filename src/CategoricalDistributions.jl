module CategoricalDistributions

import Distributions
import ScientificTypesBase: Finite, Multiclass, OrderedFactor
using OrderedCollections
using CategoricalArrays
import Missings
using Random
using UnicodePlots

const Dist = Distributions

import Distributions: pdf, logpdf, support, mode

include("utilities.jl")
include("types.jl")
include("methods.jl")
include("arrays.jl")

export UnivariateFinite, UnivariateFiniteArray

# re-eport from Distributions:
export pdf, logpdf, support, mode

# re-export from ScientificTypesBase:
export Multiclass, OrderedFactor

end
