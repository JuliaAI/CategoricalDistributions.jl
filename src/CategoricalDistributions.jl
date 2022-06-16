module CategoricalDistributions

import Distributions
import ScientificTypes: Finite, Multiclass, OrderedFactor, scitype, DefaultConvention
using OrderedCollections
using CategoricalArrays
import Missings
using Random
using UnicodePlots

const Dist = Distributions
const MAX_NUM_LEVELS_TO_SHOW_BARS = 12

import Distributions: pdf, logpdf, support, mode

include("utilities.jl")
include("types.jl")
include("methods.jl")
include("arrays.jl")
include("arithmetic.jl")

export UnivariateFinite, UnivariateFiniteArray, UnivariateFiniteVector

# re-eport from Distributions:
export pdf, logpdf, support, mode

# re-export from ScientificTypesBase:
export Multiclass, OrderedFactor

end
