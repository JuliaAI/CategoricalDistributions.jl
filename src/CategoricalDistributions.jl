module CategoricalDistributions

export UnivariateFinite

# re-eported from Distributions:
export pdf, logpdf, support, mode

import Distributions
import ScientificTypesBase
using OrderedCollections
using CategoricalArrays
import Missings
using Random

const Dist = Distributions
const STB = ScientificTypesBase

import Distributions: pdf, logpdf, support, mode

include("utilities.jl")
include("types.jl")
include("methods.jl")
# include("arrays.jl")

end
