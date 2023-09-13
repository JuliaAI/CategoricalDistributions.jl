module CategoricalDistributions

import Distributions
import ScientificTypes:
    Finite,
    Multiclass,
    OrderedFactor,
    DefaultConvention,
    Density,
    ScientificTypesBase

using OrderedCollections
using CategoricalArrays
import Missings
using Random

const Dist = Distributions

import Distributions: pdf, logpdf, support, mode, modes

include("utilities.jl")
include("types.jl")
include("scitypes.jl")
include("methods.jl")
include("arrays.jl")
include("arithmetic.jl")

export UnivariateFinite, UnivariateFiniteArray, UnivariateFiniteVector

# re-eport from Distributions:
export pdf, logpdf, support, mode, modes

# re-export from ScientificTypesBase:
export Multiclass, OrderedFactor

# for julia < 1.9
if !isdefined(Base, :get_extension)
  include("../ext/UnivariateFiniteDisplayExt.jl")
end

end
