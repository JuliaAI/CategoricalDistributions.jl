# CategoricalDistributions.jl

Probability distributions and measures for finite sample spaces whose
elements are *labeled* (consist of the class pool of a
`CategoricalArray`).

Designed for performance in machine learning applications. For
example, probabilistic classifiers in
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) typically
predict the `UnivariateFiniteVector` objects defined in this package.

For probability distributions over integers see the
[Distributions.jl](https://juliastats.org/Distributions.jl/stable/univariate/#Discrete-Distributions)
package, whose methods the current package extends.

| Linux | Coverage |
| :------------ | :------- |
| [![Build Status](https://github.com/JuliaAI/CategoricalDistributions.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/CategoricalDistributions.jl/actions) | [![Coverage](https://codecov.io/gh/JuliaAI/CategoricalDistributions.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/CategoricalDistributions.jl?branch=dev) |

## Installation

```julia
using Pkg
Pkg.add("CategoricalDistributions")
```

## Basic usage

The sample space of the `UnivariateFinite` distributions provided by
this package is the class pool of a `CategoricalArray`:

```julia
using CategoricalDistributions
using CategoricalArrays
import Distributions
import UnicodePlots # for optional pretty display
data = ["no", "yes", "no", "maybe", "maybe", "no",
       "maybe", "no", "maybe"] |> categorical
julia> d = Distributions.fit(UnivariateFinite, data)
               UnivariateFinite{Multiclass{3}}
         ┌                                        ┐
   maybe ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.4
      no ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.5
     yes ┤■■■■■■■ 0.1
         └                                        ┘
julia> pdf(d, "no")
0.5

julia> mode(d)
CategoricalValue{String, UInt32} "no"
```

A `UnivariateFinite` distribution can also be constructed directly
from a probability vector:

```julia
julia> d2 = UnivariateFinite(["no", "yes"], [0.15, 0.85], pool=data)
             UnivariateFinite{Multiclass{3}}
       ┌                                        ┐
    no ┤■■■■■■ 0.15
   yes ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.85
       └                                        ┘
```

A `UnivariateFinite` distribution tracks all classes in the pool:

```julia
levels(d2)
3-element Vector{String}:
 "maybe"
 "no"
 "yes"

julia> pdf(d2, "maybe")
0.0

julia> pdf(d2, "okay")
ERROR: DomainError with Value okay not in pool. :
```

Arrays of `UnivariateFinite` distributions are defined using the same
constructor. Broadcasting methods, such as `pdf`, are optimized for
such arrays:

```julia
julia> v = UnivariateFinite(["no", "yes"], [0.1, 0.2, 0.3, 0.4], augment=true, pool=data)
4-element UnivariateFiniteArray{Multiclass{3}, String, UInt32, Float64, 1}:
 UnivariateFinite{Multiclass{3}}(no=>0.9, yes=>0.1)
 UnivariateFinite{Multiclass{3}}(no=>0.8, yes=>0.2)
 UnivariateFinite{Multiclass{3}}(no=>0.7, yes=>0.3)
 UnivariateFinite{Multiclass{3}}(no=>0.6, yes=>0.4)

julia> pdf.(v, "no")
4-element Vector{Float64}:
 0.9
 0.8
 0.7
 0.6

```

Query the `UnivariateFinite` doc-string for advanced constructor options.

A (non-standard) implementation of `pdf` allows for extraction of the full
probability array:

```julia
julia> L = levels(data)
3-element Vector{String}:
 "maybe"
 "no"
 "yes"

julia> pdf(v, L)
4×3 Matrix{Float64}:
 0.0  0.9  0.1
 0.0  0.8  0.2
 0.0  0.7  0.3
 0.0  0.6  0.4
```

## Measures over finite labeled sets

There is, in fact, no enforcement that probabilities in a
`UnivariateFinite` distribution sum to one, only that they be belong
to a type `T` for which `zero(T)` is defined. In particular
`UnivariateFinite` objects implement arbitrary non-negative, signed,
or complex measures over a finite labeled set.

## What does this package provide?

- A new type `UnivariateFinite{S}` for representing probability
  distributions over the pool of a `CategoricalArray`, that is, over
  finite *labeled* sets. Here `S` is a subtype of `OrderedFactor`
  from ScientificTypesBase.jl, if the pool is ordered, or of
  `Multiclass` if the pool is unordered.

- A new array type `UnivariateFiniteArray{S} <:
  AbstractArray{<:UnivariateFinite{S}}` for efficiently manipulating
  arrays of `UnivariateFinite` distributions.

- Implementations of `rand` for generating random samples of a
  `UnivariateFinite` distribution.

- Implementations of the `pdf`, `logpdf`, `mode` and `modes` methods of
  Distributions.jl, with efficient broadcasting over the new array
  type.

- Implementation of `Distributions.fit` from Distributions.jl for
  `UnivariateFinite` distributions.

- A single constructor for constructing `UnivariateFinite`
    distributions and arrays thereof, from arrays of probabilities.

## Acknowledgements

The initial release of this package is based almost entirely on code
originally residing in
[MLJBase.jl](https://github.com/JuliaAI/MLJBase.jl) with contributions
from Anthony Blaom, Thibaut Lienart, Samuel Okon, and Chad
Scherrer. These contributions are not reflected in the current
repository's commit history.
