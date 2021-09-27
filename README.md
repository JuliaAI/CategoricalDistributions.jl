# CategoricalDistributions.jl

Probability distributions and measures for finite sample spaces whose
elements are labelled. Designed for performance in machine learning
applications to classification.

| Linux | Coverage |
| :------------ | :------- |
| [![Build Status](https://github.com/JuliaAI/CategoricalDistributions.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/CategoricalDistributions.jl/actions) | [![Coverage](https://codecov.io/gh/JuliaAI/CategoricalDistributions.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/CategoricalDistributions.jl?branch=master) |

## Very basic usage

The sample space of the `UnivariateFinite` distributions provided by
this package is the class pool of a `CategoricalArray`:

```julia
using CategoricalDistributions
using CategoricalArrays
julia> data = rand(["yes", "no", "maybe"], 10) |> categorical
10-element CategoricalArray{String,1,UInt32}:
 "maybe"
 "maybe"
 "no"
 "yes"
 "maybe"
 "no"
 "no"
 "no"
 "no"
 "yes"

julia> d = fit(UnivariateFinite, data)
UnivariateFinite{Multiclass{3}}(maybe=>0.3, no=>0.5, yes=>0.2)

julia> pdf(d, "no")
0.5

julia> mode(d)
CategoricalValue{String, UInt32} "no"
```

Efficient *arrays* of `UnivariateFinite` distributions can also be
constructed and efficiently manipulated:

```
julia> v = UnivariateFinite(["no", "yes"], [0.1, 0.2, 0.3], augment=true, pool=data)
3-element UnivariateFiniteVector{Multiclass{3}, String, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(no=>0.9, yes=>0.1)
 UnivariateFinite{Multiclass{3}}(no=>0.8, yes=>0.2)
 UnivariateFinite{Multiclass{3}}(no=>0.7, yes=>0.3)

julia> pdf.(v, "no")
3-element Vector{Float64}:
 0.9
 0.8
 0.7

julia> pdf.(v, "maybe")
3-element Vector{Float64}:
 0.0
 0.0
 0.0
```

A non-standard implementation of `pdf` allows extraction of the full
probability array:

```julia
julia> L = levels(data)
3-element Vector{String}:
 "maybe"
 "no"
 "yes"
 
julia> pdf(v, L)
3×3 Matrix{Float64}:
 0.0  0.9  0.1
 0.0  0.8  0.2
 0.0  0.7  0.3
```

## What does this package provide?

- A new type `UnivariateFinite{S}` for representing probability
  distributions over the pool of a `CategoricalArray`, that is, over
  finite *labelled* sets. Here `S` is a subtype of `OrderedFactor`
  from ScientificTypesBase.jl, if the pool is ordered, or of
  `Multiclass` if the pool is unordered.
  
- A new array type `UnivariateFiniteArray{S} <:
  AbstractArray{<:UnivariateFinite{S}}` for efficiently manipulating
  arrays of `UnivariateFinite` distributions.
  
- Implementations of `rand` for generating random samples of a
  `UnivariateFinite` distribution.
  
- Implementations of the `pdf`, `logpdf` and `mode` methods of
  Distributions.jl, with efficient broadcasting over the new array
  type.

- Implementation of `fit` from Distributions.jl for `UnivariateFinite`
  distributions.
  
- A single constructor for constructing `UnivariateFinite`
  distributions and arrays thereof, from arrays of probabilities.

There is, in fact, no enforcement that probablilities in a
`UnivariateFinite` distribution sum to one and they can therefore be
more properly understood as implementations of arbitrary finite
measures over labelled sets.
