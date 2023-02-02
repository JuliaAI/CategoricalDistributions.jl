using Test
using CategoricalDistributions
using CategoricalArrays
using Distributions
using Random
using ScientificTypes
using StableRNGs # for RNGs stable across all julia versions
rng = StableRNGs.StableRNG(123)

import CategoricalDistributions: classes, decoder, int

ambiguities_vec = Test.detect_ambiguities(CategoricalDistributions,
                                          recursive=true)
if !isempty(ambiguities_vec)
    @warn "$(length(ambiguities_vec)) method ambiguities detected"
end

@testset "utilities" begin
     @test include("utilities.jl")
end

@testset "types" begin
     @test include("types.jl")
end

@testset "scitypes" begin
     @test include("scitypes.jl")
end

@testset "methods" begin
     @test include("methods.jl")
end

@testset "arrays.jl" begin
     @test include("arrays.jl")
end

@testset "arithmetic.jl" begin
     @test include("arithmetic.jl")
end
