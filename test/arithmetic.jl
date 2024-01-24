module TestArithmetic

using Test
import BenchmarkTools: @belapsed
using CategoricalDistributions
using StableRNGs
rng = StableRNG(123)

macro belapsed1(ex)
    :(@belapsed $ex seconds=1 evals=1)
end

L = ["yes", "no"]
d1 = UnivariateFinite(L, rand(rng, 2), pool=missing)
d2 = UnivariateFinite(L, rand(rng, 2), pool=missing)
df32 = UnivariateFinite(L, rand(rng, Float32, 2), pool=missing)

@testset "arithmetic" begin

    # addition and subtraction:
    for op in [:+, :-]
        quote
            s = $op(d1, d2 )
            s2 = $op(d1, df32 )
            @test $op(pdf.(d1, L), pdf.(d2, L)) ≈ pdf.(s, L)
            @test $op(pdf.(d1, L), pdf.(df32, L)) ≈ pdf.(s2, L)
        end |> eval
    end

    # negative:
    d_neg = -d1
    @test pdf.(d_neg, L) == -pdf.(d1, L)

    # multiplication by scalar:
    d3 = d1*42
    @test pdf.(d3, L) ≈ pdf.(d1, L)*42
    d3 = 42*d1
    @test pdf.(d3, L) ≈ pdf.(d1, L)*42

    # division by scalar:
    d3 = d1/42
    @test pdf.(d3, L) ≈ pdf.(d1, L)/42

    # "probabilities" that aren't:
    d = UnivariateFinite(L, randn(rng, 2) + im*randn(2), pool=missing)
    @test pdf.(3*d -(4*d)/2, L) ≈ pdf.(d, L)
end

p = [0.1, 0.9]
P = vcat(fill(p', 10^5)...);
slow = fill(UnivariateFinite(L, p, pool=missing), 10^5);
fast = UnivariateFinite(L, P, pool=missing);
# @assert pdf(slow, L) == pdf(fast, L)

@testset "performant arithmetic for UnivariateFiniteArray" begin
    @test pdf(slow + slow, L) == pdf(fast + fast, L)
    t_slow = @belapsed1 $slow + $slow
    t_fast = @belapsed1 $fast + $fast
    @test t_slow/t_fast > 10

    @test pdf(slow - slow, L) == pdf(fast - fast, L)
    t_slow = @belapsed1 $slow - $slow
    t_fast = @belapsed1 $fast - $fast
    @test t_slow/t_fast > 10

    @test pdf(42*slow, L) == pdf(42*fast, L)
    @test pdf(slow*42, L) == pdf(fast*42, L)
    t_slow = @belapsed1 42*$slow
    t_fast = @belapsed1 42*$fast
    @test t_slow/t_fast > 10
    t_slow = @belapsed1 $slow*42
    t_fast = @belapsed1 $fast*42
    @test t_slow/t_fast > 10

    @test pdf(slow/42, L) == pdf(fast/42, L)
    t_slow = @belapsed1 $slow/42
    t_fast = @belapsed1 $fast/42
    @test t_slow/t_fast > 10
end

end # module

true
