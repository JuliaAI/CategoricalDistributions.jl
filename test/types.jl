module TestTypes

using Test
using CategoricalDistributions
using CategoricalArrays
using StableRNGs
using ScientificTypesBase
import Random

# coverage of constructor testing is expanded in the other test files

@testset "constructors" begin
    @test_throws ArgumentError UnivariateFinite(Dict('f'=>0.7, 'q'=>0.2))
    @test_throws ArgumentError UnivariateFinite(Dict('f'=>0.7, 'q'=>0.2),
                                                pool=missing)
    v = categorical(collect("asqfasqffqsaaaa"), ordered=true)
    f = v[4]
    @test_logs((:warn, r"Ignoring"),
               UnivariateFinite(Dict('f'=>0.7, 'q'=>0.3),
                                pool=f, ordered=true))

    @test_logs((:warn, r"No "),
               UnivariateFinite(['f', 'q', 's'],  [0.7, 0.2, 0.1]))

end

@testset "array constructors" begin

    rng = StableRNG(111)
    n   = 10
    c   = 3

    probs  = rand(rng, n)
    supp = ["class1", "class2"]

    UnivariateFinite(supp, probs, pool=missing, augment=true);

    # dimension mismatches:
    badprobs = rand(40, 3)
    @test_throws(CategoricalDistributions.err_dim(supp, badprobs),
                 UnivariateFinite(supp, badprobs, pool=missing))

    # dimension mismatch, augmented case:
    probs2 = [0.1 0.5 0.1;
              0.3 0.2 0.1]
    supp2 = ["no", "yes", "maybe"]
    @test_throws(CategoricalDistributions.err_dim_augmented(supp2, probs2),
                 UnivariateFinite(supp2, probs2, augment=true, pool=missing))

    # not augmentable:
    @test_throws(CategoricalDistributions.ERR_AUG,
                 UnivariateFinite(["no", "yes", "maybe"],
                                  [0.6 0.5;   # sum exceeding one!
                                   0.3 0.2],
                                  augment=true,
                                  pool=missing))

    # autosupport:
    u = UnivariateFinite(probs, pool=missing, augment=true);
    probs = probs ./ sum(probs)
    u = UnivariateFinite(probs, pool=missing);
    @test u isa UnivariateFinite
    probs = rand(10, 2)
    probs = probs ./ sum(probs, dims=2)
    u = UnivariateFinite(probs, pool=missing);
    @test u.scitype == Multiclass{2}
    u = UnivariateFinite(probs, pool=missing, augment=true);
    @test u.scitype == Multiclass{3}

    probs = [-1,0,1]
    @test_throws(CategoricalDistributions.ERR_01,
                 UnivariateFinite(probs, pool=missing, augment=true))

    v = categorical(1:3)
    @test_logs((:warn, r"Ignoring"),
               UnivariateFinite(v[1:2], rand(3), augment=true, pool=missing));
    @test_logs((:warn, r"Ignoring"),
               UnivariateFinite(v[1:2], rand(3), augment=true, ordered=true));

    # using `UnivariateFiniteArray` as a constructor just falls back
    # to `UnivariateFinite` constructor:
    probs  = rand(rng, n)
    d1 = UnivariateFiniteArray(supp, probs, pool=missing, augment=true)
    d2 = UnivariateFinite(supp, probs, pool=missing, augment=true)
    @test d1.prob_given_ref == d2.prob_given_ref
end

@testset "display" begin
    d = UnivariateFinite(["x", "y"], [0.3, 0.7], pool=missing)
    v = UnivariateFinite(["x", "y"], rand(3), augment=true, pool=missing)
    io = IOBuffer()
    print(io, d, v, [d, ], [v, ])
end

end

true
