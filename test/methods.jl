module TestUnivariateFiniteMethods

using Test
using CategoricalDistributions
using CategoricalArrays
import Distributions
using StableRNGs
import Random
rng = StableRNG(123)
using ScientificTypes
import Random.default_rng

import CategoricalDistributions: classes, ERR_NAN_FOUND

v = categorical(collect("asqfasqffqsaaaa"), ordered=true)
V = categorical(collect("asqfasqffqsaaaa"))
a, s, q, f = v[1], v[2], v[3], v[4]
A, S, Q, F = V[1], V[2], V[3], V[4]

@testset "set 1" begin

    # ordered (OrderedFactor)
    dict = Dict(s=>0.1, q=>0.2, f=>0.7)
    d    = UnivariateFinite(dict)
    dict_bimodal = Dict(a=>0.1, s=>0.1, q=>0.4, f=>0.4)
    d_bimodal    = UnivariateFinite(dict_bimodal)
    @test classes(d) == [a, f, q, s]
    @test classes(d) == classes(s)
    @test levels(d) == levels(s)
    @test support(d) == [f, q, s]
    @test support(d) == [CategoricalDistributions.fast_support(d)...]
    @test CategoricalDistributions.sample_scitype(d) == OrderedFactor{4}
    # levels!(v, reverse(levels(v)))
    # @test classes(d) == [s, q, f, a]
    # @test support(d) == [s, q, f]
    @test ismissing(pdf(d, missing))
    @test pdf(d, s) ≈ 0.1
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, q) ≈ 0.2
    @test pdf(d, 'q') ≈ 0.2
    @test pdf(d, f) ≈ 0.7
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, a) ≈ 0.0
    @test logpdf(d, s) ≈ log(0.1)
    @test logpdf(d, 's') ≈ log(0.1)
    @test logpdf(d, q) ≈ log(0.2)
    @test logpdf(d, 'q') ≈ log(0.2)
    @test logpdf(d, f) ≈ log(0.7)
    @test logpdf(d, 'f') ≈ log(0.7)
    @test isinf(logpdf(d, a))
    @test mode(d) == f
    @test modes(d_bimodal) == [f, q]

    @test UnivariateFinite(support(d), [0.7, 0.2, 0.1]) ≈ d

    N = 50
    rng = StableRNG(125)
    samples = [rand(rng, d) for i in 1:N];
    rng = StableRNG(125)
    @test samples == [rand(rng, d) for i in 1:N]

    N = 10000
    samples = rand(StableRNG(123), d, N);
    @test Set(samples) == Set(support(d))
    freq = Distributions.countmap(samples)
    @test isapprox(freq[f]/N, 0.7, atol=0.05)
    @test isapprox(freq[s]/N, 0.1, atol=0.05)
    @test isapprox(freq[q]/N, 0.2, atol=0.05)

    # test unnormalized case gives same answer:
    dd = UnivariateFinite(support(d), [70, 20, 10])
    samples = rand(StableRNG(123), dd, N);
    @test Set(samples) == Set(support(d))
    ffreq = Distributions.countmap(samples)
    @test isapprox(freq[f]/N, ffreq[f]/N)
    @test isapprox(freq[s]/N, ffreq[s]/N)
    @test isapprox(freq[q]/N, ffreq[q]/N)

    # unordered (Multiclass):
    dict = Dict(S=>0.1, Q=>0.2, F=>0.7)
    d    = UnivariateFinite(dict)
    @test classes(d) == [a, f, q, s]
    @test classes(d) == classes(s)
    @test levels(d) == levels(s)
    @test support(d) == [f, q, s]
    @test CategoricalDistributions.sample_scitype(d) == Multiclass{4}
    # levels!(v, reverse(levels(v)))
    # @test classes(d) == [s, q, f, a]
    # @test support(d) == [s, q, f]

    @test pdf(d, S) ≈ 0.1
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, Q) ≈ 0.2
    @test pdf(d, 'q') ≈ 0.2
    @test pdf(d, F) ≈ 0.7
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, A) ≈ 0.0
    @test logpdf(d, S) ≈ log(0.1)
    @test logpdf(d, 's') ≈ log(0.1)
    @test logpdf(d, Q) ≈ log(0.2)
    @test logpdf(d, 'q') ≈ log(0.2)
    @test logpdf(d, F) ≈ log(0.7)
    @test logpdf(d, 'f') ≈ log(0.7)
    @test isinf(logpdf(d, A))
    @test mode(d) == F

    @test UnivariateFinite(support(d), [0.7, 0.2, 0.1]) ≈ d

    N = 50
    rng = StableRNG(661)
    samples = [rand(rng,d) for i in 1:50];
    rng = StableRNG(661)
    @test samples == [rand(rng, d) for i in 1:N]

    N = 10000
    samples = rand(rng, d, N);
    @test Set(samples) == Set(support(d))
    freq = Distributions.countmap(samples)
    @test isapprox(freq[F]/N, 0.7, atol=0.05)
    @test isapprox(freq[S]/N, 0.1, atol=0.05)
    @test isapprox(freq[Q]/N, 0.2, atol=0.05)

    # no support specified:
    @test_logs (:warn, r"No ") UnivariateFinite([0.7, 0.3])
    d = UnivariateFinite([0.7, 0.3], pool=missing)
    @test pdf(d, "class_1") == 0.7
end

@testset "broadcasting pdf over single UnivariateFinite object" begin
    d = UnivariateFinite(["a", "b"], [0.1, 0.9], pool=missing);
    @test pdf.(d, ["a", "b"]) == [0.1, 0.9]
end

@testset "constructor arguments not categorical values" begin
    @test_throws ArgumentError UnivariateFinite(Dict('f'=>0.7, 'q'=>0.2))
    @test_throws ArgumentError UnivariateFinite(Dict('f'=>0.7, 'q'=>0.2),
                                                pool=missing)
    @test_logs((:warn, r"Ignoring"),
               UnivariateFinite(Dict('f'=>0.7, 'q'=>0.3),
                                pool=f, ordered=true))

    d = UnivariateFinite(Dict('f'=>0.7, 'q'=>0.2, 's'=>0.1), pool=v)
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, 'q') ≈ 0.2
    @test_logs((:warn, r"No "),
               UnivariateFinite(['f', 'q', 's'],  [0.7, 0.2, 0.1]))
    d = UnivariateFinite(['f', 'q', 's'],  [0.7, 0.2, 0.1], pool=missing)
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, 'q') ≈ 0.2

    d = UnivariateFinite(Dict('f'=>0.7, 'q'=>0.2, 's'=>0.1), pool=f)
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, 'q') ≈ 0.2

    d = UnivariateFinite(['f', 'q', 's'],  [0.7, 0.2, 0.1], pool=a)
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, 'q') ≈ 0.2

    d = UnivariateFinite(Dict('f'=>0.7, 'q'=>0.2, 's'=>0.1), pool=v)
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, 'q') ≈ 0.2

    d = UnivariateFinite(['f', 'q', 's'],  [0.7, 0.2, 0.1], pool=v)
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, 'q') ≈ 0.2
end

@testset "Univariate mode" begin
    v = categorical(1:101)
    p = rand(rng,101)
    s = sum(p) - p[42]
    p[42] = 0.5
    p = p/2/s
    p[42] = 0.5
    d = UnivariateFinite(v, p)
    @test mode(d) == 42

    # `mode` of `Univariate` objects containing `NaN` in probs.
    unf = UnivariateFinite([0.1, 0.2, NaN, 0.1, NaN], pool=missing)
    @test_throws ERR_NAN_FOUND mode(unf)
end

@testset "Univariate modes, bimodal" begin
    v = categorical(1:101)
    p = rand(rng,101)
    p[24] = 2*maximum(p)
    p[42] = p[24]
    p = p/sum(p)
    d = UnivariateFinite(v, p)
    @test modes(d) == [24, 42]

    # `mode` of `Univariate` objects containing `NaN` in probs.
    unf = UnivariateFinite([0.1, 0.2, NaN, 0.1, NaN], pool=missing)
    @test_throws ERR_NAN_FOUND modes(unf)
end

@testset "UnivariateFinite methods" begin
    y = categorical(["yes", "no", "yes", "yes", "maybe"])
    yes = y[1]
    no = y[2]
    maybe = y[end]
    prob_given_class = Dict(yes=>0.7, no=>0.3)
    d = UnivariateFinite(prob_given_class)
    @test pdf(d, yes) ≈ 0.7
    @test pdf(d, no) ≈ 0.3
    @test pdf(d, maybe) ≈ 0

    v = categorical(collect("abcd"))
    d = UnivariateFinite(v, [0.2, 0.3, 0.1, 0.4])
    sample = rand(rng,d, 10^4)
    freq_given_class = Distributions.countmap(sample)
    pairs = collect(freq_given_class)
    sort!(pairs, by=pair->pair[2], alg=QuickSort)
    sorted_classes = first.(pairs)
    @test sorted_classes == ['c', 'a', 'b', 'd']

    junk = categorical(['j',])
    j = junk[1]
    v = categorical(['a', 'b', 'a', 'b', 'c', 'b', 'a', 'a', 'f'])
    a = v[1]
    f = v[end]
    # remove f from sample:
    v = v[1 : end - 1]
    d = Distributions.fit(UnivariateFinite, v)
    @test pdf(d, a) ≈ 0.5
    @test pdf(d, 'a') ≈ 0.5
    @test pdf(d, 'b') ≈ 0.375
    @test pdf(d, 'c') ≈ 0.125
    @test pdf(d, 'f') == 0
    @test pdf(d, f) == 0
    @test_throws DomainError pdf(d, 'j')

    d2 = Distributions.fit(UnivariateFinite, v, nothing)
    @test d2 ≈ d

    # with weights:
    w = [2, 3, 2, 3, 5, 3, 2, 2]
    d = Distributions.fit(UnivariateFinite, v, w)
    @test pdf(d, a) ≈ 8/22
    @test pdf(d, 'a') ≈ 8/22
    @test pdf(d, 'b') ≈ 9/22
    @test pdf(d, 'c') ≈ 5/22
    @test pdf(d, 'f') == 0
    @test pdf(d, f) == 0
    @test_throws DomainError pdf(d, 'j')

    # regression test
    # https://github.com/alan-turing-institute/MLJBase.jl/pull/432/files#r502299301
    d = UnivariateFinite(classes(categorical(UInt32[0, 1])), [0.4, 0.6])
    @test pdf(d, UInt32(1)) == 0.6
end

@testset "approx for UnivariateFinite" begin
    y = categorical(["yes", "no", "maybe"])
    yes   = y[1]
    no    = y[2]
    maybe = y[3]
    @test(UnivariateFinite([yes, no, maybe], [0.1, 0.2, 0.7]) ≈
          UnivariateFinite([maybe, yes, no], [0.7, 0.1, 0.2]))
    @test(!(UnivariateFinite([yes, no, maybe], [0.1, 0.2, 0.7]) ≈
          UnivariateFinite([maybe, yes, no], [0.7, 0.2, 0.1])))
end

@testset "params" begin
    d = UnivariateFinite(["x", "y"], [0.3, 0.7], pool=missing)
    ps = Distributions.params(d)
    @test ps.levels == ["x", "y"]
    @test ps.probs == ("x" => 0.3, "y" => 0.7)
end

@testset "equality/approx" begin
    ϵ = 0.0000000000000001
    η = 0.01
    d = UnivariateFinite(["x", "y"], [0.3, 0.7], pool=missing)
    d_close = UnivariateFinite(["x", "y"], [0.3 + ϵ, 0.7 - ϵ], pool=missing)
    d_far = UnivariateFinite(["x", "y"], [0.3 + η, 0.7], pool=missing)
    @test d ≈ d_close
    @test !(d ≈ d_far)
    # v = [d, d]
    # v_close = [d_close, d_close]
    # @test v ≈ v_close
end

function displays_okay(v) # `v` a "probability" vector
    d = UnivariateFinite(v, pool=missing);
    str = sprint(show, "text/plain", d)
    contains(str, "UnivariateFinite{Multiclass{$(length(v))}}")
end

@testset "display" begin
    @test displays_okay([0.3, 0.7])
    @test displays_okay([5 + 3im, 4 - 7im])
    using UnicodePlots
    @test displays_okay([0.3, 0.7])
    @test displays_okay([5 + 3im, 4 - 7im])
end

@testset "rand signatures" begin
    dict = Dict(s=>0.1, q=>0.2, f=>0.7)
    d    = UnivariateFinite(dict)

    sampler = Random.Sampler(default_rng(), d, Val(1))
    @test sampler isa Random.SamplerTrivial
    sampler = Random.Sampler(default_rng(), d, Val(Inf))
    @test sampler isa Random.SamplerSimple

    # sampling one at a time, or all at once is the same:
    rng0 = StableRNG(123)
    samples = [rand(rng0, d) for i in 1:30]
    @test samples == [rand(StableRNG(123), d, 30)...]

    Random.seed!(123)
    samples = [rand(default_rng(), d) for i in 1:30]
    Random.seed!(123)
    @test [rand(d) for i in 1:30] == samples

    Random.seed!(123)
    samples = rand(Random.default_rng(), d, 3, 5)
    Random.seed!(123)
    @test samples == rand(d, 3, 5)
end

end # module

true
