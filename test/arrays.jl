module TestUnivariateFiniteArray

using Test
using CategoricalDistributions
using CategoricalArrays
import Distributions:pdf, logpdf, support
import Distributions
using StableRNGs
import Random
using Missings
using ScientificTypes

import CategoricalDistributions: classes, ERR_NAN_FOUND
import CategoricalArrays.unwrap

rng = StableRNG(111)
n   = 10
c   = 3

@testset "corner case 1.3 constructor" begin
    # not tested earlier as implementation depends on *array*
    # constructor:
    d = UnivariateFinite(["ying", "yang"], 0.3, augment=true,
                         ordered=true, pool=missing)
    @test pdf(d, "yang") == 0.3
    @test classes(d)[1] == "ying"
    d = UnivariateFinite(classes(d), 0.3, augment=true)
    @test pdf(d, "yang") == 0.3
end

@testset "constructing UnivariateFiniteArray objects" begin

    probs  = rand(rng, n)
    supp = ["class1", "class2"]

    # @test_throws DomainError UnivariateFinite(supp, probs, pool=missing)
    u = UnivariateFinite(supp, probs, pool=missing, augment=true)
    @test length(u) == n
    @test size(u) == (n,)
    @test pdf.(u, "class2") ≈ probs

    # autosupport:
    # @test_throws DomainError UnivariateFinite(probs, pool=missing)
    u = UnivariateFinite(probs, pool=missing, augment=true)
    @test length(u) == n
    @test size(u) == (n,)
    @test pdf.(u, "class_2") ≈ probs
    probs = probs ./ sum(probs)
    u = UnivariateFinite(probs, pool=missing)
    @test u isa UnivariateFinite
    @test pdf(u, "class_1") == probs[1]
    probs = rand(rng, 10, 2)
    probs = probs ./ sum(probs, dims=2)
    u = UnivariateFinite(probs, pool=missing)
    @test length(u) == 10
    u.scitype == Multiclass{2}
    pdf.(u, "class_1") == probs[:, 1]
    u = UnivariateFinite(probs, pool=missing, augment=true)
    @test length(u) == 10
    u.scitype == Multiclass{3}
    pdf.(u, "class_2") == probs[:, 1]

    probs = [-1,0,1]
    @test_throws(DomainError,
                 UnivariateFinite(probs, pool=missing, augment=true))

    v = categorical(1:3)
    @test_logs((:warn, r"Ignoring"),
               UnivariateFinite(v[1:2], rand(rng, 3),
                                augment=true, pool=missing))
    @test_logs((:warn, r"Ignoring"),
               UnivariateFinite(v[1:2], rand(rng, 3),
                                augment=true, ordered=true))

end

@testset "get and set" begin
    # binary
    s = rand(rng, n)
    u = UnivariateFinite(["yes", "no"], s, augment=true, pool=missing)

    @test u[1] isa UnivariateFinite
    v = u[3:4]
    @test v isa UnivariateFiniteArray
    @test v[1] ≈ u[3]

    # set:
    u[1] = UnivariateFinite(["yes", "no"], [0.1, 0.9], pool=u[1])
    @test pdf(u[1], "yes") == 0.1
    @test pdf(u[1], "no") == 0.9

    # multiclass
    P   = rand(rng, n, c)
    P ./= sum(P, dims=2)
    u   = UnivariateFinite(P, pool=missing)

    @test u[1] isa UnivariateFinite
    v = u[3:4]
    @test v isa UnivariateFiniteArray
    @test v[1] ≈ u[3]

    # set:
    s = Distributions.support(u)
    scalar = UnivariateFinite(s, P[end,:])
    u[1] = scalar
    @test u[1] ≈ u[end]
    @test pdf.(u, s[1])[2:end] == P[2:end,1]
end

n = 10
P = rand(rng, n);
all_classes = categorical(["no", "yes"], ordered=true)
u = UnivariateFinite(all_classes, P, augment=true) #uni_fin_arr

# next is not specific to `UnivariateFiniteArray` but is for any
# abstract array with eltype `UnivariateFinite`:
@testset "piratical pdf and logpdf" begin
    # test pdf(uni_fin_arr, labels) and
    # logpdf(uni_fin_arr, labels)
    @test pdf(u, ["yes", "no"]) == hcat(P, 1 .- P)
    @test isequal(logpdf(u, ["yes", "no"]), log.(hcat(P, 1 .- P)))
    @test pdf(u, reverse(all_classes)) == hcat(P, 1 .- P)
    @test isequal(logpdf(u, reverse(all_classes)), log.(hcat(P, 1 .- P)))

    # test pdf(::Array{UnivariateFinite, 1}, labels) and
    # logpdf(::Array{UnivariateFinite, labels)
    @test pdf([u...], ["yes", "no"]) == hcat(P, 1 .- P)
    @test isequal(logpdf([u...], ["yes", "no"]), log.(hcat(P, 1 .- P)))
    @test pdf([u...], all_classes) == hcat(1 .- P, P)
    @test isequal(logpdf([u...], all_classes), log.(hcat(1 .- P, P)))
end

@testset "broadcasting: pdf.(uni_fin_arr, scalar) and logpdf.(uni_fin_arr, scalar) " begin
    v = pdf.(u, missing)
    @test eltype(v) == Union{Missing,Float64}
    @test all(ismissing, v)
    v = logpdf.(u, missing)
    @test eltype(v) == Union{Missing,Float64}
    @test all(ismissing, v)

    @test pdf.(u,"yes") == P
    @test isequal(logpdf.(u,"yes"), log.(P))
    @test pdf.(u,all_classes[2]) == P
    @test isequal(logpdf.(u,all_classes[2]), log.(P))

    # check unseen probablities are a zero *array*:
    v = categorical(1:4)
    probs = rand(rng, 3)
    u2 = UnivariateFinite(v[1:2], probs, augment=true)
    @test pdf.(u2, v[3]) == zeros(3)
    @test isequal(logpdf.(u2, v[3]), log.(zeros(3)))

    ## Check that the appropriate errors are thrown
    @test_throws DomainError pdf.(u,"strange_level")
end

_skip(v) = collect(skipmissing(v))

@testset "broadcasting: pdf.(uni_fin_arr, array_same_shape) and logpdf.(uni_fin_arr, array_same_shape)" begin
    v0 = categorical(rand(rng, string.(classes(u)), n))
    vm = vcat(v0[1:end-1], [missing, ])
    for v in [v0, vm]
        @test _skip(broadcast(pdf, u, v)) ==
            _skip([pdf(u[i], v[i]) for i in 1:length(u)])
        @test _skip(broadcast(logpdf, u, v)) ==
                      _skip([logpdf(u[i], v[i]) for i in 1:length(u)])
        @test _skip(broadcast(pdf, u, unwrap.(v))) ==
            _skip([pdf(u[i], v[i]) for i in 1:length(u)])
        @test _skip(broadcast(logpdf, u, unwrap.(v))) ==
                      _skip([logpdf(u[i], v[i]) for i in 1:length(u)])
    end

    ## Check that the appropriate errors are thrown
    v1 = categorical([v0[1:end-1]...;"strange_level"])
    v2 = [v0...;rand(rng, v0)] #length(u) !== length(v2)
    v3 = categorical([vm[end:-1:begin+1]...;"strange_level"])
    @test_throws DimensionMismatch broadcast(pdf, u, v2)
    @test_throws DomainError broadcast(pdf, u, v1)
    @test_throws DomainError broadcast(pdf, u, v3)

end

@testset "broadcasting: check indexing in `getter((cv_ref, i))` see PR#375 from MLJBase" begin
    c  = categorical([0,1,1])
    d = UnivariateFinite(c[1:1], [1 1 1]')
    v = categorical([0,1,1,1])
    @test broadcast(pdf, d, v[2:end]) == [0,0,0]
end

@testset "_getindex" begin
   @test CategoricalDistributions._getindex(collect(1:4), 2) == 2
   @test CategoricalDistributions._getindex(0, 2) === 0
end

@testset "broadcasting mode" begin
    # binary
    rng = StableRNG(668)
    probs = rand(rng, n)
    u = UnivariateFinite(probs, augment = true, pool=missing)
    supp = Distributions.support(u)
    _modes = mode.(u)
    @test _modes isa CategoricalArray
    expected = [ifelse(p > 0.5, supp[2], supp[1]) for p in probs]
    @test all(_modes .== expected)

    # multiclass
    rng = StableRNG(554)
    P   = rand(rng, n, c)
    P ./= sum(P, dims=2)
    u   = UnivariateFinite(P, pool=missing)
    expected = mode.([u...])
    @test all(mode.(u) .== expected)

    # `mode` broadcasting of `Univariate` objects containing `NaN` in probs.
    unf_arr = UnivariateFinite(
        [
            0.1 0.2 NaN 0.1 NaN;
            0.2 0.1 0.1 0.4 0.2;
            0.3 NaN 0.2 NaN 0.3
        ],
        pool=missing
    )
    @test_throws ERR_NAN_FOUND mode.(unf_arr)
end

@testset "broadcasting modes" begin
    # binary
    rng = StableRNG(668)
    probs = rand(rng, n)
    u = UnivariateFinite(probs, augment = true, pool=missing)
    supp = Distributions.support(u)
    _modes = modes.(u)
    @test _modes isa Vector{<:CategoricalArray}
    expected = [ifelse(p > 0.5, [supp[2]], [supp[1]]) for p in probs]
    @test all(_modes .== expected)

    # multiclass, bimodal
    rng = StableRNG(554)
    P   = rand(rng, n, c)
    M, M_idx = findmax(P, dims=2)
    M_idx = getindex.(M_idx, 2)
    for i in axes(P,1)
        m = M[i]
        j = M_idx[i]
        while j == M_idx[i]
            j = rand(axes(P,2))
        end
        P[i,j] = m
    end
    P ./= sum(P, dims=2)
    u   = UnivariateFinite(P, pool=missing)
    expected = modes.([u...])
    @test all(modes.(u) .== expected)

    # `mode` broadcasting of `Univariate` objects containing `NaN` in probs.
    unf_arr = UnivariateFinite(
        [
            0.1 0.2 NaN 0.1 NaN;
            0.2 0.1 0.1 0.4 0.2;
            0.3 NaN 0.2 NaN 0.3
        ],
        pool=missing
    )
    @test_throws ERR_NAN_FOUND modes.(unf_arr)
end

@testset "cat for UnivariateFiniteArray" begin

    # ordered:
    v = categorical(["no", "yes", "maybe", "unseen"])
    u1 = UnivariateFinite([v[1], v[2]], rand(rng, 5), augment=true)
    u2 = UnivariateFinite([v[3], v[2]], rand(rng, 6), augment=true)
    us = (u1, u2)
    u = cat(us..., dims=1)
    @test length(u) == length(u1) + length(u2)
    @test classes(u) == classes(u1)
    supp = Distributions.support(u)
    @test Set(supp) == Set(["no", "yes", "maybe"])
    s1 = Distributions.support(u1)
    s2 = Distributions.support(u2)
    @test pdf(u, s1)[1:length(u1),:] == pdf(u1, s1)
    @test pdf(u, s2)[length(u1)+1:length(u1)+length(u2),:] ==
        pdf(u2, s2)
    @test pdf.(u, v[1])[length(u1)+1:length(u1)+length(u2)] ==
        zeros(length(u2))
    @test pdf.(u, v[3])[1:length(u1)] == zeros(length(u1))
    @test pdf.(u, v[4]) == zeros(length(u))

    # unordered:
    v = categorical(["no", "yes", "maybe", "unseen"], ordered=true)
    u1 = UnivariateFinite([v[1], v[2]], rand(rng, 5), augment=true)
    u2 = UnivariateFinite([v[3], v[2]], rand(rng, 6), augment=true)
    us = (u1, u2)
    u = cat(us..., dims=1)
    @test length(u) == length(u1) + length(u2)
    @test classes(u) == classes(u1)
    supp = Distributions.support(u)
    @test Set(supp) == Set(["no", "yes", "maybe"])
    s1 = Distributions.support(u1)
    s2 = Distributions.support(u2)
    @test pdf(u, s1)[1:length(u1),:] == pdf(u1, s1)
    @test pdf(u, s2)[length(u1)+1:length(u1)+length(u2),:] ==
        pdf(u2, s2)
    @test pdf.(u, v[1])[length(u1)+1:length(u1)+length(u2)] ==
        zeros(length(u2))
    @test pdf.(u, v[3])[1:length(u1)] == zeros(length(u1))
    @test pdf.(u, v[4]) == zeros(length(u))

    @test pdf([vcat(u1, u2)...], supp) ≈
        pdf(vcat([u1...], [u2...]), supp)
    h = hcat(u1, u1)
    h_nowrap = hcat([u1...], [u1...])
    @test size(h) == size(h_nowrap)
    # TODO: why does identity.(h) not work?
    @test h[3,2] ≈ h_nowrap[3,2]

    # errors
    v1 = categorical(1:2, ordered=true)
    v2 = categorical(v1, ordered=true)
    levels!(v2, levels(v2) |> reverse )
    probs = rand(rng, 3)
    u1 = UnivariateFinite(v1, probs, augment=true)
    u2 = UnivariateFinite(v2, probs, augment=true)
    @test_throws DomainError vcat(u1, u2)

    v1 = categorical(1:2)
    v2 = categorical(2:3)
    u1 = UnivariateFinite(v1, probs, augment=true)
    u2 = UnivariateFinite(v2, probs, augment=true)
    @test_throws DomainError vcat(u1, u2)

end

@testset "classes" begin
    v = categorical(collect("abca"), ordered=true)
    u1 = UnivariateFinite([v[1], v[2]], rand(rng, 5), augment=true)
    @test classes(u1) == collect("abc")
    u2 = [missing, u1...]
    @test classes(u2) == collect("abc")
    @test_throws(CategoricalDistributions.ERR_EMPTY_UNIVARIATE_FINITE,
                 classes(u2[1:1]))
end

function ≅(x::T, y::T) where {T<:UnivariateFinite}
    return x.decoder == y.decoder &&
           x.prob_given_ref == y.prob_given_ref &&
           x.scitype == y.scitype
end

function ≅(x::AbstractArray, y::AbstractArray)
    return all((≅).(x, y))
end

@testset "indexing of UnivariateFininiteArray (see issue #43)" begin
    u = UnivariateFinite(['x', 'z'], rand(2, 3, 2), pool=missing, ordered=true)
    v = u[1:2]
    @test v isa UnivariateFiniteArray
    @test v ≅ u[1:2, 1] ≅ u[[1,2], 1]
    w = u[2]
    @test w isa UnivariateFinite
    @test w ≅ v[2]
end

end

function ≅(x::T, y::T) where {T<:UnivariateFinite}
    return x.decoder == y.decoder &&
           x.prob_given_ref == y.prob_given_ref &&
           x.scitype == y.scitype
end

@testset "CartesianIndex" begin
    v = categorical(["a", "b"], ordered=true)
    m = UnivariateFinite(v, rand(rng, 5, 2), augment=true)
    @test m[1, 1] ≅ m[CartesianIndex(1, 1)] ≅ m[CartesianIndex(1, 1, 1)]
    @test_throws BoundsError m[CartesianIndex(1)]
    @test all(zip(Matrix(m), copy(m), m)) do (x, y, z)
        return x ≅ y ≅ z
    end
    @test Matrix(m) isa Matrix
    # TODO: probably it would be better for copy to keep it
    #       UnivariateFiniteArray but it would be breaking
    @test copy(m) isa Matrix
    @test similar(m) isa Matrix
end

@testset "broadcasted pdf" begin
    v = categorical(["a", "b"], ordered=true)
    v2 = categorical(["a", "b"], ordered=true, levels=["b", "a"])
    x = UnivariateFinite(v, rand(rng, 5), augment=true)
    @test pdf.(x, v[1]) == pdf.(x, v2[1]) == pdf.(x, "a")
    @test pdf.(x, v[2]) == pdf.(x, v2[2]) == pdf.(x, "b")

    x = UnivariateFinite(v, rand(rng, 5, 2), augment=true)
    @test size(pdf.(x, missing)) == (5, 2)

    v3 = categorical(["a" "b"], ordered=true)
    v4 = categorical(["a" "b"], ordered=true, levels=["b", "a"])
    # note that v5 and v6 have the same shape and contents as v3 and v4
    # just they are Matrix{Any} not CategoricalMatrix
    v5 = Any[v3[1] v3[2]]
    v6 = Any[v4[1] v4[2]]
    x = UnivariateFinite(v, hcat([0.1, 0.2]), augment=true)

    # these tests show that now we have corrected refpools
    # but still there is an inconsistency in behavior
    @test pdf.(x, v) == hcat([0.9, 0.2])
    @test pdf.(x, v2) == hcat([0.9, 0.2])
    @test pdf.(x, v3) == hcat([0.9, 0.2])
    @test pdf.(x, v4) == hcat([0.9, 0.2])
    @test pdf.(x, v5) == [0.9 0.1; 0.8 0.2]
    @test pdf.(x, v6) == [0.9 0.1; 0.8 0.2]
end

@testset "pdf with various types" begin
    v = categorical(["a", "b"], ordered=true)
    a = view("a", 1:1) # quite common case when splitting strings
    b = view("b", 1:1)
    x = UnivariateFinite(v, [0.1, 0.2, 0.3], augment=true)
    @test pdf.(x, a) == pdf.(x, "a") == pdf.(x, v[1])
    @test logpdf.(x, a) == logpdf.(x, "a") == logpdf.(x, v[1])
    @test pdf(x, [a, b]) == pdf(x, ["a", "b"]) == pdf(x, v)
    @test logpdf(x, [a, b]) == logpdf(x, ["a", "b"]) == logpdf(x, v)

    x = UnivariateFinite(v, 0.1, augment=true)
    @test pdf.(x, a) == pdf.(x, "a") == pdf.(x, v[1]) == 0.9
    @test logpdf.(x, a) == logpdf.(x, "a") == logpdf.(x, v[1]) == log(0.9)
    @test pdf(x, a) == pdf(x, "a") == pdf(x, v[1]) == 0.9
    @test logpdf(x, a) == logpdf(x, "a") == logpdf(x, v[1]) == log(0.9)
end
true
