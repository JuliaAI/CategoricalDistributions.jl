@testset "scitype" begin
    d = UnivariateFinite([0.1, 0.2, 0.7], pool=missing)
    @test scitype(d)  == Density{Multiclass{3}}
    d2 = UnivariateFinite([0.1, 0.2, 0.7], pool=missing, ordered=true)
    @test scitype(d2)  == Density{OrderedFactor{3}}
    d = UnivariateFinite(rand(3), pool=missing)

    v = UnivariateFinite(rand(3), augment=true, pool=missing)
    @test scitype(v) == AbstractVector{Density{Multiclass{2}}}
    v2 = UnivariateFinite(rand(3), augment=true, pool=missing, ordered=true)
    @test scitype(v2) == AbstractVector{Density{OrderedFactor{2}}}

    @test scitype([d, d]) == AbstractVector{Density{Multiclass{3}}}
    @test scitype([d2, d2]) == AbstractVector{Density{OrderedFactor{3}}}
end

true
