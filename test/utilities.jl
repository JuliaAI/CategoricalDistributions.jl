@testset "classes" begin
    v = categorical(collect("asqfasqffqsaaaa"), ordered=true)
    @test classes(v[1]) == levels(v)
    @test classes(v) == levels(v)
    levels!(v, reverse(levels(v)))
    @test classes(v[1]) == levels(v)
    @test classes(v) == levels(v)
end

@testset "int, classes, decoder" begin
    N = 10
    mix = shuffle(rng, 0:N - 1)

    Xraw = broadcast(x->mod(x,N), rand(rng, Int, 2N, 3N))
    Yraw = string.(Xraw)

    # to turn a categ matrix into a ordinary array with categorical
    # elements. Needed because broacasting the identity gives a
    # categorical array in CategoricalArrays >0.5.2
    function matrix_(X)
        ret = Array{Any}(undef, size(X))
        for i in eachindex(X)
            ret[i] = X[i]
        end
        return ret
    end

    X = categorical(Xraw)
    x = X[1]
    Y = categorical(Yraw)
    y = Y[1]
    V = matrix_(X)
    W = matrix_(Y)

    # broadcasted encoding:
    @test int(X) == int(V)
    @test int(Y) == int(W)

    # encoding is right-inverse to decoding:
    d = decoder(x)
    @test d(int(V)) == V # ie have the same elements
    e = decoder(y)
    @test e(int(W)) == W

    @test int(classes(y)) == 1:length(classes(x))

    # int is based on ordering not index
    v = categorical(['a', 'b', 'c'], ordered=true)
    @test int(v) == 1:3
    levels!(v, ['c', 'a', 'b'])
    @test int(v) == [2, 3, 1]

    # Errors
    @test_throws DomainError int("g")
end

true
