using LabeledArrays
using Test

@testset "LabeledArrays.jl" begin
    # Write your own tests here.
    @testset "NBijection" begin
        x = NBijection('a':'z')
        @test x[13] == 'm'
        @test eltype(x) == Char
        @test collect(x) == ['a':'z'...,'ö']
        @test get!(x,'m') == 13
        @test get!(x,'ö') == 27
        @test keytype(x) == Char
        @test keys(x) == vcat('a':'z',['ö'])
        @test valtype(x) == Int
        @test values(x) == 1:27
        @test indexin('m', x) == indexin('m', collect(x))
    end
end
