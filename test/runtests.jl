using MySRCore
using Test

@testset "MySRCore package identity" begin
    @test nameof(MySRCore) == :MySRCore
    @test isdefined(MySRCore, :Options)
    @test isdefined(MySRCore, :equation_search)
end
