using AlfvenDetectors
using Test
using Random
using Pkg

Random.seed!(12345)

@testset "AlfvenDetectors" begin

@info "Testing utilities"
include("samplers.jl")

@info "Testing models"
include("ae.jl")

if "CuArrays" in keys(Pkg.installed())
	@info "Testing GPU support"
	include("gpu.jl")
end

end