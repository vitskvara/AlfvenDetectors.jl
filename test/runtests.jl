using AlfvenDetectors
using Test
using Random
using Pkg

@testset "AlfvenDetectors" begin

@testset "utilities" begin
	@info "Testing utilities"
	include("data.jl")
	include("samplers.jl")
	include("flux_utils.jl")
	include("model_utils.jl")
end

@testset "Models" begin
	@info "Testing models"
	include("ae.jl")
	include("vae.jl")
	include("tsvae.jl")
end

if "CuArrays" in keys(Pkg.installed())
	@testset "GPU support" begin
		@info "Testing GPU support"
		include("gpu.jl")
	end
end

@testset "Experiments" begin
	@info "Testing experiments"
	include("experiments.jl")
end

end