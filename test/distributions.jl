using Test
using AlfvenDetectors
using Random
using Pkg

@testset "distributions" begin
	X = AlfvenDetectors.binormal(Float32, 10)
	@test size(X) == (10,)
	@test eltype(X) == Float32

	X = AlfvenDetectors.binormal(Float32, 4,5)
	@test size(X) == (4,5)
	@test eltype(X) == Float32
	 
	X = AlfvenDetectors.binormal(Float64, 4,5)
	@test eltype(X) == Float64

	X = AlfvenDetectors.binormal(4)
	@test size(X) == (4,)
	@test eltype(X) == Float32
	
	X = AlfvenDetectors.binormal(4,3)
	@test size(X) == (4,3)
	@test eltype(X) == Float32

	if "CuArrays" in keys(Pkg.installed())
		using CuArrays
		X = AlfvenDetectors.binormal_gpu(Float32,4,10)
		@test size(X) == (4,10)
		@test typeof(X) <: CuArray
		@test eltype(X) == Float32
	end

	
end