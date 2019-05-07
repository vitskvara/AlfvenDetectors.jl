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

	# GMs
	means = [[-1f0,-1f0], [1f0,1f0]]
	sigmas = [0.1, 1.0]
	gm = AlfvenDetectors.GM(means, sigmas)
	@test gm.n == 2
	@test gm.w == [0.5, 0.5]

	means = [[-1f0,-1f0], [1f0,1f0]]
	sigmas = [0.1, 1.0]
	gm = AlfvenDetectors.GM(means, sigmas, [0.0, 1.0])
	@test gm.n == 2
	@test gm.w == [0.0, 1.0]
	
	@test size(AlfvenDetectors.sample(gm,100)) == (2,100)
	@test eltype(AlfvenDetectors.sample(gm,10)) == Float32
	@test size(AlfvenDetectors.sample(gm,Float64,100)) == (2,100)
	@test eltype(AlfvenDetectors.sample(gm,Float64,10)) == Float64
	# this is a deprecated caller for backward compatibility purposes
	@test size(AlfvenDetectors.sample(gm,Float64,5,100)) == (2,100)
	
	# cubecorners
	@test size(AlfvenDetectors.cubecorners(4)) == (4,4^2)
	cs = AlfvenDetectors.cubecorners_rand(24,4,seed=1)
	@test size(cs) == (24,4)
	cs2 = AlfvenDetectors.cubecorners_rand(24,4)
	cs3 = AlfvenDetectors.cubecorners_rand(24,4,seed=1)
	@test cs3 == cs != cs2 

	# cubeGM
	dim = 2
	ncomp = 3
	σ = [1.0,0.1,0.001]
	ws = [0.1,0.3,0.6]
	cgm = AlfvenDetectors.cubeGM(dim,ncomp,σ,ws)
	X = AlfvenDetectors.sample(cgm,10)
	@test size(X) == (dim,10)
	@test eltype(X) == Float32
	@test cgm.w == ws
	@test cgm.σ == σ
	
	cgm = AlfvenDetectors.cubeGM(dim,ncomp)
	@test length(cgm.μ) == 3
	@test length(cgm.σ) == 3 	
	@test cgm.σ == [0.1f0, 0.1f0, 0.1f0]
	@test cgm.w == fill(1/ncomp,ncomp) 	
end