using Test
using AlfvenDetectors
using Flux

xdim = 3
ldim = 1
N = 10
model = Flux.Chain(Flux.Dense(xdim, ldim), Flux.Dense(ldim, xdim))


@testset "flux utils" begin 
	m32 = AlfvenDetectors.adapt(Float32, model)
	@test typeof(m32.layers[1].W.data[1]) == Float32
	m64 = AlfvenDetectors.adapt(Float64, model)
	@test typeof(m64.layers[1].W.data[1]) == Float64
	m32 = AlfvenDetectors.adapt(Float32, model)
	@test typeof(m32.layers[1].W.data[1]) == Float32

	mf = AlfvenDetectors.freeze(model)
	@test length(collect(params(mf))) == 0
	# also test that the layer is really frozen after training?
end