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

	m = AlfvenDetectors.layerbuilder([5,4,3,2], fill(Flux.Dense, 3), [Flux.relu, Flux.relu, Flux.relu])
	@test length(m.layers) == 3
	x = randn(5,10)
	@test size(m.layers[1](x)) == (4,10)
	@test size(m.layers[2](m.layers[1](x))) == (3,10)
	@test size(m(x)) == (2,10)
	@test typeof(m.layers[1].σ) == typeof(relu)
	
	m = AlfvenDetectors.layerbuilder(5,4,2,4, fill(Flux.Dense, 4), fill(Flux.relu, 4))
	@test length(m.layers) == 4
	x = randn(5,10)
	@test size(m.layers[1](x)) == (4,10)
	@test size(m.layers[2](m.layers[1](x))) == (4,10)
	@test size(m.layers[3](m.layers[2](m.layers[1](x)))) == (4,10)
	@test size(m(x)) == (2,10)
	@test typeof(m.layers[1].σ) == typeof(relu)

	m = AlfvenDetectors.layerbuilder([5,4,3,2], "relu", "linear", "Dense")
	@test length(m.layers) == 3
	x = randn(5,10)
	@test size(m.layers[1](x)) == (4,10)
	@test size(m.layers[2](m.layers[1](x))) == (3,10)
	@test size(m(x)) == (2,10)
	@test typeof(m.layers[1].σ) == typeof(relu)
	@test typeof(m.layers[3].σ) == typeof(identity)

	m = AlfvenDetectors.aelayerbuilder([5,4,4,2], relu, Dense)	
	@test length(m.layers) == 3
	x = randn(5,10)
	@test size(m.layers[1](x)) == (4,10)
	@test size(m.layers[2](m.layers[1](x))) == (4,10)
	@test size(m(x)) == (2,10)
	@test typeof(m.layers[1].σ) == typeof(relu)
	@test typeof(m.layers[3].σ) == typeof(identity)

	X = AlfvenDetectors.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	opt = ADAM(0.01)
	loss(x) = Flux.mse(model(x), x)
	cb(m,d,l,o) = nothing
	data = fill(X,100)
	l = Flux.Tracker.data(loss(X))
	frozen_params = map(x->copy(Flux.Tracker.data(x)), collect(params(model)))
	AlfvenDetectors.train!(model, data, loss, opt, cb)
	_l = Flux.Tracker.data(loss(X))
	@test _l < l
	# were the layers realy trained?
	for (fp, p) in zip(frozen_params, collect(params(model)))
		@test fp!=p
	end
end