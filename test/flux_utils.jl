using Test
using AlfvenDetectors
using Flux
using ValueHistories

xdim = 3
ldim = 1
N = 10
model = Flux.Chain(Flux.Dense(xdim, ldim), Flux.Dense(ldim, xdim))

paramchange(frozen_params, params) = 
	map(x-> x[1] != x[2], zip(frozen_params, params))

@testset "flux utils" begin 
	# adapt
	m32 = AlfvenDetectors.adapt(Float32, model)
	@test typeof(m32.layers[1].W.data[1]) == Float32
	m64 = AlfvenDetectors.adapt(Float64, model)
	@test typeof(m64.layers[1].W.data[1]) == Float64
	m32 = AlfvenDetectors.adapt(Float32, model)
	@test typeof(m32.layers[1].W.data[1]) == Float32

	# freeze
	mf = AlfvenDetectors.freeze(model)
	@test length(collect(params(mf))) == 0

	#iscuarray
	@test !AlfvenDetectors.iscuarray(randn(4,10))
	_x = model(randn(xdim,10))
	@test !AlfvenDetectors.iscuarray(_x)

	# layerbuilder
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

	#aelayerbuilder
	m = AlfvenDetectors.aelayerbuilder([5,4,4,2], relu, Dense)	
	@test length(m.layers) == 3
	x = randn(5,10)
	@test size(m.layers[1](x)) == (4,10)
	@test size(m.layers[2](m.layers[1](x))) == (4,10)
	@test size(m(x)) == (2,10)
	@test typeof(m.layers[1].σ) == typeof(relu)
	@test typeof(m.layers[3].σ) == typeof(identity)

	# update!&train!
	X = AlfvenDetectors.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
	opt = ADAM(0.01)
	loss(x) = Flux.mse(model(x), x)
	cb(m,d,l,o) = nothing
	data = fill(X,100)
	L = loss(X)
	l = Flux.Tracker.data(L)
	frozen_params = map(x->copy(Flux.Tracker.data(x)), collect(params(model)))
	#update!
	Flux.back!(L)
	AlfvenDetectors.update!(model,opt)
	@test all(paramchange(frozen_params, collect(params(model))))
	# train!
	AlfvenDetectors.train!(model, data, loss, opt, cb)
	_l = Flux.Tracker.data(loss(X))
	@test _l < l
	# were the layers realy trained?
	@test all(paramchange(frozen_params, collect(params(model))))

	# fast callback
	@test AlfvenDetectors.fast_callback(AlfvenDetectors.AE(4,3,2), 1, 2, 3) == nothing
	@test AlfvenDetectors.fast_callback(AlfvenDetectors.VAE(4,3,2), 1, 2, 3) == nothing
	@test AlfvenDetectors.fast_callback(AlfvenDetectors.TSVAE(4,3,(2,2)), 1, 2, 3) == nothing
	
	# basic callback
	hist = MVHistory()
	cb=AlfvenDetectors.basic_callback(hist,true,0.0001,100; train_length=10,epoch_size=5)
	@assert typeof(cb) == AlfvenDetectors.basic_callback

	# upscaling stuff
	# oneszeros
	x = AlfvenDetectors.oneszeros(2,3,2)
	@test x == [0.0; 0.0; 1.0; 1.0; 0.0; 0.0]
	@test size(x) == (6,)
	x = AlfvenDetectors.oneszeros(2,3,1)
	@test x == [1.0; 1.0; 0.0; 0.0; 0.0; 0.0]
	x = AlfvenDetectors.oneszeros(Float32,2,3,1)
	@test typeof(x[1]) == Float32
	# voneszeros
	@test AlfvenDetectors.voneszeros(2,3,2) == [0.0; 0.0; 1.0; 1.0; 0.0; 0.0]
	x = AlfvenDetectors.voneszeros(Float32,2,3,2)
	@test size(x) == (6,)
	@test typeof(x[1]) == Float32
	# honeszeros
	@test AlfvenDetectors.honeszeros(2,3,2) == [0.0 0.0 1.0 1.0 0.0 0.0]
	x = AlfvenDetectors.honeszeros(Float32,2,3,2)
	@test size(x) == (1,6)
	@test typeof(x[1]) == Float32
	# vscalemat
	X = AlfvenDetectors.vscalemat(2,2)
	@test X == [1.0 0.0; 1.0 0.0; 0.0 1.0; 0.0 1.0]
	X = AlfvenDetectors.vscalemat(Float32,4,3)
	@test typeof(X[1]) == Float32
	@test size(X) == (12,3)
	# vscalemat
	X = AlfvenDetectors.hscalemat(2,2)
	@test X == [1.0 1.0 0.0 0.0; 0.0 0.0 1.0 1.0]
	X = AlfvenDetectors.hscalemat(Float32,4,3)
	@test typeof(X[1]) == Float32
	@test size(X) == (3,12)
	# upscaling
	a = Tracker.collect(Flux.Tracker.TrackedReal.(Float32.([1.0 2.0; 3.0 4.0])))
	a = reshape(a,2,2,1,1)
	# 2D
	X = AlfvenDetectors.upscale(a[:,:,1,1],(3,2))
	@test size(X) == (6,4)
	@test typeof(X) <: Flux.TrackedArray
	@test X.data[3,1] == 1.0
	@test X.data[4,1] == 3.0
	@test X.data[3,4] == 2.0
	@test X.data[4,4] == 4.0
	# 3D
	X = AlfvenDetectors.upscale(a[:,:,:,1],(3,2))
	@test size(X) == (6,4,1)
	@test typeof(X) <: Flux.TrackedArray
	@test X.data[3,1,1] == 1.0
	@test X.data[4,1,1] == 3.0
	@test X.data[3,4,1] == 2.0
	@test X.data[4,4,1] == 4.0
	# 4D
	X = AlfvenDetectors.upscale(a,(3,2))
	@test size(X) == (6,4,1,1)
	@test typeof(X) <: Flux.TrackedArray
	@test X.data[3,1,1,1] == 1.0
	@test X.data[4,1,1,1] == 3.0
	@test X.data[3,4,1,1] == 2.0
	@test X.data[4,4,1,1] == 4.0
	# also test if propagation through the upscale layer works
	X = randn(Float32,24,24,1,1)
	global model = Flux.Chain(
	    # 24x24x2x1
	    Flux.Conv((3,3), 1=>4, pad=(1,1)),
	    # 24x24x4x1
	    x->Flux.maxpool(x,(8,6)),
	    # 3x4x4x1
	    x->AlfvenDetectors.upscale(x,(8,6)),
	    # 24x24x4x1
	    Flux.Conv((3,3), 4=>1, pad=(1,1))
	)
	frozen_params = map(x->copy(Flux.Tracker.data(x)), collect(params(model)))
	Y = model(X)
	@test size(Y) == size(X)
	loss(x) = Flux.mse(x,model(x))
	opt = Flux.ADAM()
	L = loss(X)
	Flux.back!(L)
	AlfvenDetectors.update!(model, opt)
	@test all(paramchange(frozen_params, collect(params(model))))

	# padding
	a = Tracker.collect(Flux.Tracker.TrackedReal.(Float32.([1.0 2.0; 3.0 4.0])))
	a = reshape(a,2,2,1,1)
	# 2D
	X = AlfvenDetectors.zeropad(a[:,:,1,1],[1,2,2,3])
	@test size(X) == (5,7)
	@test typeof(X)  <:Flux.TrackedArray
	# 3D
	X = AlfvenDetectors.zeropad(a[:,:,:,1],[1,2,2,3])
	@test size(X) == (5,7,1)
	@test typeof(X)  <:Flux.TrackedArray
	# 4D
	X = AlfvenDetectors.zeropad(a,[1,2,2,3])
	@test size(X) == (5,7,1,1)
	@test typeof(X)  <:Flux.TrackedArray
	# backprop
	X = randn(Float32,4,4,1,1)
	global model = Flux.Chain(
	    # 4x4x1x1
	    Flux.Conv((3,3), 1=>4, pad=(1,1)),
	    # 4x4x4x1
	    x->Flux.maxpool(x,(2,2)),
	    # 2x2x4x1
	    x->AlfvenDetectors.zeropad(x,(1,1,1,1)),
	    # 4x4x4x1
	    Flux.Conv((3,3), 4=>1, pad=(1,1))
	)
	frozen_params = map(x->copy(Flux.Tracker.data(x)), collect(params(model)))
	Y = model(X)
	@test size(Y) == size(X)
	loss(x) = Flux.mse(x,model(x))
	opt = Flux.ADAM()
	L = loss(X)
	Flux.back!(L)
	AlfvenDetectors.update!(model, opt)
	@test all(paramchange(frozen_params, collect(params(model))))
	
	# convmaxpool
	X = randn(12,6,2,5)
	layer = AlfvenDetectors.convmaxpool(3,2=>4,2)
	@test size(layer(X)) == (6,3,4,5)
	layer = AlfvenDetectors.convmaxpool(5,2=>8,(3,2))
	@test size(layer(X)) == (4,3,8,5)
	layer = AlfvenDetectors.convmaxpool(3,2=>8,2;stride=3)
	@test size(layer(X)) == (2,1,8,5)

	# convupscale
	X = randn(2,4,4,10)
	layer = AlfvenDetectors.convupscale(3,4=>2,2)
	@test size(layer(X)) == (4,8,2,10)
	layer = AlfvenDetectors.convupscale(5,4=>1,(4,3))
	@test size(layer(X)) == (8,12,1,10)
	layer = AlfvenDetectors.convupscale(3,4=>2,2;stride=2)
	@test size(layer(X)) == (2,4,2,10)
	
end