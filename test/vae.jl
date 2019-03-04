using AlfvenDetectors
using Test
using ValueHistories
using Flux
using Random

xdim = 5
ldim = 1
N = 10

@testset "VAE" begin
    println("           variational autoencoder")

	x = AlfvenDetectors.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
    Random.seed!(12345)

	# this has unit variance on output
    model = AlfvenDetectors.VAE([xdim,2,2*ldim], [ldim,2,xdim])
	# for training check
	frozen_params = map(x->copy(Flux.Tracker.data(x)), collect(params(model)))
	_x = model(x)
	# test correct construction
	@test size(model.encoder.layers,1) == 2
	@test size(model.decoder.layers,1) == 2
	# test basic functionality
	@test size(model.encoder(x)) == (2*ldim, N)
	@test size(_x) == (xdim,N)
	# test output types
	@test typeof(_x) <: Flux.TrackedArray{AlfvenDetectors.Float,2}

	# loss functions
	kl = AlfvenDetectors.KL(model,x)
	@test typeof(kl) == Flux.Tracker.TrackedReal{AlfvenDetectors.Float}
	ll = AlfvenDetectors.loglikelihood(model,x)
	@test typeof(ll) == Flux.Tracker.TrackedReal{AlfvenDetectors.Float}
	llm = AlfvenDetectors.loglikelihood(model,x,10)
	@test typeof(llm) == Flux.Tracker.TrackedReal{AlfvenDetectors.Float}
	l = AlfvenDetectors.loss(model, x, 10, 0.01)
	@test typeof(l) == Flux.Tracker.TrackedReal{AlfvenDetectors.Float}
	ls = AlfvenDetectors.getlosses(model, x, 10, 0.01)
	@test sum(map(x->abs(x[1]-x[2]), zip(ls, (kl-llm, -llm, kl))))/3 < 2e-1
	# its never the same because of the middle stochastic layer
	# tracking
	hist = MVHistory()
	AlfvenDetectors.track!(model, hist, x, 10, 0.01)
	AlfvenDetectors.track!(model, hist, x, 10, 0.01)
	is, ls = get(hist, :loss)
	@test abs(ls[1] - l) < 2e-1
	@test abs(ls[1] - ls[2]) < 1e-1
	# training
	AlfvenDetectors.fit!(model, x, 5, 100, β =0.1, cbit=5, history = hist, verb = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	for (fp, p) in zip(frozen_params, collect(params(model)))
		@test fp!=p
	end
	# sample
	gx = AlfvenDetectors.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{AlfvenDetectors.Float,1}
	gx = AlfvenDetectors.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{AlfvenDetectors.Float,2}
	@test size(gx) == (xdim,5)

	###########################################################
    ### VAE with estimated diagonal of covariance on output ###
    ###########################################################
    model = AlfvenDetectors.VAE([xdim,2,2*ldim], [ldim,2,xdim*2], variant = :diag)
	_x = model(x)
	# test correct construction
	@test size(model.encoder.layers,1) == 2
	@test size(model.decoder.layers,1) == 2
	# test basic functionality
	@test size(model.encoder(x)) == (2*ldim, N)
	@test size(_x) == (2*xdim,N)
	# loss functions
	prels = AlfvenDetectors.getlosses(model, x, 10, 0.01)
	AlfvenDetectors.fit!(model, x, 5, 500, β =0.1, runtype = "fast")
	postls = AlfvenDetectors.getlosses(model, x, 10, 0.01)
	@test any(x->x[1]>x[2], zip(prels, postls))
	gx = AlfvenDetectors.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{AlfvenDetectors.Float,1}
	gx = AlfvenDetectors.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{AlfvenDetectors.Float,2}
	@test size(gx) == (xdim,5)

	##########################################
    ### VAE with scalar variance on output ###
    ##########################################
    model = AlfvenDetectors.VAE([xdim,2,2*ldim], [ldim,2,xdim + 1], variant = :scalar)
	_x = model(x)
	# test correct construction
	@test size(model.encoder.layers,1) == 2
	@test size(model.decoder.layers,1) == 2
	# test basic functionality
	@test size(model.encoder(x)) == (2*ldim, N)
	@test size(_x) == (1+xdim,N)
	# loss functions
	prels = AlfvenDetectors.getlosses(model, x, 10, 0.01)
	AlfvenDetectors.fit!(model, x, 5, 100, β =0.1, runtype = "fast")
	postls = AlfvenDetectors.getlosses(model, x, 10, 0.01)
	@test all(x->x[1]>x[2], zip(prels, postls))
	gx = AlfvenDetectors.sample(model)
	@test typeof(gx) <: Flux.TrackedArray{AlfvenDetectors.Float,1}
	gx = AlfvenDetectors.sample(model,5)
	@test typeof(gx) <: Flux.TrackedArray{AlfvenDetectors.Float,2}
	@test size(gx) == (xdim,5)

	# alternative constructor test
	model = AlfvenDetectors.VAE(xdim, ldim, 4)
	@test length(model.encoder.layers) == 4
	@test length(model.decoder.layers) == 4
	@test size(model.encoder(x)) == (ldim*2, N)
	@test size(model(x)) == (xdim, N)

	model = AlfvenDetectors.VAE(xdim, ldim, 4, variant = :scalar)
	@test size(model(x)) == (xdim+1, N)

	model = AlfvenDetectors.VAE(xdim, ldim, 4, variant = :diag)
	@test size(model(x)) == (xdim*2, N)
end