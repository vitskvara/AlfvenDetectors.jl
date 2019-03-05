using AlfvenDetectors
using Test
using ValueHistories
using Flux
using Random

xdim = 5
ldim = 1
N = 100

@testset "TSVAE" begin
    println("           two-stage VAE")
    Random.seed!(12345)
   	x = AlfvenDetectors.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
   	model = AlfvenDetectors.TSVAE(
   		[[xdim, 3, ldim*2], [ldim, 3, xdim+1]],
   		[[ldim,ldim,ldim*2], [ldim, ldim, ldim+1]])
   	_x = model(x)
   	z = model.m1.sampler(model.m1.encoder(x))
   	@test size(_x) == (xdim+1, N)
   	@test size(z) == (ldim,N)
   	@test size(model.m1(x)) == size(_x)
   	@test size(model.m2(z)) == (ldim+1, N)
   	@test size(model.m2.encoder(z)) == (ldim*2,N)

	model = AlfvenDetectors.TSVAE(xdim, ldim, (3,2))
   	_x = model(x)
   	z = model.m1.sampler(model.m1.encoder(x))
   	@test size(_x) == (xdim+1, N)
   	@test size(z) == (ldim,N)
   	@test size(model.m1(x)) == size(_x)
   	@test size(model.m2(z)) == (ldim+1, N)
   	@test size(model.m2.encoder(z)) == (ldim*2,N)
   	@test length(model.m1.encoder.layers) == 3
   	@test length(model.m1.decoder.layers) == 3
   	@test length(model.m2.encoder.layers) == 2
   	@test length(model.m2.decoder.layers) == 2

   	# fit!
    frozen_params = map(x->copy(Flux.Tracker.data(x)), collect(params(model)))
    @test length(frozen_params) == 20
   	history = (MVHistory(),MVHistory())
   	m1ls, m2ls = AlfvenDetectors.getlosses(model,x,10,1.0)
   	AlfvenDetectors.fit!(model, x, 5, 500; history = history, verb = false)
   	post_m1ls, post_m2ls = AlfvenDetectors.getlosses(model,x,10,1.0)
	@test exp(post_m1ls[2]) < 1e-6
	@test m1ls[1] > post_m1ls[1]
	@test any(x->x[1]>x[2], zip(m1ls, post_m1ls))
	@test any(x->x[1]>x[2], zip(m2ls, post_m2ls))
	# were the layers realy trained?
	for (fp, p) in zip(frozen_params, collect(params(model)))
		@test fp!=p
	end
	_,l1h = get(history[1],:loss)
	_,l2h = get(history[2],:loss)
	@test length(l1h) == 10000
	@test length(l2h) == 10000
end
