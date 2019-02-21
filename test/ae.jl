using AlfvenDetectors
using Flux
using ValueHistories

xdim = 50
ldim = 1
N = 10

x = AlfvenDetectors.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))
model = AlfvenDetectors.AE([xdim,2,ldim], [ldim,2,xdim])
_x = model(x)

@testset "AE" begin
	println("           autoencoder")

	# test correct construction
	@test size(model.encoder.layers,1) == 2
	@test size(model.decoder.layers,1) == 2
	# test basic functionality
	@test size(model.encoder(x)) == (ldim, N)
	@test size(_x) == (xdim,N)
	# test output types
	@test typeof(_x) <: Flux.TrackedArray{AlfvenDetectors.Float,2}
	@test typeof(AlfvenDetectors.loss(model, x)) == Flux.Tracker.TrackedReal{AlfvenDetectors.Float}    
	# test loss functions
	l = AlfvenDetectors.getlosses(model, x)[1]
	@test typeof(l) == AlfvenDetectors.Float
	@test AlfvenDetectors.loss(model, x) == l
	# test basic loss tracking
	hist = MVHistory()
	AlfvenDetectors.track!(model, hist, x)
	AlfvenDetectors.track!(model, hist, x)
	is, ls = get(hist, :loss)
	@test ls[1] == l
	@test ls[1] == ls[2]
	# test training
	AlfvenDetectors.fit!(model, x, 5, 1000, cbit=100, history = hist)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	@test ls[end] < 1e-6
end

