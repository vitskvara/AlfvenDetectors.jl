using AlfvenDetectors
using Test
using ValueHistories
using Flux
using Random

xdim = 50
ldim = 1
N = 10

x = AlfvenDetectors.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2))))

@testset "VAE" begin
    println("           variational autoencoder")
    Random.seed!(12345)
    model = AlfvenDetectors.VAE([xdim,2,2*ldim], [ldim,2,xdim])
	_x = model(x)
	# for training check
	frozen_params = map(x->copy(Flux.Tracker.data(x)), collect(params(model)))

	using CuArrays
	gx = x |> gpu
	gmodel = model |> gpu

	gmodel(gx)
end