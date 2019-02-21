using AlfvenDetectors
using Flux
using ValueHistories
using CuArrays

xdim = 50
ldim = 1
N = 10

@testset "AE-GPU" begin
	println("           autoencoder")

	x = AlfvenDetectors.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2)))) |> gpu
	model = AlfvenDetectors.AE([xdim,2,ldim], [ldim,2,xdim]) |> gpu
	_x = model(x)

	@test typeof(x) == CuArray{AlfvenDetectors.Float,2}
	@test typeof(_x) <: TrackedArray{AlfvenDetectors.Float,2}    
	hist = MVHistory()
	AlfvenDetectors.fit!(model, x, 5, 1000, cbit=100, history = hist)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	@test ls[end] < 1e-6
end
