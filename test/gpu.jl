using AlfvenDetectors
using Flux
using CuArrays

@testset "AE-GPU" begin

	x = AlfvenDetectors.Float.(randn(xdim,N)) |> gpu
	model = AlfvenDetectors.AE([xdim,2,ldim], [ldim,2,xdim]) |> gpu
	_x = model(x)

	@test typeof(x) == CuArray{AlfvenDetectors.Float,2}
	@test typeof(_x) <: TrackedArray{AlfvenDetectors.Float,2}    
	
end
