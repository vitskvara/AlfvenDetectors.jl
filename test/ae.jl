using AlfvenDetectors
using Flux
using Pkg

xdim = 3
ldim = 1
N = 10

x = AlfvenDetectors.Float.(randn(xdim,N))
model = AlfvenDetectors.AE([xdim,2,ldim], [ldim,2,xdim])
_x = model(x)

@testset "AE" begin

	@test size(model.encoder.layers,1) == 2
	@test size(model.decoder.layers,1) == 2
	@test size(model.encoder(x)) == (ldim, N)
	@test size(_x) == (xdim,N)
	@test typeof(x) == Array{AlfvenDetectors.Float,2}
	@test typeof(_x) <: Flux.TrackedArray{AlfvenDetectors.Float,2}
    
end
