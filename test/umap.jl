using AlfvenDetectors
using Test
using Random

@testset "UMAP" begin
	dim = 2
    model = AlfvenDetectors.UMAP(dim)
    if model == nothing
    	@warn "The UMAP Python package was not found, exiting the UMAP testset."
    else
    	X = randn(4,100)
    	Y = AlfvenDetectors.fit!(model,X)
    	@test size(Y) == (dim, size(X,2))
    	Z = randn(4,200)
    	@test size(AlfvenDetectors.transform(model, Z)) == (dim, size(Z,2))
    end
end