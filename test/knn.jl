using AlfvenDetectors
using Test
using Random

@testset "kNN" begin
	xdim, N = 2, 50
	X = hcat(randn(xdim, N/2)-[10,10], randn(xdim, N/2)+[10,10])
    
end