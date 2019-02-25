using AlfvenDetectors
using Test

M = 2
N = 9
X = rand(M,N)
uniter = 3
ubatchsize = 4
batchsize = 5
nepochs = 10

@testset "samplers" begin
	@test AlfvenDetectors.checkbatchsize(10, 3, true) == 3
	@test AlfvenDetectors.checkbatchsize(10, 3, false) == 3
	@test AlfvenDetectors.checkbatchsize(3, 10, true) == 10
	@test AlfvenDetectors.checkbatchsize(3, 10, false) == 3
	
	usampler = AlfvenDetectors.UniformSampler(X, uniter, ubatchsize)
	@test usampler.data == X 
	@test usampler.M == M
	@test usampler.N == N
	@test usampler.niter == uniter
	@test usampler.batchsize == ubatchsize
	@test usampler.iter == 0
	@test usampler.replace == false
	batch = AlfvenDetectors.next!(usampler)
	@test usampler.iter == 1
	@test size(batch) == (M, ubatchsize)
	ixs = AlfvenDetectors.enumerate(usampler)
	@test usampler.iter == uniter
	@test length(ixs) == uniter - 1
	@test size(ixs[1][2]) == (M, ubatchsize)
	@test size(ixs[2][2]) == (M, ubatchsize)
	AlfvenDetectors.reset!(usampler)
	xs = AlfvenDetectors.collect(usampler)
	@test length(xs) == uniter
	@test size(xs[1]) == (M,ubatchsize)
	@test size(xs[end]) == (M,ubatchsize)

	esampler = AlfvenDetectors.EpochSampler(X, nepochs, batchsize)
	@test esampler.M == M
	@test esampler.N == N
	@test esampler.batchsize == batchsize
	@test esampler.nepochs == nepochs
	@test esampler.data == X
	@test esampler.iter == 0
	@test esampler.epochsize == ceil(Int,N/batchsize)
	batch = AlfvenDetectors.next!(esampler)
	@test esampler.iter == 0
	@test size(batch) == (M,batchsize)
	batch = AlfvenDetectors.next!(esampler)
	@test size(batch) == (M,N%batchsize)
	batch = AlfvenDetectors.next!(esampler)
	@test size(batch) == (M,batchsize)
	ixs = AlfvenDetectors.enumerate(esampler)
	@test esampler.iter == nepochs
	@test length(ixs) == (nepochs-1)*esampler.epochsize-1
	@test size(ixs[1][2]) == (M,N%batchsize)
	@test size(ixs[2][2]) == (M,batchsize)
	AlfvenDetectors.reset!(esampler)
	xs = AlfvenDetectors.collect(esampler)
	@test length(xs) == nepochs*esampler.epochsize
	@test size(xs[1]) == (M,batchsize)
	@test size(xs[2]) == (M,N%batchsize)
end