using AlfvenDetectors

M = 2
N = 9
X = rand(M,N)
batchsize = 5
nepochs = 10

@testset "samplers" begin
	esampler = AlfvenDetectors.EpochSampler(X, nepochs, batchsize)
	@test esampler.M == M
	@test esampler.N == N
	@test esampler.batchsize == batchsize
	@test esampler.nepochs == nepochs
	@test esampler.data == X
	@test esampler.iter == 0
	@test esampler.epochsize == ceil(Int,N/batchsize)
	batch = AlfvenDetectors.next!(esampler)
	@test size(batch) == (M,batchsize)
	batch = AlfvenDetectors.next!(esampler)
	@test size(batch) == (M,N%batchsize)
	batch = AlfvenDetectors.next!(esampler)
	@test size(batch) == (M,batchsize)
	ixs = AlfvenDetectors.enumerate(esampler)
	@test length(ixs) == (nepochs-1)*esampler.epochsize-1
	@test size(ixs[1][2]) == (M,N%batchsize)
	@test size(ixs[2][2]) == (M,batchsize)
	AlfvenDetectors.reset!(esampler)
	xs = AlfvenDetectors.collect(esampler)
	@test length(xs) == nepochs*esampler.epochsize
	@test size(xs[1]) == (M,batchsize)
	@test size(xs[2]) == (M,N%batchsize)
end