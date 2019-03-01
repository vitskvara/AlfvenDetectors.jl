using AlfvenDetectors
using Flux
using ValueHistories
using CuArrays
using Test
using Random

xdim = 5
ldim = 1
N = 10

sim(x,y) = abs(x-y) < 1e-6

@testset "flux utils" begin
	# iscuarray
	x = randn(4,10)
	@test !AlfvenDetectors.iscuarray(x)
	x = x |> gpu
	@test AlfvenDetectors.iscuarray(x)
	model = Flux.Chain(Flux.Dense(4, ldim), Flux.Dense(ldim, 4)) |> gpu
	_x = model(x)
	@test AlfvenDetectors.iscuarray(_x)
end

@testset "model utils" begin
	x = fill(0.0,5,5) |> gpu
	sd = fill(1.0,5) |> gpu
	# the version where sigma is a vector (scalar variance)
	@test sim(AlfvenDetectors.loglikelihood(x,x,sd), 0.0 - AlfvenDetectors.l2pi/2*5)
	@test sim(AlfvenDetectors.loglikelihood(x,x,sd), 0.0 - AlfvenDetectors.l2pi/2*5)
end

@testset "AE-GPU" begin
	x = AlfvenDetectors.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2)))) |> gpu
	Random.seed!(12345)
	model = AlfvenDetectors.AE([xdim,2,ldim], [ldim,2,xdim]) |> gpu
	_x = model(x)
	# for training check
	frozen_params = map(x->copy(Flux.Tracker.data(x)), collect(params(model)))

	@test typeof(x) == CuArray{AlfvenDetectors.Float,2}
	@test typeof(_x) <: TrackedArray{AlfvenDetectors.Float,2}    
	hist = MVHistory()
	AlfvenDetectors.fit!(model, x, 5, 1000, cbit=100, history = hist, verb=false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	@test ls[end] < 1e-4
	# were the layers realy trained?
	for (fp, p) in zip(frozen_params, collect(params(model)))
		@test fp!=p
	end
end

@testset "VAE-GPU" begin
	x = AlfvenDetectors.Float.(hcat(ones(xdim, Int(N/2)), zeros(xdim, Int(N/2)))) |> gpu

	# unit VAE
	Random.seed!(12345)
    model = AlfvenDetectors.VAE([xdim,2,2*ldim], [ldim,2,xdim]) |> gpu
	_x = model(x)
	# for training check
	frozen_params = map(x->copy(Flux.Tracker.data(x)), collect(params(model)))

	@test typeof(x) == CuArray{AlfvenDetectors.Float,2}
	@test typeof(_x) <: TrackedArray{AlfvenDetectors.Float,2}    
	hist = MVHistory()
	AlfvenDetectors.fit!(model, x, 5, 50, β =0.1, cbit=5, history = hist, verb = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	for (fp, p) in zip(frozen_params, collect(params(model)))
		@test fp!=p
	end

	# diag VAE
	Random.seed!(12345)
    model = AlfvenDetectors.VAE([xdim,2,2*ldim], [ldim,2,xdim*2], variant = :diag) |> gpu
	_x = model(x)
	# for training check
	frozen_params = map(x->copy(Flux.Tracker.data(x)), collect(params(model)))

	@test typeof(x) == CuArray{AlfvenDetectors.Float,2}
	@test typeof(_x) <: TrackedArray{AlfvenDetectors.Float,2}    
	hist = MVHistory()
	AlfvenDetectors.fit!(model, x, 5, 50, β =0.1, cbit=5, history = hist, verb = false)
	is, ls = get(hist, :loss)
	@test ls[1] > ls[end] 
	# were the layers realy trained?
	for (fp, p) in zip(frozen_params, collect(params(model)))
		@test fp!=p
	end

end