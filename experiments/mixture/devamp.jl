using GenModels
using Flux

# decoder ensemble with vamp 
mutable struct DEVAMP
	encoder
	decoders
	pseudoinputs
	α
	
	# alpha may be a function
	DEVAMP(E,Ds,A,xdim::Union{Int,Tuple},K::Int) = 
		(length(Ds)==K) ? new(E,Ds,Flux.param(randn(Float32, xdim...,K)), A) :
		error("number of decoders not consistent with requested prior size $K")
end

Flux.@treelike(DEVAMP)

ncomponents(m::DEVAMP) = size(m.pseudoinputs)[end]

function sample_latent(m::DEVAMP, x)
	ex = m.encoder(x)
	μ, σ2 = GenModels.mu(ex), GenModels.sigma2(ex)
	σ2 = min.(σ2,Float32(1e10))
	z = GenModels.samplenormal(μ, σ2)
end

function sample_latent(e, x)
	ex = e(x)
	μ, σ2 = GenModels.mu(ex), GenModels.sigma2(ex)
	σ2 = min.(σ2,Float32(1e10))
	z = GenModels.samplenormal(μ, σ2)
end

function reconstruct(m::DEVAMP, x)
	z = sample_latent(m, x)
	xhats = map(d-> d(z), m.decoders)
	α = m.α(x)
	if ndims(α) == 1 # if alpha is a vector
		xhat = reduce(+, xhats .* α)
	else # if it is an array - an alpha for each sample
		nc = ncomponents(m)
		n = size(α, 2)
		if ndims(xhats[1]) == 2 
			xhat = dropdims(sum(cat(xhats..., dims=3) .* reshape(α, (1, n, nc)), dims=3), dims = 3)
		elseif ndims(xhats[1]) == 4 
			xhat = dropdims(sum(cat(xhats..., dims=5) .* reshape(α, (1, 1, 1, n, nc)), dims=5), dims = 5)
		else
			error("not implemented for input of this shape")
		end
	end
	return xhat
end

function sampleVamp(m::DEVAMP, n::Int)
	ids = rand(1:ncomponents(m), n)
	return _sampleVamp(m, ids)
end

function sampleVamp(m::DEVAMP, n::Int, k::Int)
	nc = ncomponents(m)
	(k>nc) ? error("Requested component id $k is larger than number of available components $nc") : nothing
	ids = repeat([k], n)
	return _sampleVamp(m, ids)
end

function _sampleVamp(m::DEVAMP, ids::Vector)
	x = m.pseudoinputs[repeat([:], ndims(m.pseudoinputs)-1)..., ids]
	ex = m.encoder(x)
	μ, σ2 = GenModels.mu(ex), GenModels.sigma2(ex)
	σ2 = min.(σ2,Float32(1e10))
	return GenModels.samplenormal(μ, σ2)
end

function loss(m::DEVAMP, x, β; fixa = true)
	z = sample_latent(m, x)
	zp = sampleVamp(m, size(z,2))
	xhats = map(d-> d(z), m.decoders)
	α = (fixa) ? Flux.Tracker.data(m.α(x)) : m.α(x)
	if ndims(α) == 1 # if alpha is a vector
		xhat = reduce(+, xhats .* α)
	else # if it is an array - an alpha for each sample
		nc = ncomponents(m)
		n = size(α, 2)
		if ndims(xhats[1]) == 2 
			xhat = dropdims(sum(cat(xhats..., dims=3) .* reshape(α, (1, n, nc)), dims=3), dims = 3)
		elseif ndims(xhats[1]) == 4 
			xhat = dropdims(sum(cat(xhats..., dims=5) .* reshape(α, (1, 1, 1, n, nc)), dims=5), dims = 5)
		else
			error("not implemented for input of this shape")
		end
	end
	return Flux.mse(x, xhat) + Float32(β)*GenModels.MMD(GenModels.imq, z, zp, Float32(0.01))
end

function loss(m::DEVAMP, x, Y, β; fixa = true)
	z = sample_latent(m, x)
	zp = sampleVamp(m, size(z,2))
	xhats = map(d-> d(z), m.decoders)
	α = m.α(x)
	nc = ncomponents(m)
	n = size(α, 2)
	_α = (fixa) ? Flux.Tracker.data(α) : α
	if ndims(xhats[1]) == 2
		xhat = dropdims(sum(cat(xhats..., dims=3) .* reshape(_α, (1, n, nc)), dims=3), dims = 3)
	elseif  ndims(xhats[1]) == 4
		xhat = dropdims(sum(cat(xhats..., dims=5) .* reshape(_α, (1, 1, 1, n, nc)), dims=5), dims = 5)
	else
		error("not implemented for input of this shape")
	end
	return Flux.mse(x, xhat) + Float32(β)*GenModels.MMD(GenModels.imq, z, zp, Float32(0.01)) + Flux.crossentropy(α, Y)
end

function ssloss(m::DEVAMP, x, β; fixa = true)
	_x,_y = x[1], x[2]
	if _y == nothing
		return loss(m, _x, β; fixa = fixa)
	else
		return loss(m, _x, _y, β; fixa = fixa)
	end
end

function new_loss(m::DEVAMP, x, Y, β)
	z = sample_latent(m, x)
	zp = sampleVamp(m, size(x,2))
	xhats = map(d-> d(z), m.decoders)
	# here i need to compute the alphas as 
	α = m.α(x)
	nc = ncomponents(m)
	n = size(α, 2)
	xhat = dropdims(sum(cat(xhats..., dims=3) .* reshape(Flux.Tracker.data(α), (1, n, nc)), dims=3), dims = 3)
	return Flux.mse(x, xhat) + Float32(β)*GenModels.MMD(GenModels.imq, z, zp, 0.01) + Flux.crossentropy(α, Y)
end

function plot_model_results(m, X)
	x = copy(X)
	z = Flux.Tracker.data(sample_latent(m, x))
	zp = sampleVamp(m, size(x,ndims(x))).data
	N = size(X)[end]
	α = m.α(x)
	xhats = map(d-> d(z), m.decoders)
	if ndims(α) == 1 # if alpha is a vector
		xhat = reduce(+, xhats .* α)
	else # if it is an array - an alpha for each sample
		nc = ncomponents(m)
		n = size(α, 2)
		if ndims(xhats[1]) == 2 
			xhat = dropdims(sum(cat(xhats..., dims=3) .* reshape(α, (1, n, nc)), dims=3), dims = 3)
		elseif ndims(xhats[1]) == 4 
			xhat = dropdims(sum(cat(xhats..., dims=5) .* reshape(α, (1, 1, 1, n, nc)), dims=5), dims = 5)
		else
			error("not implemented for input of this shape")
		end
	end
	if ndims(x) == 2
		mse = map(i -> Flux.mse(x[:,i], xhat[:,i]).data, 1:N)
		mse1 = map(i -> Flux.mse(x[:,i], m.decoders[1](z[:,i])).data, 1:N)
		mse2 = map(i -> Flux.mse(x[:,i], m.decoders[2](z[:,i])).data, 1:N)
	else	
		mse = map(i -> Flux.mse(x[:,:,:,i], xhat[:,:,:,i]).data, 1:N)
		mse1 = map(i -> Flux.mse(x[:,:,:,i], m.decoders[1](z[:,i])).data, 1:N)
		mse2 = map(i -> Flux.mse(x[:,:,:,i], m.decoders[2](z[:,i])).data, 1:N)
	end
		#mse1 = map(i -> Flux.mse(x[:,i], α[1,i] * m.decoders[1](z[:,i])).data, 1:N)
	#mse2 = map(i -> Flux.mse(x[:,i], α[2,i] * m.decoders[2](z[:,i])).data, 1:N)

	N2 = Int(N/2)
	figure(figsize=(5,10))
	subplot(611)
	title("z distribution")
	hist(vec(z[:,1:N2]), alpha=0.5)
	hist(vec(z[:,N2+1:end]), alpha=0.5)

	subplot(612)
	title("prior distribution")
	Npz = 1000
	zp1 = sampleVamp(m, Npz, 2).data
	zp2 = sampleVamp(m, Npz, 1).data
	hist(vec(zp1), alpha=0.5)
	hist(vec(zp2), alpha=0.5)

	subplot(613)
	title("decoder 1 mse distribution")
	hist(mse1[1:N2], alpha=0.5, label="1")
	hist(mse1[1+N2:end], alpha=0.5, label="2")
	legend()

	subplot(614)
	title("decoder 2 mse distribution")
	hist(mse2[1:N2], alpha=0.5)
	hist(mse2[1+N2:end], alpha=0.5)

	subplot(615)
	title("alpha[1] distribution")
	if ndims(α) == 1
		nothing
	else
		α = Flux.Tracker.data(α)
		hist([α[1,1:N2], α[1,1+N2:end]], alpha=0.5)
	end

	subplot(616)
	title("alpha[2] distribution")
	if ndims(α) == 1
		nothing
	else
		hist([α[2,1:N2], α[2,1+N2:end]], alpha=0.5)
	end

	tight_layout()
end

function plot_conv_model_results(m, x)
	figure(figsize=(5,10))
	subplot(711)
	z = Flux.Tracker.data(sample_latent(m, x))
	scatter(z[1,1:N2], z[2,1:N2], label="positive")
	scatter(z[1,N2+1:end], z[2,N2+1:end], label="negative")
	legend()
	title("latent representations - train")

	subplot(712)
	tx = test_data |> gpu
	tz = Flux.Tracker.data(sample_latent(m, tx))
	scatter(tz[1,1:N2], tz[2,1:N2], label="positive")
	scatter(tz[1,N2+1:end], tz[2,N2+1:end], label="negative")
	legend()
	title("latent representations - test")

	subplot(713)
	title("prior distribution")
	Npz = 1000
	zp1 = cpu(sampleVamp(m, Npz, 2).data)
	zp2 = cpu(sampleVamp(m, Npz, 1).data)
	zp = cat(zp1, zp2, dims=2)
	hist2D(zp[1,:], zp[2,:], 20)

	mse1 = map(i -> Flux.mse(x[:,:,:,i], m.decoders[1](z[:,i])).data, 1:N)
	mse2 = map(i -> Flux.mse(x[:,:,:,i], m.decoders[2](z[:,i])).data, 1:N)
	subplot(714)
	title("decoder 1 mse distribution")
	hist(mse1[1:N2], alpha=0.5, label="1")
	hist(mse1[1+N2:end], alpha=0.5, label="2")
	legend()

	subplot(715)
	title("decoder 2 mse distribution")
	hist(mse2[1:N2], alpha=0.5)
	hist(mse2[1+N2:end], alpha=0.5)

	α = m.α(gpu(train_data))
	subplot(716)
	title("alpha[1] distribution - train")
	α = Flux.Tracker.data(α)
	hist([α[1,1:N2], α[1,1+N2:end]], alpha=0.5)

	α = m.α(gpu(test_data))
	subplot(717)
	title("alpha[1] distribution - test")
	α = Flux.Tracker.data(α)
	hist([α[1,1:N2], α[1,1+N2:end]], alpha=0.5)

	tight_layout()
end

function plot_reconstructions(m,X,train_data)
	xhat = cpu(Flux.Tracker.data(reconstruct(m, X)));
	figure()
	suptitle("test reconstructions")
	for i in 1:4
		subplot(4,2,2i-1)
		pcolormesh(X[:,:,1,i])

		subplot(4,2,2i)
		pcolormesh(xhat[:,:,1,i])
	end
	tight_layout()

	figure()
	suptitle("train reconstructions")
	for i in 1:4
		ipatch = rand(1:N)
		subplot(4,2,2i-1)
		pcolormesh(train_data[:,:,1,ipatch])

		xhat = Flux.Tracker.data(cpu(reconstruct(m, gpu(train_data[:,:,:,ipatch:ipatch]))))
		subplot(4,2,2i)
		pcolormesh(xhat[:,:,1,1])
	end
	tight_layout()
end
