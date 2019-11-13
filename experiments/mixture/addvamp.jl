using GenModels
using Flux

# decoder ensemble with vamp 
mutable struct ADDVAMP
	encoders
	decoders
	pseudoinputs
	α
	
	# alpha may be a function
	ADDVAMP(Es,Ds,A,xdim::Union{Int,Tuple},K::Int) = 
		(length(Es)==length(Ds)==K) ? new(Es,Ds,Flux.param(randn(Float32, xdim...,K)), A) :
		error("number of decoders not consistent with requested prior size $K")
end

Flux.@treelike(ADDVAMP)

ncomponents(m::ADDVAMP) = size(m.pseudoinputs)[end]

function reconstruct(m::ADDVAMP, x, y::Int)
	if y == 0
		z = sample_latent(m, x, 1)
		xhat = m.decoders[1](z)
	elseif y == 1
		z1 = sample_latent(m, x, 1)
		z2 = sample_latent(m, x, 2)
		xhat = m.decoders[1](z1) + m.decoders[2](z2)
	end
	return xhat
end

function sampleVamp(m::ADDVAMP, n::Int, k::Int)
	nc = ncomponents(m)
	(k>nc) ? error("Requested component id $k is larger than number of available components $nc") : nothing
	x = m.pseudoinputs[repeat([:], ndims(m.pseudoinputs)-1)..., k:k]
	ex = m.encoders[k](x)
	μ, σ2 = GenModels.mu(ex), GenModels.sigma2(ex)
	σ2 = min.(σ2,Float32(1e10))
	return hcat([GenModels.samplenormal(μ, σ2) for i in 1:n]...)
end

function sample_latent(e, x)
	ex = e(x)
	μ, σ2 = GenModels.mu(ex), GenModels.sigma2(ex)
	σ2 = min.(σ2,Float32(1e10))
	z = GenModels.samplenormal(μ, σ2)
end

sample_latent(m::ADDVAMP, x, k) = sample_latent(m.encoders[k], x)

function loss(m::ADDVAMP, x, y, β)
	if y == 0
		z = sample_latent(m, x, 1)
		zp = sampleVamp(m, size(z,2), 1)
		xhat = m.decoders[1](z)
		return Flux.mse(x, xhat) + Float32(β)*GenModels.MMD(GenModels.imq, z, zp, Float32(0.01)) 
	else
		z1 = sample_latent(m, x, 1)
		z2 = sample_latent(m, x, 2)
		zp1 = sampleVamp(m, size(z1,2), 1)
		zp2 = sampleVamp(m, size(z2,2), 2)
		xhat = m.decoders[1](z1) + m.decoders[2](z2)
		return Flux.mse(x, xhat) + Float32(β)*GenModels.MMD(GenModels.imq, z1, zp1, Float32(0.01)) + Float32(β)*GenModels.MMD(GenModels.imq, z2, zp2, Float32(0.01)) 
	end
end

function estimate_y(m,X)
	(size(X,4) != 1) ? error("input only individual samples") : nothing
	es = [Flux.Tracker.data(Flux.mse(X, reconstruct(m, X, y))) for y in [0,1]]
	return argmin(es)-1
end

function loss(m::ADDVAMP, x, β)
	y = map(i->estimate_y(m, x[:,:,:,i:i]), 1:size(x,4))
	x1 = x[:,:,:,y.==1]
	x0 = x[:,:,:,y.==0]
	return loss(m,x1,1) + loss(m,x0,0)
end

function plot_reconstructions(m,x,y)
	xhat = cpu(Flux.Tracker.data(reconstruct(m,x,y)));
	figure()
	for i in 1:4
		subplot(4,2,2i-1)
		pcolormesh(x[:,:,1,i])

		subplot(4,2,2i)
		title(Flux.mse(cpu(x[:,:,1,i]), xhat[:,:,1,i]))
		pcolormesh(xhat[:,:,1,i])
	end
	tight_layout()
end

function update_model(m, x, y, β, opt)
	l = loss(m, x, y, β)
	Flux.back!(l)
	GenModels.update!(m, opt)
end
function train_addvamp(m, data, β, opt)
	for _data in data
		x, y = _data
		if y != nothing
			update_model(m, gpu(x), y, β, opt)
		else
			ys = map(i->estimate_y(m, gpu(x[:,:,:,i:i])), 1:size(x,4))
			x1 = x[:,:,:,ys.==1]
			x0 = x[:,:,:,ys.==0]
			update_model(m, gpu(x1), 1, β, opt)
			update_model(m, gpu(x0), 0, β, opt)
		end
		l1 = loss(m, X1, Y1, β).data
		l0 = loss(m, X0, Y0, β).data
		println("$l1     $l0")
	end
end
function batch_mse(m, X)
	es = map(i -> [Flux.Tracker.data(Flux.mse(gpu(X[:,:,:,i:i]), 
			reconstruct(m, gpu(X[:,:,:,i:i]), y))) for y in [0,1]], 1:size(X,4))
	cat(es..., dims=2)
end
