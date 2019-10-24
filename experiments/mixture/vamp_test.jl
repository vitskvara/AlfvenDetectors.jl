using GenModels
using Flux

struct VAMP
	encoder
	decoder
	pseudoinputs
	
	VAMP(E,D,xdim::Int,K::Int) = new(E,D,Flux.param(randn(xdim,K)))
end

Flux.@treelike(VAMP)

ncomponents(m::VAMP) = size(m.pseudoinputs,2)

function sampleVamp(m::VAMP, n::Int)
	ids = rand(1:ncomponents(m), n)
	return _sampleVamp(m, ids)
end

function sampleVamp(m::VAMP, n::Int, k::Int)
	nc = size(m.pseudoinputs,2) 
	(k>nc) ? error("Requested component id $k is larger than number of available components $nc") : nothing
	ids = repeat([k], n)
	return _sampleVamp(m, ids)
end

function _sampleVamp(m::VAMP, ids::Vector)
	x = m.pseudoinputs[:, ids]
	ex = m.encoder(x)
	μ, σ2 = GenModels.mu(ex), GenModels.sigma2(ex)
	return GenModels.samplenormal(μ, σ2)
end


function loss(m::VAMP, x, β)
	ex = m.encoder(x)
	μ, σ2 = GenModels.mu(ex), GenModels.sigma2(ex)
	z = GenModels.samplenormal(μ, σ2)
	zp = sampleVamp(m, size(x,2))
	xhat = m.decoder(z)
	return Flux.mse(x, xhat) + Float32(β)*GenModels.MMD(GenModels.imq, z, zp, 0.01)
end

# test
using PyPlot

M = 3
N = 100
X = randn(M,N)/1
X[:,1:Int(N/2)] .-= 1.0
X[:,Int(N/2)+1:end] .+= 1.0

zdim = 1
hdim = 4
K = 2
encoder = GenModels.aelayerbuilder([M, hdim, zdim*2], Flux.tanh, Flux.Dense)
decoder = GenModels.aelayerbuilder([zdim, hdim, M], Flux.tanh, Flux.Dense)

m = VAMP(encoder, decoder, M, K)

β = 1.0

loss(x) = loss(m,x,β)

data = [X for i in 1:1000]
opt = ADAM()
cb(m,d,l,o) = println(loss(X).data)

GenModels.train!(m, data, loss, opt, cb)

ex = m.encoder(x)
μ, σ2 = GenModels.mu(ex), GenModels.sigma2(ex)
z = GenModels.samplenormal(μ, σ2).data
zp = sampleVamp(m, size(x,2)).data

N2 = Int(N/2)
figure()
subplot(211)
hist(vec(z[:,1:N2]), alpha=0.5)
hist(vec(z[:,N2+1:end]), alpha=0.5)

subplot(212)
Npz = 1000
zp1 = sampleVamp(m, Npz, 1).data
zp2 = sampleVamp(m, Npz, 2).data
hist(vec(zp1), alpha=0.5)
hist(vec(zp2), alpha=0.5)
