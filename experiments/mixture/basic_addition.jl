using GenModels
using Flux

struct TAE
	encoder
	decoders
end

Flux.@treelike TAE

function reconstruct(m::TAE, x)
	z = m.encoder(x)
	xhat = reduce(+, map(d->d(z), m.decoders))
#	xhat = m.d1(z) + m.d2(z)
	return xhat
end

loss(m::TAE, x) = Flux.mse(x, reconstruct(m,x))

# net
M = 3
N = 100
X = randn(M,N)/5
X[:,1:Int(N/2)] .-= 1.0
X[:,Int(N/2)+1:end] .+= 1.0

zdim = 1
hdim = 50
K = 2

e = Chain(Dense(M, hdim, Flux.tanh), Dense(hdim, zdim))
ds = Tuple([Chain(Dense(zdim, hdim, Flux.tanh), Dense(hdim, M)) for k in 1:K])

m = TAE(e,ds)

loss(x) = loss(m,x)

data = [X for i in 1:1000]
opt = ADAM()
cb(m,d,l,o) = println(loss(X).data)


GenModels.train!(m, data, loss, opt, cb)

plot_model_results(m, X)

struct decoders
	ds
end

Flux.@treelike decoders

md = decoders(ds)
Flux.params(m::decoders) = [[params(x) for x in m.ds]...]

m = TAE(e,md)
