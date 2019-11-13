include("devamp.jl")

# test it 
using PyPlot


M = 3
N = 100
X = randn(M,N)/5
X[:,1:Int(N/2)] .-= 1.0
X[:,Int(N/2)+1:end] .+= 1.0

zdim = 1
hdim = 50
K = 2
encoder = GenModels.aelayerbuilder([M, hdim, zdim*2], Flux.tanh, Flux.Dense)
decoders = Tuple([GenModels.aelayerbuilder([zdim, hdim, M], Flux.tanh, Flux.Dense) for k in 1:K])
a(x) = ones(K)/K 

m = DEVAMP(encoder, decoders, a, M, K)

β = 1.0

loss(x) = loss(m,x,β)

data = [X for i in 1:1000]
opt = ADAM()
cb(m,d,l,o) = println(loss(X).data)


GenModels.train!(m, data, loss, opt, cb)

plot_model_results(m, X)

# trainable alpha
encoder = GenModels.aelayerbuilder([M, hdim, zdim*2], Flux.tanh, Flux.Dense)
decoders = Tuple([GenModels.aelayerbuilder([zdim, hdim, M], Flux.tanh, Flux.Dense) for k in 1:K])
alpha = Flux.Chain(Dense(M, hdim*2, tanh), Dense(hdim*2, K),  softmax)

m = DEVAMP(encoder, decoders, alpha, M, K)

β = 1.0

loss(x) = loss(m,x,β)

data = [X for i in 1:1000]
opt = ADAM()
cb(m,d,l,o) = println(loss(X).data)

GenModels.train!(m, data, loss, opt, cb)

plot_model_results(m, X)

# (semi)supervised training
N2 = Int(N/2)
Y = zeros(K, N)
Y[1,1:N2] .= 1
Y[2,1+N2:end] .= 1

encoder = GenModels.aelayerbuilder([M, hdim, zdim*2], Flux.tanh, Flux.Dense)
decoders = Tuple([GenModels.aelayerbuilder([zdim, hdim, M], Flux.tanh, Flux.Dense) for k in 1:K])
ac = Flux.Chain(Dense(M, hdim, tanh), Dense(hdim, K),  softmax)
a(x) = Y .* ac(x)
#a(x) = Y

m = DEVAMP(encoder, decoders, a, M, K)

β = 1.0

loss(x) = loss(m,x,β)

data = [X for i in 1:500]
opt = ADAM()
cb(m,d,l,o) = println(loss(X).data)

GenModels.train!(m, data, loss, opt, cb)

plot_model_results(m, X)

# randomly assign labels
#Ys = vcat([rand([[1,0], nothing]) for i in 1:N2], [rand([[0,1], nothing]) for i in 1:N2])
Ys = ones(K, N)
Ys[1,1:N2] = rand([1000,1], N2)
Ys[2,1+N2:end] = rand([1000,1], N2)
# a [1000, 1] couple after softmax is [1, 0], however a [1, 1] is [0.5, 0.5]
ac = Flux.Chain(Dense(M, hdim*2, relu), Dense(hdim*2, K), x-> x .* Ys, softmax)
encoder = GenModels.aelayerbuilder([M, hdim, zdim*2], Flux.relu, Flux.Dense)
decoders = Tuple([GenModels.aelayerbuilder([zdim, hdim, M], Flux.relu, Flux.Dense) for k in 1:K])

me = GenModels.freeze(encoder)
mex = me(X)

m = DEVAMP(encoder, decoders, ac, M, K)

β = 1.0

loss(x) = loss(m,x,β)

data = [X for i in 1:1000]
opt = ADAM()
cb(m,d,l,o) = println(loss(X).data)

GenModels.train!(m, data, loss, opt, cb)

plot_model_results(m, X)

# what if the alpha was computed from the encoding instead of a separate chain
Ys = ones(K, N)
Ys[1,1:N2] = rand([10000,1], N2)
Ys[2,1+N2:end] = rand([10000,1], N2)
# a [1000, 1] couple after softmax is [1, 0], however a [1, 1] is [0.5, 0.5]
encoder = GenModels.aelayerbuilder([M, hdim, zdim*2], Flux.tanh, Flux.Dense)
decoders = Tuple([GenModels.aelayerbuilder([zdim, hdim, M], Flux.tanh, Flux.Dense) for k in 1:K])
# this does not work very well
# ac = Flux.Chain(encoder, Dense(zdim*2, K, Flux.tanh), x-> x .* Ys, softmax)
ac = Flux.Chain(GenModels.freeze(encoder), Dense(zdim*2, hdim*2, Flux.tanh), Dense(hdim*2, K, Flux.tanh),
 x-> x .* Ys, softmax)

m = DEVAMP(encoder, decoders, ac, M, K)

β = 1.0

loss(x) = loss(m,x,β)

data = [X for i in 1:1000]
opt = ADAM()
cb(m,d,l,o) = println(loss(X).data)

GenModels.train!(m, data, loss, opt, cb)

plot_model_results(m, X)

# try a different way of computing the alpha
Yn = []
for i in 1:size(Y,2)
	y = (rand() > 0.7) ? Y[:,i] : nothing
	push!(Yn, y)
end
encoder = GenModels.aelayerbuilder([M, hdim, zdim*2], Flux.tanh, Flux.Dense)
decoders = Tuple([GenModels.aelayerbuilder([zdim, hdim, M], Flux.tanh, Flux.Dense) for k in 1:K])
ac = Flux.Chain(Dense(M, hdim*2, Flux.tanh), Dense(hdim*2, K), softmax)

m = DEVAMP(encoder, decoders, ac, M, K)

β = 1.0

loss(x) = ssloss(m,x,β)

opt = ADAM()
cb(m,d,l,o) = println(loss((X, nothing)).data)

# start with unlabeled data
data = [(X, nothing) for i in 1:1000]

GenModels.train!(m, data, loss, opt, cb)

plot_model_results(m, X)

# try it will fully labeled data
encoder = GenModels.aelayerbuilder([M, hdim, zdim*2], Flux.tanh, Flux.Dense)
decoders = Tuple([GenModels.aelayerbuilder([zdim, hdim, M], Flux.tanh, Flux.Dense) for k in 1:K])
ac = Flux.Chain(Dense(M, hdim*2, Flux.tanh), Dense(hdim*2, K), softmax)

m = DEVAMP(encoder, decoders, ac, M, K)
loss(x) = ssloss(m,x,β)

opt = ADAM()

cb(m,d,l,o) = println(loss((X, Y)).data)
data = [(X, Y) for i in 1:1000]

GenModels.train!(m, data, loss, opt, cb)

plot_model_results(m, X)

# finally only use a small number of labels
t = 0.8 # how much data is gonna be unlabeled
encoder = GenModels.aelayerbuilder([M, hdim, zdim*2], Flux.tanh, Flux.Dense)
decoders = Tuple([GenModels.aelayerbuilder([zdim, hdim, M], Flux.tanh, Flux.Dense) for k in 1:K])
ac = Flux.Chain(Dense(M, hdim*2, Flux.tanh), Dense(hdim*2, K), softmax)
β = 1.0

m = DEVAMP(encoder, decoders, ac, M, K)
loss(x) = ssloss(m,x,β)

opt = ADAM()

cb(m,d,l,o) = println(loss((X, Y)).data)
data = [((rand() > t) ? (X, Y) : (X, nothing)) for i in 1:200]

GenModels.train!(m, data, loss, opt, cb)

plot_model_results(m, X)

# get alpha from z
t = 0.8 # how much data is gonna be unlabeled
encoder = GenModels.aelayerbuilder([M, hdim, zdim*2], Flux.tanh, Flux.Dense)
decoders = [GenModels.aelayerbuilder([zdim, hdim, M], Flux.tanh, Flux.Dense) for k in 1:K]
ac = Flux.Chain(Dense(zdim, hdim*2, Flux.tanh), Dense(hdim*2, K), softmax)
function clss(X) 
	ex = m.encoder(x)
	μ, σ2 = GenModels.mu(ex), GenModels.sigma2(ex)
	z = GenModels.samplenormal(μ, σ2)
	return ac(z)
end

β = 0.1

m = DEVAMP(encoder, decoders, clss, M, K)
loss(x) = ssloss(m,x,β)

opt = ADAM()

cb(m,d,l,o) = println(loss((X, Y)).data)
data = [((rand() > t) ? (X, Y) : (X, nothing)) for i in 1:200]

GenModels.train!(m, data, loss, opt, cb)

plot_model_results(m, X)

# learn additive signal
# have a Grim component
# add one docedoer that is always present
t = 0.8 # how much data is gonna be unlabeled
encoder = GenModels.aelayerbuilder([M, hdim, zdim*2], Flux.tanh, Flux.Dense)
decoders = [GenModels.aelayerbuilder([zdim, hdim, M], Flux.tanh, Flux.Dense) for k in 1:K]

for i in 1:K
	x = m.pseudoinputs[:, ids]
	ex = m.encoder(x)
	μ, σ2 = GenModels.mu(ex), GenModels.sigma2(ex)
end

function alphas(m,X)
	ex = m.encoder(x)
	μ, σ2 = GenModels.mu(ex), GenModels.sigma2(ex)
	z = GenModels.samplenormal(μ, σ2)
	for k in 1:K
		px = m.pseudoinputs[:, repeat([k], size(X,2))] 
		epx = m.encoder(px)
		μ, σ2 = GenModels.mu(epx), GenModels.sigma2(epx)
	end

end

ac = Flux.Chain(Dense(M, hdim*2, Flux.tanh), Dense(hdim*2, K), softmax)

β = 0.1

m = DEVAMP(encoder, decoders, ac, M, K)
loss(x) = ssloss(m,x,β)

opt = ADAM()

cb(m,d,l,o) = println(loss((X, Y)).data)
data = [((rand() > t) ? (X, Y) : (X, nothing)) for i in 1:200]

GenModels.train!(m, data, loss, opt, cb)

plot_model_results(m, X)

const l2pi = Float(log(2*pi)) # the model converges the same with zero or correct value

- sum((μ - z).^2 ./σ2 + log.(σ2) .+ log(2*pi),dims = 1))/2

function loglikelihood(X::AbstractMatrix, μ::AbstractMatrix, σ2::AbstractVector) 
    # again, this has to be split otherwise it is very slow
    y = (μ - X).^2
    y = (one(Float) ./σ2)' .* y 
    - StatsBase.mean(sum( y .+ reshape(log.(σ2), 1, length(σ2)) .+ l2pi,dims = 1))*half
end
