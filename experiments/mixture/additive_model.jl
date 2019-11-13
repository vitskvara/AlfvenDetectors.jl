include("devamp.jl")

using PyPlot
using CuArrays

# first create the data
function circle(i,j,patchsize,r)
	x = zeros(patchsize, patchsize)
	for _i in i-r:i+r
		for _j in j-r:j+r
			((_i-i)^2 + (_j-j)^2 <= r^2) ? x[_i,_j] = 1 : nothing
		end
	end
	x
end

function generate_data(N,patchsize,R)
	x = randn(patchsize, patchsize,1,N)/10
	for n in 1:N
		i = rand(R+1:patchsize-R)
		j = rand(R+1:patchsize-R)
		x[:,:,1,n] += circle(i,j,patchsize,R)
	end
	x
end

patchsize = 32
N = 100
N2 = Int(N/2)
R = 3
train_data = Float32.(cat(generate_data(N2,patchsize,R), randn(patchsize, patchsize, 1, N2)/10, dims=4))
train_labels = hcat(ones(Int,1,N2), zeros(Int,1,N2))
train_labels = Int32.(vcat(train_labels, 1 .- train_labels))
train_data_unlabeled = Float32.(cat(generate_data(N2,patchsize,R), randn(patchsize, patchsize, 1, N2)/10, dims=4))
test_data = Float32.(cat(generate_data(N2,patchsize,R), randn(patchsize, patchsize, 1, N2)/10, dims=4))
test_labels = hcat(ones(Int,1,N2), zeros(Int,1,N2))
test_labels = Int32.(vcat(test_labels, 1 .- test_labels))

t = 0.5
data = [((rand() > t) ? (gpu(train_data), gpu(train_labels)) : (gpu(train_data_unlabeled), nothing)) for i in 1:200]
X = train_data_unlabeled[:,:,:,49:52] |> gpu
Y = train_labels[:,49:52] |> gpu

K = 2
hdim = 40


# now try a devamp model
ldim = 1
xsize = size(train_data)[1:3]
encoder = GenModels.convencoder(xsize, ldim*2, 2, 3, (8,16), 1) |> gpu
decoders = Tuple([GenModels.convdecoder(xsize, ldim, 2, 3, (16,8), 1) |> gpu for k in 1:K])
ac = Flux.Chain(
	Flux.Conv((3,3), 1=>4, relu, pad=1), 
	MaxPool((2,2)), 
	x->reshape(x, Int(patchsize/2)^2*4, :),
	Dense(Int(patchsize/2)^2*4, K),
	softmax
	) |> gpu


β = 0.1
m = DEVAMP(encoder, decoders, ac, xsize, K) 
m.pseudoinputs = gpu(m.pseudoinputs)
loss(x) = ssloss(m,x,β)
opt = ADAM()
cb(m,d,l,o) = println(loss((X, Y)).data)


GenModels.train!(m, data, loss, opt, cb)

plot_model_results(m, X)

plot_reconstructions(m,X,train_data)


# try it with a larger ldim
ldim = 2
xsize = size(train_data)[1:3]
encoder = GenModels.convencoder(xsize, ldim*2, 2, 3, (8,16), 1) |> gpu
decoders = Tuple([GenModels.convdecoder(xsize, ldim, 2, 3, (16,8), 1) |> gpu for k in 1:K])
ac = Flux.Chain(
	Flux.Conv((3,3), 1=>4, relu, pad=1), 
	MaxPool((2,2)), 
	x->reshape(x, Int(patchsize/2)^2*4, :),
	Dense(Int(patchsize/2)^2*4, K),
	softmax
	) |> gpu

β = 0.1
m = DEVAMP(encoder, decoders, ac, xsize, K) 
m.pseudoinputs = gpu(m.pseudoinputs)
loss(x) = ssloss(m,x,β)
opt = ADAM()
cb(m,d,l,o) = println(loss((X, Y)).data)

GenModels.train!(m, data, loss, opt, cb)

plot_conv_model_results(m, gpu(train_data))

plot_reconstructions(m,X,train_data)

# this one is based on the latent code
ldim = 2
xsize = size(train_data)[1:3]
encoder = GenModels.convencoder(xsize, ldim*2, 2, 3, (8,16), 1) |> gpu
decoders = Tuple([GenModels.convdecoder(xsize, ldim, 2, 3, (16,8), 1) |> gpu for k in 1:K])
ac = Flux.Chain(
	Dense(ldim, hdim, relu),
	Dense(hdim, K),
	softmax
	) |> gpu
class(x) = ac(sample_latent(encoder, x))


β = 0.1
m = DEVAMP(encoder, decoders, class, xsize, K) 
m.pseudoinputs = gpu(m.pseudoinputs)
loss(x) = ssloss(m,x,β,fixa=false)
opt = ADAM()
cb(m,d,l,o) = println(loss((X, Y)).data)

for i in 1:1
	GenModels.train!(m, data, loss, opt, cb)
end

plot_conv_model_results(m, gpu(train_data))

plot_reconstructions(m,X,train_data)

# now lets compute alpha as the loglikelihood with respect to the vamp components
function loglikelihood(X::AbstractMatrix, μ::AbstractVector, σ2::AbstractVector)
	y = (X .- μ).^2 ./ σ2 
	y = y .+ log.(σ2)
	y = y .+ log(Float32(2*pi))
	- sum(y, dims=1)/2f0
end

function vamp_alpha(m, X)
	pe = m.encoder(m.pseudoinputs)
	pμ, pσ2 = GenModels.mu(pe), GenModels.sigma2(pe)

	z = sample_latent(m,X)
	llh = cat([loglikelihood(z, pμ[:,i], pσ2[:,i]) for i in 1:size(pe,2)]..., dims=1)
#	lh = exp.(llh)

	normalize_columns(llh)
end

normalize_columns(u::AbstractMatrix) = u ./ sum(u, dims=1)

ldim = 2
xsize = size(train_data)[1:3]
encoder = GenModels.convencoder(xsize, ldim*2, 2, 3, (8,16), 1) |> gpu
decoders = Tuple([GenModels.convdecoder(xsize, ldim, 2, 3, (16,8), 1) |> gpu for k in 1:K])
class(x) = vamp_alpha(m, x)

β = 1.0
m = DEVAMP(encoder, decoders, class, xsize, K) 
m.pseudoinputs = gpu(m.pseudoinputs)
loss(x) = ssloss(m,x,β,fixa=false)
opt = ADAM()
cb(m,d,l,o) = println(loss((X, Y)).data)


for i in 1:1
	GenModels.train!(m, data, loss, opt, cb)
end

plot_conv_model_results(m, gpu(train_data))

plot_reconstructions(m,X,train_data)

l = loss((X,nothing))
l = loss((X,Y))
Flux.back!(l)
GenModels.update!(m,opt)
α = m.α(X)