"""
	VAE{encoder, sampler, decoder, variant}

Flux-like structure for the variational autoencoder.
"""
struct VAE
	encoder
	sampler
	decoder
	variant::Symbol
end

VAE(E,S,D) = VAE(E,S,D,:unit)

# make the struct callable
(vae::VAE)(X) = vae.decoder(vae.sampler(vae.encoder(X)))

# and make it trainable
Flux.@treelike VAE #encoder, decoder

"""
    KL(μ, σ2)

KL divergence between a normal distribution and unit gaussian.
"""
KL(μ, σ2) = Float(1/2)*mean(sum(σ2 + μ.^2 - log.(σ2) .- 1, dims = 1))

"""
    likelihood(X, μ, [σ2])

Likelihood of a sample X given mean and variance.
"""
likelihood(X, μ) = - mean(sum((μ - X).^2,dims = 1))/2
likelihood(X, μ, σ2) = - mean(sum((μ - X).^2 ./σ2 + log.(σ2),dims = 1))/2

"""
    mu(X)

Extract mean as the first horizontal half of X.
"""
mu(X) = X[1:Int(size(X,1)/2),:]

"""
    sigma2(X)

Extract sigma^2 as the second horizontal half of X. 
"""
sigma2(X) = softplus.(X[Int(size(X,1)/2+1):end,:]) .+ Float(1e-6)

"""
    logps(x)

Is the logarithm of the standard pdf of x.
"""
logps(x) = abs.(-1/2*x.^2 - 1/2*log(2*pi))

"""
    samplenormal(X)

Sample normal distribution with mean and sigma2 extracted from X.
"""
function samplenormal(X)
    μ, σ2 = mu(X), sigma2(X)
	ϵ = Float.(randn(size(μ)))
    # if cuarrays are loaded and X is on GPU, convert eps to GPU as well
    if iscuarray(μ.data)
    	ϵ = ϵ |> gpu
    end
    return μ .+  ϵ .* sqrt.(σ2)
end

"""
	VAE(esize, dsize; [activation, layer, variant])

Initialize a variational autoencoder with given encoder size and decoder size.

esize - vector of ints specifying the width anf number of layers of the encoder
\ndsize - size of decoder
\nactivation [Flux.relu] - arbitrary activation function
\nlayer [Flux.Dense] - type of layer
\nvariant [:unit] - :unit - output has unit variance
\n 		          - :sigma - the variance of the output is estimated
"""
function VAE(esize::Array{Int64,1}, dsize::Array{Int64,1}; activation = Flux.relu,
		layer = Flux.Dense, variant = :unit)
	@assert variant in [:unit, :sigma]
	@assert size(esize, 1) >= 3
	@assert size(dsize, 1) >= 3
	@assert esize[end] == 2*dsize[1]
	(variant==:unit) ? (@assert esize[1] == dsize[end]) :
		(@assert esize[1]*2 == dsize[end])

	# construct the encoder
	encoder = aelayerbuilder(esize, activation, layer)

	# construct the decoder
	decoder = aelayerbuilder(dsize, activation, layer)

	# finally construct the ae struct
	vae = VAE(encoder, samplenormal, decoder, variant)

	return vae
end

################
### training ###
################

"""
	KL(vae, X)

KL divergence between the encoder output and unit gaussian.
"""
function KL(vae::VAE, X) 
	ex = vae.encoder(X)
	KL(mu(ex), sigma2(ex))
end

"""
	likelihood(vae, X)

Likelihood of an autoencoded sample X.
"""
function likelihood(vae::VAE, X)
	if vae.variant == :unit
		μ = vae(X)
		return likelihood(X,μ)
	else	
		vx = vae(X)
		μ, σ2 = mu(vx), sigma2(vx)
		return likelihood(X,μ, σ2)
	end
end

"""
	likelihood(vae, X, M)

Likelihood of an autoencoded sample X sampled M times.
"""
likelihood(vae::VAE, X, M) = mean([likelihood(vae, X) for m in 1:M])

### anomaly score jako pxvita?
# muz, sigmaz = encoder(x)
# lognormal(x, decoder(muz), 1.0)

"""
	loss(vae, X, M, lambda)

Loss function of the variational autoencoder. Lambda is scaling parameter of
the KLD, 1 = full KL, 0 = no KL.
"""
loss(vae::VAE, X, M, lambda) = -likelihood(vae,X,M) + Float(lambda)*KL(vae, X)

"""
	evalloss(vae, X, M, lambda)

Print vae loss function values.
"""
function evalloss(vae::VAE, X, M, lambda) 
	l, lk, kl = getlosses(vae, X, M, lambda)
	print("loss: ", l,
	"\nlikelihood: ", lk,
	"\nKL: ", kl, "\n\n")
end

"""
	getlosses(vae, X, M, lambda)

Return the numeric values of current losses.
"""
getlosses(vae::VAE, X, M, lambda) = (
	Flux.Tracker.data(loss(vae, X, M, lambda)),
	Flux.Tracker.data(-likelihood(vae,X,M)),
	Flux.Tracker.data(KL(vae, X))
	)

"""
	fit!(vae, X, batchsize, [M, iterations, cbit, nepochs, 
	verb, lambda, rdelta, history, eta])

Trains the VAE neural net.

vae - a VAE object
\nX - data array with instances as columns
\nbatchsize - batchsize
\nM [1] - number of samples for likelihood
\niterations [1000] - number of iterations
\ncbit [200] - after this # of iterations, output is printed
\nnepochs [nothing] - if this is supplied, epoch training will be used instead of fixed iterations
\nverb [true] - if output should be produced
\nlambda [1] - scaling for the KLD loss
\nrdelta [Inf] - stopping condition for likelihood
\nhistory [nothing] - a dictionary for training progress control
\neta [eta] - learning rate
"""
function fit!(vae::VAE, X, batchsize; M=1, iterations=1000, cbit = 200, 
	nepochs = nothing, verb::Bool = true, lambda = 1,
	rdelta = Inf, history = nothing, eta = 0.001)
	# settings
	opt = ADAM(params(vae), eta)

	# sampler
	# sampler
	if nepochs == nothing
		sampler = UniformSampler(X,iterations,batchsize)
	else
		sampler = EpochSampler(X,nepochs,batchsize)
		cbit = sampler.epochsize
		iterations = nepochs*cbit
	end
	# it might be smaller than the original one if there is not enough data
	batchsize = sampler.batchsize 

	# using ProgressMeter 
	if verb
		p = Progress(iterations, 0.3)
		x = next!(sampler)
		reset!(sampler)
		_l, _lk, _kl = getlosses(vae, x, M, lambda)
	end

	# train
	for (i,x) in enumerate(sampler)
		# gradient computation and update
		l = loss(vae, x, M, lambda)
		Flux.Tracker.back!(l)
		opt()

		# progress
		if verb 
			if (i%cbit == 0 || i == 1)
				_l, _lk, _kl = getlosses(vae, x, M, lambda)
			end
			ProgressMeter.next!(p; showvalues = [(:loss,_l),(:likelihood, _lk),(:KL, _kl)])
		end

		# save actual iteration data
		if history != nothing
			track!(vae, history, x, M, lambda)
		end

		# if stopping condition is present
		if rdelta < Inf
			re = Flux.Tracker.data(-likelihood(vae, x))[1]
			if re < rdelta
				println("Training ended prematurely after $i iterations,\n",
					"likelihood $re < $rdelta")
				break
			end
		end
	end
end

"""
	track!(vae, history, X, M, lambda)

Save current progress.
"""
function track!(vae::VAE, history::MVHistory, X, M, lambda)
	l, lk, kl = getlosses(vae, X, M, lambda)
	push!(history, :loss, l)
	push!(history, :KLD, kl)
	push!(history, :likelihood, lk)
end
