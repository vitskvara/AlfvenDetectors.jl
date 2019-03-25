"""
	VAE{encoder, sampler, decoder, variant}

Flux-like structure for the variational autoencoder.
"""
struct VAE <: FluxModel
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
	VAE(esize, dsize; [activation, layer, variant])

Initialize a variational autoencoder with given encoder size and decoder size.

esize - vector of ints specifying the width anf number of layers of the encoder
dsize - size of decoder
activation [Flux.relu] - arbitrary activation function
layer [Flux.Dense] - type of layer
variant [:unit] 
	:unit - output has unit variance
	:scalar - a scalar variance of the output is estimated
	:diag - the diagonal of covariance of the output is estimated
"""
function VAE(esize::Array{Int64,1}, dsize::Array{Int64,1}; activation = Flux.relu,
		layer = Flux.Dense, variant = :unit)
	@assert variant in [:unit, :diag, :scalar]
	@assert size(esize, 1) >= 3
	@assert size(dsize, 1) >= 3
	@assert esize[end] == 2*dsize[1]
	(variant==:unit) ? (@assert esize[1] == dsize[end]) :
		((variant==:diag) ? (@assert esize[1]*2 == dsize[end]) :
			(@assert esize[1] + 1 == dsize[end]) )

	# construct the encoder
	encoder = aelayerbuilder(esize, activation, layer)

	# construct the decoder
	decoder = aelayerbuilder(dsize, activation, layer)

	# finally construct the ae struct
	vae = VAE(encoder, samplenormal, decoder, variant)

	return vae
end

"""
	VAE(xdim, zdim, nlayers; [activation, layer, variant])

Initialize a variational autoencoder given input and latent dimension 
and numberof layers. The width of layers is linearly interpolated 
between xdim and zdim.

	xdim = input size
	zdim = code size
	nlayers = number of layers
	activation [Flux.relu] = arbitrary activation function
	layer [Flux.Dense] = layer type
	variant [:unit] 
		:unit - output has unit variance
		:scalar - a scalar variance of the output is estimated
		:diag - the diagonal of covariance of the output is estimated
"""
function VAE(xdim::Int, zdim::Int, nlayers::Int; activation = Flux.relu,
		layer = Flux.Dense, variant = :unit)
	@assert nlayers >= 2

	esize = ceil.(Int, range(xdim, zdim, length=nlayers+1))
	dsize = reverse(esize)
	esize[end] = esize[end]*2
	if variant == :scalar
		dsize[end] = dsize[end] + 1
	elseif variant == :diag
		dsize[end] = dsize[end]*2
	end

	VAE(esize,dsize; activation=activation, layer=layer, variant=variant)
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
	loglikelihood(vae, X)

Loglikelihood of an autoencoded sample X.
"""
function loglikelihood(vae::VAE, X)
	if vae.variant == :unit
		μ = vae(X)
		return loglikelihoodopt(X,μ)
	elseif vae.variant == :scalar
		vx = vae(X)
		μ, σ2 = mu_scalarvar(vx), sigma2_scalarvar(vx)
		return loglikelihoodopt(X,μ, σ2)
	elseif vae.variant == :diag
		vx = vae(X)
		μ, σ2 = mu(vx), sigma2(vx)
		return loglikelihoodopt(X,μ, σ2)
	end
end

"""
	loglikelihood(vae, X, L)

Loglikelihood of an autoencoded sample X sampled L times.
"""
loglikelihood(vae::VAE, X, L) = StatsBase.mean([loglikelihood(vae, X) for m in 1:L])

### anomaly score jako pxvita?
# muz, sigmaz = encoder(x)
# lognormal(x, decoder(muz), 1.0)

"""
	loss(vae, X, L, β)

Loss function of the variational autoencoder. β is scaling parameter of
the KLD, 1 = full KL, 0 = no KL.
"""
loss(vae::VAE, X, L, β) = -loglikelihood(vae,X,L) + Float(β)*KL(vae, X)

"""
	evalloss(vae, X, L, β)

Print vae loss function values.
"""
function evalloss(vae::VAE, X, L, β) 
	l, lk, kl = getlosses(vae, X, L, β)
	print("total loss: ", l,
	"\n-loglikelihood: ", lk,
	"\nKL: ", kl, "\n\n")
end

"""
	getlosses(vae, X, L, β)

Return the numeric values of current losses.
"""
getlosses(vae::VAE, X, L, β) = (
	Flux.Tracker.data(loss(vae, X, L, β)),
	Flux.Tracker.data(-loglikelihood(vae,X,L)),
	Flux.Tracker.data(KL(vae, X))
	)

"""
	track!(vae, history, X, L, β)

Save current progress.
"""
function track!(vae::VAE, history::MVHistory, X, L, β)
	l, lk, kl = getlosses(vae, X, L, β)
	push!(history, :loss, l)
	push!(history, :loglikelihood, lk)
	push!(history, :KL, kl)
end

########### callback #################

"""
	(cb::basic_callback)(m::VAE, d, l, opt, L::Int, β::Real)

Callback for the train! function.
TODO: stopping condition, change learning rate.
"""
function (cb::basic_callback)(m::VAE, d, l, opt, L::Int, β::Real)
	# update iteration count
	cb.iter_counter += 1
	# save training progress to a MVHistory
	if cb.history != nothing
		track!(m, cb.history, d, L, β)
	end
	# verbal output
	if cb.verb 
		# if first iteration or a progress print iteration
		# recalculate the shown values
		if (cb.iter_counter%cb.show_it == 0 || cb.iter_counter == 1)
			ls = getlosses(m, d, L, β)
			cb.progress_vals = Array{Any,1}()
			push!(cb.progress_vals, ceil(Int, cb.iter_counter/cb.epoch_size))
			push!(cb.progress_vals, cb.iter_counter%cb.epoch_size)
			push!(cb.progress_vals, ls[1])
			push!(cb.progress_vals, ls[2])
			push!(cb.progress_vals, ls[3])
		end
		# now give them to the progress bar object
		ProgressMeter.next!(cb.progress; showvalues = [
			(:epoch,cb.progress_vals[1]),
			(:iteration,cb.progress_vals[2]),
			(:loss,cb.progress_vals[3]),
			(Symbol("-loglikelihood"),cb.progress_vals[4]),
			(:KL,cb.progress_vals[5])
			])
	end
end

"""
	fit!(vae::VAE, X, batchsize::Int, nepochs::Int; 
		L=1, β::Real= Float(1.0), cbit::Int=200, history = nothing, 
		verb::Bool = true, η = 0.001, runtype = "experimental", 
		[usegpu, memoryefficient])

Trains the VAE neural net.

vae - a VAE object
X - data array with instances as columns
batchsize - batchsize
nepochs - number of epochs
L [1] - number of samples for likelihood
β [1.0] - scaling for the KLD loss
cbit [200] - after this # of iterations, progress is updated
history [nothing] - a dictionary for training progress control
verb [true] - if output should be produced
η [0.001] - learning rate
runtype ["experimental"] - if fast is selected, no output and no history is written
usegpu - if X is not already on gpu, this will put the inidvidual batches into gpu memory rather 
		than all data at once
memoryefficient - calls gc after every batch, again saving some memory but prolonging computation
"""
function fit!(vae::VAE, X, batchsize::Int, nepochs::Int; 
	L=1, β::Real= Float(1.0), cbit::Int=200, history = nothing, 
	verb::Bool = true, η = 0.001, runtype = "experimental", trainkwargs...)
	@assert runtype in ["experimental", "fast"]
	# sampler
	sampler = EpochSampler(X,nepochs,batchsize)
	epochsize = sampler.epochsize
	# it might be smaller than the original one if there is not enough data
	batchsize = sampler.batchsize 

	# loss
	# use default loss

	# optimizer
	opt = ADAM(η)

	# callback
	if runtype == "experimental"
		cb = basic_callback(history,verb,η,cbit; 
			train_length = nepochs*epochsize,
			epoch_size = epochsize)
		_cb(m::VAE,d,l,o) =  cb(m,d,l,o,L,β)
	elseif runtype == "fast"
		_cb = fast_callback 
	end

	# train
	train!(
		vae,
		collect(sampler),
		x->loss(vae,x,L,β),
		opt,
		_cb;
		trainkwargs...
		)
end

##### auxiliarry functions #####
"""
	sample(vae::VAE, [M::Int])

Get samples generated by the VAE.
"""
function StatsBase.sample(vae::VAE)
	if vae.variant == :unit
		X = vae.decoder(randn(Float, size(vae.decoder.layers[1].W,2)))
	elseif vae.variant == :scalar
		X = samplenormal_scalarvar(vae.decoder(randn(Float, size(vae.decoder.layers[1].W,2)))) 
		X = reshape(X, size(X,1))
	elseif vae.variant == :diag
		X = samplenormal(vae.decoder(randn(Float, size(vae.decoder.layers[1].W,2))))
		X = reshape(X, size(X,1))
	end
	return X
end
function StatsBase.sample(vae::VAE, M::Int)
	if vae.variant == :unit
		return vae.decoder(randn(Float, size(vae.decoder.layers[1].W,2),M))
	elseif vae.variant == :scalar
		return samplenormal_scalarvar(vae.decoder(randn(Float, size(vae.decoder.layers[1].W,2),M)))
	elseif vae.variant == :diag
		return samplenormal(vae.decoder(randn(Float, size(vae.decoder.layers[1].W,2),M)))
	end
end
