##########################
### ae NN construction ###
##########################

"""
	AE{encoder, sampler, decoder}

Flux-like structure for the basic autoencoder.
"""
struct AE{E, D} <: FluxModel
	encoder::E
	decoder::D
end

# make the struct callable
(ae::AE)(X) = ae.decoder(ae.encoder(X))

# and make it trainable
Flux.@treelike AE

"""
	AE(esize, dsize; [activation, layer])

Initialize an autoencoder with given encoder size and decoder size.

	esize = vector of ints specifying the width anf number of layers of the encoder
	dsize = size of decoder
	activation [Flux.relu] = arbitrary activation function
	layer [Flux.Dense] = layer type
"""
function AE(esize::Array{Int64,1}, dsize::Array{Int64,1}; activation = Flux.relu,
		layer = Flux.Dense)
	@assert size(esize, 1) >= 3
	@assert size(dsize, 1) >= 3
	@assert esize[end] == dsize[1] 
	@assert esize[1] == dsize[end]

	# construct the encoder
	encoder = aelayerbuilder(esize, activation, layer)

	# construct the decoder
	decoder = aelayerbuilder(dsize, activation, layer)

	# finally construct the ae struct
	ae = AE(encoder, decoder)

	return ae
end

"""
	AE(xdim, zdim, nlayers; [activation, layer])

Initialize an autoencoder given input and latent dimension and number
of layers. The width of layers is linearly interpolated between
xdim and zdim.

	xdim = input size
	zdim = code size
	nlayers = number of layers
	activation [Flux.relu] = arbitrary activation function
	layer [Flux.Dense] = layer type
"""
function AE(xdim::Int, zdim::Int, nlayers::Int; activation = Flux.relu,
		layer = Flux.Dense)
	@assert nlayers >= 2

	esize = ceil.(Int, range(xdim, zdim, length=nlayers+1))
	dsize = reverse(esize)

	AE(esize,dsize; activation=activation, layer=layer)
end

################
### training ###
################

"""
	loss(ae, X)

Reconstruction error.
"""
loss(ae::AE, X) = Flux.mse(ae(X), X)

"""
	evalloss(ae, X)

Print ae loss function values.	
"""
evalloss(ae::AE, X) = println("loss: ", getlosses(ae,X)[1], "")

"""
	getlosses(ae, X)

Return the numeric values of current losses.
"""
getlosses(ae::AE, X) = (
	Flux.Tracker.data(loss(ae, X))
	)

"""
	track!(m, history, X)

Save current progress.
"""
function track!(m::AE, history::MVHistory, X)
	push!(history, :loss, Flux.Tracker.data(loss(m,X)))
end

"""
	(cb::basic_callback)(m::AE, d, l, opt)

Callback for the train! function.
TODO: stopping condition, change learning rate.
"""
function (cb::basic_callback)(m::AE, d, l, opt)
	# update iteration count
	cb.iter_counter += 1
	# save training progress to a MVHistory
	if cb.history != nothing
		track!(m, cb.history, d)
	end
	# verbal output
	if cb.verb 
		# if first iteration or a progress print iteration
		# recalculate the shown values
		if (cb.iter_counter%cb.show_it == 0 || cb.iter_counter == 1)
			ls = getlosses(m, d)
			cb.progress_vals = Array{Any,1}()
			push!(cb.progress_vals, ceil(Int, cb.iter_counter/cb.epoch_size))
			push!(cb.progress_vals, cb.iter_counter%cb.epoch_size)
			push!(cb.progress_vals, ls[1])			
		end
		# now give them to the progress bar object
		ProgressMeter.next!(cb.progress; showvalues = [
			(:epoch,cb.progress_vals[1]),
			(:iteration,cb.progress_vals[2]),
			(:loss,cb.progress_vals[3])
			])
	end
end

"""
	fit!(m::AE, X, batchsize::Int, nepochs::Int; 
	cbit::Int=200, history = nothing, verb = true, eta = 0.001,
	runtype = "experimental")

Fit an autoencoder.
"""
function fit!(m::AE, X, batchsize::Int, nepochs::Int; 
	cbit::Int=200, history = nothing, verb = true, eta = 0.001,
	runtype = "experimental")
	@assert runtype in ["experimental", "fast"]
	# sampler
	sampler = EpochSampler(X,nepochs,batchsize)
	epochsize = sampler.epochsize
	# it might be smaller than the original one if there is not enough data
	batchsize = sampler.batchsize 

	# loss
	# specified as an anonymous function in the call
	#loss(x) = loss(m, x[2]) # since first element of x is the index from enumerate

	# optimizer
	opt = ADAM(eta)
	
	# callback
	if runtype == "experimental"
		cb = basic_callback(history,verb,eta,cbit; 
			train_length = nepochs*epochsize,
			epoch_size = epochsize)
	elseif runtype == "fast"
		cb = fast_callback 
	end

	train!(
		m,
		collect(sampler), 
		x->loss(m, x), 
		opt, 
		cb
		)
end
