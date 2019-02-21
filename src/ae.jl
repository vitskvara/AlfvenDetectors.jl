##########################
### ae NN construction ###
##########################

"""
	AE{encoder, sampler, decoder}

Flux-like structure for the basic autoencoder.
"""
struct AE{E, D}
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


mutable struct basic_callback
	history
	iter_counter::Int
end

function (cb::basic_callback)(m::AE, d, l, opt)
	if cb.history != nothing
		track!(m, cb.history, d)
	end
end

"""
	fit!(m::AE, X, batchsize::Int, nepochs::Int; 
	cbit::Int=200, history = nothing, verb = true, eta = 0.001)

Fit an autoencoder.
"""
function fit!(m::AE, X, batchsize::Int, nepochs::Int; 
	cbit::Int=200, history = nothing, verb = true, eta = 0.001)
	# optimizer
	opt = ADAM(eta)

	# sampler
	sampler = EpochSampler(X,nepochs,batchsize)
	epochsize = sampler.epochsize
	# it might be smaller than the original one if there is not enough data
	batchsize = sampler.batchsize 

	# loss
	# specified as an anonymous function in the call
	#loss(x) = loss(m, x[2]) # since first element of x is the index from enumerate

	# callback
	cb = basic_callback(history,1)

	train!(
		m,
		collect(sampler), 
		x->loss(m, x), 
		opt, 
		cb
		)
end


# to be deprecated
"""
	fit!(ae, X, batchsize, [ cbit, nepochs, verb, rdelta, history, eta])

Trains the AE.

	ae = AE type object
	X = data array with instances as columns
	batchsize = batchsize
	iterations [1000] = number of iterations
	cbit [200] = after this # of iterations, output is printed
	nepochs [nothing] = if this is supplied, epoch training will be used instead of fixed iterations
	verb [true] = if output should be produced
	rdelta [Inf] = stopping condition for reconstruction error
	history [nothing] = MVHistory() to be filled with data of individual iterations
	eta [0.001] = learning rate
"""
function _fit!(ae::AE, X, batchsize::Int, nepochs::Int; cbit::Int = 200,
	verb = true, rdelta = Inf, history = nothing, eta = 0.001)
	# optimizer
	opt = ADAM(params(ae), eta)

	# sampler
	sampler = EpochSampler(X,nepochs,batchsize)
	epochsize = sampler.epochsize

	# it might be smaller than the original one if there is not enough data
	batchsize = sampler.batchsize 

	# using ProgressMeter 
	if verb
		p = Progress(nepochs*sampler.epochsize, 0.3)
		# get data for intial loss value
		x = next!(sampler)
		reset!(sampler)
		_l = getlosses(ae, x)
		_e = 1
		_i = 1
	end
	

	# training
	for (i,x) in enumerate(sampler)
		# gradient computation and update
		l = loss(ae, x)
		Flux.Tracker.back!(l)
		opt()

		# progress
		if verb 
			if (i%cbit == 0 || i == 1)
				_l = getlosses(ae, x)
				_e = ceil(Int, i/epochsize)
				_i = i%epochsize
			end
			ProgressMeter.next!(p; showvalues = [
				(:epoch,_e),
				(:iteration,_i),
				(:loss,_l[1])
				])
		end

		# save loss data
		if history != nothing
			track!(ae, history, x)
		end

		# if stopping condition is present
		if rdelta < Inf
			re = Flux.Tracker.data(l)[1]
			if re < rdelta
				if verb
					println("Training ended prematurely after $i iterations,",
						"reconstruction error $re < $rdelta")
				end
				break
			end
		end
	end	
end

