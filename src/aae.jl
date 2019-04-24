"""
	AAE{encoder, decoder, discirminator, pz}

Flux-like structure for the basic autoencoder.
"""
struct AAE{E, FE, DE, DS, FDS, PZ} <: FluxModel
	encoder::E 
	f_encoder::FE # frozen encoder copy
	decoder::DE
	discriminator::DS
	f_discriminator::FDS # frozen discriminator copy
	pz::PZ
end
AAE(e, de, ds, pz) = AAE(e, freeze(e), de, ds, freeze(ds), pz) # default constructor 

# make the struct callable
(aae::AAE)(X) = aae.decoder(aae.encoder(X))

# and make it trainable
Flux.@treelike AAE

AAE

"""
	AAE(esize, decsize, dissize; [activation, layer])

Initialize an adversarial autoencoder.

	esize = vector of ints specifying the width anf number of layers of the encoder
	decsize = size of decoder
	dissize = size of discriminator
	activation [Flux.relu] = arbitrary activation function
	layer [Flux.Dense] = layer type
"""
function AAE(esize::Array{Int64,1}, decsize::Array{Int64,1}, dissize::Array{Int64,1}; 
	pz = randn, activation = Flux.relu,	layer = Flux.Dense)
	@assert size(esize, 1) >= 3
	@assert size(decsize, 1) >= 3
	@assert size(dissize, 1) >= 3
	@assert esize[end] == decsize[1] 
	@assert esize[1] == decsize[end]
	@assert esize[end] == dissize[1]
	@assert 1 == dissize[end]	

	# construct the encoder
	encoder = aelayerbuilder(esize, activation, layer)

	# construct the decoder
	decoder = aelayerbuilder(decsize, activation, layer)

	# cinstruct the discriminator
	discriminator = discriminatorbuilder(dissize, activation, layer)

	# finally construct the ae struct
	aae = AAE(encoder, decoder, discriminator, n->pz(Float,esize[end],n))

	return aae
end

"""
	AAE(xdim, zdim, ae_nlayers, disc_nlayers; [activation, layer])

Initialize an adversarial autoencoder given input and latent dimension 
and number of layers. The width of layers is linearly interpolated 
between xdim and zdim.

	xdim = input size
	zdim = code size
	ae_nlayers = number of layers of the autoencoder
	disc_nlayers = number of layers of the discriminator
	activation [Flux.relu] = arbitrary activation function
	layer [Flux.Dense] = layer type
"""
function AAE(xdim::Int, zdim::Int, ae_nlayers::Int, disc_nlayers::Int; 
	pz = randn, activation = Flux.relu, layer = Flux.Dense)
	@assert ae_nlayers >= 2
	@assert disc_nlayers >= 2

	# this will create the integer array specifying the width of individual layers using linear interpolations
	esize = ceil.(Int, range(xdim, zdim, length=ae_nlayers+1))
	decsize = reverse(esize)
	dissize = ceil.(Int, range(zdim, 1, length=disc_nlayers+1))

	# finally return the structure
	AAE(esize,decsize, dissize; pz=pz, activation=activation, layer=layer)
end

################
### training ###
################

"""
	aeloss(AAE, X)

Autoencoder loss.
"""
aeloss(aae::AAE,X) = Flux.mse(X,aae(X))

"""
	dloss(AAE,X[,Z])

Discriminator loss given code Z and original sample X. If Z not given, 
it is autoamtically generated using the prescribed pz.
"""
function dloss(aae::AAE,X,Z=nothing) 
	if Z == nothing
		Z = aae.pz(size(X,2))
	end
	return - half*(mean(log.(aae.discriminator(Z) .+ eps(Float))) + 
		mean(log.(1 .- aae.discriminator(aae.f_encoder(X)) .+ eps(Float))))
end

"""
	gloss(AAE,X)

Encoder/generator loss.
"""
gloss(aae::AAE,X) = - mean(log.(aae.f_discriminator(aae.encoder(X)) .+ eps(Float)))

"""
	loss(AAE,X)

Adversarial autoencoder loss (MSE).
"""
loss(aae::AAE,X) = aeloss(aae,X)

"""
	getlosses(AAE, X, Z)

Return the numeric values of current losses.
"""
getlosses(aae::AAE, X, Z) =  (
		Flux.Tracker.data(aeloss(aae,X)),
		Flux.Tracker.data(dloss(aae,X,Z)),
		Flux.Tracker.data(gloss(aae,X))
		)

"""
	getlosses(AAE, X)

Return the numeric values of current losses.
"""
getlosses(aae::AAE, X) = getlosses(aae::AAE, X, aae.pz(size(X,2)))

"""
	evalloss(AAE, X[, Z])

Print AAE losses.
"""
function evalloss(aae::AAE, X, Z=nothing) 
	ael, dl, gl = (Z == nothing) ?  getlosses(aae, X) : getlosses(aae, X, Z)
	print("autoencoder loss: ", l,
	"\ndiscriminator loss: ", dl,
	"\ngenerator loss: ", gl, "\n\n")
end

"""
	getlsize(AAE)

Return size of the latent code.
"""
getlsize(aae::AAE) = size(aae.encoder.layers[end].W,1)

"""
	track!(AAE, history, X)

Save current progress.
"""
function track!(aae::AAE, history::MVHistory, X)
	ael, dl, gl = getlosses(aae, X)
	push!(history, :aeloss, ael)
	push!(history, :dloss, dl)
	push!(history, :gloss, gl)
end

"""
	(cb::basic_callback)(AAE, d, l, opt)

Callback for the train! function.
TODO: stopping condition, change learning rate.
"""
function (cb::basic_callback)(m::VAE, d, l, opt)
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
			push!(cb.progress_vals, cb.iter_counter)
			push!(cb.progress_vals, ls[1])
			push!(cb.progress_vals, ls[2])
			push!(cb.progress_vals, ls[3])
		end
		# now give them to the progress bar object
		ProgressMeter.next!(cb.progress; showvalues = [
			(:epoch,cb.progress_vals[1]),
			(:iteration,cb.progress_vals[2]),
			(:aeloss,cb.progress_vals[3]),
			(:dloss,cb.progress_vals[4]),
			(:gloss,cb.progress_vals[5])
			])
	end
end

