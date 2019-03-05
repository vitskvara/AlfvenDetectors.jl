"""
	TSVAE{encoder, sampler, decoder, variant}

Flux-like structure for the two-stage variational autoencoder.
"""
struct TSVAE <: FluxModel
	m1::VAE
	m2::VAE
end

(tsvae::TSVAE)(X) = tsvae.m1(X)

# is this necesarry?
Flux.@treelike TSVAE #encoder, decoder

"""
	TSVAE(m1size, m2size; [activation, layer, variant])

Initialize a variational autoencoder with given encoder size and decoder size.

m1size - a tuple of two vectors specifying the size of model 1
m2size - a tuple of two vectors specifying the size of model 2
activation [Flux.relu] - arbitrary activation function
layer [Flux.Dense] - type of layer
"""
function TSVAE(m1size::AbstractVector, m2size::AbstractVector; 
	activation = Flux.relu,	layer = Flux.Dense)
	@assert m1size[2][1] == m2size[1][1]

	# construct models
	m1 = VAE(m1size[1], m1size[2], activation=activation, layer=layer,
		variant = :scalar)
	m2 = VAE(m2size[1], m2size[2], activation=activation, layer=layer,
		variant = :scalar)
	
	return TSVAE(m1,m2)
end

"""
	TSVAE(xdim::Int, zdim::Int, nlayers::Union{Int, Tuple}; 
	activation = Flux.relu,	layer = Flux.Dense)

A lightweight constructor for TSVAE.
"""
function TSVAE(xdim::Int, zdim::Int, nlayers::Union{Int, Tuple}; 
	activation = Flux.relu,	layer = Flux.Dense)
	# if nlayers is scalar (both nets are to be the same depth)
	# create a tuple anyway
	nlayers = scalar2tuple(nlayers)

	m1 = VAE(xdim, zdim, nlayers[1]; activation=activation, layer=layer,
		variant=:scalar)
	m2 = VAE(zdim, zdim, nlayers[2]; activation=activation, layer=layer,
		variant=:scalar)
	return TSVAE(m1,m2)
end

"""
	getlosses(tsvae, X, L, β)

Return the numeric values of current losses as a tuple.
"""
function getlosses(tsvae::TSVAE, X, L, β) 
	m1ls = getlosses(tsvae.m1, X, L, β)
	Z = tsvae.m1.sampler(tsvae.m1.encoder(X))
	m2ls = getlosses(tsvae.m2, Z, L, β)
	return m1ls, m2ls
end

function fit!(tsvae::TSVAE, X, batchsize::Union{Int, Tuple}, 
	nepochs::Union{Int, Tuple}; 
	L::Union{Int, Tuple}=(1,1), β::Union{Real, Tuple}= Float(1.0), 
	cbit::Union{Int, Tuple}=(200,200), history = nothing, 
	verb::Bool = true, η::Union{Real, Tuple} = (0.001,0.001), 
	runtype = "experimental")
	@assert runtype in ["experimental", "fast"]

	# some argument may be scalar or a tuple
	# this will convert them all to tuples
	batchsize = scalar2tuple(batchsize)
	nepochs = scalar2tuple(nepochs)
	L = scalar2tuple(L)
	β = scalar2tuple(β)
	cbit = scalar2tuple(cbit)
	history = scalar2tuple(history)
	η = scalar2tuple(η)

	if verb
		println("Training the first stage...")
	end
	fit!(tsvae.m1, X, batchsize[1], nepochs[1]; L = L[1], β = β[1], cbit=cbit[1],
		history = history[1], verb = verb, η = η[1], runtype = runtype)
	
	if verb
		println("Training the second stage...")
	end
	Z = tsvae.m1.sampler(tsvae.m1.encoder(X).data)
	fit!(tsvae.m2, Z, batchsize[2], nepochs[2]; L = L[2], β = β[2], cbit=cbit[2],
		history = history[2], verb = verb, η = η[2], runtype = runtype)
end
