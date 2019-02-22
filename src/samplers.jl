"""
	UniformSampler

A uniformly distributed sampler from a given data (Matrix).

Fields:

	data = original data
	M = number of rows (features)
	N = number of columns (samples)
	niter = how many iterations
	batchsize = how many samples in iteration
	iter = iteration counter
	replace = sample with replacement?
"""
mutable struct UniformSampler
	data
	M
	N
	niter
	batchsize
	iter
	replace
end

"""
	checkbatchsize(N,batchsize,replace)

Chaecks if batchsize is not larger than number of samples if replace = false.
"""
function checkbatchsize(N,batchsize,replace)
	if batchsize > N && !replace
		@warn "batchsize too large, setting to $N"
		batchsize = N
	end
	return batchsize
end

"""
	UniformSampler(X::Matrix, niter::Int, batchsize::Int; replace = false)

A standard constructor.
"""
function UniformSampler(X::Matrix, niter::Int, batchsize::Int; replace = false)
	M,N = size(X)
	batchsize = checkbatchsize(N,batchsize,replace)
	return UniformSampler(X,M,N,niter,batchsize,0, replace)
end

"""
	next!(s::UniformSampler)

Returns next batch.
"""
function next!(s::UniformSampler)
	if s.iter < s.niter
		s.iter += 1
		return s.data[:,sample(1:s.N,s.batchsize,replace=s.replace)]
	else
		return nothing
	end
end

"""
	reset!(s::UniformSampler)

Set iteration counter to zero.
"""
function reset!(s::UniformSampler)
	s.iter = 0
end

"""
	enumerate(s::UniformSampler)

Returns an iterable over indices and batches.
"""
function enumerate(s::UniformSampler)
	return [(i,next!(s)) for i in 1:s.niter]
end

"""
	EpochSampler

Sample in batches that cover the entire dataset for a given number of epochs.

Fields:

	data = original data matrix
	M = number of rows (features)
	N = number of columns (samples)
	nepochs = how many epochs
	epochsize = how many iterations are in an epoch
	batchsize = how many samples in iteration
	iter = iteration counter
	buffer = list of indices yet unused in the current epoch
"""
mutable struct EpochSampler
	data
	M
	N
	nepochs
	epochsize
	batchsize
	iter
	buffer
end

"""
	EpochSampler(X, nepochs::Int, batchsize::Int)

Default constructor.
"""
function EpochSampler(X, nepochs::Int, batchsize::Int)
	M,N = size(X) 
	batchsize = checkbatchsize(N,batchsize,false)
	return EpochSampler(X,M,N,nepochs,Int(ceil(N/batchsize)),batchsize,0,
		sample(1:N,N,replace = false))
end

"""
	next!(s::EpochSampler)

Returns the next batch.
"""
function next!(s::EpochSampler)
	if s.iter < s.nepochs
		L = length(s.buffer)
		if  L > s.batchsize
			inds = s.buffer[1:s.batchsize]
			s.buffer = s.buffer[s.batchsize+1:end]
		else
			inds = s.buffer
			# reshuffle the indices again
			s.buffer = sample(1:s.N,s.N,replace = false)
			s.iter += 1
		end
		# using views is very memory efficient, however it slows down training and is unusable for GPU
		# return @views s.data[:,inds]
		return s.data[:,inds]
	else
		return nothing
	end
end

"""
	collect(s::EpochSampler)

Colect all samples in an array.
"""
function collect(s::EpochSampler)
	return [next!(s) for i in 1:(s.nepochs*s.epochsize)]
end

"""
	enumerate(s::EpochSampler)

Returns an iterable over indices and batches.
"""
function enumerate(s::EpochSampler)
	return [(i,next!(s)) for i in 1:(s.nepochs*s.epochsize)]
end

"""
	enumerate(s::UniformSampler)

Returns an iterable over indices and batches.
"""
function reset!(s::EpochSampler)
	s.iter = 0
	s.buffer = sample(1:s.N,s.N,replace=false)
end
