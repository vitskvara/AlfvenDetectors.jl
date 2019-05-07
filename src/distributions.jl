"""
	binormal([T], m[, n])

Generate n samples from double gaussian m-dimensional distribution.
"""
function binormal(T::DataType,m::Int)
	i = rand(1:2)
	μ = T.([-1.0f0, 1.0f0])
	σ = T(0.1)
	return randn(T,m)*σ .+ μ[i]
end
binormal(m::Int) = binormal(Float32,m)
binormal(T::DataType,m::Int,n::Int) = hcat(map(x->binormal(T,m),1:n)...)
binormal(m::Int,n::Int) = binormal(Float32,m,n)
binormal_gpu(T::DataType,m::Int,n::Int) = binormal(T,m,n) |> gpu

"""
	quadnormal([T], m[, n])

Generate n samples from four-component gaussian m-dimensional mixture.
"""
function quadnormal(T::DataType,m::Int)
	@assert m>=2
	i = rand(1:4)
	μ = T.(hcat(fill(-1.0,m), fill(-1.0,m), fill(1.0,m), fill(1.0,m)))
	μ[1,1] = 1.0
	μ[1,3] = -1.0
	σ = T(0.1)
	return randn(T,m)*σ .+ μ[:,i]
end
quadnormal(m::Int) = quadnormal(Float32,m)
quadnormal(T::DataType,m::Int,n::Int) = hcat(map(x->quadnormal(T,m),1:n)...)
quadnormal(m::Int,n::Int) = quadnormal(Float32,m,n)
quadnormal_gpu(T::DataType,m::Int,n::Int) = quadnormal(T,m,n) |> gpu

"""
	GM{M,S,N,W,G}

Gaussian-mixture structure.
"""
mutable struct GM{M,S,N,W,G}
	μ::M
	σ::S
	w::W
	n::N
	gpu::G
end

"""
	GM(M,S[,W][;gpu])

Constructor with means, sigmas and weights.
"""
GM(M::AbstractVector,S::AbstractVector,W::AbstractVector;gpu=false) = GM(M,S,Weights(W),length(M),gpu)
GM(M::AbstractVector,S::AbstractVector;gpu=false) = GM(M,S,fill(1/length(M),length(M)),gpu=gpu)

"""
	sample(GM[, T], m[, n])

Sample from a Gaussian Mixture.
"""
function sample(gm::GM,T::DataType)
	i = sample(1:gm.n,gm.w,1)[1]
	return randn(T,size(gm.μ[1],1))*T.(gm.σ[i]) + T.(gm.μ[i])
end
sample(gm::GM,T::DataType,n::Int) = hcat(map(x->sample(gm,T),1:n)...)
sample(gm::GM,n::Int) = sample(gm,Float32,n)
# this is basically useless but is still good to have to be compatible with the rest of the code
sample(gm::GM,T::DataType,m::Int,n::Int) = sample(gm,T,n)
sample_gpu(gm::GM,T::DataType,m::Int,n::Int) = sample(gm,T,m,n) |> gpu

(gm::GM)(T::DataType, m::Int, n::Int) = gm.gpu ? sample_gpu(gm, T, m, n) : sample(gm, T, m, n)

"""
	cubecorners(m)

Returns all the possible combinations for corners of an m-dimensional cube of side length 2.
Is very slow for m>=10.
"""
function cubecorners(m::Int)
	corners_stack = fill(1.0f0,m)
	corners = copy(corners_stack)
	for i in 1:m
		corners_stack[i] = -1.0f0
		corners = cat(corners, unique([x for x in Combinatorics.permutations(corners_stack)])..., dims=2) 
	end
	return corners
end
"""
	cubecorners_rand(m,n[;seed])

Returns n randomly selected corners of a m-dimensional cube of side length 2.
"""
function cubecorners_rand(m::Int,n::Int;seed=nothing)
	@assert n<=m^2
	(seed == nothing) ? nothing : Random.seed!(seed)
	corners = []
	corner_candidate = collect(sample([-1f0,1f0],m))
	for i in 1:n
		while (corner_candidate in corners)
			corner_candidate = collect(sample([-1f0,1f0],m))
		end
		push!(corners, corner_candidate)
	end
	Random.seed!()
	return cat(corners..., dims=2)
end

"""
	cubeGM(m,n[,σ,weights][; seed,gpu])

Returns a Gaussian Mixture object with means randomly selected as corners of a cube.

	m = dimensionality of the space
	n = number of components <= m^2
	σ = standard deviation of the components (vector or scalar)
	weights = weights of the components
	seed = for pseudorandom initialization
	gpu = should the generated numbers be on gpu?
"""
function cubeGM(m::Int,n::Int,σ::AbstractVector,weights::AbstractVector; seed=nothing, gpu=false)
	μ = cubecorners_rand(m,n;seed=seed)
	μ = [μ[:,i] for i in 1:n] 
	return GM(μ,σ,weights;gpu=gpu)
end
cubeGM(m::Int,n::Int,σ::AbstractVector; kwargs...) = cubeGM(m,n,σ,fill(1/n,n);kwargs...)
cubeGM(m::Int,n::Int,σ::Real,weights::AbstractVector; kwargs...) = cubeGM(m,n,fill(σ,n),weights;kwargs...)
cubeGM(m::Int,n::Int,σ::Real; kwargs...) = cubeGM(m,n,σ,fill(1/n,n);kwargs...)
cubeGM(m::Int,n::Int; kwargs...) = cubeGM(m,n,0.1f0;kwargs...)
