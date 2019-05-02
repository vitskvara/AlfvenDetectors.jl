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
