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


