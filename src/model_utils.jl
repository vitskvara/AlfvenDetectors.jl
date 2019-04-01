const l2pi = Float(log(2*pi)) # the model converges the same with zero or correct value
const δ = Float(1e-6)
const half = Float(0.5)

"""
    KL(μ, σ2)

KL divergence between a normal distribution and unit gaussian.
"""
KL(μ::Real, σ2::Real) = Float(1/2)*(σ2 + μ^2 - log(σ2) .- Float(1.0))
KL(μ, σ2) = Float(1/2)*StatsBase.mean(sum(σ2 + μ.^2 - log.(σ2) .- Float(1.0), dims = 1))
# have this for the general full covariance matrix normal distribution?

"""
    loglikelihood(X, μ, [σ2])

Loglikelihood of a normal sample X given mean and variance.
"""
loglikelihood(X::Real, μ::Real) = - ((μ - X)^2 + l2pi)*half
loglikelihood(X::Real, μ::Real, σ2::Real) = - ((μ - X)^2/σ2 + log(σ2) + l2pi)*half
loglikelihood(X, μ) = - StatsBase.mean(sum((μ - X).^2 .+ l2pi,dims = 1))*half
loglikelihood(X, μ, σ2) = - StatsBase.mean(sum((μ - X).^2 ./σ2 + log.(σ2) .+ l2pi,dims = 1))*half
# in order to work on gpu and for faster backpropagation, dont use .+ here for arrays
# see also https://github.com/FluxML/Flux.jl/issues/385
function loglikelihood(X::AbstractMatrix, μ::AbstractMatrix, σ2::AbstractVector) 
    # again, this has to be split otherwise it is very slow
    y = (μ - X).^2
    y = (one(Float) ./σ2)' .* y 
    - StatsBase.mean(sum( y .+ reshape(log.(σ2), 1, length(σ2)) .+ l2pi,dims = 1))*half
end

"""
    loglikelihoodopt(X, μ, [σ2])

Loglikelihood of a normal sample X given mean and variance without the constant term. For
optimalization the results is the same and this is faster.
"""
loglikelihoodopt(X::Real, μ::Real) = - ((μ - X)^2)*half
loglikelihoodopt(X::Real, μ::Real, σ2::Real) = - ((μ - X)^2/σ2 + log(σ2))*half
loglikelihoodopt(X, μ) = - StatsBase.mean(sum((μ - X).^2,dims = 1))*half
loglikelihoodopt(X, μ, σ2) = - StatsBase.mean(sum( (μ - X).^2 ./σ2 + log.(σ2),dims = 1))*half
function loglikelihoodopt(X::AbstractArray{T,4}, μ::AbstractArray{T,4}, σ2::AbstractArray{T,4}) where T
    y = (μ - X).^2
    y = y ./ σ2
    - StatsBase.mean(sum( y .+ log.(σ2),dims = 1))*half
end
# in order to work on gpu and for faster backpropagation, dont use .+ here
# see also https://github.com/FluxML/Flux.jl/issues/385
function loglikelihoodopt(X::AbstractMatrix, μ::AbstractMatrix, σ2::AbstractVector) 
    # again, this has to be split otherwise it is very slow
    y = (μ - X).^2
    y = (one(Float) ./σ2)' .* y 
    - StatsBase.mean(sum( y .+ reshape(log.(σ2), 1, length(σ2)),dims = 1))*half
end

"""
    mu(X)

Extract mean as the first horizontal half of X.
"""
mu(X) = X[1:Int(size(X,1)/2),:]
mu(X::AbstractArray{T,4}) where T = X[:,:,1:Int(size(X,3)/2),:]

"""
    mu_scalarvar(X)

Extract mean as all but the last rows of X.
"""
mu_scalarvar(X) = X[1:end-1,:]
mu_scalarvar(X::AbstractArray{T,4}) where T = X[:,:,1:Int(size(X,3)/2),:]

"""
    sigma2(X)

Extract sigma^2 as the second horizontal half of X. 
"""
sigma2(X) = softplus.(X[Int(size(X,1)/2+1):end,:]) .+ δ
sigma2(X::AbstractArray{T,4}) where T = softplus.(X[:,:,Int(size(X,3)/2+1):end,:]) .+ δ

"""
    sigma2_scalarvar(X)

Extract sigma^2 as the last row of X. 
"""
sigma2_scalarvar(X) = softplus.(X[end,:]) .+ δ
sigma2_scalarvar(X::AbstractArray{T,4}) where T = StatsBase.mean(softplus.(X[:,:,Int(size(X,3)/2+1):end,:]) .+ δ, dims=[1,2])

"""
   samplenormal(μ, σ2)

Sample  a normal distribution with given mean and standard deviation.
"""
function samplenormal(μ, σ2)
    ϵ = randn(Float, size(μ))
    # if cuarrays are loaded and X is on GPU, convert eps to GPU as well
    if iscuarray(μ)
        ϵ = ϵ |> gpu
    end
    return μ +  ϵ .* sqrt.(σ2)
end
function samplenormal(μ::AbstractMatrix, σ2::AbstractVector)
    ϵ = randn(Float, size(μ))
    # if cuarrays are loaded and X is on GPU, convert eps to GPU as well
    if iscuarray(μ)
        ϵ = ϵ |> gpu
    end
    return μ +  sqrt.(σ2)' .* ϵ  
end

"""
    samplenormal(X)

Sample normal distribution with mean and sigma2 extracted from X.
"""
function samplenormal(X)
    μ, σ2 = mu(X), sigma2(X)
    return samplenormal(μ, σ2)
end

"""
   samplenormal_scalarvar(X)

Sample normal distribution from X where variance is the last row. 
"""
function samplenormal_scalarvar(X)
    μ, σ2 = mu_scalarvar(X), sigma2_scalarvar(X)
    return samplenormal(μ, σ2)
end

"""
   scalar2tuple(x)

If x is scalar, return a tuple containing (x,deepcopy(x)). 
"""
function scalar2vec(x)
    if x == nothing
        return Array{Any,1}([nothing,nothing])
    elseif length(x) == 1
        return Array{Any,1}([x,deepcopy(x)])
    end
    return x
end