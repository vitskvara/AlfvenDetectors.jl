const l2pi = Float.(log(2*pi)) # the model converges the same with zero or correct value
#const l2pi = Float.(0.0)

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
loglikelihood(X::Real, μ::Real) = - ((μ - X)^2 + l2pi)/2
loglikelihood(X::Real, μ::Real, σ2::Real) = - ((μ - X)^2/σ2 + log(σ2) + l2pi)/2
loglikelihood(X, μ) = - StatsBase.mean(sum((μ - X).^2 .+ l2pi,dims = 1))/2
loglikelihood(X, μ, σ2) = - StatsBase.mean(sum((μ - X).^2 ./σ2 .+ log.(σ2) .+ l2pi,dims = 1))/2

"""
    loglikelihoodopt(X, μ, [σ2])

Loglikelihood of a normal sample X given mean and variance without the constant term. For
optimalization the results is the same and this is faster.
"""
loglikelihoodopt(X::Real, μ::Real) = - ((μ - X)^2)/2
loglikelihoodopt(X::Real, μ::Real, σ2::Real) = - ((μ - X)^2/σ2 + log(σ2))/2
loglikelihoodopt(X, μ) = - StatsBase.mean(sum((μ - X).^2,dims = 1))/2
loglikelihoodopt(X, μ, σ2) = - StatsBase.mean(sum((μ - X).^2 ./σ2 .+ log.(σ2),dims = 1))/2
# maybe define a different behaviour for vectors and matrices?

"""
    mu(X)

Extract mean as the first horizontal half of X.
"""
mu(X) = X[1:Int(size(X,1)/2),:]

"""
    sigma2(X)

Extract sigma^2 as the second horizontal half of X. 
"""
sigma2(X) = softplus.(X[Int(size(X,1)/2+1):end,:]) .+ Float(1e-6)

"""
    samplenormal(X)

Sample normal distribution with mean and sigma2 extracted from X.
"""
function samplenormal(X)
    μ, σ2 = mu(X), sigma2(X)
	ϵ = randn(Float, size(μ))
    # if cuarrays are loaded and X is on GPU, convert eps to GPU as well
    if iscuarray(μ)
    	ϵ = ϵ |> gpu
    end
    return μ .+  ϵ .* sqrt.(σ2)
end
