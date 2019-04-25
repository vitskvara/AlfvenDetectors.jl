using Flux
using PyPlot
using AlfvenDetectors
using StatsBase
using Random
using ValueHistories

# artificial data
m1 = [-1.0;-1.0]
m2 = [1.0;1.0]
v1 = v2 = 0.1
N1 = N2 = 50
N = N1+N2
M = 2
X = hcat(randn(M,N1)*v1 .+ m1, randn(M,N2)*v2 .+ m2)

# construct the generating distribution
function binormal(T::DataType,m::Int)
	i = rand(1:2)
	μ = T.([-1.0f0, 1.0f0])
	σ = T(0.1)
	return randn(T,m)*σ .+ μ[i]
end
binormal(m::Int) = binormal(Float32,m)
binormal(T::DataType,m::Int,n::Int) = hcat(map(x->binormal(T,m),1:n)...)
binormal(m::Int,n::Int) = binormal(Float32,m,n)

# define the parts of the network
ldim = 1
hdim = 50
nonlinearity = Flux.relu
model = AlfvenDetectors.AAE(M, ldim, 3, 3, binormal, hdim = hdim, activation=nonlinearity)
hist = MVHistory()
AlfvenDetectors.fit!(model, X, 50, 2000, history=hist,verb=true)

rX = model(X).data
figure(figsize=(10,5))
subplot(1,2,1)
title("original data and reconstructions")
scatter(X[1,:],X[2,:])
scatter(rX[1,1:N1],rX[2,1:N1],c="r")
scatter(rX[1,N1+1:end],rX[2,N1+1:end],c="g")

Z = model.encoder(X).data
subplot(1,2,2)
title("distribution of latent code")
if ldim == 1
	plt.hist(vec(Z[1:N1]),20,color="r")
	plt.hist(vec(Z[N1+1:end]),20,color="g")
else
	plt.hist2d(Z[1,:],Z[2,:],32)
end
show()

# now lets do the diagonal gaussian example
M = 2
N = 200
s = Float32.([30 10; 10 1])
X = s*randn(Float32,M,N)
figure()
scatter(X[1,:],X[2,:])
show()

# construct and train the model
ldim = 1
hdim = 50
nonlinearity = Flux.relu
model = AlfvenDetectors.AAE(M, ldim, 3, 3, randn, hdim = hdim, activation=nonlinearity)
hist = MVHistory()
AlfvenDetectors.fit!(model, X, 50, 2000, history=hist,verb=true);

rX = model(X).data
figure(figsize=(10,5))
subplot(1,2,1)
title("original data and reconstructions")
scatter(X[1,:],X[2,:])
scatter(rX[1,:],rX[2,:])

Z = model.encoder(X).data
subplot(1,2,2)
title("distribution of latent code")
if ldim == 1
	plt.hist(vec(Z),32)
else
	plt.hist2d(Z[1,:],Z[2,:],32)
end
show()
