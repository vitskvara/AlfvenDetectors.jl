using AlfvenDetectors
using EvalCurves
using PyPlot
using GenModels
using ValueHistories
using JLD2
using FileIO
using Flux
using StatsBase
using LinearAlgebra

# get the paths
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
trainpath = "/home/vit/vyzkum/alfven/cdb_data/oneclass_data/training/128_normalized"
modelpath = "/home/vit/vyzkum/alfven/experiments/oneclass/first_runs/models"
models = readdir(modelpath)
mf = joinpath(modelpath, models[end])

# load a model
model = GenModels.construct_model(mf)
params = AlfvenDetectors.parse_params(mf)

# get testing data
seed = 1
patchsize = 128
readfun = AlfvenDetectors.readnormlogupsd
testing_data = AlfvenDetectors.collect_testing_data_oneclass(datapath, readfun, patchsize; seed=seed);
labels = 1 .- testing_data[3]; # switch the labels here - positive class is actually the normal one
patches = testing_data[1];
positive_patches = patches[:,:,:,labels.==1];
negative_patches = patches[:,:,:,labels.==0];

# get training data
training_data = load(joinpath(trainpath, "seed-$(seed).jld2"))

# eval what we need
# auc at test, mse at test 0/1, mse at train, auc at test with jacodeco
# means and sds of these over seed iterations
mse(model,x) = Flux.mse(x, model(x)).data
mse(model,x,M) = StatsBase.mean([mse(model, x[:,:,:,((i-1)*M+1):min(i*M, size(x,4))]) for i in 1:ceil(Int, size(x,4)/M)])

# get the mses
M = 100
train_mse = mse(model, training_data["patches"][:,:,:,1:1000], M)
test1_mse = mse(model, positive_patches, M)
test0_mse = mse(model, negative_patches, M)
test_mse = mse(model, patches, M)

# get auc based on plain mse
score_mse(model, x) = map(i->mse(model, x[:,:,:,i:i]), 1:size(x,4))
auc(model,x,y,sf) = EvalCurves.auc(EvalCurves.roccurve(sf(model,x), y)...)
auc_mse = auc(model, patches, labels, score_mse)

# show some patches
i = rand(1:10000)
X = training_data["patches"][:,:,:,i:i]
figure()
subplot(1,2,1)
pcolormesh(X[:,:,1,1])
subplot(1,2,2)
pcolormesh(model(X)[:,:,1,1].data)


# jacodeco
log_normal(x, μ, σ2::T, d::Int) where {T<:Real}  = (d > 0) ? - sum((@. ((x - μ)^2) ./ σ2), dims=1)/2 .- d * (log.(σ2) + log(2π) )/2 : 0
log_normal(x::AbstractArray{T}) where T = log_normal(x, zeros(T,size(x)), T(1), size(x,1))
function LinearAlgebra.logabsdet(f, x::Vector)    
    Σ = Flux.data(Flux.Tracker.jacobian(f, x))
    S = svd(Σ)
    mask = [S.S .!= 0]
    2*sum(log.(abs.(S.S[mask])))
end
nobs(x::AbstractArray{T,4}) where T = size(x,4)
nobs(x::AbstractArray{T,2}) where T = size(x,2)
logpx(m, x, z) = Flux.mse(x, m.decoder(z))
jacodeco(model, x::Vector, z::Vector) = sum(log_normal(z)) + sum(logpx(model, x, z)) - logabsdet(model.decoder, z)
function jacodeco(m, x, z)
    @assert nobs(x) == nobs(z)
    [jacodeco(m, getobs(x, i), getobs(z, i)) for i in 1:nobs(x)]
end
jacodeco(m, x) = jacodeco(m, x, m.g(x))
Flux.Tracker.jacobian(model.decoder, randn(128)) # this does not work

