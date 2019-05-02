using BSON
using FewShotAnomalyDetection
using Flux
using MLDataPattern
using PyPlot
using StatsBase
using EvalCurves

# extract the data
infile = "zdata.bson"
data = BSON.load(infile)
labels = data[:labels]
Z = data[:Z]

# now split the data into training and testing parts
a = 0.8
Z0 = Z[:,labels.==0]
Z1 = Z[:,labels.==1]
N0 = size(Z0,2)
N1 = size(Z1,2)
Z0 = Z0[:,sample(1:N0,N0,replace=false)]
Z1 = Z1[:,sample(1:N1,N1,replace=false)]
trN0 = floor(Int,N0*a)
trN1 = floor(Int,N1*a) 
trZ, trY = hcat(Z0[:,1:trN0], Z1[:,1:trN1]), vcat(fill(0,trN0), fill(1,trN1))
tstZ, tstY = hcat(Z0[:,trN0+1:end], Z1[:,trN1+1:end]), vcat(fill(0,N0-trN0), fill(1,N1-trN1))
train = (trZ, trY)
test = (tstZ, tstY)

# setup the svae
hiddenDim = 32
latentDim = 2
numLayers = 3
nonlinearity = "relu"
layerType = "Dense"
β = 0.1 # ratio between reconstruction error and the distance between p(z) and q(z)
α = 0.1 # threshold in the memory that does not matter to us at the moment!
loss_α = 0.1 # importance ratio between anomalies and normal data in mem_loss
memorySize = 256
k = 256
labelCount = 1

batchSize = 128
numBatches = 10000

##############################################
# SVAE itself
##############################################
inputdim = size(train[1], 1)
svae = SVAEbase(inputdim, hiddenDim, latentDim, numLayers, nonlinearity, layerType)
mem = KNNmemory{Float32}(memorySize, inputdim, k, labelCount, (x) -> zparams(svae, x)[1], α)

# Basic Wasserstein loss to train the svae on unlabelled data
σ = 0.1 # width of imq kernel
trainRepresentation(data) = wloss(svae, data, β, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, σ))
# inserts the data into the memory
remember(data, labels) = trainQuery!(mem, data, labels)
# Expects anomalies in the data with correct label (some of them
trainWithAnomalies(data, labels) = FewShotAnomalyDetection.mem_wloss(svae, mem, data, labels, β, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, 1), loss_α)

# Unsupervised learning
opt = Flux.Optimise.ADAM(1e-5)
cb = Flux.throttle(() -> println("SVAE: $(trainRepresentation(train[1]))"), 5)
# there is a hack with RandomBatches because so far I can't manage to get them to work without the tuple - I have to find a different sampling iterator
Flux.train!((x) -> trainRepresentation(getobs(x)), Flux.params(svae), RandomBatches((train[1],), size = batchSize, count = numBatches), opt, cb = cb)
println("Train err: $(trainRepresentation(train[1])) vs test error: $(trainRepresentation(test[1]))")

# Adding stuff into the memory
remember(train[1], train[2])

numBatches = 1000 # it will take a looong time

# learn with labels
cb = Flux.throttle(() -> println("SVAE mem loss: $(trainWithAnomalies(train[1], train[2]))"), 60)
# there is a hack with RandomBatches because so far I can't manage to get them to work without the tuple - I have to find a different sampling iterator
Flux.train!(trainWithAnomalies, Flux.params(svae), RandomBatches((train[1], train[2]), size = batchSize, count = numBatches), opt, cb = cb)

# get the anomaly score
as=vec(-FewShotAnomalyDetection.pxexpectedz(svae, test[1]))
roc = EvalCurves.roccurve(as,test[2])
auc = EvalCurves.auc(roc...)
figure()
title("AUC = $auc")
plot(roc...)
plot([0,1],[0,1],c="k", alpha=0.3)

# plot the latent space
u=zparams(svae,Z)[1].data
figure(figsize=(10,5))
subplot(121)
scatter(u[1,labels.==1],u[2,labels.==1],label="alfven",s=10)
legend()
subplot(122)
scatter(u[1,labels.==0],u[2,labels.==0],label="no alfven",s=10)
legend()
