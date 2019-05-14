using BSON
using FewShotAnomalyDetection
using Flux
using MLDataPattern
using PyPlot
using StatsBase
using EvalCurves
using AlfvenDetectors

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
inputdim = size(train[1],1)
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
s = 0.1 # width of imq kernel

##############################################
# SVAE itself
##############################################
model = AlfvenDetectors.SVAEMem(inputdim, hiddenDim, latentDim, numLayers, 
		memorySize, k, labelCount, α; nonlinearity=nonlinearity, layerType=layerType)
# first train the svae
AlfvenDetectors.fit!(model, train[1], batchSize, numBatches, β, s);
# then train the memory
numBatches = 500 # it will take a looong time
AlfvenDetectors.fit!(model, train[1], train[2], batchSize, numBatches, β, 1, loss_α, cbtime=20);

# get the anomaly score
as = AlfvenDetectors.as_logpxgivenz(model, test[1])
roc = EvalCurves.roccurve(as,test[2])
auc = EvalCurves.auc(roc...)
auc_svae = auc
figure()
title("AUC = $auc")
plot(roc...)
plot([0,1],[0,1],c="k", alpha=0.3)

# plot the latent space
u=zparams(model.svae,Z)[1].data
figure(figsize=(10,5))
subplot(121)
scatter(u[1,labels.==1],u[2,labels.==1],label="alfven",s=10)
legend()
subplot(122)
scatter(u[1,labels.==0],u[2,labels.==0],label="no alfven",s=10)
legend()

# also, compare it to other methods
# KNN
model_knn = AlfvenDetectors.KNN()
AlfvenDetectors.fit!(model_knn, train[1], train[2])
aucm = []
aucmw = []
kvec = collect(1:2:33)
for k in kvec
	asm = AlfvenDetectors.as_mean(model_knn, test[1], k)
	asmw = AlfvenDetectors.as_mean_weighted(model_knn, test[1], k)
	push!(aucm, EvalCurves.auc(EvalCurves.roccurve(asm,test[2])...))
	push!(aucmw, EvalCurves.auc(EvalCurves.roccurve(asmw,test[2])...))
end
figure()
plot(kvec, aucm, label="kNN AUC")
plot(kvec, aucmw, label="kNN AUC weighted")
xlabel("k")
ylabel("AUC")
legend()

# GMM
asfvec = ["as_max_ll_mse", "as_mean_ll_mse", "as_med_ll_mse", "as_ll_maxarg"]
Nclustvec = collect(2:12)
label = 1
auc_gmm = zeros(length(Nclustvec), length(asfvec))
for (i,Nclust) in enumerate(Nclustvec)
	model_gmm = AlfvenDetectors.GMMModel(Nclust; kind=:diag, method=:kmeans)
	AlfvenDetectors.fit!(model_gmm, hcat(train[1], test[1]))
	AlfvenDetectors.fit!(model_gmm, train[1], train[2], refit=false)
	for (j,asf) in enumerate(asfvec)
		as = eval(Meta.parse("AlfvenDetectors."*asf))(model_gmm, test[1], label)
		auc_gmm[i,j] = EvalCurves.auc(EvalCurves.roccurve(as,test[2])...)
	end
end
figure()
for (j,asf) in enumerate(asfvec)
	plot(Nclustvec, auc_gmm[:,j],label=asf)
end
xlabel("N components")
ylabel("AUC")
legend()
