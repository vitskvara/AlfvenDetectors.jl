using AlfvenDetectors
using Flux
using ValueHistories
using StatsBase
using GenerativeModels
using Dates
using BSON
using PyPlot
using GaussianMixtures
using EvalCurves

# now get some data
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
patchsize = 128
readfun = AlfvenDetectors.readnormlogupsd
shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()
patchdata = map(x->AlfvenDetectors.get_patch(datapath,x[1], x[2], x[3], patchsize, readfun;
	memorysafe = true)[1],	zip(shotnos, tstarts, fstarts))
data = cat(patchdata..., dims=4)

# split them into test and train
a = 0.6
X = copy(data)
N = size(X,4)
shuffle_inds = sample(1:N, N, replace=false) 
X = X[:,:,:,shuffle_inds]
Y = labels[shuffle_inds]
trN = floor(Int, N*a)
(trX,trY) = X[:,:,:,1:trN], Y[1:trN]
(tstX,tstY) = X[:,:,:,trN+1:end], Y[trN+1:end]

# get the model 
mf = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/wae_binormal/ConvWAE_channels-[8,16]_patchsize-128_nepochs-200_2019-05-01T21:00:27.121.bson"
model_data = BSON.load(mf)
model = Flux.testmode!(model_data[:model])
exp_args = model_data[:experiment_args]
model_args = model_data[:model_args]
model_kwargs = model_data[:model_kwargs]
history = model_data[:history]

# create the latent space representations
trZ = model.encoder(trX).data
figure(figsize=(10,10))
subplot(221)
title("histogram of z space")
plt.hist2d(trZ[1,:], trZ[2,:],20)
subplot(222)
title("labeled z space")
scatter(trZ[1,trY.==1], trZ[2,trY.==1], label="alfven",s=5)
scatter(trZ[1,trY.==0], trZ[2,trY.==0], label="no alfven",s=5)
legend()
subplot(223)
title("histogram of alfven samples")
plt.hist2d(trZ[1,trY.==1], trZ[2,trY.==1],20)
subplot(224)
title("histogram of no alfven samples")
plt.hist2d(trZ[1,trY.==0], trZ[2,trY.==0],20)

# now fit the 2 component GMM to the Z space representations
trZt = Array(trZ')
kind = :diag
gmm_model = GaussianMixtures.GMM(2, trZt, kind=kind)

# now the anomaly score is going to be the loglikelihood of the datapoint given the first component
tstZ = model.encoder(tstX).data
tstZt = Array(tstZ')
llh = llpg(gmm_model,tstZt)
as = llh[:,1]

roc = EvalCurves.roccurve(as, tstY)
auc = EvalCurves.auc(roc...)
figure()
title("AUC = $auc")
plot(roc...)
plot([0,1],[0,1],c="k", alpha=0.3)