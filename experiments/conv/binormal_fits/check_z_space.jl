using AlfvenDetectors
using Flux
using ValueHistories
using StatsBase
using GenerativeModels
using Dates
using BSON
using PyPlot

# now get some data
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
patchsize = 128
readfun = AlfvenDetectors.readnormlogupsd
shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()
patchdata = map(x->AlfvenDetectors.get_patch(datapath,x[1], x[2], x[3], patchsize, readfun;
	memorysafe = true)[1],	zip(shotnos, tstarts, fstarts))
data = cat(patchdata..., dims=4)

#
modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe"
subpath = "wae_binormal2/"
mpath = joinpath(modelpath, subpath) 
models = readdir(mpath)

# 
imodel = 3
model_data = BSON.load(joinpath(mpath,models[imodel]))
model = Flux.testmode!(model_data[:model])
exp_args = model_data[:experiment_args]
model_args = model_data[:model_args]
model_kwargs = model_data[:model_kwargs]
history = model_data[:history]

# plot training history
figure()
for key in keys(history)
	is,ls = get(history, key)
	plot(is,ls,label=string(key))
end
legend()
show()

# look at the Z space
Z_g = model.encoder(data).data
figure(figsize=(10,10))
subplot(221)
title("histogram of z space")
plt.hist2d(Z_g[1,:], Z_g[2,:],20)
subplot(222)
title("labeled z space")
scatter(Z_g[1,labels.==1], Z_g[2,labels.==1], label="alfven",s=5)
scatter(Z_g[1,labels.==0], Z_g[2,labels.==0], label="no alfven",s=5)
legend()
subplot(223)
title("histogram of alfven samples")
plt.hist2d(Z_g[1,labels.==1], Z_g[2,labels.==1],20)
subplot(224)
title("histogram of no alfven samples")
plt.hist2d(Z_g[1,labels.==0], Z_g[2,labels.==0],20)

# check some reconstructions
ipatch = 10
cmap = "plasma"
patch = data[:,:,:,ipatch:ipatch]
rpatch = model(patch).data
e = Flux.mse(patch, rpatch)
figure(figsize=(10,5))
subplot(121)
title("original")
pcolormesh(patch[:,:,1,1],cmap=cmap)
subplot(122)
title("reconstruction, error = $(e)")
pcolormesh(rpatch[:,:,1,1],cmap=cmap)

# check the reconstruction error progress
is, ls = get(history, :aeloss)
plot(ls[1000:end])