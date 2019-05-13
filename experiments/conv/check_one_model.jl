using AlfvenDetectors
using Flux
using ValueHistories
using StatsBase
using GenerativeModels
using Dates
using BSON
#using PyPlot
using Plots
plotly()
#using CuArrays

# now get some data
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
patchsize = 128
patch_f = joinpath(dirname(pathof(AlfvenDetectors)), "../experiments/conv/data/labeled_patches_$patchsize.bson")
if isfile(patch_f)
	patchdata = BSON.load(patch_f)
	data = patchdata[:data];
	shotnos = patchdata[:shotnos];
	labels = patchdata[:labels];
	tstarts = patchdata[:tstarts];
	fstarts = patchdata[:fstarts];
else
	readfun = AlfvenDetectors.readnormlogupsd
	shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()
	patchdata = map(x->AlfvenDetectors.get_patch(datapath,x[1], x[2], x[3], patchsize, readfun;
		memorysafe = true)[1],	zip(shotnos, tstarts, fstarts))
	data = cat(patchdata..., dims=4)
end

#
modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe"
subpath = "waae_64_8_16_32_32_lambda-10_sigma-1_cube-8/1"
mpath = joinpath(modelpath, subpath) 
models = readdir(mpath)
#imode = 46
imodel = 50
mf = joinpath(mpath,models[imodel])

# or load it directly
#mf = "/home/vit/.julia/environments/v1.1/dev/AlfvenDetectors/experiments/conv/ConvAAE_channels-[2,2]_patchsize-128_nepochs-10_2019-05-06T10:03:25.287.bson"
#mf = "./ConvWAE_channels-[2,2]_patchsize-128_nepochs-2_2019-05-07T09:16:59.027.bson"

# 
model_data = BSON.load(mf)
exp_args = model_data[:experiment_args]
model_args = model_data[:model_args]
model_kwargs = model_data[:model_kwargs]
history = model_data[:history]
if haskey(model_data, :model)
	model = model_data[:model]
else
	model = Flux.testmode!(GenerativeModels.construct_model(mf))
end

# plot training history
for key in keys(history)
	is,ls = get(history, key)
	plot!(is,ls,label=string(key)*"=$(ls[end])")
end
title!("")

# look at the Z space
Z_pt = model.pz(1000)
batchsize = 128
Z_g = GenerativeModels.encode(model, data, batchsize).data
if size(Z_g,1) == 3
	scatter(Z_pt[1,:],Z_pt[2,:],Z_pt[3,:], label="samples from pz")
	scatter!(Z_g[1,:],Z_g[2,:],Z_g[3,:], label="encoded data")
else
	scatter(Z_pt[1,:],Z_pt[2,:], label="samples from pz")
	scatter!(Z_g[1,:],Z_g[2,:], label="encoded data")
end

if size(Z_g,1) == 3
	scatter(Z_g[1,labels.==1],Z_g[2,labels.==1],Z_g[3,labels.==1], label="alfven data")
	scatter!(Z_g[1,labels.==0],Z_g[2,labels.==0],Z_g[3,labels.==0], label="no alfven data")
else
	scatter(Z_g[1,labels.==1],Z_g[2,labels.==1], label="alfven data")
	scatter!(Z_g[1,labels.==0],Z_g[2,labels.==0], label="no alfven data")
end
plot()

















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