using AlfvenDetectors
using Flux
using StatsBase

readfun = AlfvenDetectors.readnormlogupsd
datapath = "/home/vit/vyzkum/alfven/cdb_data/data_sample"
shots = readdir(datapath)
shots = filter(x-> any(map(y -> occursin(y,x), 
	["10370", "10514", "10800", "10866", "10870", "10893"])), 
	shots)
shots = joinpath.(datapath, shots)
patchsize = 64
data = AlfvenDetectors.collect_conv_signals(shots, readfun, patchsize)
m,n,c,k = size(data)
insize = (m,n,c)
latentdim = 64
nconv = 4
kernelsize = 3
channels = (32,32,64,64)
scaling = 2
model = AlfvenDetectors.ConvVAE(insize, latentdim, nconv, kernelsize, channels, scaling)
batchsize = 32
batch = data[:,:,:,StatsBase.sample(1:k,batchsize,replace=false)]
L = AlfvenDetectors.loss(model,batch,1,1.0)
