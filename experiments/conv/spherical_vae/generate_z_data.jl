using AlfvenDetectors
using BSON
using PyPlot
using GenModels
using ValueHistories
using Flux
using Random
using StatsBase

file = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/data_augmentation/ConvAE_channels-[8,16,32,64]_patchsize-128_nepochs-250_2019-05-01T05:45:25.604.bson"
outfile = "zdata.bson"
model_data = BSON.load(file)
model = Flux.testmode!(model_data[:model])
patchsize = 128
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
# collect unlabeled patches
available_shots = readdir(datapath)
training_shots = AlfvenDetectors.select_training_shots(10, available_shots; seed=1, use_alfven_shots=true)
testing_shots = filter(x->!any(occursin.(x,training_shots)),available_shots)
Ntesting_shots = 10
Random.seed!(1)
testing_shot_inds = sample(1:length(testing_shots), Ntesting_shots, replace=false)
Random.seed!()
shots = joinpath.(datapath, testing_shots[testing_shot_inds])
readfun = AlfvenDetectors.readnormlogupsd
data = AlfvenDetectors.collect_conv_signals(shots, readfun, patchsize; memorysafe=true)
N_unlabeled = size(data,4)
# then collect the few labeled patches
shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()
shotnos = shotnos[labels.==1]
tstarts = tstarts[labels.==1]
fstarts = fstarts[labels.==1]
labels = labels[labels.==1]
patchdata = map(x->AlfvenDetectors.get_patch(datapath, x[1], x[2], x[3], patchsize,
	readfun; memorysafe = true)[1], zip(shotnos, tstarts, fstarts))
patchdata = cat(patchdata..., dims=4)
#concat it all together
data = cat(data, patchdata, dims=4)
Ndata = size(data,4)
# finally, compute the z representations
batchsize = 64
Z = []
for i in 1:ceil(Int,Ndata/batchsize)
	batch = data[:,:,:,((i-1)*batchsize+1):min(i*batchsize,Ndata)]
	z = model.encoder(batch).data
	push!(Z, z)
end
Z = hcat(Z...)
labels = vcat(fill(0,N_unlabeled),labels)
BSON.bson(outfile, Z = Z, labels = labels)
