using AlfvenDetectors
using Flux
using ValueHistories
using ArgParse
using DelimitedFiles
using Random
using StatsBase
using GenModels
using CuArrays

# some settings, derived from ../conv/run_experiment.jl
warnings = true
iptrunc = "valid"
memorysafe = true # if true, slower but enables loading of more data - the Julia h5 library has a memory leak
readfun = AlfvenDetectors.readnormlogupsd
readfun = AlfvenDetectors.readlogupsd


seed = 1
α = 0.8 # training/all patches ratio
hostname = gethostname()
if hostname == "gpu-node"
	datapath = "/compass/Shared/Exchange/Havranek/Link to Alfven"
else
	datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
end
patchsize = 128

# get the patch info
all_info = AlfvenDetectors.labeled_patches();
shotnos, patch_labels, tstarts, fstarts = all_info;
N = length(shotnos)
positive_inds = (patch_labels .== 1);
N_positive = sum(positive_inds)
N_train = floor(Int, N_positive*α)
train_inds = sample(collect(1:N)[positive_inds], N_train);
test_inds = collect(1:N)[map(n->!(n in train_inds),1:N)];
# now get the patches
train_info = map(x->x[train_inds], all_info);
test_info = map(x->x[test_inds], all_info);
N_patches = 500
train_patches = AlfvenDetectors.collect_training_patches(datapath, train_info[1], train_info[3], 
	train_info[4], N_patches, readfun, patchsize; δ = 0.02, memorysafe = memorysafe);
train_data = train_patches[1] |> gpu
# these are for the final model valiadation
train_patches_validation = map(x->AlfvenDetectors.get_patch(datapath, x[1], x[2], x[3], patchsize, readfun; 
		memorysafe = memorysafe)[1], zip(train_info[1], train_info[3], train_info[4]));
train_patches_validation = cat(train_patches_validation..., dims=4)
test_patches = map(x->AlfvenDetectors.get_patch(datapath, x[1], x[2], x[3], patchsize, readfun; 
		memorysafe = memorysafe)[1], zip(test_info[1], test_info[3], test_info[4]));
N_test = length(test_patches)
test_patches = cat(test_patches..., dims=4)
test_positive_inds = test_info[2] .== 1
test_negative_inds = test_info[2] .== 0

x = test_patches[:,:,1,end]
maximum(x) # 0.6
minimum(x) # 0.1

# show some plots
using PyPlot
figure(figsize=(10,10))
for i in 1:9
	subplot(3,3,i)
	ip = sample(1:N_patches,1)[1]
	pcolormesh(train_patches[1][:,:,1,ip], cmap="plasma")
end

# construct the model
# 128 + upscale = 0.0076235738, 6:37
# 128 + transpose = 0.005280155, 5:42
# the precision seems to be the same, but one is a bit slower
# 256 + upscale = 0.0040311976, 4 mins on gpu-node
# 128 + upscale + nonnormalized  = 0.0046886955, 6:38
# but the negative vs positive discrimination sucks

insize = size(train_data)[1:3]
zdim = 1024 #   loss:       0.0039325454
#zdim = 128 #   loss:       0.0043666917
nconv = 2
kernelsize = 3
channels = (8, 16)
scaling = (2, 2)
#upscale_type = "transpose"
upscale_type = "upscale"
model = ConvAE(insize, zdim, nconv, kernelsize, channels, scaling; upscale_type = upscale_type) |> gpu

# and now train it
if hostname == "gpu-node"
	opt = GenModels.fit!(model, train_data, 50, 100; cbit = 5, memoryefficient = true);
	opt = GenModels.fit!(model, train_data, 50, 100; opt=opt, cbit = 5, memoryefficient = true);
else
	opt = GenModels.fit!(model, train_data, 25, 20; cbit = 5, memoryefficient = true);
	opt = GenModels.fit!(model, train_data, 25, 20; opt=opt, cbit = 5, memoryefficient = true);
	opt = GenModels.fit!(model, train_data, 25, 20; opt=opt, cbit = 5, memoryefficient = true);
end
cpu_model = model |> cpu

# save the model
if hostname == "gpu-node"
	modelpath = "/compass/home/skvara/alfven/experiments/oneclass/initial_test"
else
	modelpath = "/home/vit/vyzkum/alfven/experiments/oneclass/initial_test"
end
mkpath(modelpath)
modelname = "ConvAE"
fname = joinpath(modelpath, "$(modelname)_$(zdim)_$(upscale_type).bson")
GenModels.save_model(fname, model |> cpu, 
	modelname = modelname, 
	model_args = [
		:insize => insize, 
		:zdim => zdim, 
		:nconv => nconv, 
		:kernelsize => kernelsize, 
		:channels => channels, 
		:scaling => scaling
		],
	model_kwargs = Dict(
		:upscale_type => upscale_type
		))
#ml = GenModels.construct_model(fname)
## is it the same?
#tx = train_patches[1][:,:,:,1:1];
#cpu_model = model |> cpu 
#all(cpu_model(tx) .≈ ml(tx)) # yes

# compute the average reconstruction error on testing dataset
train_mse = Flux.mse(cpu_model(train_patches_validation), train_patches_validation).data
test_mse_positive = Flux.mse(cpu_model(test_patches[:,:,:,test_positive_inds]), test_patches[:,:,:,test_positive_inds]).data
test_mse_negative = Flux.mse(cpu_model(test_patches[:,:,:,test_negative_inds]), test_patches[:,:,:,test_negative_inds]).data
# maybe do not normalize the input?

# now check the reconstructions on the training and testing data set
figure(figsize=(10,10))
for i in 1:5
	ip = sample(1:N_patches,1)[1]
	x = train_patches[1][:,:,:,ip:ip]
	subplot(5,2,(i-1)*2+1)
	pcolormesh(x[:,:,1,1], cmap="plasma")
	subplot(5,2,i*2)
	pcolormesh(cpu_model(x).data[:,:,1,1], cmap="plasma")
end

figure(figsize=(10,10))
for i in 1:5
	ip = sample(collect(1:N_test)[test_positive_inds],1)[1]
	x = test_patches[:,:,:,ip:ip]
	subplot(5,2,(i-1)*2+1)
	pcolormesh(x[:,:,1,1], cmap="plasma")
	subplot(5,2,i*2)
	pcolormesh(cpu_model(x).data[:,:,1,1], cmap="plasma")
end
