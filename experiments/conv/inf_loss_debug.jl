# replicate this behaviour
# julia run_experiment.jl VAE 64 2 2 2 --batchnorm --gpu --memorysafe --memory-efficient 
# --eta=0.001 --nepochs=200 --savepoint=50 --savepath=vae_large --beta=0.01 --batchsize=128
using AlfvenDetectors
using Flux
using ValueHistories
using ArgParse
using DelimitedFiles
using Random
using StatsBase
using GenModels

modelname = "ConvVAE"
ldim = 32
nlayers = 2
channels = [2, 2]
patchsize = 128
kernelsize = [3,3]
scaling = [2,2]
nshots = 10
noalfven = false
measurement_type = "uprobe"
usegpu = true
batchsize = 256
batchnorm = true
outbatchnorm = false
resblock = false
eta = 0.0001
beta = 0.001
optimiser = "RMSProp"
vae_variant = :scalar
outer_nepochs = 50
inner_nepochs = 1
warnings = false
memoryefficient = true
iptrunc = "valid"
svpth = "."
savepoint = 10
memorysafe = true
positive_patch_ratio = 0f0
seed = 1
readfun = AlfvenDetectors.readnormlogupsd
ndense = 1
if usegpu
	using CuArrays
end
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
savepath = "/home/vit/vyzkum/alfven/experiments/conv/$measurement_type"

println("Loading basic training data...\n")
available_shots = readdir(datapath)
training_shots = AlfvenDetectors.select_training_shots(nshots, available_shots; seed=seed, use_alfven_shots=!noalfven)
println("Using $(training_shots)\n")
shots = joinpath.(datapath, training_shots)
data = AlfvenDetectors.collect_conv_signals(shots, readfun, patchsize; 
	warns=warnings, type=iptrunc, memorysafe=memorysafe)

xdim = size(data)
model_args = [
		:xdim => xdim[1:3],
		:ldim => ldim, 
		:nlayers => nlayers,
		:kernelsize => kernelsize,
		:channels => channels,
		:scaling => scaling
	]
model_kwargs = Dict{Symbol, Any}(
	:batchnorm => batchnorm,
	:outbatchnorm => outbatchnorm,
	:ndense => ndense
	)
fit_kwargs = Dict{Symbol, Any}(
		:usegpu => usegpu,
		:memoryefficient => memoryefficient
	)
# model-specific arguments
if occursin("VAE", modelname)
	model_kwargs[:variant] = vae_variant
	fit_kwargs[:beta] = beta
end


filename_kwargs = Dict(
	:patchsize => patchsize,
	:channels => "["*reduce((x,y)->"$(x),$(y)",channels)*"]",
	:nepochs => outer_nepochs*inner_nepochs
	)
filename = AlfvenDetectors.create_filename(modelname, [], Dict(), Dict(), 
	filename_kwargs...)
# create the model
model = GenModels.construct_model(modelname, [x[2] for x in model_args]...; model_kwargs...)
model, history, t = AlfvenDetectors.fitsave_unsupervised(data, model, batchsize, 
	outer_nepochs, inner_nepochs, model_args, model_kwargs, fit_kwargs, savepath; 
	modelname = "GenModels."*modelname, optname=optimiser, eta=eta, 
	usegpu=usegpu, savepoint=savepoint, filename=filename, experiment_args=Dict())





























using DelimitedFiles
using DelimitedFiles
X = reshape(readdlm("X.txt"), patchsize, patchsize, 1, batchsize)
m = reshape(readdlm("m.txt"), patchsize, patchsize, 1, batchsize)
s = reshape(readdlm("s.txt"), 1, 1, 1, batchsize)

function loglikelihood(X::AbstractArray{T,4}, m::AbstractArray{T,4}, s::AbstractArray{T,4}) where T
	half = 0.5f0
    y = (m - X).^2
    y = y ./ s
    y = y .+ log.(s)
    y = sum(y,dims = 1)
    - StatsBase.mean(y)*half
end
