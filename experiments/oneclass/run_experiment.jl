using AlfvenDetectors
using Flux
using ValueHistories
using ArgParse
using DelimitedFiles
using Random
using StatsBase
using GenModels
using CuArrays

# init - via argparse
# get the model name, zdim, nlayers, channels, kernelsize, batchsize, learning rate, beta, gamma, lambda, 
# savepath, seed, niter, anomaly score
# collect the training and testing data - do the shifts, add noise
# run and save the model
# use argparse to extract the command line arguments
# name of algorithm, usegpu, latentdim, no of layers, coils?
# maybe also number of epochs and batchsize?
s = ArgParseSettings()
@add_arg_table s begin
    "modelname"
		required = true
        help = "one of [AE, VAE, TSVAE, WAE, AAE, WAAE]"
    "ldimsize"
    	required = true
    	arg_type = Int
    	help = "size of latent dimension"
    "nlayers"
    	required = true
    	arg_type = Int
    	help = "number of layers"
    "channels"
    	required = true
    	arg_type = Int
    	help = "a list of channel numbers"
    	nargs = '+'
    "--patchsize"
    	default = 128
    	arg_type= Int
    	help = "size of image patch"
    "--kernelsize"
    	default = [3]
    	help = "a scalar or a vector of kernelsizes"
    	arg_type = Int
    	nargs = '+'
    "--scaling"
    	default = [2]
    	help = "a scalar or a vector of scaling factors"
    	arg_type = Int
    	nargs = '+'
    "--npatches"
    	default = 10000
    	arg_type = Int
    	help = "number of patches used for training"
    "--nshots"
    	default = 100
    	arg_type = Int
    	help = "number of shots used for training in case that the target class is without alfvens"
	"--batchsize"
		default = 100
		arg_type = Int
		help = "batch size"
	"--batchnorm"
		action = :store_true
		help = "use batchnorm in convolutional layers"
	"--outbatchnorm"
		action = :store_true
		help = "use batchnorm in the last output layer of the decoder"
	"--unnormalized"
		action = :store_true
		help = "do not normalize the input data"
	"--resblock"
		action = :store_true
		help = "use ResNet blocks for convolutional layers"
	"--eta"
		default = Float32(0.0001)
		arg_type = Float32
		help = "learning rate"
	"--beta"
		default = 1.0f0
		arg_type = Float32
		help = "value of beta for VAE loss"
	"--optimiser"
		default = "RMSProp"
		help = "optimiser type"
	"--vae-variant"
		default = "scalar"
		help = "variant of the VAE model, one of [unit, scalar, diag]"
	"--nepochs"
		default = 10
		arg_type = Int
		help = "number of outer epochs"
	"--niepochs"
		default = 1
		arg_type = Int
		help = "number of inner epochs"
	"--no-warnings"
		action = :store_true
		help = "dont print warnings"
	"--not-memory-efficient"
		action = :store_true
		help = "If not set, garbage collector is called after every epoch."
	"--test"
		action = :store_true
		help = "Test run saved in the current dir."
	"--ip-trunc"
		default = "valid"
		help = "Data truncation method based on Ip values. One of [valid, flattop]."
		range_tester = (x->x in ["valid", "flattop"])
	"--savepath"
		default = ""
		help = "alternative saving path"
	"--savepoint"
		default = 1
		arg_type = Int
		help = "how often should an intermediate state of the model be saved"
	"--not-memorysafe"
		action = :store_true
		help = "if not set, use a memory safe loading of hdf5 file which is slower but enables loading of larger datasets (requires h5py Python library)"
	"--seed"
		arg_type = Int
		default = 1
		help = "random seed for data preparation stochastics"
	"--ndense"
		arg_type = Int
		default = 1
		help = "number of dense layers"
	"--hdim"
		default = nothing
		help = "size of hidden dimension, especially useful for AAE discriminator"
	"--disc-nlayers"
		default = nothing
		help = "number of hidden layers of the AAE discriminator"
	"--kernel"
		default = "imq"
		help = "kernel of the WAE model"
	"--sigma"
		arg_type = Float32
		default = 0.1f0
		help = "scaling parameter of the WAE kernel"
	"--lambda"
		arg_type = Float32
		default = 1.0f0
		help = "scaling parameter of the MMD loss"
	"--gamma"
		arg_type = Float32
		default = 1.0f0
		help = "scaling parameter of the GAN loss in WAAE"
	"--verb"
		action = :store_true
		help = "show the training progress"
	"--h5data"
		action = :store_true
		help = "if set, the data will be randomly created from h5 raw data, this takes a lot of memory"
	"--upscale-type"
		default = "transpose"
		help = "upsacling type, one of [transpose, upscale]"
	"--normal-negative"
		action = :store_true
		help = "if set, the normal class is represented by data without alfven samples"
end
parsed_args = parse_args(ARGS, s)
modelname = "Conv"*parsed_args["modelname"]
ldim = parsed_args["ldimsize"]
nlayers = parsed_args["nlayers"]
channels = parsed_args["channels"]
patchsize = parsed_args["patchsize"]
kernelsize = parsed_args["kernelsize"]
length(kernelsize) == 1 ? kernelsize = kernelsize[1] : nothing
scaling = parsed_args["scaling"]
length(scaling) == 1 ? scaling = scaling[1] : nothing
npatches = parsed_args["npatches"]
nshots = parsed_args["nshots"]
batchsize = parsed_args["batchsize"]
batchnorm = parsed_args["batchnorm"]
outbatchnorm = (parsed_args["outbatchnorm"])
normalize = !(parsed_args["unnormalized"])
resblock = parsed_args["resblock"]
eta = parsed_args["eta"]
beta = parsed_args["beta"]
optimiser = parsed_args["optimiser"]
vae_variant = Symbol(parsed_args["vae-variant"])
outer_nepochs = parsed_args["nepochs"]
inner_nepochs = parsed_args["niepochs"]
warnings = !parsed_args["no-warnings"]
memoryefficient = !parsed_args["not-memory-efficient"]
test = parsed_args["test"]
iptrunc = parsed_args["ip-trunc"]
svpth = parsed_args["savepath"]
savepoint = parsed_args["savepoint"]
memorysafe = !(parsed_args["not-memorysafe"])
seed = parsed_args["seed"]
ndense = parsed_args["ndense"]
hdim = parsed_args["hdim"]
hdim = (hdim==nothing ? nothing : Meta.parse(hdim))
disc_nlayers = parsed_args["disc-nlayers"]
disc_nlayers = (disc_nlayers==nothing ? nlayers : Meta.parse(disc_nlayers))
kernel = eval(Meta.parse("GenModels."*parsed_args["kernel"]))
sigma = parsed_args["sigma"]
lambda = parsed_args["lambda"]
gamma = parsed_args["gamma"]
verb = parsed_args["verb"]
h5data = parsed_args["h5data"]
upscale_type = parsed_args["upscale-type"]
normal_negative = parsed_args["normal-negative"]
# data reading functions
if normalize
	readfun = AlfvenDetectors.readnormlogupsd
else
	readfun = AlfvenDetectors.readlogupsd
end
usegpu = true
pz = GenModels.randn_gpu

# decide on the paths
hostname = gethostname()
if hostname == "vit-ThinkPad-E470"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
	savepath = "/home/vit/vyzkum/alfven/experiments/oneclass"
elseif hostname == "tarbik.utia.cas.cz"
	datapath = "/home/skvara/work/alfven/cdb_data/uprobe_data"
	savepath = "/home/skvara/work/alfven/experiments/oneclass"
elseif occursin("soroban", hostname) || hostname == "gpu-node"
	datapath = "/compass/home/skvara/no-backup/uprobe_data"
	savepath = "/compass/home/skvara/alfven/experiments/oneclass"
end
if test
	savepath = "."
end
if svpth != ""
	if svpth[1] != "/"
		savepath = joinpath(savepath, svpth)
	else
		savepath = svpth
	end
end 
mkpath(savepath)

# collect all the data
if normal_negative
	patches, shotnos, labels, tstarts, fstarts = 
		AlfvenDetectors.oneclass_negative_training_data(datapath, nshots, seed, readfun, patchsize)
else
	if h5data
		patches, shotnos, labels, tstarts, fstarts = 
			AlfvenDetectors.collect_training_data_oneclass(datapath, npatches, readfun, patchsize; 
				α = 0.8, seed=seed)
	else
		norms = normalize ? "_normalized" : ""
		fname = joinpath(dirname(datapath), "oneclass_data/training/$(patchsize)$(norms)/seed-$(seed).jld2")
		patches, shotnos, labels, tstarts, fstarts = 
			AlfvenDetectors.oneclass_training_data_jld(fname, npatches)
	end
end
# if test token is given, only run with a limited number of patches
println("Total size of data: $(size(patches))")
if test
	patches = patches[:,:,:,1:min(256,size(patches,4))]
end
xdim = size(patches)

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
	:ndense => ndense,
	:upscale_type => upscale_type
	)
fit_kwargs = Dict{Symbol, Any}(
		:usegpu => usegpu,
		:memoryefficient => memoryefficient
	)
if ndense>1
	if hdim != nothing
		model_kwargs[:dsizes] = fill(hdim,ndense-1)	
	else
		model_kwargs[:dsizes] = fill(ldim*2,ndense-1)
	end
end
# model-specific arguments
if occursin("VAE", modelname)
	model_kwargs[:variant] = vae_variant
	fit_kwargs[:beta] = beta
end
if occursin("AAE", modelname)
	model_kwargs[:hdim] = hdim
	insert!(model_args, 3, :disc_nlayers => disc_nlayers)
	push!(model_args, :pz => pz)
end
if occursin("WAE", modelname)
	model_kwargs[:kernel] = kernel
	fit_kwargs[:σ] = sigma
	fit_kwargs[:λ] = lambda
	push!(model_args, :pz => pz)
end
if occursin("WAAE", modelname)
	model_kwargs[:kernel] = kernel
	fit_kwargs[:σ] = sigma
	fit_kwargs[:λ] = lambda
	fit_kwargs[:γ] = gamma
end

### run and save the model
filename_kwargs = Dict(
	:patchsize => patchsize,
	:channels => "["*reduce((x,y)->"$(x),$(y)",channels)*"]",
	:nepochs => outer_nepochs*inner_nepochs,
	:seed => seed
	)
filename = AlfvenDetectors.create_filename(modelname, [], Dict(), Dict(), 
	filename_kwargs...)
# create the model
model = GenModels.construct_model(modelname, [x[2] for x in model_args]...; model_kwargs...)
model, history, t = AlfvenDetectors.fitsave_unsupervised(patches, model, batchsize, 
	outer_nepochs, inner_nepochs, model_args, model_kwargs, fit_kwargs, savepath; 
	modelname = "GenModels."*modelname, optname=optimiser, eta=eta, verb = verb,
	usegpu=usegpu, savepoint=savepoint, filename=filename, experiment_args=parsed_args)
