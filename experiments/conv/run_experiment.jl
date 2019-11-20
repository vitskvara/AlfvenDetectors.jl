using AlfvenDetectors
using Flux
using ValueHistories
using ArgParse
using DelimitedFiles
using Random
using StatsBase
using GenModels
using Pkg

# use argparse to extract the command line arguments
# name of algorithm, usegpu, latentdim, no of layers, coils?
# maybe also number of epochs and batchsize?
s = ArgParseSettings()
@add_arg_table s begin
    "modelname"
		required = true
        help = "one of [AE, VAE, TSVAE, WAE, AAE]"
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
    "--nshots"
    	default = 10
    	arg_type = Int
    	help = "number of shots used"
    "--no-alfven"
    	action = :store_true
    	help = "dont use alfven data for training"
    "--measurement"
    	default = "uprobe"
    	help = "one of [mscamp, mscphase, mscampphase or uprobe]"
    "--gpu"
	    help = "use gpu?"
    	action = :store_true
    "--coils"
		default = [12,13,14]
		help = "a list of used coils"
		nargs = '+'
		arg_type = Int
	"--batchsize"
		default = 128
		arg_type = Int
		help = "batch size"
	"--batchnorm"
		action = :store_true
		help = "use batchnorm in convolutional layers"
	"--outbatchnorm"
		action = :store_true
		help = "use batchnorm in the last output layer of the decoder"
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
	"--memory-efficient"
		action = :store_true
		help = "If set, garbage collector is called after every epoch."
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
		default = 200
		arg_type = Int
		help = "how often should an intermediate state of the model be saved"
	"--memorysafe"
		action = :store_true
		help = "if set, use a memory safe loading of hdf5 file which is slower but enables loading of larger datasets (requires h5py Python library)"
	"--positive-patch-ratio"
		arg_type = Float32
		default = 0f0
		help = "the ratio of positively labeled patches to the rest of the data"	
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
	"--pz-components"
		arg_type = Int
		default = 1
		help = "number of pz components"
	"--pz-type"
		default = "vamp"
		help = "the type of predefined pz, one of [cube, flower, vamp]"
	"--lambda"
		arg_type = Float32
		default = 1.0f0
		help = "scaling parameter of the MMD loss"
	"--gamma"
		arg_type = Float32
		default = 1.0f0
		help = "scaling parameter of the GAN loss in WAAE"
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
nshots = parsed_args["nshots"]
noalfven = parsed_args["no-alfven"]
measurement_type = parsed_args["measurement"]
usegpu = parsed_args["gpu"]
coils = parsed_args["coils"]
batchsize = parsed_args["batchsize"]
batchnorm = parsed_args["batchnorm"]
outbatchnorm = (parsed_args["outbatchnorm"])
resblock = parsed_args["resblock"]
eta = parsed_args["eta"]
beta = parsed_args["beta"]
optimiser = parsed_args["optimiser"]
vae_variant = Symbol(parsed_args["vae-variant"])
outer_nepochs = parsed_args["nepochs"]
inner_nepochs = parsed_args["niepochs"]
warnings = !parsed_args["no-warnings"]
memoryefficient = parsed_args["memory-efficient"]
test = parsed_args["test"]
iptrunc = parsed_args["ip-trunc"]
svpth = parsed_args["savepath"]
savepoint = parsed_args["savepoint"]
memorysafe = parsed_args["memorysafe"]
positive_patch_ratio = parsed_args["positive-patch-ratio"]
seed = parsed_args["seed"]
ndense = parsed_args["ndense"]
hdim = parsed_args["hdim"]
hdim = (hdim==nothing ? nothing : Meta.parse(hdim))
disc_nlayers = parsed_args["disc-nlayers"]
disc_nlayers = (disc_nlayers==nothing ? nlayers : Meta.parse(disc_nlayers))
kernel = eval(Meta.parse("GenModels."*parsed_args["kernel"]))
sigma = parsed_args["sigma"]
pz_components = parsed_args["pz-components"]
lambda = parsed_args["lambda"]
gamma = parsed_args["gamma"]
pz_type = parsed_args["pz-type"]
if pz_type in ["flower", "cube"]
	pz_type = "AlfvenDetectors."*parsed_args["pz-type"]*"GM"
end
if measurement_type == "mscamp"
	readfun = AlfvenDetectors.readmscamp
elseif measurement_type == "mscphase"
	readfun = AlfvenDetectors.readnormmscphase
elseif measurement_type == "mscampphase"
	readfun = AlfvenDetectors.readmscampphase
elseif measurement_type == "uprobe"
	readfun = AlfvenDetectors.readnormlogupsd
end
### set the rest of the stuff
if usegpu && "CuArrays" in keys(Pkg.installed())
	using CuArrays
end

hostname = gethostname()
if hostname == "vit-ThinkPad-E470"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
	savepath = "/home/vit/vyzkum/alfven/experiments/conv/$measurement_type"
elseif hostname == "tarbik.utia.cas.cz"
	datapath = "/home/skvara/work/alfven/cdb_data/uprobe_data"
	savepath = "/home/skvara/work/alfven/experiments/conv/$measurement_type"
elseif occursin("soroban", hostname) || hostname == "gpu-node"
#	datapath = "/compass/Shared/Exchange/Havranek/Link to Alfven"
	datapath = "/compass/home/skvara/no-backup/uprobe_data"
	savepath = "/compass/home/skvara/alfven/experiments/conv/$measurement_type"
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

# decide the type of reading function
collect_fun(x) = (measurement_type == "uprobe") ?
	AlfvenDetectors.collect_conv_signals(x, readfun, patchsize; 
		warns=warnings, type=iptrunc, memorysafe=memorysafe) :
	AlfvenDetectors.collect_conv_signals(x, readfun, patchsize, coils; 
		warns=warnings, type=iptrunc, memorysafe=memorysafe)

# collect all the data
data, training_shotnos = AlfvenDetectors.collect_training_data(datapath, collect_fun, nshots,
	readfun, positive_patch_ratio, patchsize; seed=seed, use_alfven_shots=!noalfven)
# if test token is given, only run with a limited number of patches
println("Total size of data: $(size(data))")
if test
	data = data[:,:,:,1:256]
end
xdim = size(data)

# put all data into gpu only if you want to be fast and not care about memory clogging
# otherwise that is done in the train function now per batch
# data = data |> gpu

# pz
if pz_type == "vamp"
	pz = VAMP(pz_components, xdim[1:3])
	pz = (usegpu ? gpu(pz) : pz )
else
	if pz_components == 1
		prior = (usegpu ? GenModels.randn_gpu : randn)
	else
		prior = eval(Meta.parse(pz_type))(ldim, pz_components; seed=seed, gpu=usegpu)
	end
	pz(n) = prior(Float32,ldim,n)
end
println("")

### setup args
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
	fit_kwargs[:γ] = gamma
end
if occursin("WAE", modelname)
	model_kwargs[:kernel] = kernel
	push!(model_args, :pz => pz)
	fit_kwargs[:σ] = sigma
	fit_kwargs[:λ] = lambda
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
	:nepochs => outer_nepochs*inner_nepochs
	)
filename = AlfvenDetectors.create_filename(modelname, [], Dict(), Dict(), 
	filename_kwargs...)
# create the model
model = GenModels.construct_model(modelname, [x[2] for x in model_args]...; model_kwargs...)
# this is here in case that VAMP is used
if pz_type == "vamp"
	model.pz(N::Int) = GenModels.encodeSampleVamp(model.pz, model.encoder, N)
end
model, history, t = AlfvenDetectors.fitsave_unsupervised(data, model, batchsize, 
	outer_nepochs, inner_nepochs, model_args, model_kwargs, fit_kwargs, savepath; 
	modelname = "GenModels."*modelname, optname=optimiser, eta=eta, 
	usegpu=usegpu, savepoint=savepoint, filename=filename, experiment_args=parsed_args)
