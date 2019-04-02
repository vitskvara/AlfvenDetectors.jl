using AlfvenDetectors
using Flux
using ValueHistories
using ArgParse

# use argparse to extract the command line arguments
# name of algorithm, usegpu, latentdim, no of layers, coils?
# maybe also number of epochs and batchsize?
s = ArgParseSettings()
@add_arg_table s begin
    "modelname"
		required = true
        help = "one of [AE, VAE, TSVAE]"
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
    	default = 64
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
	"--nepochs"
		default = 10
		arg_type = Int
		help = "number of outer epochs"
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
measurement_type = parsed_args["measurement"]
usegpu = parsed_args["gpu"]
coils = parsed_args["coils"]
batchsize = parsed_args["batchsize"]
outer_nepochs = parsed_args["nepochs"]
inner_nepochs = 1
warnings = !parsed_args["no-warnings"]
memoryefficient = parsed_args["memory-efficient"]
test = parsed_args["test"]
iptrunc = parsed_args["ip-trunc"]
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

if usegpu
	using CuArrays
end

hostname = gethostname()
if hostname == "vit-ThinkPad-E470"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/data_sample"
	savepath = "/home/vit/vyzkum/alfven/experiments/conv/$measurement_type"
elseif hostname == "tarbik.utia.cas.cz"
	datapath = "/home/skvara/work/alfven/cdb_data/data_sample"
	savepath = "/home/skvara/work/alfven/experiments/conv/$measurement_type"
elseif hostname == "soroban-node-03"
	datapath = "/compass/Shared/Exchange/Havranek/Link to Alfven"
	savepath = "/compass/home/skvara/alfven/experiments/conv/$measurement_type"
end
mkpath(savepath)

shots = readdir(datapath)
shots = joinpath.(datapath, shots)

if test
	savepath = "."
	shots = shots[1:10]
end

if measurement_type == "uprobe"
	data = AlfvenDetectors.collect_conv_signals(shots, readfun, patchsize; 
		warns=warnings, type=iptrunc)
else
	data = AlfvenDetectors.collect_conv_signals(shots, readfun, patchsize, coils; 
		warns=warnings, type=iptrunc)
end
# put all data into gpu only if you want to be fast and not care about memory clogging
# otherwise that is done in the train function now per batch
# data = data |> gpu

### setup args
xdim = size(data)
model_args = [
		:xdim => xdim[1:3],
		:ldim => ldim, 
		:nlayers => nlayers,
		:kernelsize => kernelsize,
		:channels => channels,
		:scaling => scaling
	]
if modelname == "ConvVAE"
	model_kwargs = Dict(
		:variant => :scalar
		)
else
	model_kwargs = Dict(
		)
end
fit_kwargs = Dict(
		:usegpu => usegpu,
		:memoryefficient => memoryefficient
	)

### run and save the model
model, history, t = AlfvenDetectors.fitsave_unsupervised(data, modelname, batchsize, 
	outer_nepochs, inner_nepochs, model_args, model_kwargs, fit_kwargs, savepath; 
	usegpu=usegpu)
