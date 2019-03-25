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
        help = "One of [AE, VAE, TSVAE]."
    "ldimsize"
    	required = true
    	arg_type = Int
    	help = "Size of latent dimension."
    "nlayers"
    	required = true
    	arg_type = Int
    	help = "Number of layers."
    "--measurement"
    	default = "mscamp"
    	help = "One of [mscamp, mscphase or uprobe]."
    "--gpu"
	    help = "Use gpu?"
    	action = :store_true
    "--coils"
		default = [12,13,14]
		help = "A list of used coils."
		nargs = '+'
		arg_type = Int
	"--batchsize"
		default = 128
		arg_type = Int
		help = "Batch size."
	"--nepochs"
		default = 10
		arg_type = Int
		help = "Number of outer epochs."
	"--no-warnings"
		action = :store_true
		help = "Dont print warnings."
	"--memory-efficient"
		action = :store_true
		help = "If set, garbage collector is called after every epoch."
	"--test"
		action = :store_true
		help = "Test run saved in the current dir."
end
parsed_args = parse_args(ARGS, s)
modelname = parsed_args["modelname"]
usegpu = parsed_args["gpu"]
ldim = parsed_args["ldimsize"]
nlayers = parsed_args["nlayers"]
coils = parsed_args["coils"]
batchsize = parsed_args["batchsize"]
outer_nepochs = parsed_args["nepochs"]
inner_nepochs = 1
warnings = !parsed_args["no-warnings"]
memoryefficient = parsed_args["memory-efficient"]
measurement_type = parsed_args["measurement"]
test = parsed_args["test"]
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
	savepath = "/home/vit/vyzkum/alfven/experiments/gpu_test/$measurement_type"
elseif hostname == "tarbik.utia.cas.cz"
	datapath = "/home/skvara/work/alfven/cdb_data/data_sample"
	savepath = "/home/skvara/work/alfven/experiments/gpu_test/$measurement_type"
elseif hostname == "soroban-node-03"
	datapath = "/compass/Shared/Exchange/Havranek/Link to Alfven"
	savepath = "/compass/home/skvara/alfven/experiments/gpu_test/$measurement_type"
end
mkpath(savepath)
if test
	savepath = "."
end

shots = readdir(datapath)
shots = joinpath.(datapath, shots)[1:ceil(Int,length(shots)/10)]
if measurement_type == "uprobe"
	rawdata = AlfvenDetectors.collect_signals(shots, readfun; warns=warnings)
else
	rawdata = AlfvenDetectors.collect_signals(shots, readfun, coils; warns=warnings)
end
data = rawdata
### setup args
xdim = size(data,1)
model_args = [
	:xdim => xdim,
	:ldim => ldim, 
	:nlayers => nlayers
	]
if modelname == "VAE"
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
model, history, t = AlfvenDetectors.fitsave_unsupervised(data, modelname, batchsize, outer_nepochs, inner_nepochs,
	 model_args, model_kwargs, fit_kwargs, savepath)
