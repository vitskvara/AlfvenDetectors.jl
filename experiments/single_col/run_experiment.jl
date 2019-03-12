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
    "--measurement"
    	default = "mscamp"
    	help = "one of [mscamp, mscphase or uprobe]"
    "--gpu"
	    help = "use gpu?"
    	action = :store_true
    "--coils"
		default = [12,13,14]
		help = "a list of used coil"
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
measurement_type = parsed_args["measurement"]
if measurement_type == "mscamp"
	readfun = AlfvenDetectors.readmscamp
elseif measurement_type == "mscphase"
	readfun = AlfvenDetectors.readnormmscphase
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
	savepath = "/home/vit/vyzkum/alfven/experiments/single_col/$measurement_type"
elseif hostname == "tarbik.utia.cas.cz"
	datapath = "/home/skvara/work/alfven/cdb_data/data_sample"
	savepath = "/home/skvara/work/alfven/experiments/single_col/$measurement_type"
elseif hostname == "soroban-node-03"
	datapath = "/compass/Shared/Exchange/Havranek/Link to Alfven"
	savepath = "/compass/home/skvara/alfven/experiments/single_col/$measurement_type"
end
mkpath(savepath)

shots = readdir(datapath)
shots = joinpath.(datapath, shots)
if measurement_type == "uprobe"
	rawdata = AlfvenDetectors.collect_signals(shots, readfun)
else
	rawdata = AlfvenDetectors.collect_signals(shots, readfun, coils)
end
data = rawdata |> gpu

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
	)

### run and save the model
model, history, t = AlfvenDetectors.fitsave_unsupervised(data, modelname, batchsize, outer_nepochs, inner_nepochs,
	 model_args, model_kwargs, fit_kwargs, savepath)
