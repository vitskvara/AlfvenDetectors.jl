using AlfvenDetectors
using Flux
using ValueHistories
using ArgParse

modelname = ARGS[1]
usegpu = true
ldim = 20
nlayers = 3
coils = [12,13,14]

if usegpu
	using CuArrays
end

hostname = gethostname()
if hostname == "vit-ThinkPad-E470"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/data_sample"
	savepath = "/home/vit/vyzkum/alfven/experiments/single_col/basic"
elseif hostname == "tarbik"
	datapath = "/home/skvara/work/alfven/cdb_data/data_sample"
	savepath = "/home/skvara/work/alfven/experiments/single_col/basic"
end
mkpath(savepath)

shots = readdir(datapath)
shots = joinpath.(datapath, shots)
rawdata = AlfvenDetectors.collect_mscamps(shots, coils) 
data = rawdata |> gpu

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
batchsize = 64
outer_nepochs = 10
inner_nepochs = 1
fit_kwargs = Dict(
	)

model, history, t = AlfvenDetectors.fitsave_mscamps(data, modelname, batchsize, outer_nepochs, inner_nepochs,
	 model_args, model_kwargs, fit_kwargs, savepath)
