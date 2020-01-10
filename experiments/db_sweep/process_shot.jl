# run as julia process_shot.jl infile
using AlfvenDetectors
using GenModels
using Flux
using ValueHistories
using JLD2, FileIO
using FileIO
using Statistics
using CuArrays

# setup paths
hst = gethostname()
if hst == "vit-ThinkPad-E470"
	basepath = "/home/vit/vyzkum/alfven/experiments"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
	savepath = ""
elseif hst in ["gpu-node", "gpu-titan"]
	basepath = "/compass/home/skvara/alfven/experiments"
	datapath = "/compass/home/skvara/no-backup/uprobe_data"
end

# get model and data files
modelpath = joinpath(basepath,"db_sweep_models")
modelfiles = readdir(modelpath)
savepath = joinpath(basepath, "db_sweep_results/scores")
mkpath(savepath)
shotfs = readdir(datapath)

function load_model(mf)
	model, exp_args, model_args, model_kwargs, history = AlfvenDetectors.load_model(mf)
	Flux.testmode!(model)
	model = model |> gpu
end

function load_data(df)
	psd = AlfvenDetectors.readlogupsd(df,memorysafe=true);
	t = AlfvenDetectors.readtupsd(df,memorysafe=true);
	f = AlfvenDetectors.readfupsd(df,memorysafe=true)/1000000;
	psd, t, f
end

score(model, patches) = vec(mean((patches - model(patches).data).^2, dims=(1,2,3))) |> cpu
score(model, patches, batchsize) = vcat(map(i -> score(model, patches[:,:,:,(i-1)*batchsize+1:min((i*batchsize), 
	size(patches,4))]), 1:ceil(Int,size(patches,4)/batchsize))...)


function collect_patches(psd, tinds, finds, patchsize)
	map(fi -> cat([psd[fi:fi+patchsize-1,ti:ti+patchsize-1] for ti in tinds]..., dims=4) |> gpu, finds)
end

function compute_scores(model, psd, t, f)
	patchsize = 128
	tstep = 10
	fstep = 32
	tinds = 1:tstep:length(t)-patchsize
	finds = 1:fstep:length(f)-patchsize

	patches = collect_patches(psd, tinds, finds, patchsize);

	batchsize = 10
	scores = map(p -> score(model, p, batchsize), patches);

	return cat(scores..., dims=2), t[tinds], f[finds]
end

function compute_scores_and_save(mf, psd, t, f)
	model = load_model(mf)
	scores, tp, fp = compute_scores(model, psd, t, f)

	# and save it
	fname = joinpath(datadir, split(split(basename(mf), "_")[end],".bson")[1]*".jld2")
	@save fname scores tp fp
	println("$fname saved succesfuly!")
end

# now do the computing
if length(ARGS) > 0
	shotf = ARGS[1]
else
	shotf = shotfs[1]
end
shotf = joinpath(datapath, shotf) # data file is argument of the script
datadir = joinpath(savepath, split(split(basename(shotf), "_")[2], ".")[1])
mkpath(datadir)
psd, t, f = load_data(shotf)

# do it over all models
map(x->compute_scores_and_save(x, psd, t, f), joinpath.(modelpath, modelfiles));