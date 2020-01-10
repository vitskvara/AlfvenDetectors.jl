using AlfvenDetectors
using GenModels
using Flux
using ValueHistories
using JLD2
using FileIO
using CSV
using DataFrames
using Statistics
using CuArrays

hst = gethostname()
if hst == "vit-ThinkPad-E470"
	basepath = "/home/vit/vyzkum/alfven/experiments/oneclass"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
elseif hst in ["gpu-node", "gpu-titan"]
	basepath = "/compass/home/skvara/alfven/experiments/oneclass"
	datapath = "/compass/home/skvara/no-backup/uprobe_data"
end
ps = [
	"supervised",
	"unsupervised_additional"
	]
fs = joinpath.(basepath, ps, "eval/models_eval.csv")
dfs = map(CSV.read, fs)
bigdf = vcat(dfs...)

# ow we could simply take the top 10 models from bigdf...?
# yes, since there are only 2 models in maxmodeldf
subcols = [:model,:channels,:ldim,:nepochs,:normalized,:neg,:β,:λ,:γ,:σ,:prec_50_mse,:file]
maxmodeldf = filter(row -> row[:neg] == true, bigdf)[!,subcols]
maxmodeldf = maxmodeldf[sortperm(maxmodeldf[!,:prec_50_mse], rev=true)[1:10],:]
modelfiles = maxmodeldf[:file]
modelfiles_base = basename.(modelfiles)
modelfiles = joinpath.(dirname(basepath),"db_sweep_models", modelfiles_base)

map(isfile, modelfiles)

# set the model file
models = []
for mf in modelfiles
	model, exp_args, model_args, model_kwargs, history = AlfvenDetectors.load_model(mf)
	Flux.testmode!(model)
	model = model |> gpu;
	push!(models, model)
	println(exp_args["unnormalized"])
end
# YES


# tynhle byly asi pouzity pro ten obrazek v clanku
"ConvWAE_channels-[32,64]_patchsize-128_nepochs-30_seed-1_2019-11-17T19:50:08.347.bson"
"ConvAE_channels-[32,32,64]_patchsize-128_nepochs-30_seed-1_2019-11-18T15:40:18.273.bson"

# get data and try if the models can actually identify the maximum as in the paper
shots = readdir(datapath);
shotno = "10870"
shotf = joinpath(datapath, filter(x->occursin(shotno, x),shots)[1])
psd = AlfvenDetectors.readlogupsd(shotf,memorysafe=true);
t = AlfvenDetectors.readtupsd(shotf,memorysafe=true);
f = AlfvenDetectors.readfupsd(shotf,memorysafe=true)/1000000;

# now we have the data, lets limit ourselves to some time and frequency frame
patchsize = 128
f0 = 0.9
i0 = ((1:length(f))[f .> f0])[1];
ts = [0.98, 1.3]
tinds = ts[1] .< t .<ts[2];
finds = i0:patchsize-1+i0;
ot = t[tinds];
of = f[finds] ;
opsd = psd[finds,tinds];

# now gather the patches
stepsize = 10
t0inds = collect(1:stepsize:length(ot)-128);
t0s = [ot[t:t+patchsize-1] for t in t0inds];
patches = cat([opsd[:,t:t+patchsize-1] for t in t0inds]..., dims=4) |> gpu;
batchsize = 10

score(model, patches) = vec(mean((patches - model(patches).data).^2, dims=(1,2,3))) |> cpu
score(model, patches, batchsize) = vcat(map(i -> score(model, patches[:,:,:,(i-1)*batchsize+1:min((i*batchsize), 
	size(patches,4))]), 1:ceil(Int,size(patches,4)/batchsize))...)

function median_filter(x,wl)
	y = similar(x, length(x)-wl+1)
	for i in 1:length(y)
		y[i] = median(x[i:i+wl-1])
	end
	y
end

scores = map(m -> score(m, patches, batchsize), models);
wl = 30
med_scores = map(x->median_filter(x,wl), scores);

# plot it
figure()
for (i,s) in enumerate(med_scores)
	subplot(5,2,i)
	plot(s)
end

# now do the whole procedure with on f-t grid
patchsize = 128
tstep = 10
fstep = 32
tinds = 1:tstep:length(t)-patchsize
finds = 1:fstep:length(f)-patchsize

function collect_patches(psd, tinds, finds)
	map(fi -> cat([psd[fi:fi+patchsize-1,ti:ti+patchsize-1] for ti in tinds]..., dims=4) |> gpu, finds)
end

patches = collect_patches(psd, tinds, finds);
model = models[1]

scores = map(p -> score(model, p, batchsize), patches);
wl = 30
med_scores = map(x->median_filter(x,wl), scores);

figure()
for (i,s) in enumerate(med_scores)
	subplot(7,2,i)
	plot(s)
end
