using PyPlot
using AlfvenDetectors
using GenModels
using Flux
using ValueHistories
using CSV
using Statistics
using JLD2
using FileIO

evaldatapath = "/home/vit/vyzkum/alfven/cdb_data/"
basepath = "/home/vit/vyzkum/alfven/experiments/oneclass"
p =	"unsupervised_additional"

# get the model
df = CSV.read(joinpath(basepath, p, "eval/models_eval.csv"))
#imodel = argmax(df[!,:auc_mse])
imodel = argmax(df[!,:prec_50_mse])
mf = df[imodel, :file]
mf = joinpath(basepath, p, "models/ConvWAE_channels-[32,64]_patchsize-128_nepochs-30_seed-1_2019-11-17T19:50:08.347.bson")
mf = joinpath(basepath, p, "models/ConvAE_channels-[32,32,64]_patchsize-128_nepochs-30_seed-1_2019-11-18T15:40:18.273.bson")
model, exp_args, model_args, model_kwargs, history = AlfvenDetectors.load_model(mf)
Flux.testmode!(model)

# get the data
using PyCall
using HDF5

# setup
outpath = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/paper/figs_tables"
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
shots = readdir(datapath)
shotno = "10870"
shotf = joinpath(datapath, filter(x->occursin(shotno, x),shots)[1])
rawf = joinpath(dirname(datapath), "raw_signals/$(shotno).h5")

# load the data
if exp_args["unnormalized"]
	psd = AlfvenDetectors.readlogupsd(shotf,memorysafe=true);
else
	psd = AlfvenDetectors.readnormlogupsd(shotf,memorysafe=true);
end
t = AlfvenDetectors.readtupsd(shotf,memorysafe=true);
f = AlfvenDetectors.readfupsd(shotf,memorysafe=true)/1000000;
I = AlfvenDetectors.readip(shotf,memorysafe=true);
signal = h5read(rawf, "f")
tsignal = h5read(rawf, "t")/1000
sinds = minimum(t) .<= tsignal .<= maximum(t)
signal = signal[sinds]
tsignal = tsignal[sinds]

# plot params
cmap = "plasma" # colormap
matplotlib.rc("font", family = "normal",
    weight = "bold",
    size = 16
)
PyCall.PyDict(matplotlib."rcParams")["text.usetex"] = true
PyCall.PyDict(matplotlib."rcParams")["font.family"] = "serif"

f0 = 0.9
patchsize = 128
i0 = ((1:length(f))[f .> f0])[1]
ts = [0.98, 1.3]

figure()
pcolormesh(t,f,psd)

tinds = ts[1] .< t .<ts[2]
finds = i0:patchsize-1+i0
ot = t[tinds]
of = f[finds] 
opsd = psd[finds,tinds]

figure(figsize=(8,4))
pcolormesh(ot,of,opsd,cmap=cmap)

stepsize = 10
t0inds = collect(1:stepsize:length(ot)-128)
t0s = [ot[t:t+patchsize-1] for t in t0inds]
patches = cat([opsd[:,t:t+patchsize-1] for t in t0inds]..., dims=4);

# get reconstructions
batchsize = 10
rpatches = cat(map(i -> model(patches[:,:,:,(i-1)*batchsize+1:min((i*batchsize), 
	size(patches,4))]).data, 1:ceil(Int,size(patches,4)/batchsize))..., dims=4)

run_time = split(split(mf,"_")[end], ".")[1]
f = joinpath(outpath, "detector_data_$(run_time).jld2")
if !isfile(f)
	scores = vec(mean((patches - rpatches).^2, dims=(1,2,3)))
	plot_t = ot[t0inds]
	plot_data = Dict(
		"scores" => scores,
		"plot_t" => plot_t,
		"f0" => f0,
		"mf" => mf
		) 
	save(f, plot_data)
else
	plot_data = load(f)
	scores = plot_data["scores"]
	f0 = plot_data["f0"]
	plot_t = plot_data["plot_t"]
	mf = plot_data["mf"]
end

# now plot the mf
figure(figsize=(8,4))
subplot(211)
pcolormesh(ot,of,opsd,cmap=cmap)
subplot(212)
plot(plot_t, scores)
xlim(ts)

using Statistics
function median_filter(x,wl)
	y = similar(x, length(x)-wl+1)
	for i in 1:length(y)
		y[i] = median(x[i:i+wl-1])
	end
	y
end

wl = 10
wl2 = Int(wl/2)
med_scores = median_filter(scores,wl)

figure(figsize=(8,6))
subplot(311)
pcolormesh(ot,of,opsd,cmap=cmap)
subplot(312)
plot(plot_t, scores)
xlim(ts)
subplot(313)
plot(plot_t[wl2:end-wl2], med_scores)
xlim(ts)

figure(figsize=(8,4))
subplot(211)
pcolormesh(ot,of,opsd,cmap=cmap)
subplot(212)
plot(plot_t, scores, alpha=0.3)
plot(plot_t[wl2:end-wl2], med_scores)
xlim(ts)
savefig(joinpath(outpath, "detector_$(run_time).eps"))
