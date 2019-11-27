# show roc and pr curves for selected models
# show it for best oneclass alfvén/nonalfvén and two stage knn/GMM model
# first we have to find the best models
# then export their roc and pr curves 
# and do the plots
using PyPlot
using CSV
using DataFrames
using Statistics
using PaperUtils
using AlfvenDetectors
using GenModels
using Flux
using ValueHistories
using JLD2
using FileIO
using EvalCurves

outpath = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/paper/figs_tables/"

evaldatapath = "/home/vit/vyzkum/alfven/cdb_data/"
basepath = "/home/vit/vyzkum/alfven/experiments/oneclass"
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"

ps = [
	"supervised",
	"unsupervised_additional"
	]
fs = joinpath.(basepath, ps, "eval/models_eval.csv")

mse(model,x) = Flux.mse(x, model(x)).data
score_mse(model, x) = map(i->mse(model, x[:,:,:,i:i]), 1:size(x,4))

function get_roc_prc(mf)
	model, exp_args, model_args, model_kwargs, history = AlfvenDetectors.load_model(mf)
	Flux.testmode!(model)

	# get the data
	seed = exp_args["seed"]
	norms = exp_args["unnormalized"] ? "" : "_normalized"
	readfun = exp_args["unnormalized"] ? AlfvenDetectors.readnormlogupsd : AlfvenDetectors.readlogupsd
	normal_negative = get(exp_args, "normal-negative", false)
	if !(normal_negative)
		testing_data = load(joinpath(evaldatapath, "oneclass_data/testing/128$(norms)/seed-$(seed).jld2"));
	else
		testing_data = load(joinpath(evaldatapath, "oneclass_data_negative/testing/128$(norms)/data.jld2"));
	end

	patches = testing_data["patches"]
	if normal_negative
		labels = testing_data["labels"]
	else
		labels = 1 .- testing_data["labels"]
	end 

	scores = score_mse(model, patches)
	roc = EvalCurves.roccurve(scores, labels)
	if !normal_negative
		scores = - scores
	end
	prc = EvalCurves.prcurve(scores, labels)

	return roc, prc
end

mfs = [
	joinpath(basepath, ps[1], "models/ConvWAE_channels-[32,64]_patchsize-128_nepochs-10_seed-10_2019-11-18T02:23:23.796.bson"),
	joinpath(basepath, ps[2], "models/ConvWAAE_channels-[32,64]_patchsize-128_nepochs-10_seed-1_2019-11-15T04:55:09.238.bson")
]

results = map(get_roc_prc, mfs)
out_data = DataFrame(
	model_file = mfs,
	roc = [x[1] for x in results],
	prc = [x[2] for x in results]
	)
oneclass_f = joinpath(outpath, "roc_prc_oneclass.csv")
CSV.write(oneclass_f, out_data)

# this is where we got the best models from
dfs = map(CSV.read, fs)

# first get the supervised model
df = filter(row -> row[:model] == "ConvWAE", dfs[1])
imodel = argmax(df[!,:auc_mse_pos])
row = df[imodel,:]
mf = row[:file]
mf = joinpath(basepath, ps[1], "models/ConvWAE_channels-[32,64]_patchsize-128_nepochs-10_seed-10_2019-11-18T02:23:23.796.bson")

df = filter(row -> row[:model] == "ConvWAAE", dfs[2])
imodel = argmax(df[!,:auc_mse])
row = df[imodel,:]
mf = row[:file]
mf = joinpath(basepath, ps[2], "models/ConvWAAE_channels-[32,64]_patchsize-128_nepochs-10_seed-1_2019-11-15T04:55:09.238.bson")

# this is the best supervised model
mf = joinpath(basepath, ps[1], "models/ConvWAE_channels-[32,64]_patchsize-128_nepochs-10_seed-10_2019-11-18T02:23:23.796.bson")
inds = rand(1:43,4)
figure()
for (i,j) in enumerate(inds)
	subplot(4,2,(i-1)*2+1)
	pcolormesh(patches[:,:,1,j])
	r = model(patches[:,:,:,j:j]).data
	subplot(4,2,i*2)
	pcolormesh(r[:,:,1,1])
end

inds = rand(44:191,4)
figure()
for (i,j) in enumerate(inds)
	subplot(4,2,(i-1)*2+1)
	pcolormesh(patches[:,:,1,j])
	r = model(patches[:,:,:,j:j]).data
	subplot(4,2,i*2)
	pcolormesh(r[:,:,1,1])
end
