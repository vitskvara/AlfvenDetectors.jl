using PyPlot

include("eval.jl")

# get the paths
hostname = gethostname()
if hostname == "gpu-node"
	evaldatapath = "/compass/home/skvara/no-backup/" 
	basepath = "/compass/home/skvara/alfven/experiments/oneclass"
else
	evaldatapath = "/home/vit/vyzkum/alfven/cdb_data/"
	basepath = "/home/vit/vyzkum/alfven/experiments/oneclass"
end

ps = [
	"eval_tuning", 
	"ldim_runs",
	"nobatchnorm_convsize_runs",
	"nobatchnorm_runs",
	"opt_runs",
	"negative_runs",
	"noresblock_runs",
	"unsupervised",
	"supervised",
	"unsupervised_additional"
	]
fs = joinpath.(basepath, ps, "eval/models_eval.csv")
# nejlepsi vysledek v #7 a #8 - dobre se uci jen ta nulova trida

dfs = map(CSV.read, fs)

idf = 9
df = dfs[idf]
df = filter(row -> row[:prec_50_mse_pos] > 0.9, df)
i = argmin(df[!,:auc_mse_pos] - df[!,:prec_50_mse_pos])
#i = 383
row = df[i,:]
mf = joinpath(basepath, ps[idf], "models", basename(row[:file]))
mf = joinpath(basepath, "supervised/models/ConvWAE_channels-[32,64]_patchsize-128_nepochs-10_seed-1_2019-11-14T22:22:57.306.bson")

datapath = joinpath(basepath, ps[idf])
modelpath = joinpath(datapath, "models")
evalpath = joinpath(datapath, "eval")
mkpath(evalpath)

model = GenModels.construct_model(mf) |> gpu
Flux.testmode!(model)
model_data = load(mf)
exp_args = model_data[:experiment_args]
seed = exp_args["seed"]
norms = exp_args["unnormalized"] ? "" : "_normalized"
readfun = exp_args["unnormalized"] ? AlfvenDetectors.readnormlogupsd : AlfvenDetectors.readlogupsd
normal_negative = get(exp_args, "normal-negative", false)
if !(normal_negative)
	testing_data = load(joinpath(evaldatapath, "oneclass_data/testing/128$(norms)/seed-$(seed).jld2"));
	training_data = load(joinpath(evaldatapath, "oneclass_data/training/128$(norms)/seed-$(seed).jld2"));
	training_patches = training_data["patches"] |> gpu;
else
	testing_data = load(joinpath(evaldatapath, "oneclass_data_negative/testing/128$(norms)/data.jld2"));
	training_data = AlfvenDetectors.oneclass_negative_training_data(datapath, 20, seed, readfun, 
		exp_args["patchsize"])
	training_patches = training_data[1] |> gpu;
end

if normal_negative
	labels = testing_data["labels"];
else
	labels = 1 .- testing_data["labels"];
end
patches = testing_data["patches"]   |> gpu;

scores = score_mse(model, patches)
roc = EvalCurves.roccurve(scores, labels)
auroc = EvalCurves.auc(roc...)
prec_50 = EvalCurves.precision_at_k(scores, labels, 50)
prc = EvalCurves.prcurve(scores, labels)

figure()
plot(roc...)
plot(prc...)

ids = sortperm(scores, rev=true) 
scores[ids]
labels[ids]
sum(labels[ids][1:50])/50

roc = 
EvalCurves.auc(EvalCurves.roccurve(-scores, labels)...)

prec_50_s = EvalCurves.precision_at_k(-scores, slabels, min(50, sum(slabels)))
prec_50_s = EvalCurves.precision_at_k(-scores, labels, min(50, sum(slabels)))

ids = sortperm(scores)
slabels[ids]


