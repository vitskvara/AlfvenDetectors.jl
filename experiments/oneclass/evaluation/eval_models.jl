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

df = dfs[9]

figure()
subplot(321)
scatter(df[!,:train_mse], df[!,:auc_mse])
xlabel("train mse")
ylabel("auc mse")

subplot(322)
scatter(df[!,:train_mse], df[!,:auc_mse])
xlabel("train mse")
ylabel("auc mse")
xlim([0, 0.1])
ylim([0.4,0.7])

subplot(323)
scatter(df[!,:test0_mse], df[:auc_mse])
xlabel("test0 mse")
ylabel("auc mse")

subplot(324)
scatter(df[!,:test1_mse], df[:auc_mse])
xlabel("test1 mse")
ylabel("auc mse")

subplot(325)
scatter(df[!,:test1_mse]-df[!,:test0_mse], df[:auc_mse])
xlabel("test1 - test0 mse")
ylabel("auc mse")

subplot(326)
scatter(df[!,:test1_mse]-df[!,:test0_mse], df[:auc_mse])
xlabel("test1 - test0 mse")
ylabel("auc mse")
xlim([-0.02, 0.02])
ylim([0.4,0.7])

tight_layout()
figf = "/home/vit/vyzkum/alfven/experiments/oneclass/waae_runs/eval/models_eval.eps"
savefig(figf)

# look at why we have auc 0.83 and prec@50 is 0.9
idf = 9
df = dfs[idf]
df = filter(row -> row[:prec_50_mse_pos] > 0.9, df)
i = argmin(df[!,:auc_mse_pos] - df[!,:prec_50_mse_pos])
#i = 383
row = df[i,:]
mf = joinpath(basepath, ps[idf], "models", basename(row[:file]))

datapath = joinpath(basepath, ps[idf])
modelpath = joinpath(datapath, "models")
evalpath = joinpath(datapath, "eval")
mkpath(evalpath)

# get a model
"models = readdir(modelpath)
imodel = 5
mf = joinpath(modelpath, models[imodel])
"
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
positive_patches = testing_data["patches"][:,:,:,labels.==1];
negative_patches = testing_data["patches"][:,:,:,labels.==0];

function jacobian(m::GenModels.GenerativeModel, z::AbstractArray{T,1}) where T
	dec = Flux.Chain(
			model.decoder,
			y -> reshape(y, :)
		)
	return Flux.Tracker.jacobian(dec, z)
end

function plot_4(model, patches, scores, labels, inds)
	figure()
	for (i, ind) in enumerate(inds)
		x = patches[:,:,:,ind:ind]
		y = model(x).data
		subplot(4,2,i*2-1)
		title("$(labels[ind]), $(scores[ind]) \n patch mean = $(mean(x))")
		pcolormesh(x[:,:,1,1], cmap="plasma")
		subplot(4,2,i*2)
		title("patch mean = $(mean(y))")
		pcolormesh(y[:,:,1,1], cmap="plasma")
	end
	tight_layout()
end

scores = score_mse(model, patches)
roc = EvalCurves.roccurve(scores, labels)
auroc = EvalCurves.auc(roc...)
prec_50 = EvalCurves.precision_at_k(scores, labels, 50)
prec_50_b = EvalCurves.precision_at_k(score_mse(model,patches), labels, min(50, sum(labels)))
prc = EvalCurves.prcurve(scores, labels)

inds = [3, 4, 5, 6]
plot_4(model, patches, scores, labels, inds)

inds = [1, 2, 300, 301]
plot_4(model, patches, scores, labels, inds)

inds = [1, 2, 180, 181]
plot_4(model, patches, scores, labels, inds)

inds = vcat(sortperm(scores)[1:2], sortperm(scores)[end-1:end])
plot_4(model, patches, scores, labels, inds)

figure()
scatter(df6[!,:ldim], df6[!,:auc_mse])



# try to compute seed averages
df = dfs[7]
subcols = [:model,:channels,:ldim,:nepochs,:normalized,:neg,:λ,:γ,:σ,:auc_mse]
subdf = df[!,subcols]
agcols = filter(x -> !(x in [:seed, :auc_mse]), subcols)
agdf = aggregate(subdf, agcols, mean)

