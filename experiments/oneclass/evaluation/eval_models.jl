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
datapath = joinpath(basepath, "nobatchnorm_ldim_runs")
modelpath = joinpath(datapath, "models")
evalpath = joinpath(datapath, "eval")
mkpath(evalpath)

f1 = joinpath(basepath, "eval_tuning/eval/models_eval.csv")
f2 = joinpath(basepath, "opt_runs/eval/models_eval.csv")
f3 = joinpath(basepath, "negative_runs/eval/models_eval.csv")

ft1 = joinpath(basepath, "eval_tuning/eval/models_eval_testmode.csv")
ft2 = joinpath(basepath, "opt_runs/eval/models_eval_testmode.csv")
ft3 = joinpath(basepath, "negative_runs/eval/models_eval_testmode.csv")

df1 = CSV.read(f1)
df2 = CSV.read(f2)
df3 = CSV.read(f3)

dft1 = CSV.read(ft1)
dft2 = CSV.read(ft2)
dft3 = CSV.read(ft3)

df = df3

# compare batchnorm and no batchnorm runs
fl = joinpath(basepath, "ldim_runs/eval/models_eval_testmode.csv")
flnb = joinpath(basepath, "nobatchnorm_ldim_runs/eval/models_eval.csv")

# no batchnorm is clearly superior
dfl = CSV.read(fl)
dflnb = CSV.read(flnb)

# again
f = joinpath(basepath, "opt_runs/eval/models_eval_testmode.csv")
fnb = joinpath(basepath, "nobatchnorm_runs/eval/models_eval_testmode.csv")
ft = joinpath(basepath, "opt_runs/eval/models_eval.csv")

# no batchnorm is clearly superior
df = CSV.read(f)
dfnb = CSV.read(fnb)
dft = CSV.read(ft)

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

# get a model
models = readdir(modelpath)
mf = joinpath(modelpath, models[2])
model = GenModels.construct_model(mf) |> gpu
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

labels = testing_data["labels"];
patches = testing_data["patches"]  |> gpu;
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

inds = [3, 4, 5, 6]
plot_4(model, patches, scores, labels, inds)

inds = [1, 2, 300, 301]
plot_4(model, patches, scores, labels, inds)

inds = [1, 2, 180, 181]
plot_4(model, patches, scores, labels, inds)

inds = vcat(sortperm(scores)[1:2], sortperm(scores)[end-1:end])
plot_4(model, patches, scores, labels, inds)

# now test the model with testmode
tmodel = Flux.testmode!(model)

tscores = score_mse(tmodel, patches)
troc = EvalCurves.roccurve(tscores, labels)
tauroc = EvalCurves.auc(troc...)

inds = [3, 4, 5, 6]
plot_4(tmodel, patches, tscores, labels, inds)

inds = [1, 2, 300, 301]
plot_4(tmodel, patches, tscores, labels, inds)

inds = vcat(sortperm(tscores)[1:2], sortperm(tscores)[end-1:end])
plot_4(tmodel, patches, tscores, labels, inds)

