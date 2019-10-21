using PyPlot

include("eval.jl")

# get the paths
hostname = gethostname()
if hostname == "gpu-node"
	evaldatapath = "/compass/home/skvara/no-backup/oneclass_data" 
	datapath = "/compass/home/skvara/alfven/experiments/oneclass/opt_runs"
else
	evaldatapath = "/home/vit/vyzkum/alfven/cdb_data/oneclass_data" 
	datapath = "/home/vit/vyzkum/alfven/experiments/oneclass/opt_runs"
end
modelpath = joinpath(datapath, "models")
evalpath = joinpath(datapath, "eval")
mkpath(evalpath)


f1 = "/compass/home/skvara/alfven/experiments/oneclass/eval_tuning/eval/models_eval.csv"
f2 = "/compass/home/skvara/alfven/experiments/oneclass/opt_runs/eval/models_eval.csv"

df1 = CSV.read(f1)
df2 = CSV.read(f2)

df = df2

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
model = GenModels.construct_model(mf)
model_data = load(mf)
exp_args = model_data[:experiment_args]
seed = exp_args["seed"]
norms = exp_args["unnormalized"] ? "" : "_normalized"
testing_data = load(joinpath(evaldatapath, "testing/128$(norms)/seed-$(seed).jld2"));
training_data = load(joinpath(evaldatapath, "training/128$(norms)/seed-$(seed).jld2"));
training_patches = training_data["patches"];
labels = 1 .- testing_data["labels"]; # switch the labels here - positive class is actually the normal one
patches = testing_data["patches"];
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

inds = [3, 4, 5, 6]
plot_4(model, patches, scores, labels, inds)

inds = [1, 2, 180, 181]
plot_4(model, patches, scores, labels, inds)

inds = vcat(sortperm(scores)[1:2], sortperm(scores)[end-1:end])
plot_4(model, patches, scores, labels, inds)

