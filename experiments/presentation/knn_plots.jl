using AlfvenDetectors
using Flux
using ValueHistories
using StatsBase
using GenerativeModels
using Dates
using BSON
using Random
using EvalCurves
using PyPlot
using DataFrames
using CSV

# plot params
outpath = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/presentation/images"
cmap = "plasma" # colormap
matplotlib.rc("font", family = "normal",
    weight = "bold",
    size = 16
)

# get some data
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
patchsize = 128
patch_f = joinpath(dirname(pathof(AlfvenDetectors)), 
	"../experiments/conv/data/labeled_patches_$patchsize.bson")
if isfile(patch_f)
	patchdata = BSON.load(patch_f)
	data = patchdata[:data];
	shotnos = patchdata[:shotnos];
	labels = patchdata[:labels];
	tstarts = patchdata[:tstarts];
	fstarts = patchdata[:fstarts];
else
	readfun = AlfvenDetectors.readnormlogupsd
	shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()
	patchdata = map(x->AlfvenDetectors.get_patch(datapath,x[1], x[2], x[3], patchsize, readfun;
		memorysafe = true)[1],	zip(shotnos, tstarts, fstarts))
	data = cat(patchdata..., dims=4)
end

modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/benchmarks"
modelpath = joinpath(modelpath, "waae_8_16_16_32_lambda-1.0_gamma-0.0_sigma-0.01/1")
models = readdir(modelpath)
#imode = 46
imodel = length(models)
mf = joinpath(modelpath,models[imodel])

# or load it directly
#mf="/home/vit/.julia/environments/v1.1/dev/AlfvenDetectors/experiments/conv/ConvAAE_channels-[2,2]_patchsize-128_nepochs-10_2019-05-06T10:03:25.287.bson"
#mf="./ConvWAE_channels-[2,2]_patchsize-128_nepochs-2_2019-05-07T09:16:59.027.bson"

# 
model_data = BSON.load(mf)
exp_args = model_data[:experiment_args]
model_args = model_data[:model_args]
model_kwargs = model_data[:model_kwargs]
history = model_data[:history]
if haskey(model_data, :model)
	model = model_data[:model]
else
	model = Flux.testmode!(GenerativeModels.construct_model(mf))
end

seed = model_data[:experiment_args]["seed"];
train_info, train_inds, test_info, test_inds = AlfvenDetectors.split_patches(0.5, shotnos, labels, tstarts, 
	fstarts; seed=seed);
train = (data[:,:,:,train_inds], train_info[2]);
test = (data[:,:,:,test_inds], test_info[2]);

# test the separation by fitting kNN
knn_model = AlfvenDetectors.KNN(:KDTree);
fx(m,x) = nothing # there is no point in fitting the unlabeled samples
fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y) ;
k = 5
asf(m,x) = AlfvenDetectors.as_mean(m,x,k);
fsmodel = AlfvenDetectors.FewShotModel(model, knn_model, fx, fxy, asf);
AlfvenDetectors.fit!(fsmodel, train[1], train[1], train[2]);
as = AlfvenDetectors.anomaly_score(fsmodel, test[1]);
auc = EvalCurves.auc(EvalCurves.roccurve(as, test[2])...)
println("AUC (kNN 5) = $auc")

# now save the most anomalous samples
fname = "anomalous_patches.png"
sortinds=sortperm(as,rev=true)
figure(figsize=(8,4))
for i in 1:4
	subplot(2,2,i)
	pcolormesh(test[1][:,:,1,sortinds[i]],cmap=cmap)
	ax = gca()
	ax.get_xaxis().set_visible(false)
	ax.get_yaxis().set_visible(false)
	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
end
tight_layout()
savefig(joinpath(outpath, fname),dpi=500)

# now do the comparison of knn on encoded data and on the original samples
auc_patches = CSV.read("auc_patches.csv")
auc_latent = CSV.read("auc_latent.csv")

# first plot everything over and over
fname = "knn_patches_vs_latent.eps"
figure()
function plot_lines(df, label, color)
	for seed in unique(df[:seed])
		subdf = filter(x->x[:seed]==seed, df)
		if seed==1
			plot(subdf[:k], subdf[:auc], label=label, c=color)
		else
			plot(subdf[:k], subdf[:auc], c=color)
		end
	end
end
plot_lines(auc_patches, "full patches", "r")
plot_lines(auc_latent, "latent", "b")
ylim([0.6, 1.0])
xlabel("k")
ylabel("AUC")
legend()
tight_layout()
savefig(joinpath(outpath, fname))

# now plot means and sds
function plot_mean_sd(df, label, color, nsd)
	kvec = unique(df[:k])
	means = map(k->StatsBase.mean((filter(row->row[:k]==k,df))[:auc]), kvec)
	sds = map(k->sqrt(StatsBase.var((filter(row->row[:k]==k,df))[:auc])), kvec)
	plot(kvec, means, label=label, c=color)
	fill_between(kvec, means-nsd*sds, means+nsd*sds, color=color, alpha=0.3, linewidth=0)
end
figure()
plot_mean_sd(auc_patches, "full patches", "r",1)
plot_mean_sd(auc_latent, "latent", "b",1)
ylim([0.6, 1.0])
xlabel("k")
ylabel("AUC")
legend()
tight_layout()
fname = "knn_patches_vs_latent_means.pdf"
savefig(joinpath(outpath, fname))
