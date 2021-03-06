using AlfvenDetectors
using Flux
using ValueHistories
using StatsBase
using GenModels
using Dates
using BSON
using Random
using EvalCurves
#using PyPlot
#using CuArrays
plotstuff = false

if plotstuff
	using Plots
	plotly()
end

# now get some data
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

#
modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/benchmarks"
#subpath = subpath*"/1"
subpath = "waae_8_16_16_32_lambda-10.0_gamma-0.0_sigma-1.0/1"
mpath = joinpath(modelpath, subpath) 
models = readdir(mpath)
#imode = 46
imodel = length(models)
mf = joinpath(mpath,models[imodel])

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
	model = Flux.testmode!(GenModels.construct_model(mf))
end

if plotstuff
	# plot training history
	for key in keys(history)
		is,ls = get(history, key)
		plot!(is,ls,label=string(key)*"=$(ls[end])")
	end
	title!("")

	# look at the Z space
	batchsize = 128
	Zqz = GenModels.encode(model, data, batchsize).data;
	ldim = size(Zqz,1)
	if :pz in fieldnames(typeof(model))
		Zpz = model.pz(1000);
	else
		Zpz = Array{Float32,2}(undef,ldim,0)
	end
	# if the ldim is larger than 3, reduce it with UMAP for plotting purposes
	if ldim > 3
		pdim = 3
		global umap_model = AlfvenDetectors.UMAP(pdim, n_neighbors=5, min_dist=0.4)
		Zt = AlfvenDetectors.fit!(umap_model, hcat(Zpz, Zqz));
		Zpzp = Zt[:,1:size(Zpz,2)];
		Zqzp = Zt[:,size(Zpz,2)+1:end];
	else
		pdim = ldim
		Zpzp = copy(Zpz);
		Zqzp = copy(Zqz);
	end
	if pdim == 3
		scatter(Zpzp[1,:],Zpzp[2,:],Zpzp[3,:], label="samples from pz")
		scatter!(Zqzp[1,:],Zqzp[2,:],Zqzp[3,:], label="encoded data")
	else
		scatter(Zpzp[1,:],Zpzp[2,:], label="samples from pz")
		scatter!(Zqzp[1,:],Zqzp[2,:], label="encoded data")
	end
	if pdim == 3
		scatter(Zqzp[1,labels.==1],Zqzp[2,labels.==1],Zqzp[3,labels.==1], label="alfven data")
		scatter!(Zqzp[1,labels.==0],Zqzp[2,labels.==0],Zqzp[3,labels.==0], label="no alfven data")
	else
		scatter(Zqzp[1,labels.==1],Zqzp[2,labels.==1], label="alfven data")
		scatter!(Zqzp[1,labels.==0],Zqzp[2,labels.==0], label="no alfven data")
	end
	plot()
end

# get  training and testing data
seed = model_data[:experiment_args]["seed"];
train_info, train_inds, test_info, test_inds = AlfvenDetectors.split_patches_unique(0.5, shotnos, 
	labels, tstarts, fstarts; seed=seed);
train = (data[:,:,:,train_inds], train_info[2]);
test = (data[:,:,:,test_inds], test_info[2]);

# test the separation by fitting kNN
knn_model = AlfvenDetectors.KNN(:KDTree);
fx(m,x) = nothing # there is no point in fitting the unlabeled samples
fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y) ;
k = 3
asf(m,x) = AlfvenDetectors.as_mean(m,x,k);
fsmodel = AlfvenDetectors.FewShotModel(model, knn_model, fx, fxy, asf);
AlfvenDetectors.fit!(fsmodel, train[1], train[1], train[2]);
as = AlfvenDetectors.anomaly_score(fsmodel, test[1]);
auc = EvalCurves.auc(EvalCurves.roccurve(as, test[2])...)
println("AUC (kNN 3) = $auc")

















if false
	# check some reconstructions
	ipatch = 10
	cmap = "plasma"
	patch = data[:,:,:,ipatch:ipatch]
	rpatch = model(patch).data
	e = Flux.mse(patch, rpatch)
	figure(figsize=(10,5))
	subplot(121)
	title("original")
	pcolormesh(patch[:,:,1,1],cmap=cmap)
	subplot(122)
	title("reconstruction, error = $(e)")
	pcolormesh(rpatch[:,:,1,1],cmap=cmap)

	# check the reconstruction error progress
	is, ls = get(history, :aeloss)
	plot(ls[1000:end])
end