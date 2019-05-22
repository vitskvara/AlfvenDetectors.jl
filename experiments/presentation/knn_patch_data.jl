using AlfvenDetectors
using Flux
using ValueHistories
using StatsBase
using GenerativeModels
using Dates
using BSON
using Random
using EvalCurves
using DataFrames
using CSV

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

# now do the same with raw data and kNN
auc_patches=[]
for seed in 1:10
	train_info, train_inds, test_info, test_inds = AlfvenDetectors.split_patches_unique(0.5, shotnos, labels, tstarts, 
		fstarts; seed=seed);
	# now vectorize the inputs
	train = (reshape(data[:,:,:,train_inds],:,length(train_info[2])), train_info[2]);
	test = (reshape(data[:,:,:,test_inds],:,length(test_info[2])), test_info[2]);

	# do the knn classification
	knn_model = AlfvenDetectors.KNN(:BruteTree);
	println("fit $seed")
	@time AlfvenDetectors.fit!(knn_model,train[1],train[2]) ;
	kvec = collect(1:2:21)
	aucs = []
	println("predict $seed")
	@time for k in kvec
		as = AlfvenDetectors.as_mean(knn_model,test[1],k);
		auc = EvalCurves.auc(EvalCurves.roccurve(as, test[2])...)
		push!(aucs, auc)
	end
	df = DataFrame(seed=fill(seed, length(aucs)), k=kvec, auc=aucs)
	push!(auc_patches, df)
end
auc_patches = vcat(auc_patches...)
fname = "auc_patches_unique.csv"
CSV.write(fname, auc_patches)
