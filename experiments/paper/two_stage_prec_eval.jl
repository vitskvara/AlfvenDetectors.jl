using Distributed
@everywhere begin
	using AlfvenDetectors
	using GenerativeModels
    using ValueHistories
    using StatsBase
end
# savepath
hostname = gethostname()
if hostname == "vit-ThinkPad-E470"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
	modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/benchmarks"
	savepath = "/home/vit/vyzkum/alfven/experiments/eval/conv/uprobe/benchmarks/individual_experiments"
	savepath = "."
elseif hostname == "tarbik.utia.cas.cz"
	datapath = "/home/skvara/work/alfven/cdb_data/uprobe_data"
	modelpath = "/home/skvara/work/alfven/experiments/conv/uprobe/benchmarks"
	savepath = "/home/skvara/work/alfven/experiments/eval/conv/uprobe/benchmarks/individual_experiments"
elseif occursin("soroban", hostname)
	datapath = "/compass/home/skvara/no-backup/uprobe_data"
	modelpath = "/compass/home/skvara/alfven/experiments/conv/uprobe/benchmarks_limited"
	savepath = "/compass/home/skvara/alfven/experiments/eval/conv/uprobe/benchmarks_limited/individual_experiments"
end
mkpath(savepath)

# models and their adresses
models = joinpath.(modelpath, readdir(modelpath))
Nmodels = length(models)
println("Found a total of $(Nmodels) saved models.")

# get labeled data
patchsize = 128
data, shotnos, labels, tstarts, fstarts = AlfvenDetectors.get_labeled_validation_data(patchsize);
println("loaded labeled validation data of size $(size(data)), with $(sum(labels)) positively labeled "*
 "samples and $(length(labels)-sum(labels)) negatively labeled samples")

# set the same number of unlabeled shots used for training the second stage - same for all models
unlabeled_nshots = 0

pmap(mf->AlfvenDetectors.eval_save(mf, AlfvenDetectors.fit_knn, "KNN", data, shotnos, labels, 
	tstarts, fstarts, savepath, datapath, unlabeled_nshots), models)

# fit GMM
unlabeled_nshots = 50

# get the motherfrickin model file and do the magic
# possibly paralelize this
map(mf->AlfvenDetectors.eval_save(mf, AlfvenDetectors.fit_gmm, "GMM", data, shotnos, labels, 
	tstarts, fstarts, savepath, datapath, unlabeled_nshots), models)
