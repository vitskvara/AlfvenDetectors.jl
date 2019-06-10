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
	modelpath = "/compass/home/skvara/alfven/experiments/conv/uprobe/benchmarks"
	savepath = "/compass/home/skvara/alfven/experiments/eval/conv/uprobe/benchmarks/individual_experiments"
end
mkpath(savepath)

# MAIN 

# models and their adresses
exdirs1 = joinpath.(modelpath,readdir(modelpath));
exdirs2 = vcat(map(x->joinpath.(x,readdir(x)), exdirs1)...);
filter!(x->length(readdir(x))!=0,exdirs2)
models = vcat(map(x->joinpath.(x,readdir(x)[end]), exdirs2)...);
Nmodels = length(models)
println("Found a total of $(Nmodels) saved models.")

# get labeled data
patchsize = 128
data, shotnos, labels, tstarts, fstarts = AlfvenDetectors.get_labeled_validation_data(patchsize);
println("loaded labeled validation data of size $(size(data)), with $(sum(labels)) positively labeled "*
 "samples and $(length(labels)-sum(labels)) negatively labeled samples")
