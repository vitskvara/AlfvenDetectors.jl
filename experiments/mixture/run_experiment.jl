include("devamp.jl")
using AlfvenDetectors
using BSON

hostname = gethostname()
if hostname == "vit-ThinkPad-E470"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
	savepath = "/home/vit/vyzkum/alfven/experiments/oneclass"
elseif hostname == "tarbik.utia.cas.cz"
	datapath = "/home/skvara/work/alfven/cdb_data/uprobe_data"
	savepath = "/home/skvara/work/alfven/experiments/oneclass"
elseif occursin("soroban", hostname) || hostname == "gpu-node"
	datapath = "/compass/home/skvara/no-backup/uprobe_data"
	savepath = "/compass/home/skvara/alfven/experiments/oneclass"
end

normalize = false

norms = normalize ? "_normalized" : ""
fname = joinpath(dirname(datapath), "oneclass_data/training/$(patchsize)$(norms)/seed-$(seed).jld2")
patches, shotnos, labels, tstarts, fstarts = 
	AlfvenDetectors.oneclass_training_data_jld(fname, npatches)

