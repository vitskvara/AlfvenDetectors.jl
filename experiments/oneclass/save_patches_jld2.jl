# this is to load data for oneclass experiment and save them to a different format 
# than h5 due to memory issues
using JLD2
using FileIO
using AlfvenDetectors
datapath = "/compass/home/skvara/no-backup/uprobe_data"

# settings
try
	global nargs = length(ARGS)
catch e
	global nargs = 0
end
seed = (nargs > 0) ? Int(Meta.parse(ARGS[1])) : 1
npatches = (nargs > 1) ? Int(Meta.parse(ARGS[2])) : 10
normalized = false
patchsize = 128
if normalized
	readfun = AlfvenDetectors.readnormlogupsd
else
	readfun = AlfvenDetectors.readlogupsd
end

# read the data
patches, shotnos, labels, tstarts, fstarts = 
	AlfvenDetectors.collect_training_data_oneclass(datapath, npatches, readfun, patchsize; 
		α = 0.8, seed=seed)

# save them
outpath = "/compass/home/skvara/no-backup/oneclass_data/training"
if normalized
	outpath = joinpath(outpath, "$(patchsize)_normalized")
else
	outpath = joinpath(outpath, "$(patchsize)")
end
mkpath(outpath)
fname = joinpath(outpath, "seed-$(seed).jld2")
if isfile(fname)
	jlddata = load(fname)
	patches = cat(jlddata["patches"], patches, dims=4)
	shotnos = cat(jlddata["shotnos"], shotnos, dims=1)
	labels = cat(jlddata["labels"], labels, dims=1)
	tstarts = cat(jlddata["tstarts"], tstarts, dims=1)
	fstarts = cat(jlddata["fstarts"], fstarts, dims=1)
end
save(fname, Dict(
	"patches" => patches,
	"shotnos" => shotnos,
	"labels" => labels,
	"tstarts" => tstarts,
	"fstarts" => fstarts
	))
