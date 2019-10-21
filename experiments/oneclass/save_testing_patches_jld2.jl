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
normalized = true
patchsize = 128
if normalized
	readfun = AlfvenDetectors.readnormlogupsd
else
	readfun = AlfvenDetectors.readlogupsd
end

# read the data
patches, shotnos, labels, tstarts, fstarts = 
	AlfvenDetectors.collect_testing_data_oneclass(datapath, readfun, patchsize; α = 0.8, seed=seed);

# save them
outpath = "/compass/home/skvara/no-backup/oneclass_data/testing"
if normalized
	outpath = joinpath(outpath, "$(patchsize)_normalized")
else
	outpath = joinpath(outpath, "$(patchsize)")
end
mkpath(outpath)
fname = joinpath(outpath, "seed-$(seed).jld2")
save(fname, Dict(
	"patches" => patches,
	"shotnos" => shotnos,
	"labels" => labels,
	"tstarts" => tstarts,
	"fstarts" => fstarts
	))
