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
normalized = true
patchsize = 128
if normalized
	readfun = AlfvenDetectors.readnormlogupsd
else
	readfun = AlfvenDetectors.readlogupsd
end

# read the data
shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()
available_inds = AlfvenDetectors.filter_available_shots(datapath, shotnos)
shotnos, labels, tstarts, fstarts = map(x->x[available_inds], 
	(shotnos, labels, tstarts, fstarts))

# get the final data
patches = map(x->AlfvenDetectors.get_patch(datapath, x[1], x[2], x[3], patchsize, readfun; 
	memorysafe=true)[1], zip(shotnos, tstarts, fstarts))
patches = cat(patches..., dims=4)

# save them
outpath = "/compass/home/skvara/no-backup/oneclass_data_negative/testing"
if normalized
	outpath = joinpath(outpath, "$(patchsize)_normalized")
else
	outpath = joinpath(outpath, "$(patchsize)")
end
mkpath(outpath)
fname = joinpath(outpath, "data.jld2")
save(fname, Dict(
	"patches" => patches,
	"shotnos" => shotnos,
	"labels" => labels,
	"tstarts" => tstarts,
	"fstarts" => fstarts
	))
