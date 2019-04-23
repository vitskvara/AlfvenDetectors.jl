module AlfvenDetectors

using Flux
using HDF5
using Suppressor
using ValueHistories
using Adapt
using StatsBase # for samplers
using ProgressMeter
using Dates # for experiments
using BSON
using SparseArrays
using DelimitedFiles
using Random
# for alternative .h5 file loading
using PyCall
# PyCall modules are pointers
const h5py = PyNULL()
# used in the get_signal() function
function _init_h5py(warns=true)
	try
		copy!(h5py, pyimport("h5py"))
	catch e
		warns ? @warn("h5py Python library was not loaded and memory safe .h5 loading is therefore not available") : nothing
	end
end

const Float = Float32

include("data.jl")
include("samplers.jl")
include("flux_utils.jl")
include("model_utils.jl")
include("ae.jl")
include("vae.jl")
include("tsvae.jl")
include("experiments.jl")

end # module
