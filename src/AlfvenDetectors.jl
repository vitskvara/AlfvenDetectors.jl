module AlfvenDetectors

using HDF5
using Suppressor
using ValueHistories
using Flux
using StatsBase # for samplers
using Dates # for experiments
using BSON
using DelimitedFiles
using Random
using GenerativeModels
using Combinatorics
#using PyPlot
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
import Base.collect
import StatsBase.sample

const Float = Float32

include("data.jl")
include("experiments.jl")
include("distributions.jl")
include("evaluation.jl")

end # module
