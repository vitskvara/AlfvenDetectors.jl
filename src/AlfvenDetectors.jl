module AlfvenDetectors

using Flux
using HDF5
using Suppressor
using ValueHistories
using Adapt
using StatsBase # for samplers
using ProgressMeter

const Float = Float32

include("data.jl")
include("flux_utils.jl")
include("samplers.jl")
include("ae.jl")
include("vae.jl")

end # module
