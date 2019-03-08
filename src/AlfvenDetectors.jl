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
include("samplers.jl")
include("flux_utils.jl")
include("model_utils.jl")
include("ae.jl")
include("vae.jl")
include("tsvae.jl")

end # module
