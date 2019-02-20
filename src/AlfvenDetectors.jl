module AlfvenDetectors

using Flux
using CuArrays
using HDF5
using Suppressor
using ValueHistories
using Adapt

const Float = Float32

include("data.jl")
include("flux_utils.jl")
include("samplers.jl")
include("ae.jl")

end # module
