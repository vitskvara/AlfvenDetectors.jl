"""
    adapt(T, m)

Creates a copy of m (a chain or a single layer) with all inner arrays converted to type T.
"""
adapt(T, m) = mapleaves(x -> Adapt.adapt(T, x), m)

"""
    freeze(m)

Creates a non-trainable copy of a Flux object.
"""
freeze(m) = Flux.mapleaves(Flux.Tracker.data,m)

# from FluxExtensions
"""
    function layerbuilder(d::Int,k::Int,o::Int,n::Int,ftype::String,lastlayer::String = "",ltype::String = "Dense")

Create a chain with `n` layers of with `k` neurons with transfer function `ftype`.
Input and output dimension is `d` / `o`.
If lastlayer is no specified, all layers use the same function.
If lastlayer is "linear", then the last layer is forced to be Dense.

It is also possible to specify dimensions in a vector.
"""
layerbuilder(k::Vector{Int},l::Vector,f::Vector) = Flux.Chain(map(i -> i[1](i[3],i[4],i[2]),zip(l,f,k[1:end-1],k[2:end]))...)

layerbuilder(d::Int,k::Int,o::Int,n::Int, args...) =
    layerbuilder(vcat(d,fill(k,n-1)...,o), args...)

function layerbuilder(ks::Vector{Int},ftype::String,lastlayer::String = "",ltype::String = "Dense")
    ftype = (ftype == "linear") ? "identity" : ftype
    ls = Array{Any}(fill(eval(:($(Symbol(ltype)))),length(ks)-1))
    fs = Array{Any}(fill(eval(:($(Symbol(ftype)))),length(ks)-1))
    if !isempty(lastlayer)
        fs[end] = (lastlayer == "linear") ? identity : eval(:($(Symbol(lastlayer))))
        ls[end] = (lastlayer == "linear") ? Dense : ls[end]
    end
    layerbuilder(ks,ls,fs)
end

"""
    aelayerbuilder(lsize, activation, layer)

Construct encoder/decoder using a layer builder. Output of last layer
is always identity.
"""
aelayerbuilder(lsize::Vector, activation, layer) = adapt(Float, 
    layerbuilder(lsize, 
        Array{Any}(fill(layer, size(lsize,1)-1)), 
        Array{Any}([fill(activation, size(lsize,1)-2); identity]))
    )


"""
    train!(model, data, loss, optimiser, callback)

Basics taken from the Flux train! function. Callback is any function
of the remaining arguments that gets called every iteration - 
use it to store or print training progress, stop training etc. 
"""
function train!(model, data, loss, optimiser, callback)
    for _data in data
        try
            l = loss(_data)
            Flux.Tracker.back!(l)
            for p in params(model)
                Δ = Flux.Optimise.apply!(optimiser, p.data, p.grad)
                p.data .-= Δ
                Δ .= 0
            end
            # now call the callback function
            # can be an object so it can store some values between individual calls
            callback(model, _data, loss, optimiser)
        catch e
            # setup a special kind of exception for known cases with a break
            rethrow(e)
        end
    end
end