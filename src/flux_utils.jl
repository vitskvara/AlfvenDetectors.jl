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

"""
    iscuarray(X)

Is X a CuArray?
"""
function iscuarray(X) 
    if typeof(X) <: TrackedArray
        return (string(typeof(X.data)) in ["CuArray{$(Float),2}", "CuArray{$(Float),1}"])
    else
        return (string(typeof(X)) in ["CuArray{$(Float),2}", "CuArray{$(Float),1}"])
    end
end

# this should be done properly but I dont know how
# now it detects whether X is not a (Tracked)Array

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
    FluxModel

Abstract type to share some methods between models.
"""
abstract type FluxModel end

"""
    update(model, optimiser)

Update model parameters using optimiser.
"""
function update!(model, optimiser)
    for p in params(model)
        Δ = Flux.Optimise.apply!(optimiser, p.data, p.grad)
        p.data .-= Δ
        Δ .= 0
    end
end

"""
    train!(model, data, loss, optimiser, callback; [usegpu])

Basics taken from the Flux train! function. Callback is any function
of the remaining arguments that gets called every iteration - 
use it to store or print training progress, stop training etc. 
"""
function train!(model, data, loss, optimiser, callback; 
    usegpu = false, memoryefficient = false)
    for _data in data
        try
            if usegpu
             _data = _data |> gpu
            end
            l = loss(_data)
            Flux.Tracker.back!(l)
            update!(model, optimiser)
            # now call the callback function
            # can be an object so it can store some values between individual calls
            callback(model, _data, loss, optimiser)
        catch e
            # setup a special kind of exception for known cases with a break
            rethrow(e)
        end
        if memoryefficient
            GC.gc();
        end
    end
end

"""
    fast_callback(m::FluxModel, d, l, opt)

A callback for fast training with no overhead.
"""
fast_callback(m::FluxModel, d, l, opt) = nothing

"""
    basic_callback

Basic experimental callback doing lots of extra stuff, probably 
unnecesarily slow. Shows and stores current loss, maybe provides 
a stopping condition or changes learning rate. Is called in every 
loop in train! and serves to store and change information in 
between iterations.
"""
mutable struct basic_callback
    history
    eta::Real
    iter_counter::Int
    progress
    progress_vals
    verb::Bool
    epoch_size::Int
    show_it::Int
end

"""
    basic_callback(hist,verb::Bool,eta::Real,show_it::Int; 
        train_length::Int=0, epoch_size::Int=1)

Initial constructor.
"""
function basic_callback(hist,verb::Bool,eta::Real,show_it::Int; 
    train_length::Int=0, epoch_size::Int=1) 
    p = Progress(train_length, 0.3)
    basic_callback(hist,eta,0,p,Array{Any,1}(),verb,epoch_size,show_it)
end

#####################################################
### function for convolutional networks upscaling ###
#####################################################
"""
    oneszeros([T],segment,length,i)

Create a vector of type T of size `length*segment` where `i`th
segment is made out of ones and the rest is zero. 
"""
function oneszeros(T::DataType,segment,length,i)
    res = zeros(T,segment*length)
    res[((i-1)*segment+1):i*segment] = ones(T,segment)
    return res
end
function oneszeros(segment,length,i)
    res = zeros(segment*length)
    res[((i-1)*segment+1):i*segment] = ones(segment)
    return res
end
"""
    voneszeros([T,]segment,length,i)

Create a vector of type T of size `length*segment` where `i`th
segment is made out of ones and the rest is zero. 
"""
voneszeros(T::DataType,segment,length,i) = oneszeros(T,segment,length,i)
voneszeros(segment,length,i) = oneszeros(segment,length,i)

"""
    honeszeros([T,]segment,length,i)

Create a horizontal vector of type T of size `length*segment` where `i`th
segment is made out of ones and the rest is zero. 
"""
honeszeros(T::DataType,segment,length,i) = Array(voneszeros(T,segment,length,i)')
honeszeros(segment,length,i) = Array(voneszeros(segment,length,i)')

"""
    vscalemat([T,]scale,n)

Vertical scaling matrix. `Scale` is the (integer) scaling factor and `n` is the 
vertical size of the original matrix.
"""
vscalemat(T,scale::Int,n::Int) = hcat([voneszeros(T,scale,n,i) for i in 1:n]...)
vscalemat(scale::Int,n::Int) = hcat([voneszeros(scale,n,i) for i in 1:n]...)

"""
    hscalemat([T,]scale,n)

Horizontal scaling matrix. `Scale` is the (integer) scaling factor and `n` is the 
horizontal size of the original matrix.
"""
hscalemat(T,scale::Int,n::Int) = vcat([honeszeros(T,scale,n,i) for i in 1:n]...)
hscalemat(scale::Int,n::Int) = vcat([honeszeros(scale,n,i) for i in 1:n]...)

"""
    upscale(x::AbstractArray, scale)

Upscales a 2D array by the integer scales given in the `scale` tuple. 
Works for 3D and 4D array in the first two dimensions.
"""
function upscale(x::AbstractArray{T,2}, scale) where T
    m,n = size(x)
    V = vscalemat(T,scale[1],m)
    H = hscalemat(T,scale[2],n)
    return V*x*H
end
function upscale(x::AbstractArray{T,3}, scale) where T
    M,N,C = size(x)
    # this is important - the array must be of the same type as x, not T
    res = Array{eltype(x),3}(undef,M*scale[1],N*scale[2],C)
    for c in 1:C
        res[:,:,c] = upscale(x[:,:,c],scale)
    end
    return Tracker.collect(res)
end
function upscale(x::AbstractArray{T,4}, scale) where T
    M,N,C,K = size(x)
    # this is important - the array must be of the same type as x, not T
    res = Array{eltype(x),4}(undef,M*scale[1],N*scale[2],C,K)
    for c in 1:C
        for k in 1:K
            res[:,:,c,k] = upscale(x[:,:,c,k],scale)
        end
    end
    return Tracker.collect(res)
end

"""
    zeropad(x::AbstractArray,widths)

widths = [top, right, bottom, left] padding size
"""
function zeropad(x::AbstractArray{T,2},widths) where T
    M,N = size(x)
    # first do vertical padding
    y = [zeros(T, widths[1], N); x; zeros(T, widths[3], N)]
    # then the horizontal
    y = [zeros(T, M+widths[1]+widths[3], widths[4]) y zeros(T, M+widths[1]+widths[3], widths[2])]
    return y
end
function zeropad(x::AbstractArray{T,3},widths) where T
    M,N,C = size(x)
    res = Array{eltype(x),3}(undef,M+widths[1]+widths[3],N+widths[2]+widths[4],C)
    for c in 1:C
        res[:,:,c] = zeropad(x[:,:,c],widths)
    end
    return Tracker.collect(res)
end
function zeropad(x::AbstractArray{T,4},widths) where T
    M,N,C,K = size(x)
    res = Array{eltype(x),4}(undef,M+widths[1]+widths[3],N+widths[2]+widths[4],C,K)
    for c in 1:C
        for k in 1:K
            res[:,:,c,k] = zeropad(x[:,:,c,k],widths)
        end
    end
    return Tracker.collect(res)
end