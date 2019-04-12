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
        return (string(typeof(X.data)) in ["CuArray{$(Float),$i}" for i in 1:10])
    else
        return (string(typeof(X)) in ["CuArray{$(Float),$i}" for i in 1:10])
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

######################################################
### functions for convolutional networks upscaling ###
######################################################
"""
    oneszeros([T],segment,length,i)

Create a vector of type T of size `length*segment` where `i`th
segment is made out of ones and the rest is zero. 
"""
function oneszeros(T::DataType,segment::Int,length::Int,i::Int)
    res = zeros(T,segment*length)
    res[((i-1)*segment+1):i*segment] = ones(T,segment)
    return res
end
function oneszeros(segment::Int,length::Int,i::Int)
    res = zeros(segment*length)
    res[((i-1)*segment+1):i*segment] = ones(segment)
    return res
end
"""
    voneszeros([T,]segment,length,i)

Create a vector of type T of size `length*segment` where `i`th
segment is made out of ones and the rest is zero. 
"""
voneszeros(T::DataType,segment::Int,length::Int,i::Int) = oneszeros(T,segment,length,i)
voneszeros(segment::Int,length::Int,i::Int) = oneszeros(segment,length,i)

"""
    honeszeros([T,]segment,length,i)

Create a horizontal vector of type T of size `length*segment` where `i`th
segment is made out of ones and the rest is zero. 
"""
honeszeros(T::DataType,segment::Int,length::Int,i::Int) = Array(voneszeros(T,segment,length,i)')
honeszeros(segment::Int,length::Int,i::Int) = Array(voneszeros(segment,length,i)')

"""
    vscalemat([T,]scale,n)

Vertical scaling matrix. `Scale` is the (integer) scaling factor and `n` is the 
vertical size of the original matrix.
"""
vscalemat(T,scale::Int,n::Int) = hcat([voneszeros(T,scale,n,i) for i in 1:n]...)
vscalemat(scale::Int,n::Int) = hcat([voneszeros(scale,n,i) for i in 1:n]...)
vscalemat_sparse(T,scale::Int, n::Int) = sparse(1:scale*n, vcat([fill(i,scale) for i in 1:n]...), fill(one(T),scale*n))
vscalemat_sparse(scale::Int, n::Int) = sparse(1:scale*n, vcat([fill(i,scale) for i in 1:n]...), fill(1.0,scale*n))

"""
    hscalemat([T,]scale,n)

Horizontal scaling matrix. `Scale` is the (integer) scaling factor and `n` is the 
horizontal size of the original matrix.
"""
hscalemat(T,scale::Int,n::Int) = vcat([honeszeros(T,scale,n,i) for i in 1:n]...)
hscalemat(scale::Int,n::Int) = vcat([honeszeros(scale,n,i) for i in 1:n]...)
hscalemat_sparse(T,scale::Int, n::Int) = sparse(vcat([fill(i,scale) for i in 1:n]...), 1:scale*n, fill(one(T),scale*n))
hscalemat_sparse(scale::Int, n::Int) = sparse(vcat([fill(i,scale) for i in 1:n]...), 1:scale*n, fill(1.0,scale*n))

"""
    upscale(x::AbstractArray, scale)

Upscales a 2D array by the integer scales given in the `scale` tuple. 
Works for 3D and 4D array in the first two dimensions.
"""
function upscale(x::AbstractArray{T,2}, scale::Tuple{Int,Int}) where T
    m,n = size(x)
    # using the sparse matrices here only slows things down
    V = vscalemat(T,scale[1],m)
    H = hscalemat(T,scale[2],n)
    return V*x*H
end
function upscale(x::AbstractArray{T,3}, scale::Tuple{Int,Int}) where T
    M,N,C = size(x)
    # this is important - the array must be of the same type as x, not T
    res = Array{eltype(x),3}(undef,M*scale[1],N*scale[2],C)
    for c in 1:C
        res[:,:,c] = upscale(x[:,:,c],scale)
    end
    return Tracker.collect(res)
end
function upscale(x::AbstractArray{T,4}, scale::Tuple{Int,Int}) where T
    M,N,C,K = size(x)
    # this is important - the array must be of the same type as x, not T
    res = Array{eltype(x),4}(undef,M*scale[1],N*scale[2],C,K)
    for k in 1:K
        for c in 1:C
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
    for k in 1:K
        for c in 1:C
            res[:,:,c,k] = zeropad(x[:,:,c,k],widths)
        end
    end
    return Tracker.collect(res)
end

"""
    convmaxpool(ks, channels, scales; [activation, stride, batchnorm])

Create a simple two layered net consisting of a convolutional layer
and a subsequent dowscaling using maxpooling layer.

    layer = convmaxpool(3, 8=>16, (4,4))

This will have convolutional kernel of size (3,3), produce 16 channels out of 8 and 
downscale 4 time in each dimension. If batchnorm is true, batch normalisation is going to be used.
"""
function convmaxpool(ks::Int, channels::Pair, scales::Union{Tuple,Int}; 
    activation = relu, stride::Int=1, batchnorm = false)
    if !(typeof(scales) <: Tuple)   
        scales = (scales,scales)
    end
    padwidth = floor(Int,ks/2)
    if batchnorm
        return Flux.Chain(
                    Flux.Conv((ks,ks), channels, activation, pad=(padwidth,padwidth),
                        stride = (stride,stride)),
                    x->maxpool(x,scales),
                    BatchNorm(channels[2])
                )
    else
        return Flux.Chain(
                    Flux.Conv((ks,ks), channels, activation, pad=(padwidth,padwidth),
                        stride = (stride,stride)),
                    x->maxpool(x,scales)
                )
    end
end

"""
    convencoder(ins,ds,das,ks,cs,scs,as,sts,bns[,lastbatchnorm])

Create a convolutional encoder with dense output layer(s).

ins = (height,width,no channels) of input
ds = vector of widths of dense layers    
das = vector of activations of dense layers - 1 element shorter then ds (last is always unit)
ks = vector of kernel sizes
cs = vector of channel pairs
scs = vector of scale factors
cas = vector of convolutional activations
sts = vector of strides
bns = binary vector of batchnorm usage
outbatchnorm [false] = boolean - should batchnorm be used after the output layer?
"""
function convencoder(ins,ds::AbstractVector, das::AbstractVector,
    ks::AbstractVector, cs::AbstractVector, 
    scs::AbstractVector, cas::AbstractVector, sts::AbstractVector,
    bns::AbstractVector; outbatchnorm=false)
    conv_layers = Flux.Chain(map(x->convmaxpool(x[1],x[2],x[3];activation=x[4],stride=x[5],batchnorm=x[6])
        ,zip(ks,cs,scs,cas,sts,bns))...)
    # there is a problem with automatic determination of the input size of the last dense layer
    # it can be computed from the indims and the size, padding, stride and scale of the convmaxpool layer
    # however that is quit complicated so lets use a trick - we initialize a random array of indim size,
    # pass it through the conv layers and get its size
    testin = randn(Float32,ins...,1)
    testout = conv_layers(testin)
    convoutdim = reduce(*,size(testout)[1:3]) # size of the first dense layer
    ds = vcat([convoutdim], ds)
    ndl = length(ds)-1 # no of dense layers
    # and also the previous ones if needed
    if outbatchnorm
        dense_layers = Flux.Chain(
                map(i->Flux.Dense(ds[i],ds[i+1],((i==ndl) ? identity : das[i])),1:ndl)...,
                BatchNorm(ds[end]))
    else
        dense_layers = Flux.Chain(
                map(i->Flux.Dense(ds[i],ds[i+1],((i==ndl) ? identity : das[i])),1:ndl)...)
    end
    # finally put it all into one chain
    Flux.Chain(conv_layers,
        x->reshape(x,:,size(x,4)),
        dense_layers
        )
end
"""
    convencoder(insize, latentdim, nconv, kernelsize, channels, scaling,
        [ndense, dsizes, activation, stride, batchnorm, outbatchnorm])

Create a convolutional encoder.

insize = (height,width,no channels) of input
latentdim = size of latent space
nconv = number of conv layers
kernelsize = scalar, tuple or a list of those
channels = a list of channel numbers, same length as nconv
scaling = scalar, tuple or a list of those
ndense = number of dense layers (default 1)
dsizes = if ndense > 1, specify a list of latent layer widths of length = ndense - 1
activation = default relu
lstride = length of stride, default 1, can be a scalar or a list of scalars
batchnorm = boolean
outbatchnorm = boolean - should batchnorm be used after the output layer?
"""
function convencoder(insize, latentdim::Int, nconv::Int, kernelsize, channels, 
    scaling; ndense::Int=1, dsizes=nothing, activation=relu, lstride=1, batchnorm=false,
    outbatchnorm=false)
    # construct ds - vector of widths of dense layers
    if ndense>1
        (dsizes==nothing) ? error("If more than one Dense layer is require, specify their widths in dsizes.") : nothing 
        ds = vcat(dsizes, [latentdim])
    else
        ds = [latentdim]
    end
    # das - vector of dense activations
    das = fill(activation, length(ds)-1)
    # construct ks - vector of kernel sizes
    if typeof(kernelsize) <: AbstractVector
        @assert length(kernelsize) == nconv
        ks = kernelsize
    else
        ks = fill(kernelsize,nconv)
    end
    # construct cs - vector of channel pairs
    # first does not have to be specified as it is in insize
    @assert length(channels)==nconv
    cs = vcat(insize[3]=>channels[1], map(i->channels[i]=>channels[i+1],1:length(channels)-1))
    # construct scs - vector of scaling factors
    if typeof(scaling) <: AbstractVector
        @assert length(scaling) == nconv
        scs = scaling
    else
        scs = fill(scaling,nconv)
    end
    # cas - vector of convolutional activations
    cas = fill(activation, nconv)
    # sts - vector of strides
    if typeof(lstride) <: AbstractVector
        @assert length(lstride) == nconv
        sts = lstride
    else
        sts = fill(lstride,nconv)
    end
    # bns - vector of batch norm usage
    if typeof(batchnorm) <: AbstractVector
        @assert length(batchnorm) == nconv
        bns = batchnorm
    else
        bns = fill(batchnorm,nconv)
    end
    
    return convencoder(insize, ds, das, ks, cs, scs, cas, sts, bns; outbatchnorm=outbatchnorm) 
end

"""
    upscaleconv(ks, channels, scales [,activation, stride, batchnorm]

Upscaling coupled with convolution.

    layer = upscaleconv(5, 4=>2, 2)

This will upscale the input in x and y two times and then apply 
a kernel of size 5 to reduce the number of channels from 4 to 2.
"""
function upscaleconv(ks::Int, channels::Pair, scales::Union{Tuple,Int};
    activation = relu, stride::Int=1, batchnorm = false)
    if !(typeof(scales) <: Tuple)   
        scales = (scales,scales)
    end
    padwidth = floor(Int,ks/2)
    if batchnorm
        return Flux.Chain(
                BatchNorm(channels[1]),
                x -> upscale(x,scales),
                Flux.Conv((ks,ks), channels, activation; pad=(padwidth,padwidth),
                    stride = (stride,stride))
        )
    else
        return Flux.Chain(
                    x -> upscale(x,scales),
                    Flux.Conv((ks,ks), channels, activation; pad=(padwidth,padwidth),
                        stride = (stride,stride))
            )
    end
end

"""
    convtransposeconv(ks, channels, scales [,activation, stride, batchnorm]

Transposed convolution followed by convolution for upscaling.

    layer = convtransposeconv(5, 4=>2, 2)

This will upscale the input in x and y two times and then apply 
a kernel of size 5 to reduce the number of channels from 4 to 2.
"""
function convtransposeconv(ks::Int, channels::Pair, scales::Union{Tuple,Int};
    activation = relu, stride::Int=1, batchnorm=false)
    if !(typeof(scales) <: Tuple)   
        scales = (scales,scales)
    end
    padwidth = floor(Int,ks/2)
    if batchnorm
        return Flux.Chain(
                BatchNorm(channels[1]),
                Flux.ConvTranspose(scales, channels[1]=>channels[1], stride=scales),
                Flux.Conv((ks,ks), channels, activation; pad=(padwidth,padwidth),
                    stride = (stride,stride))
        )
    else
        return Flux.Chain(
                    Flux.ConvTranspose(scales, channels[1]=>channels[1], stride=scales),
                    Flux.Conv((ks,ks), channels, activation; pad=(padwidth,padwidth),
                        stride = (stride,stride))
            )
    end
end

"""
    convdecoder(ins,ds,das,ks,cs,scs,as,sts,bns [,layertype])

Create a convolutional decoder with dense input layer(s).

outs = (height,width,no channels) of output
ds = vector of widths of dense layers    
das = vector of activations of dense layers
ks = vector of kernel sizes
cs = vector of channel pairs
scs = vector of scale factors
cas = vector of convolutional activations one shorter than the rest (last activation is always identity)
sts = vector of strides
bns = boolean vector of batch normalization switches
layertype = one of ["transpose", "upscale"]
"""
function convdecoder(outs, ds::AbstractVector, das::AbstractVector, ks::AbstractVector, 
    cs::AbstractVector, scs::AbstractVector, cas::AbstractVector, sts::AbstractVector,
    bns::AbstractVector; layertype = "transpose")
    # the second one should be basically just a reversed first one
    conv_layers = Flux.Chain(map(x->convmaxpool(x[1],x[2],x[3];activation=x[4],stride=x[5]),
        zip(reverse(ks),map(x->x[2]=>x[1],reverse(cs)),
            reverse(scs),vcat(reverse(cas),[relu]),reverse(sts)))...)
    # the last output activation is identity
    # the x[3].*x[5] is there for non-unit strides
    cas = vcat(cas, [identity])
    @assert layertype in ["transpose", "upscale"]
    if layertype == "transpose"
        uplayer = convtransposeconv
    elseif layertype == "upscale"
        uplayer = upscaleconv
    end
    upscaleconv_layers = Flux.Chain(map(x->uplayer(x[1],x[2],x[3].*x[5];activation=x[4],stride=x[5],batchnorm=x[6]),
        zip(ks,cs,scs,cas,sts,bns))...)
    # there is a problem with automatic determination of the reshape dims
    # it can be computed from the indims and the size, padding, stride and scale of the upscaleconv
    # however that is quite complicated so lets use a trick - we initialize a random array of indim size,
    # pass it through the conv layers and get its size
    testin = randn(Float32,outs...,1)
    testoutsize = size(conv_layers(testin))
    convindim = reduce(*,testoutsize[1:3])*reduce(*,sts)^2 # size of the first dense layer 
    ds = vcat(ds, [convindim])
    ndl = length(ds)-1 # no of dense layers
    # and also the previous ones if needed
    dense_layers = Flux.Chain(
            map(i->Flux.Dense(ds[i],ds[i+1],das[i]),1:ndl)...)
    # finally put it all into one chain
    Flux.Chain(
        dense_layers,
        x->reshape(x,testoutsize[1]*reduce(*,sts),testoutsize[2]*reduce(*,sts),testoutsize[3],size(x,2)),
        upscaleconv_layers
        )
end
"""
    convdecoder(insize, latentdim, nconv, kernelsize, channels, scaling,
        [ndense, dsizes, activation, stride, layertype, batchnorm])

Create a convolutional decoder.

outsize = (height,width,no channels) of output
latentdim = size of latent space
nconv = number of conv layers
kernelsize = scalar, tuple or a list of those
channels = a list of channel numbers, same length as nconv
scaling = scalar, tuple or a list of those
ndense = number of dense layers (default 1)
dsizes = if ndense > 1, specify a list of latent layer widths of length = ndense
activation = default relu
lstride = length of stride, default 1, can be a scalar or a list of scalars
layertype = one of ["transpose", "upscale"]
batchnorm = boolean
"""
function convdecoder(outsize, latentdim::Int, nconv::Int, kernelsize, channels, 
    scaling; ndense::Int=1, dsizes=nothing, activation=relu, lstride=1, 
    layertype="transpose", batchnorm = false)
    # construct ds - vector of widths of dense layers
    if ndense>1
        (dsizes==nothing) ? error("If more than one Dense layer is require, specify their widths in dsizes.") : nothing 
        ds = vcat([latentdim], dsizes)
    else
        ds = [latentdim]
    end
    # das - vector of dense activations
    das = fill(activation, length(ds))
    # construct ks - vector of kernel sizes
    if typeof(kernelsize) <: AbstractVector
        @assert length(kernelsize) == nconv
        ks = kernelsize
    else
        ks = fill(kernelsize,nconv)
    end
    # construct cs - vector of channel pairs
    # first does not have to be specified as it is in insize
    @assert length(channels)==nconv
    cs = vcat(map(i->channels[i]=>channels[i+1],1:length(channels)-1), channels[end]=>outsize[3])
    # construct scs - vector of scaling factors
    if typeof(scaling) <: AbstractVector
        @assert length(scaling) == nconv
        scs = scaling
    else
        scs = fill(scaling,nconv)
    end
    # cas - vector of convolutional activations
    cas = fill(activation, nconv-1)
    # sts - vector of strides
    if typeof(lstride) <: AbstractVector
        @assert length(lstride) == nconv
        sts = lstride
    else
        sts = fill(lstride,nconv)
    end
    # bns - vector of batchnorm swithes
    if typeof(batchnorm) <: AbstractVector
        @assert length(batchnorm) == nconv
        bns = batchnorm
    else
        bns = fill(batchnorm,nconv)
    end
    
    return convdecoder(outsize, ds, das, ks, cs, scs, cas, sts, bns; layertype=layertype) 
end

##### ResNet module ####

struct ResBlock
    conv
end

function ResBlock(ks,cs,a=identity;kwargs...)
    # compute the padding margins so that the size of the image does not change
    pad = map(x->floor(Int,x/2),ks)
    conv = Flux.Conv(ks,cs,a;pad=pad,kwargs...)
    return ResBlock(conv)
end

(rb::ResBlock)(X::Array{T,4}) where T = rb.conv(X) .+ X

Flux.@treelike ResBlock