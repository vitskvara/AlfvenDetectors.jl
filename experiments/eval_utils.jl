using PyPlot
using ValueHistories
using DelimitedFiles

"""
   parse_params(filename) 

Parses params from a saved model filename.
"""
function parse_params(filename::String)
    subss = split(basename(filename), "_")
    params = Dict()
    model = split(subss[1],"/")[end]
    params[:model] = model
    for subs in subss[2:end-1]
        key, val = split(subs, "-")
        try 
            val = eval(Meta.parse(val))
        catch e
            nothing
        end
        params[Symbol(key)] = val
    end
    time = split(subss[end],".bson")[1]
    params[:time] = time
    return params
end

"""
    plothistory(h, val [,label, inds])

Plot a training history given the MVHistory object.
"""
function plothistory(h, val; label=nothing, inds=nothing)
    is,xs = get(h,val)
    if inds == nothing
        inds = 1:length(xs)
    end
    if label==nothing
        plot(is[inds], xs[inds])
    else
        plot(is[inds], xs[inds], label=label)
    end
end
"""
    plotvae(h[, inds])

Plot VAE loss.
"""
function plotvae(h, inds=nothing)
    plothistory(h,:loss,label="loss",inds=inds)
    plothistory(h,:loglikelihood,label="-loglikelihood",inds=inds)
    plothistory(h,:KL,label="KL",inds=inds)
end
"""
    plotae(h[, inds])

Plot AE loss.
"""
function plotae(h, inds=nothing)
    plothistory(h,:loss,label="loss",inds=inds)
end
"""
    plotloss(h[,inds])

Plot the training loss, can distinguish between AE and VAE losses.
"""
function plotloss(h,inds=nothing)
    if :KL in keys(h)
        plotvae(h,inds)
    else
        plotae(h,inds)
    end
end
"""
    plotlosses(h[,inds])

Plots the loss for AE, VAE or TSVAE models.
"""
function plotlosses(h, inds=nothing)
    if length(h) == 1
        figure()
        plotloss(h,inds)
        legend()
    else
        for _h in h
            figure()
            plotloss(_h,inds)
            legend()
        end
    end 
end

"""
   pretty_params(params)

Creates a pretty stringfrom the model params Dictionary. 
"""
function pretty_params(params)
    s = ""
    for (key, val) in params
        s *= "$key = $val \n"
    end
    return s
end
