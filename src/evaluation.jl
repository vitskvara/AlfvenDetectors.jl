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


# data
function get_validation_data(patchsize)
    patch_f = joinpath(dirname(pathof(AlfvenDetectors)), 
        "../experiments/conv/data/labeled_patches_$patchsize.bson")
    if isfile(patch_f)
        patchdata = BSON.load(patch_f);
        data = patchdata[:data];
        shotnos = patchdata[:shotnos];
        labels = patchdata[:labels];
        tstarts = patchdata[:tstarts];
        fstarts = patchdata[:fstarts];
    else
        readfun = AlfvenDetectors.readnormlogupsd
        shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches();
        patchdata = map(x->AlfvenDetectors.get_patch(datapath,x[1], x[2], x[3], patchsize, 
            readfun;memorysafe = true)[1],  zip(shotnos, tstarts, fstarts));
        data = cat(patchdata..., dims=4);
    end;
    return data, shotnos, labels, tstarts, fstarts
end

# now load the first stage model
function load_model(mf)
    model_data = BSON.load(mf)
    exp_args = model_data[:experiment_args]
    model_args = model_data[:model_args]
    model_kwargs = model_data[:model_kwargs]
    history = model_data[:history]
    if haskey(model_data, :model)
        model = model_data[:model]
    else
        model = Flux.testmode!(GenerativeModels.construct_model(mf))
    end
    return model, exp_args, model_args, model_kwargs, history
end

function fit_fs_model(s1_model, s2_model, fx, fxy, asf_name, asf_args, data, shotnos, labels, 
    tstarts, fstarts)
    # now iterate over anoamly score function params and seeds
    dfs_asf_arg = []
    for asf_arg in asf_args
        println("")
        println("processing $(asf_arg)...")
        asf(m,x) = eval(Meta.parse("AlfvenDetectors."*asf_name))(m,x,asf_arg...);

        # train/test data
        # this is not entirely correct, since the seed should probably be the same as 
        # the one that the s1 model was trained with
        # however for now we can ignore this
        # seed = exp_args["seed"];
        dfs_seed = []
        for seed in 1:10
            print(" seed=$seed")
            train_info, train_inds, test_info, test_inds = 
            AlfvenDetectors.split_patches_unique(0.5, 
                shotnos, labels, tstarts, fstarts; seed=seed);
            train = (data[:,:,:,train_inds], train_info[2]);
            test = (data[:,:,:,test_inds], test_info[2]);

            # now the few-shot model
            fsmodel = AlfvenDetectors.FewShotModel(s1_model, s2_model, fx, fxy, asf);
            AlfvenDetectors.fit!(fsmodel, train[1], train[1], train[2]);
            as = AlfvenDetectors.anomaly_score(fsmodel, test[1]);
            auc = EvalCurves.auc(EvalCurves.roccurve(as, test[2])...)
            df_seed = DataFrame(seed=seed, auc=auc)
            push!(dfs_seed, df_seed)
        end
        dfs_seed = vcat(dfs_seed...)
        dfs_seed[:asf_arg] = fill(asf_arg,size(dfs_seed,1)) 
        push!(dfs_asf_arg, dfs_seed)
    end
    df_exp = vcat(dfs_asf_arg...)
    return df_exp
end

function add_info(df_exp, exp_args, history, s2_model_name, s2_args, s2_kwargs, asf_name, mf)
    Nrows = size(df_exp,1)
    df_exp[:S1_model] = exp_args["modelname"]
    df_exp[:S2_model] = s2_model_name
    df_exp[:S2_model_args] = fill(s2_args, Nrows)
    df_exp[:S2_model_kwargs] = s2_kwargs
    df_exp[:as_function] = asf_name
    df_exp[:S1_file] = joinpath(split(mf,"/")[end-2:end]...)
    df_exp[:ldim] = exp_args["ldimsize"]
    df_exp[:lambda] = exp_args["lambda"]
    df_exp[:gamma] = exp_args["gamma"]
    df_exp[:beta] = exp_args["beta"]
    df_exp[:sigma] = exp_args["sigma"]
    df_exp[:batchsize] = exp_args["batchsize"]
    df_exp[:S1_iterations] = length(get(history, collect(keys(history))[1])[2])
    # skip this, it can be always loaded from the model file
    #df_exp[:S1_model_args] = fill(model_args, Nrows)
    #df_exp[:S2_model_kwargs] = model_kwargs
    #df_exp[:S1_exp_args] = exp_args
    return df_exp
end

function fit_knn(mf, data, shotnos, labels, tstarts, fstarts)
    s1_model, exp_args, model_args, model_kwargs, history = load_model(mf)

    # knn model
    s2_model_name = "KNN"
    s2_args = [:BruteTree]
    s2_kwargs = Dict()
    s2_model = eval(Meta.parse("AlfvenDetectors."*s2_model_name))(s2_args...; s2_kwargs...);
    fx(m,x) = nothing # there is no point in fitting the unlabeled samples
    fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y);
    #kvec = collect(1:2:31)
    asf_name = "as_mean"
    asf_args = map(x->[x],collect(1:2:3))

    # this contains the fitted aucs and some other data
    df_exp = fit_fs_model(s1_model, s2_model, fx, fxy, asf_name, asf_args, data, shotnos, 
        labels, tstarts, fstarts)

    # now add parameters of both S1 and S2 models

    df_exp = add_info(df_exp, exp_args, history, s2_model_name, s2_args, s2_kwargs, asf_name, mf)

    return df_exp
end

function eval_save(mf, ff, data, shotnos, labels, tstarts, fstarts, savepath)
    println("processing model $mf")
    df_exp = ff(mf, data, shotnos, labels, tstarts, fstarts)
    # now save it all
    csv_name = df_exp[:S1_model][1]*"_"*df_exp[:S2_model][1]*"_"*
        reduce(*,split(split(mf,"_")[end],".")[1:end-1])* # this extracts the timestamp
        "_"*split(mf, "_")[occursin.("nepochs", split(mf, "_"))][1]*".csv" 
    # this extracts the number of epochs
    CSV.write(joinpath(savepath,csv_name), df_exp)
end
