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

"""
    load_model(file)

Loads the model, parameters and training history from a file.
"""
function load_model(mf)
    model_data = BSON.load(mf)
    exp_args = model_data[:experiment_args]
    model_args = model_data[:model_args]
    model_kwargs = model_data[:model_kwargs]
    history = model_data[:history]
    if haskey(model_data, :model)
        model = model_data[:model]
    else
        model = Flux.testmode!(GenModels.construct_model(mf))
    end
    return model, exp_args, model_args, model_kwargs, history
end

"""
    get_labeled_validation_data(patchsize)

Get labeled patch data for validation.
"""
function get_labeled_validation_data(patchsize)
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

"""
    get_unlabeled_validation_data
"""
function get_unlabeled_validation_data(datapath, nshots, test_train_patches_shotnos, patchsize, 
    measurement_type, use_alfven_shots, warns, iptrunc, memorysafe; seed=nothing)
    available_shots = readdir(datapath)
    training_shots, testing_shots = AlfvenDetectors.split_shots(nshots, available_shots, 
        test_train_patches_shotnos; seed=seed, use_alfven_shots=use_alfven_shots)
    println("loading data from")
    println(training_shots)
    shots = joinpath.(datapath, training_shots)
    # decide the type of reading function
    if measurement_type == "mscamp"
        readfun = AlfvenDetectors.readmscamp
    elseif measurement_type == "mscphase"
        readfun = AlfvenDetectors.readnormmscphase
    elseif measurement_type == "mscampphase"
        readfun = AlfvenDetectors.readmscampphase
    elseif measurement_type == "uprobe"
        readfun = AlfvenDetectors.readnormlogupsd
    end
    data = (measurement_type == "uprobe") ?
        AlfvenDetectors.collect_conv_signals(shots, readfun, patchsize; 
            warns=warns, type=iptrunc, memorysafe=true) :
        AlfvenDetectors.collect_conv_signals(shots, readfun, patchsize, coils; 
            warns=warns, type=iptrunc, memorysafe=true)
    println("Done.")
    return data, shots
end

function fit_fs_model(s1_model, s2_model, fx, fxy, asfs, asf_args, patch_data, shotnos, labels, 
    tstarts, fstarts, datapath, unlabeled_nshots, exp_args)
    # iterate over seeds
    dfs_seed = []
    for seed in 1:10
        println("")
        println(" seed=$seed")

        # train/test labeled data
        train_info, train_inds, test_info, test_inds = 
            AlfvenDetectors.split_unique_patches(0.5, 
                shotnos, labels, tstarts, fstarts; seed=seed);
        train_labeled = (patch_data[:,:,:,train_inds], train_info[2]);
        test = (patch_data[:,:,:,test_inds], test_info[2]);
        
        # train unlabeled data
        train_unlabeled_data, shotnos_unlabeled = 
            (unlabeled_nshots > 0) ?
            get_unlabeled_validation_data(datapath, unlabeled_nshots,
                #  exp_args["nshots"], this causes memory problems
                (train_info[1], test_info[1]), exp_args["patchsize"], exp_args["measurement"], 
                !exp_args["no-alfven"], !exp_args["no-warnings"], exp_args["ip-trunc"], exp_args["memorysafe"]; 
                seed=seed) : (test[1][:,:,:,1:1], [])
            #Array{eltype(test[1]),4}(undef,size(test[1])[1:3]...,0), []
        # now add the labeled training data to the unlabeled dataset as well
        # labels are zero for unlabeled data, but it does not really come into play
        train_unlabeled = (cat(train_unlabeled_data, train_labeled[1], dims=ndims(train_labeled[1])),
            vcat(zeros(size(train_unlabeled_data, ndims(train_unlabeled_data))), train_labeled[2]))

        # now the few-shot model
        fsmodel = AlfvenDetectors.FewShotModel(s1_model, s2_model, fx, fxy, nothing);
        #AlfvenDetectors.fit!(fsmodel, train_unlabeled, train_labeled[1], train_labeled[2]);
        try
            AlfvenDetectors.fit!(fsmodel, train_unlabeled[1], train_labeled[1], train_labeled[2]);
        catch e 
            df_exp = DataFrame(as_function=String[], auc=Float64[],asf_arg=Array{Any,1}(undef,0), 
                seed=Int[])
            println(e)
            return df_exp
        end

        # now iterate over anomaly score function params and seeds
        dfs_asf_arg = []
        for asf_arg in asf_args
            print("processing $(asf_arg) ")
            # iterate over anomaly score functions
            dfs_asf = []
            for asf in asfs
                as = AlfvenDetectors.anomaly_score(fsmodel, (m,x)->asf(m,x,asf_arg...), test[1]);
                auc = EvalCurves.auc(EvalCurves.roccurve(as, test[2])...)
                println("AUC=$auc")
                asf_name = string(split(string(asf),".")[end])
                df_asf = DataFrame(as_function=asf_name, auc=auc)
                push!(dfs_asf, df_asf)
            end
            global df_asf_arg = vcat(dfs_asf...)
            df_asf_arg[:asf_arg] = fill(asf_arg,size(df_asf_arg,1))
            push!(dfs_asf_arg, df_asf_arg)
        end
        df_seed = vcat(dfs_asf_arg...)
        df_seed[:seed] = seed
        push!(dfs_seed, df_seed)
    end
    df_exp = vcat(dfs_seed...)
    return df_exp
end

function add_info(df_exp, exp_args, history, s2_model_name, s2_args, s2_kwargs, mf)
    Nrows = size(df_exp,1)
    df_exp[:S1_model] = exp_args["modelname"]
    df_exp[:S2_model] = s2_model_name
    df_exp[:S2_model_args] = fill(s2_args, Nrows)
    df_exp[:S2_model_kwargs] = s2_kwargs
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

function fit_knn(mf, data, shotnos, labels, tstarts, fstarts, datapath, unlabeled_nshots)
    s1_model, exp_args, model_args, model_kwargs, history = load_model(mf)

    # knn model
    s2_model_name = "KNN"
    s2_args = [:BruteTree]
    s2_kwargs = Dict()
    s2_model = eval(Meta.parse("AlfvenDetectors."*s2_model_name))(s2_args...; s2_kwargs...);
    fx(m,x) = nothing # there is no point in fitting the unlabeled samples
    fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y);
    asfs = [AlfvenDetectors.as_mean]
    asf_args = map(x->[x],collect(1:2:31))

    # this contains the fitted aucs and some other data
    df_exp = fit_fs_model(s1_model, s2_model, fx, fxy, asfs, asf_args, data, shotnos, labels, 
        tstarts, fstarts, datapath, unlabeled_nshots, exp_args)

    # now add parameters of both S1 and S2 models
    df_exp = add_info(df_exp, exp_args, history, s2_model_name, s2_args, s2_kwargs, mf)

    return df_exp
end

function fit_gmm(mf, data, shotnos, labels, tstarts, fstarts, datapath, unlabeled_nshots)
    s1_model, exp_args, model_args, model_kwargs, history = AlfvenDetectors.load_model(mf)

    # GMM model
    s2_model_name = "GMMModel"
    df_exps = []
    for Nclust in collect(2:2:8)
        s2_args = [Nclust]
        s2_kwargs = Dict(
            :kind => :diag,
            :method => :kmeans)
        s2_model = eval(Meta.parse("AlfvenDetectors."*s2_model_name))(s2_args...; s2_kwargs...);
        fx(m,x) = AlfvenDetectors.fit!(m,x)
        fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y);
        asfs = [AlfvenDetectors.as_max_ll_mse, AlfvenDetectors.as_mean_ll_mse, 
            AlfvenDetectors.as_med_ll_mse, AlfvenDetectors.as_ll_maxarg]
        asf_args = [[1]]

        # this contains the fitted aucs and some other data
        df_exp = fit_fs_model(s1_model, s2_model, fx, fxy, asfs, asf_args, data, 
            shotnos, labels, tstarts, fstarts, datapath, unlabeled_nshots, exp_args)

        df_exp = add_info(df_exp, exp_args, history, s2_model_name, s2_args, s2_kwargs, mf)
        push!(df_exps, df_exp)
    end

    return vcat(df_exps...)
end

function fit_svae(mf, data, shotnos, labels, tstarts, fstarts, datapath, unlabeled_nshots)
    s1_model, exp_args, model_args, model_kwargs, history = AlfvenDetectors.load_model(mf)

    # S2 model for SVAE
    s2_model_name = "SVAEMem"
    inputdim = exp_args["ldimsize"]
    hiddenDim = 32
    latentDim = 8
    numLayers = 3
    # params for memory
    memorySize = 256
    α = 0.1 # threshold in the memory that does not matter to us at the moment!
    k = 256
    labelCount = 1
    s2_args = (inputdim, hiddenDim, latentDim, numLayers, memorySize, k, labelCount, α)
    s2_kwargs = Dict()
    s2_model = eval(Meta.parse("AlfvenDetectors."*s2_model_name))(s2_args...; s2_kwargs...);
    β = 0.1 # ratio between reconstruction error and the distance between p(z) and q(z)
    γ = 0.1 # importance ratio between anomalies and normal data in mem_loss
    batchsize = 128
    nbatches = 20000 # 200
    sigma = 0.1 # width of imq kernel
    fx(m,x)=AlfvenDetectors.fit!(m, x, batchsize, nbatches, β, sigma, η=0.0001,cbtime=1);
    sigma = 0.01
    batchsize = 128 # this batchsize must be smaller than the size of the labeled training data
    nbatches = 100 # 50
    fxy(m,x,y)=AlfvenDetectors.fit!(m,x,y, batchsize, nbatches, β, sigma, γ, η=0.0001, cbtime=1);
    # finally construct the anomaly score function
    asfs = [AlfvenDetectors.as_logpxgivenz]
    asf_args = [[]]

    # this contains the fitted aucs and some other data
    df_exp = fit_fs_model(s1_model, s2_model, fx, fxy, asfs, asf_args, data, shotnos, 
        labels, tstarts, fstarts, datapath, unlabeled_nshots, exp_args)

    df_exp = add_info(df_exp, exp_args, history, s2_model_name, s2_args, s2_kwargs, mf)

    return df_exp
end

function create_csv_filename(mf, s2_model_name)
    model, exp_args, model_args, model_kwargs, history = load_model(mf)
    csv_name = exp_args["modelname"]*"_"*s2_model_name*"_"*
        reduce(*,split(split(mf,"_")[end],".")[1:end-1])* # this extracts the timestamp
        "_"*split(mf, "_")[occursin.("nepochs", split(mf, "_"))][1]*".csv" 
        # this extracts the number of epochs
    return csv_name
end


function eval_save(mf, ff, s2_model_name, data, shotnos, labels, tstarts, fstarts, savepath, 
    datapath, unlabeled_nshots)
    println("processing model $mf")
    # create the filename
    csv_name = create_csv_filename(mf, s2_model_name)
    if !isfile(joinpath(savepath,csv_name))
        df_exp = ff(mf, data, shotnos, labels, tstarts, fstarts, datapath, unlabeled_nshots)
        CSV.write(joinpath(savepath,csv_name), df_exp)
    else
        println("$(csv_name) already present, skipping")
    end
    return csv_name
end

