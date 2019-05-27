using Distributed
using AlfvenDetectors
@everywhere begin
    using ValueHistories
    using StatsBase
end
# savepath
hostname = gethostname()
if hostname == "vit-ThinkPad-E470"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
	modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/benchmarks"
	savepath = "/home/vit/vyzkum/alfven/experiments/eval/conv/uprobe/benchmarks/individual_experiments"
elseif hostname == "tarbik.utia.cas.cz"
	datapath = "/home/skvara/work/alfven/cdb_data/uprobe_data"
	modelpath = "/home/skvara/work/alfven/experiments/conv/uprobe/benchmarks"
	savepath = "/home/skvara/work/alfven/experiments/eval/conv/uprobe/benchmarks/individual_experiments"
elseif occursin("soroban", hostname)
	datapath = "/compass/home/skvara/no-backup/uprobe_data"
	modelpath = "/compass/home/skvara/alfven/experiments/conv/uprobe/benchmarks"
	savepath = "/compass/home/skvara/alfven/experiments/eval/conv/uprobe/benchmarks/individual_experiments"
end
mkpath(savepath)

# MAIN 

# models and their adresses
exdirs1 = joinpath.(modelpath,readdir(modelpath));
exdirs2 = vcat(map(x->joinpath.(x,readdir(x)), exdirs1)...);
if hostname == "vit-ThinkPad-E470"
	# on laptotp, only go through the most trained models
	global models = vcat(map(x->joinpath.(x,readdir(x)[end]), exdirs2)...);
else
	global models = vcat(map(x->joinpath.(x,readdir(x)), exdirs2)...);
end
Nmodels = length(models)
println("Found a total of $(Nmodels) saved models.")

# get data
patchsize = 128
data, shotnos, labels, tstarts, fstarts = AlfvenDetectors.get_validation_data(patchsize);
println("loaded validation data of size $(size(data)), with $(sum(labels)) positively labeled "*
 "samples and $(length(labels)-sum(labels)) negatively labeled samples")

function fit_gmm(mf, data, shotnos, labels, tstarts, fstarts)
    s1_model, exp_args, model_args, model_kwargs, history = AlfvenDetectors.load_model(mf)

    # GMM model
    s2_model_name = "GMMModel"
    df_exps = []
    for Nclust in collect(2:12)
	    s2_args = [Nclust]
	    s2_kwargs = Dict(
	    	:kind => :diag,
	    	:method => :kmeans)
	    s2_model = eval(Meta.parse("AlfvenDetectors."*s2_model_name))(s2_args...; s2_kwargs...);
	    fx(m,x) = AlfvenDetectors.fit!(m,x) # there is no point in fitting the unlabeled samples
	    fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y);
	    for asf_name in ["as_max_ll_mse", "as_mean_ll_mse", "as_med_ll_mse", "as_ll_maxarg"]
	    	asf_args = [1]

		    # this contains the fitted aucs and some other data
		    df_exp = AlfvenDetectors.fit_fs_model(s1_model, s2_model, fx, fxy, asf_name, asf_args, data, 
		    	shotnos, labels, tstarts, fstarts)

		    # now add parameters of both S1 and S2 models

		    df_exp = AlfvenDetectors.add_info(df_exp, exp_args, history, s2_model_name, s2_args, s2_kwargs, 
		    	asf_name, mf)
		    push!(df_exps, df_exp)
		end
    end

    return vcat(df_exps...)
end


# get the motherfrickin model file and do the magic
# possibly paralelize this
pmap(mf->AlfvenDetectors.eval_save(mf, fit_gmm, "GMM", data, shotnos, labels, 
	tstarts, fstarts, savepath), models)
