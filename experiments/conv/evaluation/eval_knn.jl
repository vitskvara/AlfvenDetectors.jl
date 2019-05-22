using AlfvenDetectors
using BSON
using Flux
using BSON
using StatsBase
using DataFrames
using CSV
using ValueHistories
using GenerativeModels
using EvalCurves

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
			readfun;memorysafe = true)[1],	zip(shotnos, tstarts, fstarts));
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

function fit_fs_model(s1_model, s2_model, fx, fxy, asf_name, asf_args, data, shotnos, labels, tstarts, fstarts)
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
			train_info, train_inds, test_info, test_inds = AlfvenDetectors.split_patches_unique(0.5, 
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
	asf_args = map(x->[x],collect(1:2:31))

	# this contains the fitted aucs and some other data
	df_exp = fit_fs_model(s1_model, s2_model, fx, fxy, asf_name, asf_args, data, shotnos, labels, tstarts, fstarts)

	# now add parameters of both S1 and S2 models

	df_exp = add_info(df_exp, exp_args, history, s2_model_name, s2_args, s2_kwargs, asf_name, mf)

	return df_exp
end

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
data, shotnos, labels, tstarts, fstarts = get_validation_data(patchsize);
println("loaded validation data of size $(size(data)), with $(sum(labels)) positively labeled"*
 "samples and $(length(labels)-sum(labels)) negatively labeled samples")

# get the motherfrickin model file and do the magic
# possibly paralelize this
for mf in models
	println("processing model $mf")
	df_exp = fit_knn(mf, data, shotnos, labels, tstarts, fstarts)
	# now save it all
	csv_name = df_exp[:S1_model][1]*"_"*df_exp[:S2_model][1]*"_"*reduce(*,split(split(mf,"_")[end],".")[1:end-1])*".csv" 
	CSV.write(joinpath(savepath,csv_name), df_exp)
end
