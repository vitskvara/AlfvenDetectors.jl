using AlfvenDetectors
using EvalCurves
using Flux
using Statistics
using GenModels
using ValueHistories
using JLD2
using FileIO
using LinearAlgebra
using DataFrames
using CSV
using BSON
#if gethostname() == "gpu-node"
	using CuArrays
#end

# mse
mse(model,x) = Flux.mse(x, model(x)).data
mse(model,x,M) = mean([mse(model, x[:,:,:,((i-1 )*M+1):min(i*M, size(x,4))]) for i in 1:ceil(Int, size(x,4)/M)])
# output var
output_var(model, x) = var(map(i->mean(model(x[:,:,:,i:i]).data), 1:size(x,4)))
# get auc based on plain mse
score_mse(model, x) = map(i->mse(model, x[:,:,:,i:i]), 1:size(x,4))
auc(model,x,y,sf) = EvalCurves.auc(EvalCurves.roccurve(sf(model,x), y)...)
prec_at_k(model,x,y,sf,k) = EvalCurves.precision_at_k(sf(model,x), y, k)

function eval_model(mf, evaldatapath)
	model = GenModels.construct_model(mf) |> gpu
	params = AlfvenDetectors.parse_params(mf)

	hostname = gethostname()
	if occursin("soroban", hostname) || hostname == "gpu-node"
		datapath = "/compass/home/skvara/no-backup/uprobe_data"
	else
		datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
	end

	# 
	exp_args = load(mf, :experiment_args)
	seed = exp_args["seed"]
	norms = exp_args["unnormalized"] ? "" : "_normalized"
	readfun = exp_args["unnormalized"] ? AlfvenDetectors.readnormlogupsd : AlfvenDetectors.readlogupsd
	normal_negative = get(exp_args, "normal-negative", false)
	if !(normal_negative)
		testing_data = load(joinpath(evaldatapath, "oneclass_data/testing/128$(norms)/seed-$(seed).jld2"));
		training_data = load(joinpath(evaldatapath, "oneclass_data/training/128$(norms)/seed-$(seed).jld2"));
		training_patches = training_data["patches"] |> gpu;
	else
		testing_data = load(joinpath(evaldatapath, "oneclass_data_negative/testing/128$(norms)/data.jld2"));
		training_data = AlfvenDetectors.oneclass_negative_training_data(datapath, 20, seed, readfun, 
			exp_args["patchsize"])
		training_patches = training_data[1] |> gpu;
	end

	# labeled data
	if !normal_negative
		labels = 1 .- testing_data["labels"]; # switch the labels here - positive class is actually the normal one
	else
		labels = testing_data["labels"];
	end
	patches = testing_data["patches"] |> gpu;
	positive_patches = testing_data["patches"][:,:,:,labels.==1] |> gpu;
	negative_patches = testing_data["patches"][:,:,:,labels.==0] |> gpu;
	
	# get the mses
	M = 1
	train_mse = mse(model, training_patches[:,:,:,1:1000], M)
	test1_mse = mse(model, positive_patches, M)
	test0_mse = mse(model, negative_patches, M)
	test_mse = mse(model, patches, M)
	test_var = output_var(model, patches)

	# get auc	
	auc_mse = auc(model, patches, labels, score_mse)

	# other accuracy measures
	prec_10_mse = prec_at_k(model, patches, labels, score_mse, 10)

	return  mf, params, exp_args, train_mse, test_mse, test1_mse, test0_mse, test_var, auc_mse, prec_10_mse
end

