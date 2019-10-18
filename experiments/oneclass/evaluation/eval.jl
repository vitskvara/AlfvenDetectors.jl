using AlfvenDetectors
using EvalCurves
using PyPlot
using GenModels
using ValueHistories
using JLD2
using FileIO
using Flux
using StatsBase
using LinearAlgebra
using DataFrames
using CSV

# mse
mse(model,x) = Flux.mse(x, model(x)).data
mse(model,x,M) = StatsBase.mean([mse(model, x[:,:,:,((i-1 )*M+1):min(i*M, size(x,4))]) for i in 1:ceil(Int, size(x,4)/M)])
# get auc based on plain mse
score_mse(model, x) = map(i->mse(model, x[:,:,:,i:i]), 1:size(x,4))
auc(model,x,y,sf) = EvalCurves.auc(EvalCurves.roccurve(sf(model,x), y)...)
prec_at_k(model,x,y,sf,k) = EvalCurves.precision_at_k(sf(model,x), y, k)

# get the paths
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
trainpath = "/home/vit/vyzkum/alfven/cdb_data/oneclass_data/training/128_normalized"
modelpath = "/home/vit/vyzkum/alfven/experiments/oneclass/waae_runs/models_eval"

# settings
seed = 1
patchsize = 128
readfun = AlfvenDetectors.readnormlogupsd
testing_data = AlfvenDetectors.collect_testing_data_oneclass(datapath, readfun, patchsize; seed=seed);

# get training data
training_data = load(joinpath(trainpath, "seed-$(seed).jld2"))

function eval_model(mf, testing_data, training_data)
	model = GenModels.construct_model(mf)
	params = AlfvenDetectors.parse_params(mf)

	# labeled data
	labels = 1 .- testing_data[3]; # switch the labels here - positive class is actually the normal one
	patches = testing_data[1];
	positive_patches = patches[:,:,:,labels.==1];
	negative_patches = patches[:,:,:,labels.==0];
	
	# get the mses
	M = 1
	train_mse = mse(model, training_data["patches"][:,:,:,1:1000], M)
	test1_mse = mse(model, positive_patches, M)
	test0_mse = mse(model, negative_patches, M)
	test_mse = mse(model, patches, M)

	# get auc	
	auc_mse = auc(model, patches, labels, score_mse)

	# other accuracy measures
	prec_10_mse = prec_at_k(model, patches, labels, score_mse, 10)

	return  params, train_mse, test_mse, test1_mse, test0_mse, auc_mse, prec_10_mse
end

models = readdir(modelpath)
#mf = joinpath(modelpath, models[end])
#model = GenModels.construct_model(mf)
#train_mse, test_mse, test1_mse, test0_mse, auc_mse, params = eval_model(mf, testing_data, training_data)

data = []
for (i,mf) in enumerate(models[1:10])
	println("processing $i")
	_data = eval_model(joinpath(modelpath,mf), testing_data, training_data)
	push!(data, _data)
end
for (i,mf) in enumerate(models[11:20])
	println("processing $i")
	_data = eval_model(joinpath(modelpath,mf), testing_data, training_data)
	push!(data, _data)
end
for (i,mf) in enumerate(models[21:end])
	println("processing $i")
	_data = eval_model(joinpath(modelpath,mf), testing_data, training_data)
	push!(data, _data)
end

df = DataFrame(
	:model=>Any[],
	:channels=>Any[],
	:nepochs=>Int[],
	:time=>String[],
	:train_mse=>Float64[],
	:test_mse=>Float64[],
	:test1_mse=>Float64[],
	:test0_mse=>Float64[],
	:auc_mse=>Float64[],
	:prec_10_mse=>Float64[]
	)
for row in data
	push!(df, [row[1][:model], row[1][:channels], row[1][:nepochs], row[1][:time], 
		row[2], row[3], row[4], row[5], row[6], row[7]])
end

# wite/read thje results
csvf = "/home/vit/vyzkum/alfven/experiments/oneclass/waae_runs/eval/models_eval.csv"
CSV.write(csvf,df)

df = CSV.read(csvf)

figure()
subplot(321)
scatter(df[!,:train_mse], df[!,:auc_mse])
xlabel("train mse")
ylabel("auc mse")

subplot(322)
scatter(df[!,:train_mse], df[!,:auc_mse])
xlabel("train mse")
ylabel("auc mse")
xlim([0, 0.1])
ylim([0.4,0.7])

subplot(323)
scatter(df[!,:test0_mse], df[:auc_mse])
xlabel("test0 mse")
ylabel("auc mse")

subplot(324)
scatter(df[!,:test1_mse], df[:auc_mse])
xlabel("test1 mse")
ylabel("auc mse")

subplot(325)
scatter(df[!,:test1_mse]-df[!,:test0_mse], df[:auc_mse])
xlabel("test1 - test0 mse")
ylabel("auc mse")

subplot(326)
scatter(df[!,:test1_mse]-df[!,:test0_mse], df[:auc_mse])
xlabel("test1 - test0 mse")
ylabel("auc mse")
xlim([-0.02, 0.02])
ylim([0.4,0.7])

tight_layout()
figf = "/home/vit/vyzkum/alfven/experiments/oneclass/waae_runs/eval/models_eval.eps"
savefig(figf)

# the model 18 is the best in terms of auc - it also has the highest difference between 
# positive and negative scores while not having the best reconstruction
labels = 1 .- testing_data[3]; # switch the labels here - positive class is actually the normal one
patches = testing_data[1];
positive_patches = patches[:,:,:,labels.==1];
negative_patches = patches[:,:,:,labels.==0];

function plot_4(model, patches, scores, labels, inds)
	figure()
	for (i, ind) in enumerate(inds)
		x = patches[:,:,:,ind:ind]
		y = model(x).data
		subplot(4,2,i*2-1)
		title("$(labels[ind]), $(scores[ind]) \n patch mean = $(mean(x))")
		pcolormesh(x[:,:,1,1], cmap="plasma")
		subplot(4,2,i*2)
		title("patch mean = $(mean(y))")
		pcolormesh(y[:,:,1,1], cmap="plasma")
	end
	tight_layout()
end

imodel = 18 # this model is really degenerate - it recosntructs everything into the same output
mf = joinpath(modelpath, models[imodel])
model = GenModels.construct_model(mf)
scores = score_mse(model, patches)

inds = [1, 2, 180, 181]
plot_4(model, patches, scores, labels, inds)

inds = vcat(sortperm(scores)[1:2], sortperm(scores)[end-1:end])
plot_4(model, patches, scores, labels, inds)

imodel = 17
mf = joinpath(modelpath, models[imodel])
model = GenModels.construct_model(mf)
scores = score_mse(model, patches)

mean(model(randn(128,128,1,1)).data)