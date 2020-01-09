using AlfvenDetectors
using GenModels
using Flux
using ValueHistories
using JLD2
using FileIO
using CSV
using DataFrames
using Statistics

basepath = "/home/vit/vyzkum/alfven/experiments/oneclass"
ps = [
	"supervised",
	"unsupervised_additional"
	]
fs = joinpath.(basepath, ps, "eval/models_eval.csv")
dfs = map(CSV.read, fs)
bigdf = vcat(dfs...)

# ow we could simply take the top 10 models from bigdf...?
# yes, since there are only 2 models in maxmodeldf
subcols = [:model,:channels,:ldim,:nepochs,:normalized,:neg,:β,:λ,:γ,:σ,:prec_50_mse,:file]
maxmodeldf = filter(row -> row[:neg] == true, bigdf)[!,subcols]
maxmodeldf = maxmodeldf[sortperm(maxmodeldf[!,:prec_50_mse], rev=true)[1:10],:]
modelfiles = maxmodeldf[:file]
modelfiles_base = basename.(modelfiles)
modelfiles = joinpath.(dirname(basepath),"db_sweep_models", modelfiles_base)

map(isfile, modelfiles)

# set the model file
for mf in modelfiles
	model, exp_args, model_args, model_kwargs, history = AlfvenDetectors.load_model(mf)
	Flux.testmode!(model)
	println("$(basename(mf)) loaded succesfuly!")
end
# YES

# tynhle byly asi pouzity pro ten obrazek v clanku
"ConvWAE_channels-[32,64]_patchsize-128_nepochs-30_seed-1_2019-11-17T19:50:08.347.bson"
"ConvAE_channels-[32,32,64]_patchsize-128_nepochs-30_seed-1_2019-11-18T15:40:18.273.bson"
