using AlfvenDetectors
using GenModels
using Flux
using ValueHistories
using CSV
using DataFrames
using JLD2
using FileIO
using BSON

mf = "/home/vit/vyzkum/alfven/experiments/oneclass/supervised/models/ConvWAAE_channels-[32,64]_patchsize-128_nepochs-20_seed-10_2019-11-18T06:12:27.457.bson"
model = GenModels.construct_model(mf) |> gpu
Flux.testmode!(model)
model_data = load(mf)
exp_args = model_data[:experiment_args]
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

# the top data must be - not trained with, not in the labeled samples (u sure?), must contain some alfvens
# look at the data that jakub sent - they should fulfill all of these requirements
# but where are they?
