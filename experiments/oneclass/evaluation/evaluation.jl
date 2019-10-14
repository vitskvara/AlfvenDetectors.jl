using AlfvenDetectors
using EvalCurves
using PyPlot
using GenModels
using ValueHistories

# get testing data

# get the paths
modelpath = "/home/vit/vyzkum/alfven/experiments/oneclass/first_runs"
models = readdir(modelpath)
mf = joinpath(modelpath, models[1])

# load a model
model = GenModels.construct_model(mf)
