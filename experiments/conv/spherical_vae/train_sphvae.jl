using BSON
using FewShotAnomalyDetection
using Flux
using MLDataPattern
using PyPlot

infile = "zdata.bson"
data = BSON.load