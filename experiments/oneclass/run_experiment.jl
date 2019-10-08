using AlfvenDetectors
using Flux
using ValueHistories
using ArgParse
using DelimitedFiles
using Random
using StatsBase
using GenModels

# init - via argparse
# get the model name, zdim, nlayers, channels, kernelsize, batchsize, learning rate, beta, gamma, lambda, 
# savepath, seed, niter, anomaly score
# collect the training and testing data - do the shifts, add noise
# run and save the model
