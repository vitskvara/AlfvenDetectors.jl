#!/bin/bash
# run10mscmodels.sh modelname ldim nlayers
for batchsize in 8 16 32 64 128 256 
do
	julia ../run_experiment.jl TSVAE 32 4 8 16 32 64 --batchsize=$batchsize --patchsize=128 --gpu --memory-efficient --no-warnings
done