#!/bin/bash
# run10mscmodels.sh modelname ldim nlayers
for ldim in 2 8 16 32 64 128 256
do
	julia ../run_experiment.jl TSVAE $ldim 4 8 16 32 64 --patchsize=128 --gpu --memory-efficient --no-warnings
done