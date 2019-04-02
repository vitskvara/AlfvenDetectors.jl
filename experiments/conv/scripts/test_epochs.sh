#!/bin/bash
# run10mscmodels.sh modelname ldim nlayers
for nepochs in 1 2 5 10 20 50 100
do
	julia ../run_experiment.jl TSVAE 32 4 8 16 32 64 --nepochs=$nepochs --patchsize=128 --gpu --memory-efficient --no-warnings
done