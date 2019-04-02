#!/bin/bash
# run10mscmodels.sh modelname ldim nlayers
for patchsize in 32 64 128 256 512
do
	julia ../run_experiment.jl TSVAE 32 4 8 16 32 64 --patchsize=$patchsize --gpu --memory-efficient --no-warnings
done