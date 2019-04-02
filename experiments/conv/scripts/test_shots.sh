#!/bin/bash
# run10mscmodels.sh modelname ldim nlayers
for nshots in 6 10 20 50 100 
do
	julia ../run_experiment.jl TSVAE 32 4 8 16 32 64 --nshots=$nshots --patchsize=128 --gpu --memory-efficient --no-warnings
done