#!/bin/bash
for seed in 1 2 3 4 5 6 7 8 9 10
do
	julia run_experiment.jl WAE 3 3 8 16 32 --scaling 2 2 1 --gpu --memory-efficient --memorysafe --savepath=wae_3d_$seed --nepochs=200 --savepoint=40 --ndense=3 --hdim=64 --positive-patch-ratio=0.2 --pz-components=8 --eta=0.001 --batchnorm --seed=$seed
done