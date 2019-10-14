#!/bin/bash
for seed in {1..10}
do
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAE 128 2 16 32 --savepath=first_runs --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=10 --seed=$seed --lambda=0.1 --gamma=0.1
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAE 256 3 16 32 32 --savepath=first_runs --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=10 --seed=$seed --scaling 2 2 1 --lambda=0.1 --gamma=0.1
done