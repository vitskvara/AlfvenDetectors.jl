#!/bin/bash
for SEED in {1..2}
do
~/julia-1.1.1/bin/julia ../run_experiment.jl AE 8 2 32 64 --kernelsize=5 --savepath=ldim_runs/models --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=10 --unnormalized --seed=$SEED
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 8 2 32 64 --kernelsize=5 --savepath=ldim_runs/models --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=10 --beta=1.0 --unnormalized --seed=$SEED
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 8 2 32 64 --kernelsize=5 --savepath=ldim_runs/models --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=10 --beta=0.0001 --unnormalized --seed=$SEED
~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE 8 2 32 64 --kernelsize=5 --savepath=ldim_runs/models --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=10 --unnormalized --seed=$SEED

~/julia-1.1.1/bin/julia ../run_experiment.jl AE 256 2 32 64 --kernelsize=5 --savepath=ldim_runs/models --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=10 --unnormalized --seed=$SEED
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 256 2 32 64 --kernelsize=5 --savepath=ldim_runs/models --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=10 --beta=1.0 --unnormalized --seed=$SEED
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 256 2 32 64 --kernelsize=5 --savepath=ldim_runs/models --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=10 --beta=0.0001 --unnormalized --seed=$SEED
~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE 256 2 32 64 --kernelsize=5 --savepath=ldim_runs/models --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=10 --unnormalized --seed=$SEED
done
