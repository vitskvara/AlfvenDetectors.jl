#!/bin/bash
for SEED in {1..2}
do
~/julia-1.1.1/bin/julia ../run_experiment.jl AE 128 2 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --unnormalized --seed=$SEED --normal-negative
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 128 2 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --beta=1.0 --unnormalized --seed=$SEED --normal-negative
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 128 2 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --beta=0.0001 --unnormalized --seed=$SEED --normal-negative
~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE 128 2 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --unnormalized --seed=$SEED --normal-negative

~/julia-1.1.1/bin/julia ../run_experiment.jl AE 128 3 32 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --unnormalized --seed=$SEED --normal-negative
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 128 3 32 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --beta=1.0 --unnormalized --seed=$SEED --normal-negative
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 128 3 32 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --beta=0.0001 --unnormalized --seed=$SEED --normal-negative
~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE 128 3 32 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --unnormalized --seed=$SEED --normal-negative

~/julia-1.1.1/bin/julia ../run_experiment.jl AE 8 3 32 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --unnormalized --seed=$SEED --normal-negative
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 8 3 32 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --beta=1.0 --unnormalized --seed=$SEED --normal-negative
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 8 3 32 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --beta=0.0001 --unnormalized --seed=$SEED --normal-negative
~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE 8 3 32 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --unnormalized --seed=$SEED --normal-negative
  
~/julia-1.1.1/bin/julia ../run_experiment.jl AE 256 3 32 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --unnormalized --seed=$SEED --normal-negative
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 256 3 32 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --beta=1.0 --unnormalized --seed=$SEED --normal-negative
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 256 3 32 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --beta=0.0001 --unnormalized --seed=$SEED --normal-negative
~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE 256 3 32 32 64 --kernelsize=5 --savepath=unsupervised/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --unnormalized --seed=$SEED --normal-negative
done
