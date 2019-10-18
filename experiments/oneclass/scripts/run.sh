#!/bin/bash
~/julia-1.1.1/bin/julia ../run_experiment.jl AE 128 2 32 64 --kernelsize=5 --savepath=opt_runs/models --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=50 --upscale-type=upscale
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 128 2 32 64 --kernelsize=5 --savepath=opt_runs/models --eta=0.0001 --nepochs=100 --batchnorm --resblock --verb --savepoint=50 --beta=1.0
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 128 2 32 64 --kernelsize=5 --savepath=opt_runs/models --eta=0.0001 --nepochs=100 --batchnorm --resblock --verb --savepoint=50 --beta=0.001
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 128 2 32 64 --kernelsize=5 --savepath=opt_runs/models --eta=0.0001 --nepochs=100 --batchnorm --resblock --verb --savepoint=50 --beta=0.001
~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE 128 2 32 64 --kernelsize=5 --savepath=opt_runs/models --eta=0.0001 --nepochs=100 --batchnorm --resblock --verb --savepoint=50 --gamma=1.0 --lambda=1.0
~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE 128 2 32 64 --kernelsize=5 --savepath=opt_runs/models --eta=0.0001 --nepochs=100 --batchnorm --resblock --verb --savepoint=50 --gamma=0.001 --lambda=0.001
