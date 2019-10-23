#!/bin/bash
~/julia-1.1.1/bin/julia ../run_experiment.jl AE 128 2 32 64 --kernelsize=5 --savepath=nobatchnorm_runs/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --unnormalized
~/julia-1.1.1/bin/julia ../run_experiment.jl AE 128 2 32 64 --kernelsize=5 --savepath=nobatchnorm_runs/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --upscale-type=upscale
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 128 2 32 64 --kernelsize=5 --savepath=nobatchnorm_runs/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --beta=1.0
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 128 2 32 64 --kernelsize=5 --savepath=nobatchnorm_runs/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --beta=0.0001
~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE 128 2 32 64 --kernelsize=5 --savepath=nobatchnorm_runs/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --gamma=1.0 --lambda=1.0
~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE 128 2 32 64 --kernelsize=5 --savepath=nobatchnorm_runs/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --gamma=0.001 --lambda=0.001
~/julia-1.1.1/bin/julia ../run_experiment.jl VAE 128 2 32 64 --kernelsize=5 --savepath=nobatchnorm_runs/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --beta=0.0001 --unnormalized
~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE 128 2 32 64 --kernelsize=5 --savepath=nobatchnorm_runs/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --gamma=1.0 --lambda=1.0 --unnormalized
~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE 128 2 32 64 --kernelsize=5 --savepath=nobatchnorm_runs/models --eta=0.0001 --nepochs=50 --resblock --verb --savepoint=10 --gamma=0.001 --lambda=0.001 --unnormalized
