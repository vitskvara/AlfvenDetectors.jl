#!/bin/bash
# run with normalized data
# run for 30 iterations
for SEED in 1 .. 10
do	
	for LDIM in 8 128 256
	do
		for BETA in 10 1 0.1
		do
			~/julia-1.1.1/bin/julia ../run_experiment.jl VAE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=50 --verb --savepoint=10 --seed=$SEED --beta=$BETA  --normal-negative 
			~/julia-1.1.1/bin/julia ../run_experiment.jl VAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=50 --verb --savepoint=10 --seed=$SEED --beta=$BETA  --normal-negative
		done
		for LAMBDA in 10 1 0.1
		do
			for SIGMA in 1 0.1 0.01
			do
				~/julia-1.1.1/bin/julia ../run_experiment.jl WAE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=50 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA  --normal-negative --unnormalized
				~/julia-1.1.1/bin/julia ../run_experiment.jl WAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=50 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA  --normal-negative --unnormalized
			done
		done
		~/julia-1.1.1/bin/julia ../run_experiment.jl AAE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=50 --verb --savepoint=10 --seed=$SEED  --normal-negative --unnormalized
		~/julia-1.1.1/bin/julia ../run_experiment.jl AAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=50 --verb --savepoint=10 --seed=$SEED  --normal-negative --unnormalized

		~/julia-1.1.1/bin/julia ../run_experiment.jl AE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=50 --verb --savepoint=10 --seed=$SEED  --normal-negative --unnormalized
		~/julia-1.1.1/bin/julia ../run_experiment.jl AE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=50 --verb --savepoint=10 --seed=$SEED  --normal-negative --unnormalized

		for LAMBDA in 10 1 0.1
		do
			for SIGMA in 1 0.1 0.01
			do
				~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=50 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA --gamma=$LAMBDA  --normal-negative --unnormalized
				~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=50 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA --gamma=$LAMBDA  --normal-negative --unnormalized
			done
		done
	done
done
