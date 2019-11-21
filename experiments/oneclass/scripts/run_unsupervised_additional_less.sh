#!/bin/bash
# run with normalized data
# run for 30 iterations
for SEED in {2..9}
do	
	for LDIM in 8 128 256
	do
		BETA=0.01
		~/julia-1.1.1/bin/julia ../run_experiment.jl VAE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --beta=$BETA  --normal-negative 
		~/julia-1.1.1/bin/julia ../run_experiment.jl VAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --beta=$BETA  --normal-negative

		~/julia-1.1.1/bin/julia ../run_experiment.jl AAE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED  --normal-negative --unnormalized
		~/julia-1.1.1/bin/julia ../run_experiment.jl AAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED  --normal-negative --unnormalized

		~/julia-1.1.1/bin/julia ../run_experiment.jl AE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED  --normal-negative --unnormalized
		~/julia-1.1.1/bin/julia ../run_experiment.jl AE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED  --normal-negative --unnormalized
	done
	LDIM=128
	LAMBDA=10
	SIGMA=1.0
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA --gamma=$LAMBDA  --normal-negative --unnormalized
	LDIM=8
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA --gamma=$LAMBDA  --normal-negative --unnormalized
	SIGMA=0.1
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA --gamma=$LAMBDA  --normal-negative --unnormalized
	LDIM=128
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA --gamma=$LAMBDA  --normal-negative --unnormalized
	LAMBDA=1
	SIGMA=0.01		
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA --gamma=$LAMBDA  --normal-negative --unnormalized
	LDIM=8
	LAMBDA=0.1
	SIGMA=1		
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA --gamma=$LAMBDA  --normal-negative --unnormalized


	LDIM=128
	LAMBDA=0.1
	SIGMA=0.01
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA  --normal-negative --unnormalized
	LAMBDA=10
	SIGMA=1
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA  --normal-negative --unnormalized
	LAMBDA=1
	SIGMA=1
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA  --normal-negative --unnormalized
	LAMBDA=0.1
	SIGMA=0.01
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA  --normal-negative --unnormalized
	LAMBDA=10
	SIGMA=1
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAE $LDIM 2 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA  --normal-negative --unnormalized
	LDIM=256
	LAMBDA=1
	SIGMA=0.1
	~/julia-1.1.1/bin/julia ../run_experiment.jl WAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=unsupervised_additional/models --eta=0.0001 --nepochs=30 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA  --normal-negative --unnormalized
done
