#!/bin/bash
# run with normalized data
# run for 30 iterations
for SEED in {2..9}
do	
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
done