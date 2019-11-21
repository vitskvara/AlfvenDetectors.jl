#!/bin/bash
# run with normalized data
# run for 30 iterations
for SEED in {2..9}
do	
	for LDIM in 8 128 256
	do
		~/julia-1.1.1/bin/julia ../run_experiment.jl AE $LDIM 2 32 64 --kernelsize=5 --savepath=supervised/models --eta=0.0001 --nepochs=20 --verb --savepoint=10 --seed=$SEED
		~/julia-1.1.1/bin/julia ../run_experiment.jl AE $LDIM 3 32 32 64 --kernelsize=5 --savepath=supervised/models --eta=0.0001 --nepochs=20 --verb --savepoint=10 --seed=$SEED

		BETA=0.1
		~/julia-1.1.1/bin/julia ../run_experiment.jl VAE $LDIM 2 32 64 --kernelsize=5 --savepath=supervised/models --eta=0.0001 --nepochs=20 --verb --savepoint=10 --seed=$SEED --beta=$BETA
		~/julia-1.1.1/bin/julia ../run_experiment.jl VAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=supervised/models --eta=0.0001 --nepochs=20 --verb --savepoint=10 --seed=$SEED --beta=$BETA
		~/julia-1.1.1/bin/julia ../run_experiment.jl AAE $LDIM 2 32 64 --kernelsize=5 --savepath=supervised/models --eta=0.0001 --nepochs=20 --verb --savepoint=10 --seed=$SEED 
		~/julia-1.1.1/bin/julia ../run_experiment.jl AAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=supervised/models --eta=0.0001 --nepochs=20 --verb --savepoint=10 --seed=$SEED
	done
	LDIM=256
	for LAMBDA in 10 1 0.1
	do
		SIGMA=0.1
		~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE $LDIM 2 32 64 --kernelsize=5 --savepath=supervised_additional/models --eta=0.0001 --nepochs=20 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA --gamma=$LAMBDA 
		~/julia-1.1.1/bin/julia ../run_experiment.jl WAAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=supervised_additional/models --eta=0.0001 --nepochs=20 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA --gamma=$LAMBDA
		~/julia-1.1.1/bin/julia ../run_experiment.jl WAE $LDIM 2 32 64 --kernelsize=5 --savepath=supervised/models --eta=0.0001 --nepochs=20 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA
		~/julia-1.1.1/bin/julia ../run_experiment.jl WAE $LDIM 3 32 32 64 --kernelsize=5 --savepath=supervised/models --eta=0.0001 --nepochs=20 --verb --savepoint=10 --seed=$SEED --lambda=$LAMBDA --sigma=$SIGMA
	done

done

