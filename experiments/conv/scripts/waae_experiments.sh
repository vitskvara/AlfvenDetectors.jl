#!/bin/bash
SEED=1
# small net, 2D, cube -- this  does not seem to learn anything useful
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10
# small net, 2D, cube lambda 10, larger kernel
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_sigma-1_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --sigma=1
# small net, 2D, cube, lambda 100
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-100_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=100
# small net, 2D, 6flower
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=flower
# small net, 2D, 6flower, lambda 10, larger kernel
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_sigma-1_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --sigma=1 --pz-type=flower
# small net, 2D, 6flower, lambda 100
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-100_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=100 --pz-type=flower
# small net, 2D, 6flower, lambda10,gamma10
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_gamma-10_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --gamma=10 --pz-type=flower
# small net, 2D, cube, lambda10,gamma10
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_gamma-10_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --gamma=10
# large net, 2D, 6flower, lambda 10
julia run_experiment.jl WAAE 2 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_8 16_32_32_lambda-10_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=flower
