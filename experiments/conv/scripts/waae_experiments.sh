#!/bin/bash
SEED=1
# small net, 2D, cube -- this  does not seem to learn anything useful
# data is projected into a parabolla-like structure
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10
# small net, 2D, cube lambda 10, larger kernel
# MMD~1.2, gloss0.67, but all data is in a cluster between 4 corners
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_sigma-1_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --sigma=1
# small net, 2D, cube, lambda 100
# everything is close to the 4 corners, however all data are projected onto a line
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-100_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=100
# small net, 2D, 6flower
# this one has actually a pretty nice latent space, although there is no real clsutering
# even though MMD~2
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=flower
# small net, 2D, 6flower, lambda 10, larger kernel
# MMD is small~1, however the whole encoding has deformed into zeros
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_sigma-1_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --sigma=1 --pz-type=flower
# small net, 2D, 6flower, lambda 100
# MMD ~1.8, gloss~0.69, nice structure but wrong fit to pz
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-100_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=100 --pz-type=flower
# small net, 2D, 6flower, lambda10,gamma10
## MMD~1.7, gloss~0.69, but the fit is terrible (L shaped)
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_gamma-10_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --gamma=10 --pz-type=flower
# small net, 2D, cube, lambda10,gamma10
# fitted MMD and gloss, however everything is clustered together - but some structure is visible
# actually the structure becomes prevalent only after some 40 iterations
# check eg iteration 35 and 50 - it becomes less a line and more a cloud
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_gamma-10_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --gamma=10
# large net, 2D, 6flower, lambda 10
# very far from pz, MMD=2, low gloss/dloss, but around iteration 10 there is some nice 
# clustering
julia run_experiment.jl WAAE 2 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_8_16_32_32_lambda-10_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=flower
# large net, 2D, 6flower, lambda 10, sigma 1
# this is close to pz, MMD~1, glos~0.7, but no visible clustering
julia run_experiment.jl WAAE 2 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_8_16_32_32_lambda-10_sigma-1_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=flower --sigma=1
# large net, 2D, 4cube, lambda 10
# MMD~2, gloss/dloss~0, far from pz, but some nice clustering
julia run_experiment.jl WAAE 2 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_8_16_32_32_lambda-10_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube
# large net, 2D, 4cubve, lambda 10, sigma 1
# quite clsoe to one of the components, but no clusters - interesting structure though
# after 10 epochs there are some interesting "clusters"
julia run_experiment.jl WAAE 2 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_8_16_32_32_lambda-10_sigma-1_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube --sigma=1
# large net, 3D, 8cube, lambda 10
# nothing interesting, latent made out of 2 halves put together - maybe it would work
julia run_experiment.jl WAAE 3 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_3_8_16_32_32_lambda-10_cube-8/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube
# large net, 3D, 8cube, lambda 100
# fits perfectly one of the components
julia run_experiment.jl WAAE 3 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_3_8_16_32_32_lambda-100_cube-8/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=100 --pz-type=cube
# large net, 3D, 8cube, lambda 10, sigma 1
# interesting horseshoe shape, also quite close to the pz, clsutering bad
julia run_experiment.jl WAAE 3 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_3_8_16_32_32_lambda-10_sigma-1_cube-8/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube --sigma=1
# large net, 3D, 8cube, lambda 10, sigma 1, 64 ldim
julia run_experiment.jl WAAE 64 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_64_8_16_32_32_lambda-10_sigma-1_cube-8/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube --sigma=1
# large net, 3D, 8cube, lambda 10, sigma 1, 64 ldim
julia run_experiment.jl WAAE 64 4 8 16 32 32 \
	--scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_64_8_16_32_32_lambda-100_sigma-1_cube-8/$SEED --nshots=100 --nepochs=50 \
	--savepoint=1 --pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=100 \
	--pz-type=cube --sigma=1
# larger net, 3D, 8cube, lambda 10, sigma 1
julia run_experiment.jl WAAE 3 4 16 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_3_16_16_32_32_lambda-10_sigma-1_cube-8/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube --sigma=1
# largeer net, 3D, 8cube, lambda 10, sigma 1, 64 ldim
# RUN THIS
julia run_experiment.jl WAAE 64 4 16 16 32 32 \
	--scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_64_16_16_32_32_lambda-10_sigma-1_cube-8/$SEED --nshots=100 --nepochs=50 \
	--savepoint=1 --pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 \
	--pz-type=cube --sigma=1
