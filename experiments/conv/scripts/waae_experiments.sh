#!/bin/bash
SEED=1
# small net, 2D, cube -- this  does not seem to learn anything useful
# data is projected into a parabolla-like structure
# knn5 auc = 0.804
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10
# small net, 2D, cube lambda 10, larger kernel
# MMD~1.2, gloss0.67, but all data is in a cluster between 4 corners
# knn5 auc = 0.511
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_sigma-1_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --sigma=1
# small net, 2D, cube, lambda 100
# everything is close to the 4 corners, however all data are projected onto a line
# knn5 auc = 0.722
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-100_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=100
# small net, 2D, 6flower
# this one has actually a pretty nice latent space, although there is no real clsutering
# even though MMD~2
# knn5 auc = 0.766
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=flower
# small net, 2D, 6flower, lambda 10, larger kernel
# MMD is small~1, however the whole encoding has deformed into zeros
# knn5 auc = 0.5
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_sigma-1_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --sigma=1 --pz-type=flower
# small net, 2D, 6flower, lambda 100
# MMD ~1.8, gloss~0.69, nice structure but wrong fit to pz
# knn5 auc = 0.79
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-100_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=100 --pz-type=flower
# small net, 2D, 6flower, lambda10,gamma10
## MMD~1.7, gloss~0.69, but the fit is terrible (L shaped)
# knn5 auc = 0.74
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_gamma-10_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --gamma=10 --pz-type=flower
# small net, 2D, cube, lambda10,gamma10
# fitted MMD and gloss, however everything is clustered together - but some structure is visible
# actually the structure becomes prevalent only after some 40 iterations
# check eg iteration 35 and 50 - it becomes less a line and more a cloud
# knn5 auc = 0.705
julia run_experiment.jl WAAE 2 2 4 8 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_4_8_lambda-10_gamma-10_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --gamma=10
# large net, 2D, 6flower, lambda 10
# very far from pz, MMD=2, low gloss/dloss, but around iteration 10 there is some nice 
# clustering
# knn5 auc = 0.806
julia run_experiment.jl WAAE 2 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_8_16_32_32_lambda-10_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=flower
# large net, 2D, 6flower, lambda 10, sigma 1
# this is close to pz, MMD~1, glos~0.7, but no visible clustering
# knn5 auc = 0.667
julia run_experiment.jl WAAE 2 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_8_16_32_32_lambda-10_sigma-1_flower-6/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=6 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=flower --sigma=1
# large net, 2D, 4cube, lambda 10
# MMD~2, gloss/dloss~0, far from pz, but some nice clustering
# knn5 auc = 0.745
julia run_experiment.jl WAAE 2 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_8_16_32_32_lambda-10_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube
# large net, 2D, 4cubve, lambda 10, sigma 1
# quite clsoe to one of the components, but no clusters - interesting structure though
# after 10 epochs there are some interesting "clusters"
# knn5 auc = 0.67
julia run_experiment.jl WAAE 2 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_2_8_16_32_32_lambda-10_sigma-1_cube-4/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=4 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube --sigma=1
# large net, 3D, 8cube, lambda 10
# nothing interesting, latent made out of 2 halves put together - maybe it would work
# knn5 auc = 0.85
julia run_experiment.jl WAAE 3 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_3_8_16_32_32_lambda-10_cube-8/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube
# large net, 3D, 8cube, lambda 100
# fits perfectly one of the components
# knn5 auc = 0.72
julia run_experiment.jl WAAE 3 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_3_8_16_32_32_lambda-100_cube-8/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=100 --pz-type=cube
# large net, 3D, 8cube, lambda 10, sigma 1
# interesting horseshoe shape, also quite close to the pz, clsutering bad
# knn5 auc = 0.74
julia run_experiment.jl WAAE 3 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_3_8_16_32_32_lambda-10_sigma-1_cube-8/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube --sigma=1
# large net, 3D, 8cube, lambda 10, sigma 1, 64 ldim
# knn5 auc = 0.68
julia run_experiment.jl WAAE 64 4 8 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_64_8_16_32_32_lambda-10_sigma-1_cube-8/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube --sigma=1
# large net, 3D, 8cube, lambda 100, sigma 1, 64 ldim
# this fits in between the data although MMD~2 and gloss~1e-2 - interesting
# knn5 auc = 0.786
julia run_experiment.jl WAAE 64 4 8 16 32 32 \
	--scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_64_8_16_32_32_lambda-100_sigma-1_cube-8/$SEED --nshots=100 --nepochs=50 \
	--savepoint=1 --pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=100 \
	--pz-type=cube --sigma=1
# larger net, 3D, 8cube, lambda 10, sigma 1
# not so far from the pz, but no clustering
# knn5 auc = 0.75
julia run_experiment.jl WAAE 3 4 16 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_3_16_16_32_32_lambda-10_sigma-1_cube-8/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube --sigma=1
# larger net, 64, 8cube, lambda 10, sigma 1, 64 ldim
# good clustering
# knn5 auc = 0.812
julia run_experiment.jl WAAE 64 4 16 16 32 32 \
	--scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_64_16_16_32_32_lambda-10_sigma-1_cube-8/$SEED --nshots=100 --nepochs=50 \
	--savepoint=1 --pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 \
	--pz-type=cube --sigma=1
# larger net, 3D, 8cube, lambda 10, sigma 0.01
# VERY GOOD SEPARATION, but poor fit to pz
# knn5 auc = 0.836
julia run_experiment.jl WAAE 3 4 16 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=waae_3_16_16_32_32_lambda-10_sigma-0.01_cube-8/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--pz-components=8 --eta=0.001 --batchnorm --seed=$SEED --lambda=10 --pz-type=cube --sigma=0.01
# larger net, pure AE
# knn5 auc = 0.766
julia run_experiment.jl AE 3 4 16 16 32 32 \
    --scaling 2 2 1 1 --gpu --memory-efficient --memorysafe \
	--positive-patch-ratio=0.1 \
	--savepath=ae_3_16_16_32_32/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--eta=0.0001 --batchnorm --seed=$SEED
# try shorter but wider, the above model does not seem to reconstruct very well
# knn5 auc = 0.815, but quite terrible separation
julia run_experiment.jl AE 3 3 32 64 64 \
    --scaling 2 2 1  --gpu --memory-efficient --memorysafe \
	--positive-patch-ratio=0.1 \
	--savepath=ae_3_32_64_64/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--eta=0.0001 --batchnorm --seed=$SEED
# AE large, 64D
# no separation, knn5 auc = 0.813
julia run_experiment.jl AE 64 3 32 64 64 \
    --scaling 2 2 1  --gpu --memory-efficient --memorysafe \
	--positive-patch-ratio=0.1 \
	--savepath=ae_64_32_64_64/$SEED --nshots=100 --nepochs=50 --savepoint=1 \
	--eta=0.0001 --batchnorm --seed=$SEED
