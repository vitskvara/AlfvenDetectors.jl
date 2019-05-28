#!/bin/bash

SEED=1
LDIM=8
NCONV=3
C1=16
C2=16
C3=32

julia ../run_experiment.jl AE $LDIM $NCONV $C1 $C2 $C3 \
    --scaling 2 2 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=benchmarks/ae_${LDIM}_${C1}_${C1}_${C3}/$SEED \
	--nshots=100 --nepochs=50 --savepoint=5 --resblock \
	--eta=0.0001 --batchnorm --seed=$SEED

LAMBDA=10.0
SIGMA=0.01

julia ../run_experiment.jl WAE $LDIM $NCONV $C1 $C2 $C3 \
    --scaling 2 2 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1  \
	--savepath=benchmarks/wae_${LDIM}_${C1}_${C1}_${C3}_lambda-${LAMBDA}_sigma-${SIGMA}/$SEED \
	--nshots=100 --nepochs=50 --savepoint=5 --resblock --pz-type=cube --sigma=$SIGMA \
	--pz-components=8 --eta=0.0001 --batchnorm --seed=$SEED --lambda=$LAMBDA

SEED=2

julia ../run_experiment.jl AE $LDIM $NCONV $C1 $C2 $C3 \
    --scaling 2 2 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=benchmarks/ae_${LDIM}_${C1}_${C1}_${C3}/$SEED \
	--nshots=100 --nepochs=50 --savepoint=5 --resblock \
	--eta=0.0001 --batchnorm --seed=$SEED

LAMBDA=10.0
SIGMA=0.01

julia ../run_experiment.jl WAE $LDIM $NCONV $C1 $C2 $C3 \
    --scaling 2 2 1 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1  \
	--savepath=benchmarks/wae_${LDIM}_${C1}_${C1}_${C3}_lambda-${LAMBDA}_sigma-${SIGMA}/$SEED \
	--nshots=100 --nepochs=50 --savepoint=5 --resblock --pz-type=cube --sigma=$SIGMA \
	--pz-components=8 --eta=0.0001 --batchnorm --seed=$SEED --lambda=$LAMBDA

# now a model with very good reconstruction
SEED=1
LDIM=64
NCONV=2
C1=16
C2=32

julia ../run_experiment.jl AE $LDIM $NCONV $C1 $C2 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=benchmarks/ae_${LDIM}_${C1}_${C1}/$SEED \
	--nshots=100 --nepochs=100 --savepoint=10 --resblock \
	--eta=0.0001 --batchnorm --seed=$SEED

SEED=2
julia ../run_experiment.jl AE $LDIM $NCONV $C1 $C2 \
    --scaling 2 2 --gpu --memory-efficient --memorysafe \
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 \
	--savepath=benchmarks/ae_${LDIM}_${C1}_${C1}/$SEED \
	--nshots=100 --nepochs=100 --savepoint=10 --resblock \
	--eta=0.0001 --batchnorm --seed=$SEED
