#!/bin/bash

SEED=$1
LDIM=$2
NCONV=$3
C1=$4
C2=$5
C3=$6
GAMMA=$7

cmd="julia ../../run_experiment.jl AAE $LDIM $NCONV $C1 $C2 $C3
    --scaling 2 2 1 --gpu --memory-efficient --memorysafe
	--ndense=3 --hdim=64 --positive-patch-ratio=0.1 
	--savepath=benchmarks/aae_${LDIM}_${C1}_${C1}_${C3}_gamma-${GAMMA}/$SEED
	--nshots=100 --nepochs=50 --savepoint=5 --resblock --pz-type=cube
	--pz-components=8 --eta=0.0001 --batchnorm --seed=$SEED --gamma=$GAMMA
	"
echo "running this command:"
echo $cmd
eval $cmd	