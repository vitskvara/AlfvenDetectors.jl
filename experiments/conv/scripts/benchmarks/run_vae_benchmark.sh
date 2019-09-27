#!/bin/bash

SEED=$1
LDIM=$2
NCONV=$3
C1=$4
C2=$5
C3=$6
BETA=$7

cmd="julia ../../run_experiment.jl VAE $LDIM $NCONV $C1 $C2 $C3
    --scaling 2 2 1 --gpu --memory-efficient --memorysafe
	--ndense=3 --hdim=256 --positive-patch-ratio=0.1 
	--savepath=benchmarks/vae_${LDIM}_${C1}_${C1}_${C3}_beta-${BETA}/$SEED
	--nshots=100 --nepochs=50 --savepoint=5 --resblock
	--eta=0.0001 --batchnorm --seed=$SEED --beta=$BETA
	"

echo "running this command:"
echo $cmd
eval $cmd	