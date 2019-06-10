#!/bin/bash
# the purpose of this script is to benchmark similar nets of different type against each other
# run everything with a different seed about 10 times

SEED=$1
echo "using seed $SEED"
LDIM=64
NCONV=3
C1=16
C2=16
C3=32
GAMMA=0.0
LAMBDA=0.0
SIGMA=0.1

# this is AE
./run_ae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3"

# this is VAE
BETA=1.0
./run_vae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$BETA"
BETA=0.1
./run_vae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$BETA"
BETA=0.01
./run_vae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$BETA"
BETA=0.001
./run_vae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$BETA"

# this is AAE
GAMMA=0.1
./run_aae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$LAMBDA"
GAMMA=1.0
./run_aae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$LAMBDA"
GAMMA=10.0
./run_aae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$LAMBDA"

# this is WAE
SIGMA=0.01
LAMBDA=0.1
./run_wae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_wae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_wae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$LAMBDA" "$SIGMA"

SIGMA=0.1
LAMBDA=0.1
./run_wae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_wae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_wae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$LAMBDA" "$SIGMA"

SIGMA=1.0
LAMBDA=0.1
./run_wae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_wae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_wae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$LAMBDA" "$SIGMA"

# THIS IS WAAE
GAMMA=1.0
SIGMA=0.01
LAMBDA=0.1
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

GAMMA=1.0
SIGMA=0.1
LAMBDA=0.1
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

GAMMA=1.0
SIGMA=1.0
LAMBDA=0.1
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

# THIS IS WAAE
GAMMA=10.0
SIGMA=0.01
LAMBDA=0.1
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

GAMMA=10.0
SIGMA=0.1
LAMBDA=0.1
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

GAMMA=10.0
SIGMA=1.0
LAMBDA=0.1
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
