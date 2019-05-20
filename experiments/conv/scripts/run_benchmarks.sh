#!/bin/bash
# the purpose of this script is to benchmark similar nets of different type against each other
# run everything with a different seed about 10 times

SEED=1
LDIM=3
NCONV=3
C1=32
C2=64
C3=64
GAMMA=0.0
LAMBDA=0.0
SIGMA=0.1
# this is AE
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

# this is AAE
GAMMA=0.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
GAMMA=1.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
GAMMA=10.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

# this is WAE
GAMMA=0.0
SIGMA=0.01
LAMBDA=0.1
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

GAMMA=0.0
SIGMA=0.1
LAMBDA=0.1
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

GAMMA=0.0
SIGMA=1.0
LAMBDA=0.1
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

# THIS IS WAAE
GAMMA=1.0
SIGMA=0.01
LAMBDA=0.1
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

GAMMA=1.0
SIGMA=0.1
LAMBDA=0.1
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

GAMMA=1.0
SIGMA=1.0
LAMBDA=0.1
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

# THIS IS WAAE
GAMMA=10.0
SIGMA=0.01
LAMBDA=0.1
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

GAMMA=10.0
SIGMA=0.1
LAMBDA=0.1
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"

GAMMA=10.0
SIGMA=1.0
LAMBDA=0.1
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=1.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
LAMBDA=10.0
./run_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA"
