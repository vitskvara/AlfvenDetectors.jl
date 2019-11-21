PZ=vamp

GAMMA=1.0
SIGMA=0.01
LAMBDA=0.1
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
LAMBDA=1.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
LAMBDA=10.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"

GAMMA=1.0
SIGMA=0.1
LAMBDA=0.1
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
LAMBDA=1.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
LAMBDA=10.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"

GAMMA=1.0
SIGMA=1.0
LAMBDA=0.1
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
LAMBDA=1.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
LAMBDA=10.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"

# THIS IS WAAE
GAMMA=10.0
SIGMA=0.01
LAMBDA=0.1
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
LAMBDA=1.0 
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
LAMBDA=10.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"

GAMMA=10.0
SIGMA=0.1
LAMBDA=0.1
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
LAMBDA=1.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
LAMBDA=10.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"

GAMMA=10.0
SIGMA=1.0
LAMBDA=0.1
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
LAMBDA=1.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
LAMBDA=10.0
./run_waae_benchmark.sh "$SEED" "$LDIM" "$NCONV" "$C1" "$C2" "$C3" "$GAMMA" "$LAMBDA" "$SIGMA" "$PZ"
