SEED=1
echo "using seed $SEED"

NCONV=3
C1=16
C2=16
C3=32
PZ="vamp"

for LDIM in 2 16 32 64
do
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
done
