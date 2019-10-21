for SEED in {1..10}
do
	/compass/home/skvara/julia-1.1.1/bin/julia save_testing_patches_jld2.jl $SEED
	echo "iteration $SEED finished"
done