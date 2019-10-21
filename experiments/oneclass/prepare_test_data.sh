for SEED in {1..10}
do
	julia save_testing_patches_jld2.jl $SEED
	echo "iteration $SEED finished"
done