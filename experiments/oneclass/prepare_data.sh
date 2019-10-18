SEED=$1
for i in {1..10}
do
	/compass/home/skvara/julia-1.1.1/bin/julia save_patches_jld2.jl $SEED 1000
	echo "iteration $SEED/$i finished"
done