#!/bin/bash
# run10mscmodels.sh modelname ldim nlayers
for i in {1..10}
do
	echo "RUNNING MODEL" $i
	julia ../run_experiment.jl $1 $2 $3 --measurement=uprobe
done