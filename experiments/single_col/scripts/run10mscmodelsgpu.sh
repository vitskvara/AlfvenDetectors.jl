#!/bin/bash
# run10mscmodels.sh modelname ldim nlayers
for i in {1..10}
do
	echo "RUNNING MODEL" $i
	julia ../mscamp.jl $1 $2 $3 --gpu --measurement=mscamp
done