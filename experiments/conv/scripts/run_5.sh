#!/bin/bash
for i in 1 2 3 4 5
do
	julia ../run_experiment.jl "$@"
done