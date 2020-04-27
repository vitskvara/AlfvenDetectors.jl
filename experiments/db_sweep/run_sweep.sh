#!/bin/bash

JDP=/compass/home/skvara/julia-1.1.1-genmodels
JL=/compass/home/skvara/julia-1.1.1/bin/julia
for f in $(ls /compass/home/skvara/no-backup/uprobe_data); do
	JULIA_DEPOT_PATH=$JDP $JL process_shot.jl $f
done
