#!/bin/bash
/compass/home/skvara/julia-1.1.1/bin/julia ../run_experiment.jl AE 128 2 16 32 --savepath=first_runs --eta=0.0001 --nepochs=20 --batchnorm --resblock --verb
