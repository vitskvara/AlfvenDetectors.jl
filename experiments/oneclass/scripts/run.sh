#!/bin/bash
~/julia-1.1.1/bin/julia ../run_experiment.jl AE 128 2 32 64 --savepath=opt_runs/models --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=50
~/julia-1.1.1/bin/julia ../run_experiment.jl AE 128 2 32 64 --savepath=opt_runs/models --eta=0.0001 --nepochs=50 --batchnorm --resblock --verb --savepoint=50
