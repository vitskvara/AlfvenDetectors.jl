#!/bin/bash
cat eval_list.txt | xargs -n 1 -P 16 ~/julia-1.1.1/bin/julia eval_batch.jl