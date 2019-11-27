using DataFrames
using CSV

savepath = "/compass/home/skvara/alfven/experiments/eval/conv/uprobe/benchmarks_limited/individual_experiments"
files = joinpath.(savepath, readdir(savepath))
dfs = map(CSV.read, files)

df = vcat(dfs...)

CSV.write(joinpath(dirname(savepath), "all_experiments.csv"), df)
