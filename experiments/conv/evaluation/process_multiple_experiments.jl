using DataFrames
using CSV

# get the list of all the dataframes
outpath = "/home/vit/vyzkum/alfven/experiments/eval/conv/uprobe/benchmarks/"
inpath = joinpath(outpath, "individual_experiments_bad_formatting")
filenames = readdir(inpath)

# now load the dataframse and then join them together
df = []
for f in filenames
	push!(df, CSV.read(joinpath(inpath,f)))
end
df=vcat(df...)

# now write everything
outname = "all_experiments.csv"
CSV.write(joinpath(outpath,outname),df)
