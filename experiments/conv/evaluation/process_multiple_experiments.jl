using DataFrames
using CSV
using ProgressMeter

# get the list of all the dataframes
outpath = "/home/vit/vyzkum/alfven/experiments/eval/conv/uprobe/benchmarks/"
inpath = joinpath(outpath, "individual_experiments")
filenames = readdir(inpath)

# now load the dataframse and then join them together
df = []
p = Progress(length(filenames),1,"Going through the files: ")
for (i,f) in enumerate(filenames)
	push!(df, CSV.read(joinpath(inpath,f)))
	ProgressMeter.next!(p;showvalues=[(:file, f)])
end
df=vcat(df...)

# now write everything
outname = "all_experiments.csv"
CSV.write(joinpath(outpath,outname),df)
