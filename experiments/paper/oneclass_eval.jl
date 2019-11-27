using PyPlot
using CSV
using DataFrames
using Statistics
using PaperUtils

# paths
outpath = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/paper/figs_tables/"

# get the paths
hostname = gethostname()
if hostname == "gpu-node"
	evaldatapath = "/compass/home/skstda/no-backup/" 
	basepath = "/compass/home/skstda/alfven/experiments/oneclass"
else
	evaldatapath = "/home/vit/vyzkum/alfven/cdb_data/"
	basepath = "/home/vit/vyzkum/alfven/experiments/oneclass"
end

ps = [
	"eval_tuning", 
	"ldim_runs",
#	"nobatchnorm_convsize_runs",
	"nobatchnorm_runs",
	"opt_runs",
	"negative_runs",
	"noresblock_runs",
	"unsupervised",
	"supervised",
	"unsupervised_additional"
	]
ps = [
#	"unsupervised",
	"supervised",
	"unsupervised_additional"
	]
fs = joinpath.(basepath, ps, "eval/models_eval.csv")
# nejlepsi vysledek v #7 a #8 - dobre se uci jen ta nulova trida

dfs = map(CSV.read, fs)

# now lets try to join the dfs together - lets see how that goes
bigdf = vcat(dfs...)

########### this part produces maximums for each criterion independently #####################
# now lets compute the averages and find best model for a selected objective
metric = "auc_mse"
metric_short = Symbol("auc")
m1 = Symbol(metric)
m2 = Symbol(metric*"_pos")
m1m = Symbol(metric*"_mean")
m1s = Symbol(metric*"_std")
m2m = Symbol(metric*"_pos_mean")
m2s = Symbol(metric*"_pos_std")
subcols = [:model,:channels,:ldim,:nepochs,:normalized,:neg,:β,:λ,:γ,:σ,m1,m2]
subdf = bigdf[!,subcols]
agcols = filter(x -> !(x in [:seed, m1, m2]), subcols)
agdf = aggregate(subdf, agcols, [mean, std])
# filter the NaNs in stds?
agdf = filter(row -> !isnan(row[m1s]),agdf) 

maxrow(df, s) = df[argmax(df[!,s]),:]

# now extract the results
oc1df = filter(row -> row[:neg] == false, agdf)
delete!(oc1df, [m1m, m1s])
rename!(oc1df, m2m => metric_short)
rename!(oc1df, m2s => :std)

agoc1df = vcat(map(x -> maxrow(x, metric_short), groupby(oc1df, :model))...)
delete!(agoc1df, :model_1)

sortvec = "Conv" .* ["AE", "VAE", "WAE", "AAE", "WAAE"]
metricvec = ["--", "KLD", "MMD", "JSD", "MMD + JSD"]
agoc1df = vcat(map(i -> agoc1df[i, :], map(x -> agoc1df[!,:model] .== x, sortvec))...)
insertcols!(agoc1df, 2, :reg => metricvec)

# finally get the output df
oagoc1df = agoc1df[!, [:reg, metric_short, :std]]
s1df = PaperUtils.df2tex(oagoc1df)
f1 = joinpath(outpath, "oneclass_"*String(m2)*".tex")
PaperUtils.string2file(f1, s1df)

# the second type of training:
oc2df = filter(row -> row[:neg] == true, agdf)
delete!(oc2df, [m2m, m2s])
rename!(oc2df, m1m => metric_short)
rename!(oc2df, m1s => :std)

agoc2df = vcat(map(x -> maxrow(x, metric_short), groupby(oc2df, :model))...)
delete!(agoc2df, :model_1)

sortvec = "Conv" .* ["AE", "VAE", "WAE", "AAE", "WAAE"]
metricvec = ["--", "KLD", "MMD", "JSD", "MMD + JSD"]
agoc2df = vcat(map(i -> agoc2df[i, :], map(x -> agoc2df[!,:model] .== x, sortvec))...)
insertcols!(agoc2df, 2, :reg => metricvec)

# finally get the output df
oagoc2df = agoc2df[!, [:reg, metric_short, :std]]
s2df = PaperUtils.df2tex(oagoc2df)
f2 = joinpath(outpath, "oneclass_"*String(m1)*".tex")
PaperUtils.string2file(f2, s2df)

# look at the top models for supervised/unsupervised cases 
# get top 3/5 WAAE hyperparams
# then restrict the supervised/unsupervised experiments only to those and rerun more experiments to get enough 
# data for evaluation
### AUC ###
sub1df = filter(row -> row[:model] == "ConvWAAE", oc1df)
sub1df = sort(sub1df, :auc)
sub1df[end-4:end,:]
# top 5:
"
│ Row │ model    │ channels │ ldim  │ nepochs │ normalized │ neg   │ β       │ λ       │ γ       │ σ       │ auc      │ std        │
│     │ String   │ String   │ Int64 │ Int64   │ Bool       │ Bool  │ Float64 │ Float64 │ Float64 │ Float64 │ Float64  │ Float64    │
├─────┼──────────┼──────────┼───────┼─────────┼────────────┼───────┼─────────┼─────────┼─────────┼─────────┼──────────┼────────────┤
│ 1   │ ConvWAAE │ [32, 64] │ 256   │ 10      │ true       │ false │ 1.0     │ 10.0    │ 10.0    │ 0.1     │ 0.617536 │ 0.13111    │
│ 2   │ ConvWAAE │ [32, 64] │ 256   │ 10      │ true       │ false │ 1.0     │ 0.1     │ 0.1     │ 1.0     │ 0.646999 │ 0.120555   │
│ 3   │ ConvWAAE │ [32, 64] │ 256   │ 10      │ true       │ false │ 1.0     │ 0.1     │ 0.1     │ 0.1     │ 0.66939  │ 0.00311109 │
│ 4   │ ConvWAAE │ [32, 64] │ 128   │ 20      │ true       │ false │ 1.0     │ 1.0     │ 1.0     │ 0.1     │ 0.673083 │ 0.0634441  │
│ 5   │ ConvWAAE │ [32, 64] │ 256   │ 20      │ true       │ false │ 1.0     │ 10.0    │ 10.0    │ 0.1     │ 0.715352 │ 0.0863328  │

"
sub2df = filter(row -> row[:model] == "ConvWAAE", oc2df)
sub2df = sort(sub2df, :auc)
sub2df[end-4:end,:]
"
│ Row │ model    │ channels     │ ldim  │ nepochs │ normalized │ neg  │ β       │ λ       │ γ       │ σ       │ auc      │ std        │
│     │ String   │ String       │ Int64 │ Int64   │ Bool       │ Bool │ Float64 │ Float64 │ Float64 │ Float64 │ Float64  │ Float64    │
├─────┼──────────┼──────────────┼───────┼─────────┼────────────┼──────┼─────────┼─────────┼─────────┼─────────┼──────────┼────────────┤
│ 1   │ ConvWAAE │ [32, 32, 64] │ 8     │ 30      │ false      │ true │ 1.0     │ 10.0    │ 10.0    │ 1.0     │ 0.844389 │ 0.0155165  │
│ 2   │ ConvWAAE │ [32, 64]     │ 8     │ 30      │ false      │ true │ 1.0     │ 0.1     │ 0.1     │ 1.0     │ 0.846473 │ 0.0169007  │
│ 3   │ ConvWAAE │ [32, 64]     │ 128   │ 30      │ false      │ true │ 1.0     │ 1.0     │ 1.0     │ 0.01    │ 0.849362 │ NaN        │
│ 4   │ ConvWAAE │ [32, 64]     │ 8     │ 30      │ false      │ true │ 1.0     │ 10.0    │ 10.0    │ 0.1     │ 0.849725 │ 0.00453216 │
│ 5   │ ConvWAAE │ [32, 32, 64] │ 128   │ 30      │ false      │ true │ 1.0     │ 10.0    │ 10.0    │ 1.0     │ 0.858771 │ NaN        │
"

# do the same for WAE
sub1df = filter(row -> row[:model] == "ConvWAE", oc1df)
sub1df = sort(sub1df, :auc)
sub1df[end-4:end,:]
"
│ Row │ model   │ channels     │ ldim  │ nepochs │ normalized │ neg   │ β       │ λ       │ γ       │ σ       │ auc      │ std       │
│     │ String  │ String       │ Int64 │ Int64   │ Bool       │ Bool  │ Float64 │ Float64 │ Float64 │ Float64 │ Float64  │ Float64   │
├─────┼─────────┼──────────────┼───────┼─────────┼────────────┼───────┼─────────┼─────────┼─────────┼─────────┼──────────┼───────────┤
│ 1   │ ConvWAE │ [32, 32, 64] │ 256   │ 20      │ true       │ false │ 1.0     │ 1.0     │ 1.0     │ 1.0     │ 0.745443 │ NaN       │
│ 2   │ ConvWAE │ [32, 32, 64] │ 256   │ 20      │ true       │ false │ 1.0     │ 10.0    │ 1.0     │ 0.1     │ 0.752828 │ NaN       │
│ 3   │ ConvWAE │ [32, 64]     │ 256   │ 20      │ true       │ false │ 1.0     │ 1.0     │ 1.0     │ 1.0     │ 0.758485 │ NaN       │
│ 4   │ ConvWAE │ [32, 64]     │ 256   │ 20      │ true       │ false │ 1.0     │ 10.0    │ 1.0     │ 0.1     │ 0.764613 │ NaN       │
│ 5   │ ConvWAE │ [32, 64]     │ 256   │ 10      │ true       │ false │ 1.0     │ 0.1     │ 1.0     │ 0.01    │ 0.777184 │ 0.0286665 │
"
sub2df = filter(row -> row[:model] == "ConvWAE", oc2df)
sub2df = sort(sub2df, :auc)
sub2df[end-4:end,:]
"
│ Row │ model   │ channels     │ ldim  │ nepochs │ normalized │ neg  │ β       │ λ       │ γ       │ σ       │ auc      │ std     │
│     │ String  │ String       │ Int64 │ Int64   │ Bool       │ Bool │ Float64 │ Float64 │ Float64 │ Float64 │ Float64  │ Float64 │
├─────┼─────────┼──────────────┼───────┼─────────┼────────────┼──────┼─────────┼─────────┼─────────┼─────────┼──────────┼─────────┤
│ 1   │ ConvWAE │ [32, 32, 64] │ 128   │ 20      │ false      │ true │ 1.0     │ 0.1     │ 1.0     │ 0.01    │ 0.85735  │ NaN     │
│ 2   │ ConvWAE │ [32, 64]     │ 128   │ 20      │ false      │ true │ 1.0     │ 10.0    │ 1.0     │ 1.0     │ 0.866412 │ NaN     │
│ 3   │ ConvWAE │ [32, 64]     │ 128   │ 30      │ false      │ true │ 1.0     │ 1.0     │ 1.0     │ 1.0     │ 0.866601 │ NaN     │
│ 4   │ ConvWAE │ [32, 32, 64] │ 128   │ 30      │ false      │ true │ 1.0     │ 0.1     │ 1.0     │ 0.01    │ 0.868433 │ NaN     │
│ 5   │ ConvWAE │ [32, 64]     │ 128   │ 30      │ false      │ true │ 1.0     │ 10.0    │ 1.0     │ 1.0     │ 0.872158 │ NaN     │
"

# do the same for prec_50
sub1df = filter(row -> row[:model] == "ConvWAAE", oc1df)
sub1df = sort(sub1df, :prec_50)
sub1df[end-4:end,:]
# top 5:
"
│ Row │ model    │ channels     │ ldim  │ nepochs │ normalized │ neg   │ β       │ λ       │ γ       │ σ       │ prec_50 │ std       │
│     │ String   │ String       │ Int64 │ Int64   │ Bool       │ Bool  │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64   │
├─────┼──────────┼──────────────┼───────┼─────────┼────────────┼───────┼─────────┼─────────┼─────────┼─────────┼─────────┼───────────┤
│ 1   │ ConvWAAE │ [32, 32, 64] │ 256   │ 10      │ true       │ false │ 1.0     │ 1.0     │ 1.0     │ 0.1     │ 0.88    │ 0.0       │
│ 2   │ ConvWAAE │ [32, 64]     │ 256   │ 10      │ true       │ false │ 1.0     │ 1.0     │ 1.0     │ 0.01    │ 0.88    │ 0.0282843 │
│ 3   │ ConvWAAE │ [32, 64]     │ 256   │ 10      │ true       │ false │ 1.0     │ 0.1     │ 0.1     │ 0.1     │ 0.88    │ 0.0282843 │
│ 4   │ ConvWAAE │ [32, 64]     │ 128   │ 20      │ true       │ false │ 1.0     │ 1.0     │ 1.0     │ 0.1     │ 0.91    │ 0.0424264 │
│ 5   │ ConvWAAE │ [32, 64]     │ 256   │ 20      │ true       │ false │ 1.0     │ 10.0    │ 10.0    │ 0.1     │ 0.97    │ 0.0424264 │
"
sub2df = filter(row -> row[:model] == "ConvWAAE", oc2df)
sub2df = sort(sub2df, :prec_50)
sub2df[end-4:end,:]
"
│ Row │ model    │ channels     │ ldim  │ nepochs │ normalized │ neg  │ β       │ λ       │ γ       │ σ       │ prec_50 │ std       │
│     │ String   │ String       │ Int64 │ Int64   │ Bool       │ Bool │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64   │
├─────┼──────────┼──────────────┼───────┼─────────┼────────────┼──────┼─────────┼─────────┼─────────┼─────────┼─────────┼───────────┤
│ 1   │ ConvWAAE │ [32, 64]     │ 128   │ 30      │ false      │ true │ 1.0     │ 1.0     │ 1.0     │ 0.01    │ 0.88    │ NaN       │
│ 2   │ ConvWAAE │ [32, 32, 64] │ 8     │ 20      │ false      │ true │ 1.0     │ 10.0    │ 10.0    │ 1.0     │ 0.89    │ 0.0707107 │
│ 3   │ ConvWAAE │ [32, 32, 64] │ 8     │ 30      │ false      │ true │ 1.0     │ 10.0    │ 10.0    │ 1.0     │ 0.89    │ 0.0141421 │
│ 4   │ ConvWAAE │ [32, 32, 64] │ 128   │ 30      │ false      │ true │ 1.0     │ 10.0    │ 10.0    │ 1.0     │ 0.9     │ NaN       │
│ 5   │ ConvWAAE │ [32, 32, 64] │ 128   │ 20      │ false      │ true │ 1.0     │ 1.0     │ 1.0     │ 1.0     │ 0.92    │ NaN       │
"

# do the same for WAE
sub1df = filter(row -> row[:model] == "ConvWAE", oc1df)
sub1df = sort(sub1df, :prec_50)
sub1df[end-4:end,:]
"
│ Row │ model   │ channels │ ldim  │ nepochs │ normalized │ neg   │ β       │ λ       │ γ       │ σ       │ prec_50 │ std       │
│     │ String  │ String   │ Int64 │ Int64   │ Bool       │ Bool  │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64   │
├─────┼─────────┼──────────┼───────┼─────────┼────────────┼───────┼─────────┼─────────┼─────────┼─────────┼─────────┼───────────┤
│ 1   │ ConvWAE │ [32, 64] │ 256   │ 20      │ true       │ false │ 1.0     │ 10.0    │ 1.0     │ 1.0     │ 0.96    │ NaN       │
│ 2   │ ConvWAE │ [32, 64] │ 256   │ 10      │ true       │ false │ 1.0     │ 0.1     │ 1.0     │ 0.01    │ 0.97    │ 0.0141421 │
│ 3   │ ConvWAE │ [32, 64] │ 256   │ 10      │ true       │ false │ 1.0     │ 1.0     │ 1.0     │ 1.0     │ 0.98    │ NaN       │
│ 4   │ ConvWAE │ [32, 64] │ 256   │ 20      │ true       │ false │ 1.0     │ 1.0     │ 1.0     │ 1.0     │ 0.98    │ NaN       │
│ 5   │ ConvWAE │ [32, 64] │ 256   │ 20      │ true       │ false │ 1.0     │ 10.0    │ 1.0     │ 0.1     │ 1.0     │ NaN       │ # wtf is this line
"
sub2df = filter(row -> row[:model] == "ConvWAE", oc2df)
sub2df = sort(sub2df, :prec_50)
sub2df[end-4:end,:]
"
│ Row │ model   │ channels     │ ldim  │ nepochs │ normalized │ neg  │ β       │ λ       │ γ       │ σ       │ prec_50 │ std     │
│     │ String  │ String       │ Int64 │ Int64   │ Bool       │ Bool │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │
├─────┼─────────┼──────────────┼───────┼─────────┼────────────┼──────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ 1   │ ConvWAE │ [32, 32, 64] │ 128   │ 30      │ false      │ true │ 1.0     │ 0.1     │ 1.0     │ 0.01    │ 0.9     │ NaN     │
│ 2   │ ConvWAE │ [32, 64]     │ 128   │ 30      │ false      │ true │ 1.0     │ 0.1     │ 1.0     │ 1.0     │ 0.9     │ NaN     │
│ 3   │ ConvWAE │ [32, 32, 64] │ 128   │ 10      │ false      │ true │ 1.0     │ 1.0     │ 1.0     │ 0.1     │ 0.92    │ NaN     │
│ 4   │ ConvWAE │ [32, 32, 64] │ 128   │ 20      │ false      │ true │ 1.0     │ 1.0     │ 1.0     │ 1.0     │ 0.92    │ NaN     │
│ 5   │ ConvWAE │ [32, 32, 64] │ 128   │ 30      │ false      │ true │ 1.0     │ 1.0     │ 1.0     │ 0.1     │ 0.92    │ NaN     │
"

########### this part produces maximums for a selected criterion and the averages for the rest as well #####################
# now lets compute the averages and find best model for a selected objective
metric = "auc_mse"
metric_short = Symbol("auc")
m1 = Symbol(metric)
m2 = Symbol(metric*"_pos")
m1m = Symbol(metric*"_mean")
m1s = Symbol(metric*"_std")
m2m = Symbol(metric*"_pos_mean")
m2s = Symbol(metric*"_pos_std")
metrics = [:auc_mse,:auc_mse_pos, :prec_10_mse, :prec_10_mse_pos, 
	:prec_20_mse, :prec_20_mse_pos, :prec_50_mse, :prec_50_mse_pos]
subcols = vcat([:model,:channels,:ldim,:nepochs,:normalized,:neg,:β,:λ,:γ,:σ], metrics)
subdf = bigdf[!,subcols]
agcols = metrics
agdf = aggregate(subdf, agcols, [mean, std])

maxrow(df, s) = df[argmax(df[!,s]),:]

# now extract the results
oc1df = filter(row -> row[:neg] == false, agdf)
delete!(oc1df, [m1m, m1s])
rename!(oc1df, m2m => metric_short)
rename!(oc1df, m2s => :std)

agoc1df = vcat(map(x -> maxrow(x, metric_short), groupby(oc1df, :model))...)
delete!(agoc1df, :model_1)

sortvec = "Conv" .* ["AE", "VAE", "WAE", "AAE", "WAAE"]
metricvec = ["--", "KLD", "MMD", "JSD", "MMD + JSD"]
agoc1df = vcat(map(i -> agoc1df[i, :], map(x -> agoc1df[!,:model] .== x, sortvec))...)
insertcols!(agoc1df, 2, :reg => metricvec)

# finally get the output df
oagoc1df = agoc1df[!, [:reg, metric_short, :std]]
s1df = PaperUtils.df2tex(oagoc1df)
f1 = joinpath(outpath, "oneclass_"*String(m2)*"_all.tex")
PaperUtils.string2file(f1, s1df)
31
# the second type of training:
oc2df = filter(row -> row[:neg] == true, agdf)
delete!(oc2df, [m2m, m2s])
rename!(oc2df, m1m => metric_short)
rename!(oc2df, m1s => :std)

agoc2df = vcat(map(x -> maxrow(x, metric_short), groupby(oc2df, :model))...)
delete!(agoc2df, :model_1)

sortvec = "Conv" .* ["AE", "WAAE"]
#metricvec = ["--", "KLD", "MMD", "JSD", "MMD + JSD"]
metricvec = ["--", "MMD + JSD"]
agoc2df = vcat(map(i -> agoc2df[i, :], map(x -> agoc2df[!,:model] .== x, sortvec))...)
insertcols!(agoc2df, 2, :reg => metricvec)

# finally get the output df
oagoc2df = agoc2df[!, [:reg, metric_short, :std]]
s2df = PaperUtils.df2tex(oagoc2df)
f2 = joinpath(outpath, "oneclass_"*String(m1)*"_all.tex")
PaperUtils.string2file(f2, s2df)



