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
	"unsupervised"
	]
fs = joinpath.(basepath, ps, "eval/models_eval.csv")
# nejlepsi vysledek v #7 a #8 - dobre se uci jen ta nulova trida

dfs = map(CSV.read, fs)

# now lets try to join the dfs together - lets see how that goes
bigdf = vcat(dfs...)

# now lets compute the averages
subcols = [:model,:channels,:ldim,:nepochs,:normalized,:neg,:λ,:γ,:σ,:auc_mse,:auc_mse_pos]
subdf = bigdf[!,subcols]
agcols = filter(x -> !(x in [:seed, :auc_mse, :auc_mse_pos]), subcols)
agdf = aggregate(subdf, agcols, [mean, std])

maxrow(df, s) = df[argmax(df[!,s]),:]

# now extract the results
oc1df = filter(row -> row[:neg] == false, agdf)
delete!(oc1df, [:auc_mse_mean, :auc_mse_std])
rename!(oc1df, :auc_mse_pos_mean => :auc)
rename!(oc1df, :auc_mse_pos_std => :std)

agoc1df = vcat(map(x -> maxrow(x, :auc), groupby(oc1df, :model))...)
delete!(agoc1df, :model_1)

sortvec = "Conv" .* ["AE", "VAE", "WAE", "AAE", "WAAE"]
metricvec = ["--", "KLD", "MMD", "JSD", "MMD + JSD"]
agoc1df = vcat(map(i -> agoc1df[i, :], map(x -> agoc1df[!,:model] .== x, sortvec))...)
insertcols!(agoc1df, 2, :reg => metricvec)

# finally get the output df
oagoc1df = agoc1df[!, [:reg, :auc, :std]]
s1df = PaperUtils.df2tex(oagoc1df)
f1 = joinpath(outpath, "oneclass_t1.tex")
PaperUtils.string2file(f1, s1df)

# the second type of training:
oc2df = filter(row -> row[:neg] == true, agdf)
delete!(oc2df, [:auc_mse_pos_mean, :auc_mse_pos_std])
rename!(oc2df, :auc_mse_mean => :auc)
rename!(oc2df, :auc_mse_std => :std)

agoc2df = vcat(map(x -> maxrow(x, :auc), groupby(oc2df, :model))...)
delete!(agoc2df, :model_1)

sortvec = "Conv" .* ["AE", "WAAE"]
#metricvec = ["--", "KLD", "MMD", "JSD", "MMD + JSD"]
metricvec = ["--", "MMD + JSD"]
agoc2df = vcat(map(i -> agoc2df[i, :], map(x -> agoc2df[!,:model] .== x, sortvec))...)
insertcols!(agoc2df, 2, :reg => metricvec)

# finally get the output df
oagoc2df = agoc2df[!, [:reg, :auc, :std]]
s2df = PaperUtils.df2tex(oagoc2df)
f2 = joinpath(outpath, "oneclass_t2.tex")
PaperUtils.string2file(f2, s2df)
