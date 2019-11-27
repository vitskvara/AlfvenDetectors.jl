# show roc and pr curves for selected models
# show it for best oneclass alfvén/nonalfvén and two stage knn/GMM model
# first we have to find the best models
# then export their roc and pr curves 
# and do the plots
using PyPlot
using CSV
using DataFrames
using Statistics
using PaperUtils

outpath = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/paper/figs_tables/"

evaldatapath = "/home/vit/vyzkum/alfven/cdb_data/"
basepath = "/home/vit/vyzkum/alfven/experiments/oneclass"

ps = [
	"supervised",
	"unsupervised_additional"
	]
fs = joinpath.(basepath, ps, "eval/models_eval.csv")
# nejlepsi vysledek v #7 a #8 - dobre se uci jen ta nulova trida

dfs = map(CSV.read, fs)

# first get the supervised model
df = filter(row -> row[:model] == "ConvWAE", dfs[1])
imodel = argmax(df[!,:auc_mse_pos])
row = df[imodel,:]
mf = row[:file]
mf = joinpath(basepath, ps[1], "models/ConvWAE_channels-[32,64]_patchsize-128_nepochs-10_seed-1_2019-11-14T22:22:57.306.bson")



df = filter(row -> row[:model] == "ConvWAAE", dfs[2])
imodel = argmax(df[!,:auc_mse])
row = df[imodel,:]
mf = row[:file]
mf = joinpath(basepath, ps[2], "models/ConvWAAE_channels-[32,64]_patchsize-128_nepochs-10_seed-1_2019-11-15T04:55:09.238.bson")
ConvWAAE_channels-\[32,64]_patchsize-128_nepochs-10_seed-1_2019-11-15T04:55:09.238.bson