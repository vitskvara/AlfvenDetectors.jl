using PyPlot
using CSV
using DataFrames
using StatsBase

# load the data from the merged csv
infile = "/home/vit/vyzkum/alfven/experiments/eval/conv/uprobe/benchmarks/all_experiments.csv"
df = CSV.read(infile)

#names(df)
# Symbol[:seed, :auc, :asf_arg, :S1_model, :S2_model, :S2_model_args, :S2_model_kwargs, :as_function,
# :S1_file, :ldim, :lambda, :gamma, :beta, :sigma, :batchsize, :S1_iterations]

# now, average the auc over seeds
byargs = filter(x->!(x in [:seed, :auc]), names(df))
seed_avg = by(df, byargs, auc = :auc => StatsBase.mean, auc_sd = :auc => x->sqrt(StatsBase.var(x)))

# find the best model of them all
function print_find_maxauc(df)
	maxauc, imaxauc = findmax(df[:auc])
	s = "best AUC=$(maxauc) at $(df[:S1_model][imaxauc])/"*"$(df[:S2_model][imaxauc]) "*
		"ldim=$(df[:ldim][imaxauc]) "*"λ=$(df[:lambda][imaxauc]) "*"γ=$(df[:gamma][imaxauc]) "*
		"σ=$(df[:sigma][imaxauc]) "*"niter=$(df[:S1_iterations][imaxauc]) "
	if :asf_arg in names(df)
		s=s*"asf_args=$(df[:asf_arg][imaxauc]) "
	end
	println(s)
	println(df[:S1_file][imaxauc])
	return df[:S1_file][imaxauc], imaxauc, maxauc
end
saf, sai, saa = print_find_maxauc(seed_avg)

# also, average over ks
byargs = filter(x->!(x in [:seed, :auc, :asf_arg]), names(df))
seed_k_avg = by(df, byargs, auc = :auc => StatsBase.mean, auc_sd = :auc => x->sqrt(StatsBase.var(x)))
skaf, skai, skaa = print_find_maxauc(seed_k_avg)
# is this the best model?
# waae_8_16_16_32_lambda-10.0_gamma-0.0_sigma-0.01/1/ConvWAAE_channels-[16,16,32]_patchsize-128_nepochs-50_2019-05-21T16:54:44.912.bson

# now do some nice plots
figure()
scatter([eval(Meta.parse(k))[1] for k in seed_avg[:asf_arg]], seed_avg[:auc])


showall(seed_avg[[:S1_model, :asf_arg, :auc, :auc_sd, :gamma, :lambda, :sigma]])
seed_k_avg[[:S1_model, :auc, :auc_sd, :gamma, :lambda, :sigma]]
