using PyPlot
using CSV
using DataFrames
using StatsBase
using Base.Iterators

# plot params
outpath = "/home/vit/vyzkum/alfven/experiments/eval/conv/uprobe/benchmarks/plots"
cmap = "plasma" # colormap
matplotlib.rc("font", family = "normal",
    #weight = "bold",
    size = 10
)
mkpath(outpath)

# load the data from the merged csv
infile = "/home/vit/vyzkum/alfven/experiments/eval/conv/uprobe/benchmarks/all_experiments.csv"
df = CSV.read(infile)

# filter out the data from the 64D AE
df=filter(row->row[:ldim]==8,df)

# also rename the s1 models that are actually something else than WAAE
map(i->( (df[:S1_model][i]=="WAAE"&&df[:lambda][i]==0) ? df[:S1_model][i]="AAE" : nothing),1:size(df,1));
map(i->( (df[:S1_model][i]=="WAAE"&&df[:gamma][i]==0) ? df[:S1_model][i]="WAE" : nothing),1:size(df,1));
map(i->( (df[:S1_model][i]=="WAAE"&&df[:gamma][i]==0&&df[:lambda][i]==0) ? df[:S1_model][i]="AE" : nothing),1:size(df,1));
length(df[:S1_model][df[:S1_model].=="WAAE"])

# rename the models so that the names actually hold

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
	s = "$(round(maxauc,digits=3))\$\\pm\$$(round(df[:auc_sd][imaxauc],digits=3))"
	println("")
	println(s)
	println("")
	return df[:S1_file][imaxauc], imaxauc, maxauc, df[:auc_sd][imaxauc], s
end
saf, sai, saa, ssd, ss = print_find_maxauc(seed_avg)

# get the best aucs for different settings
s1regs = ["--", "KLD", "MMD", "GAN", "MMD+GAN"]
s2strings = ["KNN", "GMM", "SVAE"] 
s2models = ["KNN", "GMMModel", "SVAEMem"]
s1filters = ["AE", "VAE", "WAE", "AAE", "WAAE"]
#filters = (nothing, nothing, nothing, (:lambda,0), nothing)
filters = (nothing, nothing, nothing, nothing, nothing)

subdfs = []
result_dfs = []
global presentation_s = ""
for (s2string, s2model) in zip(s2strings, s2models)
	for (s1reg, s1f, f) in zip(s1regs, s1filters, filters)
		function ff(row) 
			if f == nothing
				return row[:S1_model]==s1f&&row[:S2_model]==s2model
			else
				return row[:S1_model]==s1f&&row[f[1]]==f[2]&&row[:S2_model]==s2model
			end
		end
		subdf = filter(ff,seed_avg)
		_, _, auc, sd, ss = print_find_maxauc(subdf)
		push!(result_dfs, DataFrame(:S1_reg => s1reg, :S2_model => s2string, :auc => auc, :sd => sd))
		global presentation_s *= "$s1reg & $s2string & $ss \\\\ \n"
		push!(subdfs, subdf)
	end
end
println(presentation_s)
result_df = vcat(result_dfs...);
showall(result_df)

# try to look at the means and histograms of the aucs for the individual problems
idf = 0
isubs = vcat(collect(1:3:15), collect(2:3:15), collect(3:3:15))
figure(figsize=(6,10))
title("AUC histograms and means for different model combinations")
for (s2string, s2model) in zip(s2strings, s2models)
	for (s1reg, s1f, f) in zip(s1regs, s1filters, filters)
		global idf += 1
		isub = isubs[idf]
		subdf = subdfs[idf]	
		subplot(5,3,isub)
		if (idf-1)%5 == 0
			title(s2string*"\n\n"*s1reg)
		else
			title(s1reg)
		end
		hist(subdf[:auc],20)
		xlim([0,1])
		# also draw the mean
		ax = gca()
		xl = ax.get_ylim()
		μ = mean(subdf[:auc])
		plot([μ,μ],xl,c="k",label="$(round(μ,digits=3))")
		legend()
	end
	tight_layout()
end
fname = "auc_hist_over_models.pdf"
savefig(joinpath(outpath, fname))

# do scatter plots for various settings - gamma, lambda, sigma, beta
function plot_param_histograms(s1models, s2models, param, seed_avg)
	subdf = filter(row->row[:S1_model] in s1models, seed_avg)
	param_vals = unique(subdf[param])
	nparams = length(param_vals)

	figure(figsize=(6,10))
	title("AUC for WAAE + WAE, different values of \$\\$(string(param))\$")
	i = 0
	for param_val in param_vals
		for (s1model, s2model) in product(s1models, s2models)
			i += 1
			_subdf = filter(row->row[:S1_model]==s1model&&row[:S2_model]==s2model&&row[param]==param_val, subdf)
			subplot(nparams, length(s2models), ceil(Int, i/2))
			hist(_subdf[:auc],20, alpha=0.5, label="$(s1model)")	
			ax = gca()
			yl = ax.get_ylim()
			μ = round(mean(_subdf[:auc]), digits=3)
			scale = (i%2==0) ? 0.2 : 0.1
			text(0.4, yl[2]*scale, "$s1model $μ")
			title("$s2model")
			if (ceil(Int, i/2)-1)%nparams==0
				ylabel("\$ \\$(string(param)) = $(param_val) \$")
			end
			xlabel("AUC")
			xlim([0.3, 1.0])
		end
	end
	legend()
	tight_layout()
	return ax
end
# LAMBDA
s1models = ["WAAE", "WAE"]
s2models = ["KNN", "GMMModel", "SVAEMem"]
param = :lambda
plot_param_histograms(s1models, s2models, param, seed_avg)
fname = "lambda_histograms.pdf"
savefig(joinpath(outpath, fname))

# SIGMA
s1models = ["WAAE", "WAE"]
param = :sigma
plot_param_histograms(s1models, s2models, param, seed_avg)
fname = "sigma_histograms.pdf"
savefig(joinpath(outpath, fname))

# GAMMA
s1models = ["WAAE", "AAE"]
param = :gamma
plot_param_histograms(s1models, s2models, param, seed_avg)
fname = "gamma_histograms.pdf"
savefig(joinpath(outpath, fname))

# BETA
s1models = ["VAE"]
param = :beta
plot_param_histograms(s1models, s2models, param, seed_avg)
fname = "beta_histograms.pdf"
savefig(joinpath(outpath, fname))


subdf = filter(row->row[:S1_model]=="WAAE"&&row[:S2_model]=="KNN", seed_avg)

# also, average over ks
byargs = filter(x->!(x in [:seed, :auc, :asf_arg]), names(df))
seed_k_avg = by(df, byargs, auc = :auc => StatsBase.mean, auc_sd = :auc => x->sqrt(StatsBase.var(x)));
skaf, skai, skaa = print_find_maxauc(seed_k_avg)
# is this the best model?
# waae_8_16_16_32_lambda-10.0_gamma-0.0_sigma-0.01/1/ConvWAAE_channels-[16,16,32]_patchsize-128_nepochs-50_2019-05-21T16:54:44.912.bson

# now do some nice plots
# figure()
# scatter([eval(Meta.parse(k))[1] for k in seed_avg[:asf_arg]], seed_avg[:auc])


#showall(seed_avg[[:S1_model, :asf_arg, :auc, :auc_sd, :gamma, :lambda, :sigma]]);
#seed_k_avg[[:S1_model, :auc, :auc_sd, :gamma, :lambda, :sigma]]
