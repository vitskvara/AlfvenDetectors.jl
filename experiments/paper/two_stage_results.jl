using PyPlot
using CSV
using DataFrames
using StatsBase
using Base.Iterators

# load the data from the merged csv
#infile = "/home/vit/vyzkum/alfven/experiments/eval/conv/uprobe/benchmarks/all_experiments.csv"
infile = "/home/vit/vyzkum/alfven/experiments/eval/conv/uprobe/benchmarks_limited/all_experiments.csv"
df = CSV.read(infile)

# filter out the data from the 64D AE
df=filter(row->row[:ldim]==8,df)

# also rename the s1 models that are actually something else than WAAE
map(i->( (df[!,:S1_model][i]=="WAAE"&&df[!,:lambda][i]==0) ? df[!,:S1_model][i]="AAE" : nothing),1:size(df,1));
map(i->( (df[!,:S1_model][i]=="WAAE"&&df[!,:gamma][i]==0) ? df[!,:S1_model][i]="WAE" : nothing),1:size(df,1));
map(i->( (df[!,:S1_model][i]=="WAAE"&&df[!,:gamma][i]==0&&df[!,:lambda][i]==0) ? df[!,:S1_model][i]="AE" : nothing),1:size(df,1));
length(df[!,:S1_model][df[!,:S1_model].=="WAAE"])

# rename the models so that the names actually hold

#names(df)
# Symbol[:seed, :auc, :asf_arg, :S1_model, :S2_model, :S2_model_args, :S2_model_kwargs, :as_function,
# :S1_file, :ldim, :lambda, :gamma, :beta, :sigma, :batchsize, :S1_iterations]

# now, average the auc over seeds
byargs = filter(x->!(x in [:seed, :auc, :prec_50]), names(df))
seed_avg = by(df, byargs, auc = :auc => StatsBase.mean, auc_sd = :auc => x->sqrt(StatsBase.var(x)), prec_50 = :prec_50 => StatsBase.mean, prec_50_sd = :prec_50 => x->sqrt(StatsBase.var(x)))

# find the best model of them all
function print_find_max(df, metric)
	msd = Symbol(String(metric)*"_sd")
	maxauc, imax = findmax(df[!,metric])
	s = "best AUC=$(maxauc) at $(df[!,:S1_model][imax])/"*"$(df[!,:S2_model][imax]) "*
		"ldim=$(df[!,:ldim][imax]) "*"λ=$(df[!,:lambda][imax]) "*"γ=$(df[!,:gamma][imax]) "*
		"σ=$(df[!,:sigma][imax]) "*"niter=$(df[!,:S1_iterations][imax]) "
	if :asf_arg in names(df)
		s=s*"asf_args=$(df[!,:asf_arg][imax]) "
	end
	println(s)
	println(df[!,:S1_file][imax])
	s = "$(round(maxauc,digits=3))\$\\pm\$$(round(df[!,msd][imax],digits=3))"
	println("")
	println(s)
	println("")
	return df[!,:S1_file][imax], imax, maxauc, df[!,msd][imax], s
end
saf, sai, saa, ssd, ss = print_find_max(seed_avg, :auc)

# get the best aucs for different settings
s1regs = ["--", "KLD", "MMD", "GAN", "MMD+GAN"]
s2strings = ["KNN", "GMM"] 
s2models = ["KNN", "GMMModel"]
s1filters = ["AE", "VAE", "WAE", "AAE", "WAAE"]
#filters = (nothing, nothing, nothing, (:lambda,0), nothing)
filters = (nothing, nothing, nothing, nothing, nothing)

function get_out_df(metric)
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
			saf, _, m, sd, ss = print_find_max(subdf, metric)
			push!(result_dfs, DataFrame(:S1_reg => s1reg, :S2_model => s2string, metric => m, :sd => sd, :S1_file => saf))
			global presentation_s *= "$s1reg & $s2string & $ss \\\\ \n"
			push!(subdfs, subdf)
		end
	end
	println(presentation_s)
	result_df = vcat(result_dfs...);
end
result_df = get_out_df(:auc)
showall(result_df)
showall(result_df[!,[:S1_reg,:S2_model,:auc,:sd]])

result_df = get_out_df(:prec_50)
showall(result_df)

out_f = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/paper/figs_tables/top_2s_models.csv"
f = open(out_f,"w")
for s in result_df[!,:S1_file] 
    write(f,s)
    write(f,"\n")
end
close(f)

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
