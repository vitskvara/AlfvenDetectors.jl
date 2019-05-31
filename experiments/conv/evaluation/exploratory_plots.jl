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
		_, _, _, _, ss = print_find_maxauc(subdf)
		global presentation_s *= "$s1reg & $s2string & $ss \\\\ \n"
		push!(subdfs, subdf)
	end
end
println(presentation_s)

# try to look at the means and histograms of the aucs for the individual problems
idf = 0
isubs = vcat(collect(1:3:15), collect(2:3:15), collect(3:3:15))
figure(figsize=(6,10))
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
