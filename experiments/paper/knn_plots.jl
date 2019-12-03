using PyPlot
using DataFrames
using CSV
using PyCall
using StatsBase

# plot params
outpath = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/paper/figs_tables"
mkpath(outpath)

cmap = "plasma" # colormap
matplotlib.rc("font", family = "normal",
    weight = "bold",
    size = 16
)
# setup the plots
PyCall.PyDict(matplotlib["rcParams"])["font.size"] = 16
PyCall.PyDict(matplotlib["rcParams"])["text.usetex"] = true
PyCall.PyDict(matplotlib["rcParams"])["font.family"] = "serif"


function plot_lines(df, label, color)
	for seed in unique(df[:seed])
		subdf = filter(x->x[:seed]==seed, df)
		if seed==1
			plot(subdf[:k], subdf[:auc], label=label, c=color)
		else
			plot(subdf[:k], subdf[:auc], c=color)
		end
	end
end
function plot_mean_sd(df, label, color, nsd)
	kvec = unique(df[:k])
	means = map(k->StatsBase.mean((filter(row->row[:k]==k,df))[:auc]), kvec)
	sds = map(k->sqrt(StatsBase.var((filter(row->row[:k]==k,df))[:auc])), kvec)
	plot(kvec, means, label=label, c=color)
	fill_between(kvec, means-nsd*sds, means+nsd*sds, color=color, alpha=0.3, linewidth=0)
end
function plot_mean_sd(df, label, color, nsd,style)
	kvec = unique(df[:k])
	means = map(k->StatsBase.mean((filter(row->row[:k]==k,df))[:auc]), kvec)
	sds = map(k->sqrt(StatsBase.var((filter(row->row[:k]==k,df))[:auc])), kvec)
	plot(kvec, means, label=label, c=color, linestyle=style)
	fill_between(kvec, means-nsd*sds, means+nsd*sds, color=color, alpha=0.2, linewidth=0)
end

# now do the comparison of knn on encoded data and on the original samples

auc_patches = CSV.read("../presentation/auc_patches.csv")
auc_latent = CSV.read("../presentation/auc_latent.csv")

# first plot everything over and over
fname = "knn_patches_vs_latent.eps"
figure()
plot_lines(auc_patches, "full patches", "r")
plot_lines(auc_latent, "latent", "b")
ylim([0.5, 1.0])
xlabel("k")
ylabel("AUC")
legend(frameon=false)
tight_layout()
savefig(joinpath(outpath, fname))

# now plot means and sds
figure()
plot_mean_sd(auc_patches, "patch space", "r",1, "-")
plot_mean_sd(auc_latent, "latent space", "b",1, "--")
ylim([0.5, 1.0])
xlabel("k")
ylabel("AUC")
legend(frameon=false)
tight_layout()
fname = "knn_patches_vs_latent_means.pdf"
savefig(joinpath(outpath, fname))

# use the validation data with uniquely split training and testing shots
auc_patches_unique = CSV.read("../presentation/auc_patches_unique.csv")
auc_latent_unique = CSV.read("../presentation/auc_latent_unique.csv")

# plot all  the lines
fname = "knn_patches_vs_latent_unique.eps"
figure()
plot_lines(auc_patches_unique, "full patches", "r")
plot_lines(auc_latent_unique, "latent", "b")
ylim([0.5, 1.0])
xlabel("k")
ylabel("AUC")
legend(frameon=false)
tight_layout()
savefig(joinpath(outpath, fname))

# now plot means and sds
figure()
plot_mean_sd(auc_patches_unique, "patch space", "r",1,"-")
plot_mean_sd(auc_latent_unique, "latent space", "b",1,"--")
#plot_mean_sd(auc_patches_unique, "original space (\$d=\$$(128^2))", "r",1)
#plot_mean_sd(auc_latent_unique, "latent space (\$d=8\$)", "b",1)
ylim([0.5, 1.0])
xlabel("k")
ylabel("AUC")
#title("kNN fit, 10 crossvalidation splits")
legend(frameon=false,loc=4)
tight_layout()
fname = "knn_patches_vs_latent_means_unique.pdf"
savefig(joinpath(outpath, fname))

# plot together the uniquely sampled patches and the original ones
fname = "knn_patches_vs_latent_unique_all.eps"
figure()
plot_lines(auc_patches, "full patches", "r")
plot_lines(auc_latent, "latent", "b")
plot_lines(auc_patches_unique, "full patches (u)", "coral")
plot_lines(auc_latent_unique, "latent (u)", "cyan")
ylim([0.5, 1.0])
xlabel("k")
ylabel("AUC")
legend(frameon=false)
tight_layout()
savefig(joinpath(outpath, fname))

figure()
plot_mean_sd(auc_patches, "full patches", "r",1)
plot_mean_sd(auc_latent, "latent", "b",1)
plot_mean_sd(auc_patches_unique, "full patches (u)", "coral",1)
plot_mean_sd(auc_latent_unique, "latent (u)", "cyan",1)
ylim([0.5, 1.0])
xlabel("k")
ylabel("AUC")
legend(frameon=false)
tight_layout()
fname = "knn_patches_vs_latent_means_all.pdf"
savefig(joinpath(outpath, fname))

# blank plot
figure()
fname = "blank.pdf"
savefig(joinpath(outpath, fname))

