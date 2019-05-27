using AlfvenDetectors
using GenerativeModels
using BSON
using ValueHistories
using PyPlot
using StatsBase

# settings
outpath = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/presentation/images"
cmap = "plasma" # colormap
matplotlib.rc("font", family = "normal",
    weight = "bold",
    size = 16
)
figsize = (4,4)

# now get some data
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
patchsize = 128
patch_f = joinpath(dirname(pathof(AlfvenDetectors)), 
	"../experiments/conv/data/labeled_patches_$patchsize.bson")
if isfile(patch_f)
	patchdata = BSON.load(patch_f)
	data = patchdata[:data];
	shotnos = patchdata[:shotnos];
	labels = patchdata[:labels];
	tstarts = patchdata[:tstarts];
	fstarts = patchdata[:fstarts];
else
	readfun = AlfvenDetectors.readnormlogupsd
	shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()
	patchdata = map(x->AlfvenDetectors.get_patch(datapath,x[1], x[2], x[3], patchsize, readfun;
		memorysafe = true)[1],	zip(shotnos, tstarts, fstarts))
	data = cat(patchdata..., dims=4)
end
patch = data[:,:,1,1]

# get the model
modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/benchmarks"
modelpath = joinpath(modelpath, "waae_8_16_16_32_lambda-10.0_gamma-0.0_sigma-0.01/1")
#modelpath = joinpath(modelpath, "ae_8_16_16_32/2")
models = readdir(modelpath)
#imode = 46
imodel = length(models)
mf = joinpath(modelpath,models[imodel])
model, exp_args, model_args, model_kwargs, history = AlfvenDetectors.load_model(mf)

# get the latent representation
batchsize = 64
Z = GenerativeModels.encode(model, data, batchsize).data
umap_model = AlfvenDetectors.UMAP(2, n_neighbors=5, min_dist=0.4)
Zt = AlfvenDetectors.fit!(umap_model, Z);


# make a plot of a single (interesting) patch
ipatch = 153
patch = data[:,:,1,ipatch]
# without labels
figure(figsize=figsize)
pcolormesh(patch,cmap=cmap)
ax = gca()
ax.get_xaxis().set_visible(false)
ax.get_yaxis().set_visible(false)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
fname="patch_in.png"
savefig(joinpath(outpath, fname),dpi=300)
close()

# with xlabel
figure(figsize=figsize)
pcolormesh(patch,cmap=cmap)
ax = gca()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
xlabel("\$x \\in \\mathbb{R}^{128 \\times 128 \\times 1}\$")
tight_layout()
fname="patch_in_xlabel.png"
savefig(joinpath(outpath, fname),dpi=300)
close()

# now the reconstruction
rpatch = model(data[:,:,:,ipatch:ipatch]).data[:,:,1,1]
# w/o labels
figure(figsize=figsize)
pcolormesh(rpatch,cmap=cmap)
ax = gca()
ax.get_xaxis().set_visible(false)
ax.get_yaxis().set_visible(false)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
fname="patch_out.png"
savefig(joinpath(outpath, fname),dpi=300)
close()

# with xlabel
figure(figsize=figsize)
pcolormesh(rpatch,cmap=cmap)
ax = gca()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
xlabel("\$\\hat{x} \\in \\mathbb{R}^{128 \\times 128 \\times 1}\$")
tight_layout()
fname="patch_out_xlabel.png"
savefig(joinpath(outpath, fname),dpi=200)
close()

# now the latent layer

# plot w/o labels
figure(figsize=figsize)
scatter(Zt[1,:],Zt[2,:],s=10)
ax = gca()
ax.spines["right"].set_visible(false)
ax.spines["top"].set_visible(false)
ax.set_xticklabels([])
ax.set_yticklabels([])
xlabel("\$z \\in \\mathbb{R}^{d}, d \\in \\lbrace 2, 4, ..., 64 \\rbrace \$")
tight_layout()
fname="zspace.eps"
savefig(joinpath(outpath, fname),dpi=200)
close()

#plot with labels
figure(figsize=figsize)
scatter(Zt[1,labels.==0],Zt[2,labels.==0],s=10,label="normal")
scatter(Zt[1,labels.==1],Zt[2,labels.==1],s=10,label="alfven")
ax = gca()
ax.spines["right"].set_visible(false)
ax.spines["top"].set_visible(false)
ax.set_xticklabels([])
ax.set_yticklabels([])
legend(frameon=false)
tight_layout()
fname="zspace_labeled.png"
savefig(joinpath(outpath, fname),dpi=300)
close()

# plot with labels
figure(figsize=(3,3))
scatter(Zt[1,labels.==1],Zt[2,labels.==1],s=10,label="alfven")
scatter(Zt[1,labels.==0],Zt[2,labels.==0],s=10,label="normal")
ax = gca()
ax.spines["right"].set_visible(false)
ax.spines["top"].set_visible(false)
ax.set_xticklabels([])
ax.set_yticklabels([])
xlabel("\$z \\in \\mathbb{R}^{d}, d \\in \\lbrace 2, 4, ..., 64 \\rbrace \$")
tight_layout()
fname="zspace_labeled_xlabel.png"
savefig(joinpath(outpath, fname),dpi=300)
close()


