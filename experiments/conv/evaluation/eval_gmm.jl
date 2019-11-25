include("eval_base.jl")

# do a reverse run as well
if length(ARGS) > 0
	global models = reverse(models)
end

# set the same number of unlabeled shots used for training the second stage - same for all models
unlabeled_nshots = 50

if hostname != "vit-ThinkPad-E470"
	# outside of laptop, go through the rest of the models as well
	models = vcat(map(x->joinpath.(x,readdir(x)), exdirs2)...);
	Nmodels = length(models)
	println("Found a total of $(Nmodels) saved models.")

	# get the motherfrickin model file and do the magic
	# possibly paralelize this
	map(mf->AlfvenDetectors.eval_save(mf, AlfvenDetectors.fit_gmm, "GMM", data, shotnos, labels, 
		tstarts, fstarts, savepath, datapath, unlabeled_nshots), models)
end

# get the motherfrickin model file and do the magic
# possibly paralelize this
map(mf->AlfvenDetectors.eval_save(mf, AlfvenDetectors.fit_gmm, "GMM", data, shotnos, labels, 
	tstarts, fstarts, savepath, datapath, unlabeled_nshots), models)
