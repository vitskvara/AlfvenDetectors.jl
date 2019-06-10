include("eval_base.jl")

# set the same number of unlabeled shots used for training the second stage - same for all models
unlabeled_nshots = 50

# get the motherfrickin model file and do the magic
# possibly paralelize this
# go through the final models (highest number of epochs) first
pmap(mf->AlfvenDetectors.eval_save(mf, AlfvenDetectors.fit_svae, "SVAE", data, shotnos, labels, 
	tstarts, fstarts, savepath, datapath, unlabeled_nshots), models)

if hostname != "vit-ThinkPad-E470"
	# outside of laptop, go through the rest of the models as well
	models = vcat(map(x->joinpath.(x,readdir(x)), exdirs2)...);
	Nmodels = length(models)
	println("Found a total of $(Nmodels) saved models.")

	# get the motherfrickin model file and do the magic
	# possibly paralelize this
	pmap(mf->AlfvenDetectors.eval_save(mf, AlfvenDetectors.fit_svae, "SVAE", data, shotnos, labels, 
		tstarts, fstarts, savepath, datapath, unlabeled_nshots), models)
end
