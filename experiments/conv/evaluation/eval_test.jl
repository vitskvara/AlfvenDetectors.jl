include("eval_base.jl")

# set the same number of unlabeled shots used for training the second stage - same for all models
unlabeled_nshots = 0
savepath = "."

map(mf->AlfvenDetectors.eval_save(mf, AlfvenDetectors.fit_knn, "KNN", data, shotnos, labels, 
	tstarts, fstarts, savepath, datapath, unlabeled_nshots), models[1:2])
