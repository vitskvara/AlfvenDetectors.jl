using AlfvenDetectors
using JLD2
using FileIO

datapath = "/compass/home/skvara/no-backup/uprobe_data"

patchsize = 128
data, shotnos, labels, tstarts, fstarts = AlfvenDetectors.get_labeled_validation_data(patchsize);

seed = 13

train_info, train_inds, test_info, test_inds = 
    AlfvenDetectors.split_unique_patches(0.5, 
        shotnos, labels, tstarts, fstarts; seed=seed);

train_labeled = (data[:,:,:,train_inds], train_info[2]);
test = (data[:,:,:,test_inds], test_info[2]);

unlabeled_nshots = 50
train_unlabeled_data, shotnos_unlabeled = AlfvenDetectors.get_unlabeled_validation_data(
	datapath, unlabeled_nshots,
    #  exp_args["nshots"], this causes memory problems
    (train_info[1], test_info[1]), 128, "uprobe", 
    true, true, "valid", true; 
    seed=seed) 

train_unlabeled = (cat(train_unlabeled_data, train_labeled[1], dims=ndims(train_labeled[1])),
    vcat(zeros(size(train_unlabeled_data, ndims(train_unlabeled_data))), train_labeled[2]))

# also get test_unnormalized
readfun = AlfvenDetectors.readlogupsd
patchdata = map(x->AlfvenDetectors.get_patch(datapath,x[1], x[2], x[3], patchsize, 
    readfun; memorysafe = true)[1],  zip(test_info[1], test_info[2], test_info[3]));
test_unnormalized = (cat(patchdata..., dims=4),test_info[2]) ;

# test on the same data, add prec@50, scores
f = "/compass/home/skvara/alfven/experiments/roc_prc_eval_data.jld2"

save(f, Dict("train_labeled" => train_labeled, "train_unlabeled" => train_unlabeled, "test" => test,
	"test_unnormalized" => test_unnormalized))
