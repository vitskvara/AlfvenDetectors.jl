using AlfvenDetectors
using GenerativeModels
using ValueHistories
using StatsBase
using CSV
using DataFrames
using EvalCurves

datapath = "/compass/home/skvara/no-backup/uprobe_data"
modelpath = "/compass/home/skvara/alfven/experiments/conv/uprobe/benchmarks_limited"

patchsize = 128
data, shotnos, labels, tstarts, fstarts = AlfvenDetectors.get_labeled_validation_data(patchsize);

# get results from the first model
mf = joinpath(modelpath, "ConvWAE_channels-[16,16,32]_patchsize-128_nepochs-50_2019-06-03T22:50:45.489.bson")
s1_model, exp_args, model_args, model_kwargs, history = AlfvenDetectors.load_model(mf)
s2_model_name = "KNN"
s2_args = [:BruteTree]
s2_kwargs = Dict()
s2_model = eval(Meta.parse("AlfvenDetectors."*s2_model_name))(s2_args...; s2_kwargs...);
fx(m,x) = nothing # there is no point in fitting the unlabeled samples
fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y);
asf = AlfvenDetectors.as_mean
k = 23 
seed = 8

train_info, train_inds, test_info, test_inds = 
    AlfvenDetectors.split_unique_patches(0.5, 
        shotnos, labels, tstarts, fstarts; seed=seed);

train_labeled = (data[:,:,:,train_inds], train_info[2]);
test = (data[:,:,:,test_inds], test_info[2]);
train_unlabeled_data, shotnos_unlabeled =  (test[1][:,:,:,1:1], [])
train_unlabeled = (cat(train_unlabeled_data, train_labeled[1], dims=ndims(train_labeled[1])),
    vcat(zeros(size(train_unlabeled_data, ndims(train_unlabeled_data))), train_labeled[2]))

fsmodel = AlfvenDetectors.FewShotModel(s1_model, s2_model, fx, fxy, nothing);
AlfvenDetectors.fit!(fsmodel, train_unlabeled[1], train_labeled[1], train_labeled[2]);
as = AlfvenDetectors.anomaly_score(fsmodel, (m,x)->asf(m,x,k), test[1]);

_labels = test[2]
roc =  EvalCurves.roccurve(as, _labels)
auc = EvalCurves.auc(roc...)
prc = EvalCurves.prcurve(as, _labels)

df_knn = DataFrame(
	:model => mf,
	:roc => roc,
	:prc => prc
	)        


# get results from the second model
#mf = joinpath(modelpath, "ConvWAAE_channels-[16,16,32]_patchsize-128_nepochs-05_2019-05-23T13:23:42.794.bson")
mf = joinpath(modelpath, "ConvAAE_channels-[16,16,32]_patchsize-128_nepochs-40_2019-06-03T13:33:41.456.bson")
s1_model, exp_args, model_args, model_kwargs, history = AlfvenDetectors.load_model(mf)
Nclust = 2 #....8
s2_model_name = "GMMModel"
s2_args = [Nclust]
s2_kwargs = Dict(
    :kind => :diag,
    :method => :kmeans)
s2_model = eval(Meta.parse("AlfvenDetectors."*s2_model_name))(s2_args...; s2_kwargs...);
fx(m,x) = AlfvenDetectors.fit!(m,x)
fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y);
asfs = [AlfvenDetectors.as_max_ll_mse, AlfvenDetectors.as_mean_ll_mse, 
    AlfvenDetectors.as_med_ll_mse, AlfvenDetectors.as_ll_maxarg]
asf = asfs[1]
label = 1

seed = 8

train_info, train_inds, test_info, test_inds = 
    AlfvenDetectors.split_unique_patches(0.5, 
        shotnos, labels, tstarts, fstarts; seed=seed);

train_labeled = (data[:,:,:,train_inds], train_info[2]);
test = (data[:,:,:,test_inds], test_info[2]);
train_unlabeled_data, shotnos_unlabeled =  (test[1][:,:,:,1:1], [])
train_unlabeled = (cat(train_unlabeled_data, train_labeled[1], dims=ndims(train_labeled[1])),
    vcat(zeros(size(train_unlabeled_data, ndims(train_unlabeled_data))), train_labeled[2]))

fsmodel = AlfvenDetectors.FewShotModel(s1_model, s2_model, fx, fxy, nothing);
AlfvenDetectors.fit!(fsmodel, train_unlabeled[1], train_labeled[1], train_labeled[2]);
as = AlfvenDetectors.anomaly_score(fsmodel, (m,x)->asf(m,x,label), test[1]);

_labels = test[2]
roc =  EvalCurves.roccurve(as, _labels)
auc = EvalCurves.auc(roc...)
prc = EvalCurves.prcurve(as, _labels)

df_gmm = DataFrame(
	:model => mf,
	:roc => roc,
	:prc => prc
	)        

df = vcat(df_knn, df_gmm)

CSV.write("/compass/home/skvara/alfven/experiments/conv/roc_prc_two_stage.csv", df)