using PyPlot
using AlfvenDetectors
using BSON
using Flux
using ValueHistories
using StatsBase
using GenModels

datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
shots = joinpath.(datapath, readdir(datapath))
shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()
patchsize = 128
readfun = AlfvenDetectors.readnormlogupsd
cmap = "plasma"

@time patchdata = map(x->AlfvenDetectors.get_patch(datapath, x[1], x[2], x[3], patchsize, readfun;
        memorysafe=true), zip(shotnos, tstarts, fstarts))
data = cat([x[1] for x in patchdata]..., dims=4);
println(size(data))

alfvendata = data[:,:,:,labels.==1]
noalfvendata = data[:,:,:,labels.==0];

#modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/"
#modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/ae-test";
#modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/batchnorm-test"
#modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/benchmark-runs"
#modelpath = "/home/vit/vyzkum/alfven/experiments/conv_old_library/uprobe/benchmark-runs"
modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/data_augmentation/"
filenames = joinpath.(modelpath, readdir(modelpath))
println("$(length(filenames)) models found in modelpath")

# use only the top 3 models as they have all the params saved
filenames = filenames[1:3]

model_params_list = []
for imodel in 1:length(filenames)
    model_params = AlfvenDetectors.parse_params(filenames[imodel])
    push!(model_params_list, model_params)
end

model_args_list = []
model_kwargs_list = []
experiment_args_list = []
for filename in filenames
    data = BSON.load(filename)
    push!(model_args_list, data[:model_args])
    push!(model_kwargs_list, data[:model_kwargs])
    push!(experiment_args_list, data[:experiment_args])
end

filter_list = [
    "patchsize" => patchsize,
    "modelname" => "AE",
    "nepochs" => 1000,
]
filter_inds = map(x->all(map(y->x[y[1]] == y[2],filter_list)), experiment_args_list)
model_params_list = model_params_list[filter_inds]
filename_list = filenames[filter_inds]
println("working with $(length(filename_list)) models")

loss_list = []
model_list = []
for (params, filename) in zip(experiment_args_list, filenames)
    model_data = BSON.load(filename)
    is, ls = get(model_data[:history], :loss)
    push!(loss_list, ls)
    if get(params, "batchnorm", false)
        model = Flux.testmode!(model_data[:model])
    else
        model = model_data[:model]
    end
    push!(model_list, model)
end
final_losses = [x[end] for x in loss_list];

alfven_loss = []
noalfven_loss = []
batchsize = 64
for (model, params) in zip(model_list, model_params_list)
    if occursin("VAE", params[:model])
        push!(alfven_loss, GenModels.loss(model, alfvendata[:,:,:,1:batchsize], 1, 1.0).data)
        push!(noalfven_loss, GenModels.loss(model, noalfvendata[:,:,:,1:batchsize], 1, 1.0).data)
    else
        push!(alfven_loss, GenModels.loss(model, alfvendata[:,:,:,1:batchsize]).data)
        push!(noalfven_loss, GenModels.loss(model, noalfvendata[:,:,:,1:batchsize]).data)
    end
end

sortinds = sortperm(final_losses);
sortinds = sortperm(alfven_loss);
isample = 1
patch = data[:,:,:,isample:isample]
#sample = convsubtestdata[:,:,:,isample:isample]
#sample = validdata[:,:,:,isample:isample]
figure()
pcolormesh(patch[:,:,1,1],cmap=cmap)
for imodel in sortinds
    figure()
    ns = model_list[imodel](patch).data
    cl = Flux.mse(patch,ns)
    title("model: $imodel, final training loss: $(round(final_losses[imodel],digits=5)),
        alfven data loss: $(round(alfven_loss[imodel],digits=5)),
        no alfven data loss: $(round(noalfven_loss[imodel],digits=5)),
        patch loss: $(round(cl,digits=5)),
        ratio of alfven training data: $(experiment_args_list[imodel]["positive-patch-ratio"])")
    pcolormesh(ns[:,:,1,1],cmap=cmap)
    text(135, 30, AlfvenDetectors.pretty_params(model_params_list[imodel]))
end
show()


filenames = joinpath.(modelpath, readdir(modelpath))
# use only the top 3 models as they have all the params saved
filenames = filenames[4:end]
submodels = []
subparams = []
subflosses = []
subalosses = []
subnalosses = []
for filename in filenames
    println(filename)
    model_params = AlfvenDetectors.parse_params(filename)
    push!(subparams, model_params)
    model_data = BSON.load(filename)
    model = Flux.testmode!(model_data[:model])
    push!(submodels, model)
    is,ls = get(model_data[:history], :loss)
    push!(subflosses, ls[end])
    push!(subalosses, GenModels.loss(model, alfvendata[:,:,:,1:batchsize]).data)
    push!(subnalosses, GenModels.loss(model, noalfvendata[:,:,:,1:batchsize]).data)
end

isample = 15
patch = data[:,:,:,isample:isample]
#sample = convsubtestdata[:,:,:,isample:isample]
#sample = validdata[:,:,:,isample:isample]
figure()
pcolormesh(patch[:,:,1,1],cmap=cmap)
for imodel in 1:length(submodels)
    figure()
    ns = submodels[imodel](patch).data
    cl = Flux.mse(patch,ns)
    title("model: $imodel, final training loss: $(round(subflosses[imodel],digits=5)),
        alfven data loss: $(round(subalosses[imodel],digits=5)),
        no alfven data loss: $(round(subnalosses[imodel],digits=5)),
        patch loss: $(round(cl,digits=5))")
    pcolormesh(ns[:,:,1,1],cmap=cmap)
end
show()

imodel = 3
file = filenames[imodel]
loss = BSON.load(file)[:history]

#plotlosses(hist)
#plot(loss[500:end])

i = 15
figure()
pcolormesh(data[:,:,1,i],cmap=cmap)
patch = data[:,:,1:1,i:i];
figure()
rp = model(patch).data[:,:,1,1]
l = Flux.mse(rp,patch)
title("loss $l")
pcolormesh(rp,cmap=cmap)

using PyCall
umap = pyimport("umap")

umap_model = umap.UMAP(n_components = 2, n_neighbors=15, min_dist=0.1)

zdata = []
N = size(data,4)
for i in 1:ceil(Int,N/10)
    _zdata = model.encoder(data[:,:,:,(i-1)*10+1:min(i*10,N)]).data
    push!(zdata,_zdata)
end
zdata = hcat(zdata...);
size(zdata)

zdata2D = Array(umap_model.fit_transform(zdata')')

figure()
scatter(zdata2D[1,labels.==1], zdata2D[2,labels.==1],label="alfven",s=5)
scatter(zdata2D[1,labels.==0], zdata2D[2,labels.==0],label="no alfven",s=5)
title("all data transformed into 2D")
legend()

lims = [-8 -2.5; 1 4]
plotbox = [lims[1,1] lims[1,2] lims[1,2] lims[1,1] lims[1,1]; lims[2,1] lims[2,1] lims[2,2] lims[2,2] lims[2,1]]
zinds = vec(all(lims[:,1] .< zdata2D .< lims[:,2], dims=1));

scatter(zdata2D[1,zinds], zdata2D[2,zinds],label="selected patches",s=15,c="k")
scatter(zdata2D[1,labels.==1], zdata2D[2,labels.==1],label="alfven",s=5)
scatter(zdata2D[1,labels.==0], zdata2D[2,labels.==0],label="no alfven",s=5)
plot(plotbox[1,:], plotbox[2,:])
title("all data transformed into 2D")
legend()

data_loss = map(i->AlfvenDetectors.loss(model,data[:,:,:,i:i]).data,collect(1:size(data,4)))

for i in collect(1:size(data,4))[zinds]
    figure(figsize=(10,5))
    subplot(1,2,1)
    suptitle("shot $(shotnos[i]), label $(labels[i]), loss $(data_loss[i])")
    pcolormesh(data[:,:,1,i],cmap=cmap)
    subplot(1,2,2)
    pcolormesh(model(data[:,:,:,i:i]).data[:,:,1,1],cmap=cmap)
end    

clusterinds = (lims[:,1] .<= zdata2D .<= lims[:,2]);
clusterinds = clusterinds[1,:] .& clusterinds[2,:];

scatter(zdata2D[1,:], zdata2D[2,:],s=3)
scatter(zdata2D[1,clusterinds], zdata2D[2,clusterinds],s=3)
plot(box[1,:], box[2,:],c="k")

clusterconvdata = data[:,:,:,clusterinds];
size(clusterconvdata)

for i in 1:size(clusterconvdata,4)
    figure()
    pcolormesh(clusterconvdata[:,:,1,i])
end

#sample = batch[:,:,1:1,2:2]
sample = chirpdata[:,:,:,2:2];
if params[:model] == "ConvTSVAE"
    m = model.m1.encoder.layers[1].layers[1](sample);
    m = model.m1.encoder.layers[1].layers[2](m);
else
    m = model.encoder.layers[1].layers[1](sample);
    m = model.encoder.layers[1].layers[2](m);
end

pcolormesh(sample[:,:,1,1])

for i in 1:size(m,3)
    figure()
    pcolormesh(m.data[:,:,i,1])
end

z = model.encoder(sample);
if params[:model] == "ConvAE"
    mx = model.decoder.layers[2](model.decoder.layers[1](z))
    mx = model.decoder.layers[3].layers[1](mx)
    #mx = model.decoder.layers[3].layers[2](mx)
end

for i in 1:size(mx,3)
    figure()
    pcolormesh(mx.data[:,:,i,1])
end

modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/"
filenames = readdir(modelpath)
params = [
#    :nepochs => 200
    :opt => "NADAM"
]
fstrings = vcat(["$(x[1])-$(x[2])" for x in params])
filenames = joinpath.(modelpath,filter(x->any(map(y->occursin(y,x),fstrings)),filenames));
println("working with a list of $(length(filenames)) files")

filename = filenames[2]
model_data = BSON.load(filename)
model = model_data[:model]
hist = model_data[:history]
params = parse_params(filename)

plotlosses(hist)

filename

isample = 1
#sample = convsubtestdata[:,:,:,isample:isample]
sample = data[:,:,:,isample:isample]
pcolormesh(sample[:,:,1,1])
ns = model(sample).data
figure()
pcolormesh(ns[:,:,1,1])

modelpath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/batchnorm-test"
filenames = joinpath.(modelpath, readdir(modelpath))
aepath = "/home/vit/vyzkum/alfven/experiments/conv/uprobe/ae-test"
aefilenames = joinpath.(aepath, readdir(aepath))
filenames = vcat(filenames, aefilenames);
println("working with a total of $(length(filenames)) files")

loss_list = []
model_params_list = []
model_list = []
for imodel in 1:length(filenames)
    model_data = BSON.load(filenames[imodel])
    is, ls = get(model_data[:history], :loss)
    push!(loss_list, ls)
    model_params = parse_params(filenames[imodel])
    push!(model_params_list, model_params)
    if get(model_params, :batchnorm, false)
        model = Flux.testmode!(model_data[:model])
    else
        model = model_data[:model]
    end
    push!(model_list, model)
end
final_losses = [x[end] for x in loss_list];

filter_list = [
    x->x[:xdim] == (patchsize,patchsize,1),
    x->x[:model] == "ConvAE",
    x->x[:opt] == RMSProp
]
filter_inds = map(x->all(map(y->y(x),filter_list)),model_params_list)
filtered_params = model_params_list[filter_inds]
filtered_models = model_list[filter_inds]
filtered_losses = loss_list[filter_inds]
filtered_filenames = filenames[filter_inds]

imin = 200
imax = 550
for (loss, params) in zip(filtered_losses, filtered_params)
    bn = get(params, :batchnorm, false)
    plot(loss[imin:imax], label = "batchnorm: $bn")
end
legend()
title("AE - convergence rate depending on the use of batch normalization")

filter_list = [
    x->!get(x,:batchnorm,false),
    x->x[:eta]==0.001,
    x->x[:model] == "ConvAE"
]
filter_inds = map(x->all(map(y->y(x),filter_list)),model_params_list)
filtered_params = model_params_list[filter_inds]
filtered_models = model_list[filter_inds]
filtered_losses = loss_list[filter_inds]
filtered_filenames = filenames[filter_inds]

filtered_losses = filtered_losses[1:4]
filtered_params = filtered_params[1:4]
imin = 5
imax = 500
for (loss, params) in zip(filtered_losses, filtered_params)
    opt = get(params, :opt, false)
    plot(loss[imin:imax], label = "optimiser: $opt")
end
legend()
title("AE - Convergence rate depending on the used optimiser")
