
using AlfvenDetectors
using PyPlot
using Flux
using CuArrays  # for GPU runs
using ValueHistories

datapath = "/home/vit/vyzkum/alfven/cdb_data/original_data/"

function get_msc_array(datapath, shot, coil, timelim = [1.0, 1.25])
    _data = AlfvenDetectors.BaseAlfvenData(joinpath(datapath,"$(shot).h5"), [coil])
    tinds = timelim[1] .<= _data.t .<= timelim[2]
    return _data.msc[coil][:,tinds], _data.t[tinds], _data.f 
end

msc, t, f = get_msc_array(datapath, 11096, 20)

pcolormesh(t,f,msc)

function collect_msc(datapath, shot, coils)
    datalist = map(x-> get_msc_array(datapath, shot, x), coils)
    return hcat([x[1] for x in datalist]...), datalist[1][3]
end

shots_coils = [
#    [10370, [12, 15, 17, 20]],
    [10370, [12, 20]],
#    [11096, [11, 8, 17, 20]]
    [11096, [11, 8, 20]]
]
datalist = map(x->collect_msc(datapath, x[1], x[2]), shots_coils)
data, f = hcat([x[1] for x in datalist]...), datalist[1][2]

pcolormesh(1:size(data,2), f, data)

M,N = size(data)
# fortunately data is already normalized in the interval (0,1)
zdim = 2
small_model = AlfvenDetectors.AE([M, 20, zdim], [zdim, 20, M])
large_model = AlfvenDetectors.AE([M, 200, zdim], [zdim, 200, M])
small_train_history = MVHistory()
large_train_history = MVHistory()
batchsize = 64
nepochs = 1000
cbit = 1
# progress bars are broken in notebooks
if occursin(@__FILE__, ".jl") 
    verb = true
else
    verb = false
end

@time AlfvenDetectors.fit!(small_model, data, batchsize, nepochs;
    cbit = cbit, history = small_train_history, verb = verb)

@time AlfvenDetectors.fit!(large_model, data, batchsize, nepochs;
    cbit = cbit, history = large_train_history, verb = verb)

plot(get(small_train_history, :loss)...)
title("Training loss - smaller model")
xlabel("iteration")
ylabel("loss")

plot(get(large_train_history, :loss)...)
title("Training loss - larger model")
xlabel("iteration")
ylabel("loss")

X = data;

pcolormesh(1:size(X,2), f, X)
title("Original data")
xlabel("t")
ylabel("f")

sX = small_model(X).data
pcolormesh(1:size(sX,2), f, sX)
title("AE output - smaller model")
xlabel("t")
ylabel("f")

lX = large_model(X).data
pcolormesh(1:size(lX,2), f, lX)
title("AE output - larger model")
xlabel("t")
ylabel("f")

# convert to CuArrays
zdim = 10
cudata = data |> gpu
cumodel = AlfvenDetectors.AE([M, 10, zdim], [zdim, 10, M]) |> gpu
cu_train_history = MVHistory()
nepochs = 200

@time AlfvenDetectors.fit!(cumodel, cudata, batchsize, nepochs;
    cbit = cbit, history = cu_train_history, verb = verb)

plot(get(cu_train_history, :loss)...)
title("Training loss")
xlabel("iteration")
ylabel("loss")

X = cudata;
_X = cumodel(X).data |> cpu
pcolormesh(1:size(_X,2), f, _X)
title("AE output with GPU training")
xlabel("t")
ylabel("f")

data1 = get_msc_array(datapath, 11096, 11)
pcolormesh(data1[2], data1[3], data1[1])

data0 = get_msc_array(datapath, 11096, 20)
pcolormesh(data0[2], data0[3], data0[1])

z1 = large_model.encoder(data1[1]).data
z0 = large_model.encoder(data0[1]).data

scatter(z1[1,:], z1[2,:], label = "positive")
scatter(z0[1,:], z0[2,:], label = "negative")
legend()
