
using AlfvenDetectors
using PyPlot
using Flux
using CuArrays  # for GPU runs
using ValueHistories
using BSON: @save, @load

host = gethostname()
if occursin("vit", host)
    datapath = "/home/vit/vyzkum/alfven/cdb_data/original_data/"
else
    datapath = "/home/skvara/work/alfven/cdb_data/original_data/"
end

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
nepochs = 500
cbit = 1
# progress bars are broken in notebooks
if occursin(".jl", @__FILE__) 
    verb = true
else
    verb = false
end

@info "Training small CPU model"
@time AlfvenDetectors.fit!(small_model, data, batchsize, nepochs;
    cbit = cbit, history = small_train_history, verb = verb)

@info "Training large CPU model"
@time AlfvenDetectors.fit!(large_model, data, batchsize, nepochs;
    cbit = cbit, history = large_train_history, verb = verb)

figure()
plot(get(small_train_history, :loss)...)
title("Training loss - smaller model")
xlabel("iteration")
ylabel("loss")

figure()
plot(get(large_train_history, :loss)...)
title("Training loss - larger model")
xlabel("iteration")
ylabel("loss")

X = data;

figure()
pcolormesh(1:size(X,2), f, X)
title("Original data")
xlabel("t")
ylabel("f")

figure()
sX = small_model(X).data
pcolormesh(1:size(sX,2), f, sX)
title("AE output - smaller model")
xlabel("t")
ylabel("f")

figure()
lX = large_model(X).data
pcolormesh(1:size(lX,2), f, lX)
title("AE output - larger model")
xlabel("t")
ylabel("f")

# convert to CuArrays
zdim = 2
cudata = data |> gpu
cumodel = AlfvenDetectors.AE([M, 20, zdim], [zdim, 20, M]) |> gpu
cu_train_history = MVHistory()
nepochs = 500

@info "Training a small GPU model"
# clear cache
GC.gc()
@time AlfvenDetectors.fit!(cumodel, cudata, batchsize, nepochs;
    cbit = cbit, history = cu_train_history, verb = verb)
# clear cache
GC.gc()

@info "CPU model(data)"
@time small_model(data);

@info "GPU model(data)"
@time cumodel(cudata);

figure()
plot(get(cu_train_history, :loss)...)
title("Training loss")
xlabel("iteration")
ylabel("loss")

figure()
X = cudata;
_X = cumodel(X).data |> cpu
pcolormesh(1:size(_X,2), f, _X)
title("AE output with GPU training")
xlabel("t")
ylabel("f")

# save/load a pretrained model
f = "large_model.bson"
if !isfile(f) 
    @save f large_model
else
    @load f large_model
end

X1, t1, f1 = get_msc_array(datapath, 11096, 11)
pcolormesh(t1, f1, X1)

X0, t0, f0 = get_msc_array(datapath, 11096, 20)
pcolormesh(t0, f0, X0)

Xα = X1[:,1.06.<=t1.<=1.22]
zα = large_model.encoder(Xα).data
z1 = large_model.encoder(X1).data
z0 = large_model.encoder(X0).data

figure()
scatter(z1[1,:], z1[2,:], label = "positive")
scatter(z0[1,:], z0[2,:], label = "negative")
scatter(zα[1,:], zα[2,:], label = "alfven mode")
legend()

function connect(zs, l)
    L = length(zs)
    return vcat([hcat(
    collect(range(zs[i][1], zs[i+1][1]; length = l)), 
    collect(range(zs[i][2], zs[i+1][2]; length = l))
        )
    for i in 1:L-1]...)
end
zs = [[-6.5,-0.5], [-7.5,-2], [-7.5,-4], [-4,-4.5], [0,0]]
zpath = Array(connect(zs, 30)');

figure()
scatter(z1[1,:], z1[2,:], label = "positive")
scatter(z0[1,:], z0[2,:], label = "negative")
scatter(zα[1,:], zα[2,:], label = "alfven mode")
plot(zpath[1,:], zpath[2,:], label = "artificial z")
legend()

Xgen = large_model.decoder(zpath).data;

figure()
pcolormesh(Xgen)
title("artificial coherence")

show()
