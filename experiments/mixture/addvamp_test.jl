include("addvamp.jl")

using PyPlot
using CuArrays
using EvalCurves
using DelimitedFiles

# first create the data
function circle(i,j,patchsize,r)
	x = zeros(patchsize, patchsize)
	for _i in i-r:i+r
		for _j in j-r:j+r
			((_i-i)^2 + (_j-j)^2 <= r^2) ? x[_i,_j] = 1 : nothing
		end
	end
	x
end

function generate_data(N,patchsize,R)
	x = randn(patchsize, patchsize,1,N)/10
	for n in 1:N
		i = rand(R+1:patchsize-R)
		j = rand(R+1:patchsize-R)
		x[:,:,1,n] += circle(i,j,patchsize,R)
	end
	x
end

patchsize = 32
N = 100
N2 = Int(N/2)
N4 = Int(N2/2)
R = 3
train_data = Float32.(cat(generate_data(N2,patchsize,R), randn(patchsize, patchsize, 1, N2)/10, dims=4));
train_labels = vcat(ones(Int,N2), zeros(Int,N2));
train_data_unlabeled = Float32.(cat(generate_data(N4,patchsize,R), randn(patchsize, patchsize, 1, N4)/10, dims=4));
test_data = Float32.(cat(generate_data(N2,patchsize,R), randn(patchsize, patchsize, 1, N2)/10, dims=4));
test_labels = vcat(ones(Int,N2), zeros(Int,N2));

trX1 = train_data[:,:,:,train_labels.==1];
trX0 = train_data[:,:,:,train_labels.==0];

t1 = 0.5
t2 = 0.1
mixed_data = [(rand() > t1) ? ((rand() > t2) ? (trX1, 1) : (trX0, 0)) : (train_data_unlabeled, nothing) for i in 1:100];
data = [((rand() > t2) ? (trX1, 1) : (trX0, 0)) for i in 1:200];

K = 2
hdim = 40

# try own learning routine
X1 = train_data_unlabeled[:,:,:,1:4] |> gpu;
Y1 = 1
X0 = train_data_unlabeled[:,:,:,26:29] |> gpu;
Y0 = 0
X = test_data[:,:,:,49:52] |> gpu;
Y = nothing

ldim = 2
xsize = size(train_data)[1:3]
encoders = Tuple([GenModels.convencoder(xsize, ldim*2, 2, 3, (8,16), 1) |> gpu for k in 1:K]);
decoders = Tuple([GenModels.convdecoder(xsize, ldim, 2, 3, (16,8), 1) |> gpu for k in 1:K]);
ac = nothing

β = 0.1
m = ADDVAMP(encoders, decoders, ac, xsize, K) 
m.pseudoinputs = gpu(m.pseudoinputs);
opt = ADAM()

train_addvamp(m, mixed_data, β, opt)

plot_reconstructions(m,X0,Y0)
plot_reconstructions(m,X1,Y1)
plot_reconstructions(m,X1,0)

et = batch_mse(m, test_data)
Y = mapslices(argmin, et, dims=1) .- 1

s1 = et[1,:] - et[2,:]
auroc1 = EvalCurves.auc(EvalCurves.roccurve(s1, test_labels)...)

function eval_addvamp(data, test_data, test_labels)
	ldim = 2
	xsize = size(train_data)[1:3]
	encoders = Tuple([GenModels.convencoder(xsize, ldim*2, 2, 3, (8,16), 1) |> gpu for k in 1:K]);
	decoders = Tuple([GenModels.convdecoder(xsize, ldim, 2, 3, (16,8), 1) |> gpu for k in 1:K]);
	ac = nothing

	β = 0.1
	m = ADDVAMP(encoders, decoders, ac, xsize, K) 
	m.pseudoinputs = gpu(m.pseudoinputs);
	opt = ADAM()

	train_addvamp(m, data, β, opt)

	et = batch_mse(m, test_data)
	Y = mapslices(argmin, et, dims=1) .- 1

	s1 = et[1,:] - et[2,:]
	auroc1 = EvalCurves.auc(EvalCurves.roccurve(s1, test_labels)...)
	return auroc1
end

# test
devamp_results = map(i->eval_addvamp(data), 1:10)
devamp_results_mixed = map(i->eval_addvamp(mixed_data, test_data, test_labels), 1:10)
open("devamp_results.txt", "w") do io
	writedlm(io, devamp_results)
end
open("devamp_results_mixed.txt", "w") do io
	writedlm(io, devamp_results_mixed)
end

# compare to ae
function batch_mse_ae(m, X)
	es = map(i -> Flux.Tracker.data(Flux.mse(gpu(X[:,:,:,i:i]), 
			m(gpu(X[:,:,:,i:i])))), 1:size(X,4))
end

function eval_ae(test_data, test_labels)
	xsize = size(X)[1:3]
	ae0 = ConvAE(xsize, ldim, 2, 3, (8,16), 1) |> gpu;
	ae1 = ConvAE(xsize, ldim, 2, 3, (8,16), 1) |> gpu;
	GenModels.fit!(ae0, gpu(trX0), N2, 200);
	GenModels.fit!(ae1, gpu(trX1), N2, 200);

	mse0 = batch_mse_ae(ae0, test_data)
	mse1 = batch_mse_ae(ae1, test_data)
	s = mse0 - mse1
	auroc = EvalCurves.auc(EvalCurves.roccurve(s, test_labels)...)
	return auroc
end

ae_results = map(i -> eval_ae(test_data, test_labels), 1:10)
open("ae_results.txt", "w") do io
	writedlm(io, ae_results)
end

# make histograms of the 3 results
devamp_results = readdlm("devamp_results.txt")
devamp_results_mixed = readdlm("devamp_results_mixed.txt")
ae_results = readdlm("ae_results.txt")
results = cat(devamp_results, devamp_results_mixed, ae_results, dims=2)

figure()
boxplot(results, labels=["devamp-supervised", "devamp-semisupervised", "ae-supervised"])
savefig("devamp_comparison.eps")

#####
yy = Flux.Tracker.data(cpu(ae0(gpu(trX1))))
imodel = 10
figure()
subplot(121)
pcolormesh(trX0[:,:,1,imodel])
subplot(122)
pcolormesh(yy[:,:,1,imodel])

yy = Flux.Tracker.data(cpu(ae1(gpu(trX0))))
imodel = 10
figure()
subplot(121)
pcolormesh(trX1[:,:,1,imodel])
subplot(122)
pcolormesh(yy[:,:,1,imodel])

# look at latent
z11 = Flux.Tracker.data(cpu(sample_latent(m, gpu(trX1), 2)))
z10 = Flux.Tracker.data(cpu(sample_latent(m, gpu(trX1), 1)))
z01 = Flux.Tracker.data(cpu(sample_latent(m, gpu(trX0), 2)))
z00 = Flux.Tracker.data(cpu(sample_latent(m, gpu(trX0), 1)))
figure(figsize=(6,10))
subplot(3,2,1)
title("histograms of latent with respect to e1")
hist(z10[1,:], alpha=0.5, label="1")
hist(z00[1,:], alpha=0.5, label="0")
legend()

subplot(3,2,2)
hist(z10[2,:], alpha=0.5)
hist(z00[2,:], alpha=0.5)

subplot(3,2,3)
title("histograms of latent with respect to e2")
hist(z11[1,:], alpha=0.5)
hist(z01[1,:], alpha=0.5)

subplot(3,2,4)
hist(z11[2,:], alpha=0.5)
hist(z01[2,:], alpha=0.5)

n = 1000
zp0 = Flux.Tracker.data(cpu(sampleVamp(m,n,1)))
zp1 = Flux.Tracker.data(cpu(sampleVamp(m,n,2)))
subplot(3,2,5)
title("histograms of vamp prior")
hist(zp1[1,:], alpha=0.5)
hist(zp0[1,:], alpha=0.5)

subplot(3,2,6)
hist(zp1[2,:], alpha=0.5)
hist(zp0[2,:], alpha=0.5)
tight_layout()
