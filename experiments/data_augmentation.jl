using AlfvenDetectors
using PyPlot
using PyPlot

hostname = gethostname()
if hostname == "vit-ThinkPad-E470"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
	savepath = "/home/vit/vyzkum/alfven/experiments/oneclass"
elseif hostname == "tarbik.utia.cas.cz"
	datapath = "/home/skvara/work/alfven/cdb_data/uprobe_data"
	savepath = "/home/skvara/work/alfven/experiments/oneclass"
elseif occursin("soroban", hostname) || hostname == "gpu-node"
	datapath = "/compass/home/skvara/no-backup/uprobe_data"
	savepath = "/compass/home/skvara/alfven/experiments/oneclass"
end

readfun = AlfvenDetectors.readnormlogupsd
patchsize = 128

shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()
patches = map(x->AlfvenDetectors.get_patch(datapath, x[1], x[2], x[3], patchsize, readfun; 
	memorysafe=true)[1], zip(shotnos, tstarts, fstarts))
patches = cat(patches..., dims=4)

μ = mean(patches)

N = size(patches, 4)
i = rand(1:N)

t = μ*0.8
figure()
subplot(211)
pcolormesh(patches[:,:,1,i])

subplot(212)
y = patches[:,:,1,i]
y[y .< t] .= 0.0
pcolormesh(y)
