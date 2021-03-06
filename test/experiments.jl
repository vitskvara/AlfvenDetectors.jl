using AlfvenDetectors
using Test
using ValueHistories
using Pkg
using Flux
using GenModels

verb = true

hostname = gethostname()
if hostname == "vit-ThinkPad-E470"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/data_sample"
	#datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
elseif hostname == "tarbik.utia.cas.cz"
	datapath = "/home/skvara/work/alfven/cdb_data/data_sample"
elseif hostname in ["soroban-node-02", "soroban-node-03", "gpu-node"]
	datapath = "/compass/Shared/Exchange/Havranek/Link to Alfven"
else 
	datapath = "xyz"
end

usegpu = false

if "CuArrays" in keys(Pkg.installed()) && usegpu
	using CuArrays
end

# only run the test if the needed data is present

if isdir(datapath)
	savepath = joinpath(dirname(@__FILE__), "tmp")
	mkpath(savepath)

	ldim = 2
	nlayers = 2
	coils = [12,13,14]
	shots = readdir(datapath)[1:2]
	shots = joinpath.(datapath, shots)

	# get conv data
	heigth = 32
	width = 16
	for readfun in [AlfvenDetectors.readmscamp,
					AlfvenDetectors.readnormmscphase,
					AlfvenDetectors.readnormmscphase]
		alldata = hcat(map(x->x[:,1:(end-size(x,2)%width)], AlfvenDetectors.collect_signals(shots, readfun, coils))...);
		convdata = AlfvenDetectors.collect_conv_signals(shots, readfun, heigth, width, coils);
		@test alldata[1:heigth,1:width] == convdata[:,:,1,1]
		@test alldata[heigth+1:2*heigth,1:width] == convdata[:,:,1,2]
		@test floor(Int,size(alldata,1)/heigth)*size(alldata,2) == size(convdata,2)*size(convdata,4)
	end
	readfun = AlfvenDetectors.readnormlogupsd
	alldata = hcat(map(x->x[:,1:(end-size(x,2)%width)], AlfvenDetectors.collect_signals(shots, readfun))...)
	convdata = AlfvenDetectors.collect_conv_signals(shots, readfun, heigth, width)
	@test alldata[1:heigth,1:width] == convdata[:,:,1,1]
	@test alldata[heigth+1:2*heigth,1:width] == convdata[:,:,1,2]
	@test floor(Int,size(alldata,1)/heigth)*size(alldata,2) == size(convdata,2)*size(convdata,4)

	# get the labeled data information
	shotnos, labels = AlfvenDetectors.labeled_data()
	@test length(shotnos) == length(labels) == 40
	shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches(only_positive=true)
	@test length(shotnos) == length(labels) == length(tstarts) == length(fstarts) == sum(labels) > 0
	shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches(only_negative=true)
	@test length(shotnos) == length(labels) == length(tstarts) == length(fstarts) > 0
	@test sum(labels) == 0
	shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()
	@test length(shotnos) == length(labels) == length(tstarts) == length(fstarts) == 362
	# this is to ensure that there are no duplicities in the data
	data = [[shotnos[i], labels[i], tstarts[i], fstarts[i]] for i in 1:length(shotnos)]
	@test length(unique(data)) == length(data)
	data = cat(data..., dims=2)
	samelengths = [sum(all(data .== data[:,i], dims = 1)) for i in 1:size(data,2)]
	data_w_inds = vcat(data, collect(1:size(data,2))')
	@test size(data_w_inds[:,samelengths .!= 1]',1) == 0

	# split patches
	train_info, train_inds, test_info, test_inds = AlfvenDetectors.split_patches(0.0, shotnos, 
			labels, tstarts, fstarts; seed=1);
	@test train_info[1] == train_inds == nothing
	train_info, train_inds, test_info, test_inds = AlfvenDetectors.split_patches(1.0, shotnos, 
			labels, tstarts, fstarts; seed=1);
	@test train_info[1] == train_inds == nothing
	train_info, train_inds, test_info, test_inds = AlfvenDetectors.split_patches(0.5, shotnos, 
			labels, tstarts, fstarts; seed=1);
	@test shotnos[train_inds] == train_info[1]
	@test labels[train_inds] == train_info[2]
	@test tstarts[train_inds] == train_info[3]
	@test fstarts[train_inds] == train_info[4]
	@test shotnos[test_inds] == test_info[1]
	@test labels[test_inds] == test_info[2]
	@test tstarts[test_inds] == test_info[3]
	@test fstarts[test_inds] == test_info[4]

	# split patches unique - according to the shot number
	train_info, train_inds, test_info, test_inds = AlfvenDetectors.split_unique_patches(0.0, shotnos, 
			labels, tstarts, fstarts; seed=1);
	@test train_info[1] == train_inds == nothing
	train_info, train_inds, test_info, test_inds = AlfvenDetectors.split_unique_patches(1.0, shotnos, 
			labels, tstarts, fstarts; seed=1);
	@test train_info[1] == train_inds == nothing
	train_info, train_inds, test_info, test_inds = AlfvenDetectors.split_unique_patches(0.5, shotnos, 
			labels, tstarts, fstarts; seed=1);
	@test shotnos[train_inds] == train_info[1]
	@test labels[train_inds] == train_info[2]
	@test tstarts[train_inds] == train_info[3]
	@test fstarts[train_inds] == train_info[4]
	@test shotnos[test_inds] == test_info[1]
	@test labels[test_inds] == test_info[2]
	@test tstarts[test_inds] == test_info[3]
	@test fstarts[test_inds] == test_info[4]
	# shot numbers should not intersect in the 
	@test intersect(unique(train_info[1]), unique(test_info[1])) == []
	@test sort(vcat(unique(train_info[1]), unique(test_info[1]))) == sort(unique(shotnos))
	
	# get a patch
	ipatch = 10
	patchsize = 128
	patch, t, f = AlfvenDetectors.get_patch(datapath, shotnos[ipatch], tstarts[ipatch],
		fstarts[ipatch], patchsize, AlfvenDetectors.readnormlogupsd)
	@test size(patch) == (patchsize,patchsize)
	@test length(t) == patchsize
	@test length(f) == patchsize

	# test the data preparation functions
	available_shots = readdir(datapath)
	shotlist = AlfvenDetectors.split_shots(5, available_shots; seed = 1)
	@test length(shotlist) == 2
	@test length(shotlist[1])  == 5
	@test length(shotlist[2])  == length(available_shots) - 5
	@test intersect(shotlist[1], shotlist[2]) == []	
	shotlist2 = AlfvenDetectors.split_shots(11, available_shots; seed = 1)
	@test length(shotlist2[1]) == 11
	@test shotlist[1] == shotlist2[1][1:5]
	@test length(shotlist2[2])  == length(available_shots) - 11
	shotlist3 = AlfvenDetectors.split_shots(9, available_shots, (train_info[1], test_info[1]),
		seed = 1)
	@test length(shotlist3[1]) == 9
	@test length(shotlist3[2])  == length(available_shots) - 9
	@test intersect(shotlist3[1], shotlist3[2]) == []	
	# the training shots must not contain data from the testing patches
	@test !any(map(x->any(occursin.(string(x), shotlist3[1])), test_info[1]))
	# but they should contain some data from the training patches
	@test any(map(x->any(occursin.(string(x), shotlist3[1])), train_info[1]))

	# collect noisy patches
	available_inds = filter(i->any(occursin.(string(train_info[1][i]), available_shots)),1:length(train_info[1]))
	noisy_patches, noisy_shotnos, noisy_tstarts, noisy_fstarts =
		AlfvenDetectors.collect_training_patches(datapath, 
			train_info[1][available_inds], train_info[3][available_inds], train_info[4][available_inds], 10, 
			AlfvenDetectors.readnormlogupsd, patchsize; δ = 0.02, seed = nothing, memorysafe=true)
	@test size(noisy_patches,4) == length(noisy_shotnos) == length(noisy_tstarts) == length(noisy_fstarts) == 10
	@test size(noisy_patches) == (patchsize,patchsize,1,10)

	# also test the whole frikkin loading of training data for experiments
	collect_fun(x) = 
		AlfvenDetectors.collect_conv_signals(x, readfun, patchsize, memorysafe=true)
 	data, train_shots = AlfvenDetectors.collect_training_data(datapath, collect_fun, 12, readfun,
		0.1, patchsize; seed=1, use_alfven_shots=true)
 	@test size(data,1) == size(data,2) == patchsize
 	@test !any(map(x->any(occursin.(string(x), train_shots)), test_info[1]))
 	@test any(map(x->any(occursin.(string(x), train_shots)), train_info[1]))

 	# test the shift function
 	shotno = 10004
 	fname = joinpath(datapath, filter(x-> occursin(string(shotno), x), readdir(datapath))[1])
 	patch_info = AlfvenDetectors.labeled_patches()
	patch_info = map(x->x[patch_info[1].==shotno],patch_info)
	patchsize = 128
	tpatch, fpatch = patch_info[3][1], patch_info[4][1]
	patch, t, f = AlfvenDetectors.get_patch(datapath, shotno, tpatch,
		fpatch, patchsize, AlfvenDetectors.readnormlogupsd)
	trange = t[end]-t[1] # trange = 0.00650239f0
	frange = f[end]-f[1] # frange = 620117.25f0
	tshifted, fshifted = AlfvenDetectors.shift_patch(tpatch, fpatch)
	@test tshifted != tpatch
	@test abs(tshifted - tpatch) <= trange/4
	@test fshifted != fpatch
	@test abs(fshifted - fpatch) <= frange/4	
	
	# test the data collection functions for one class experiments
	train_info, train_inds, test_info, test_inds = AlfvenDetectors.test_train_oneclass(datapath; α=0.8, seed = 1)
	tri = [(train_info[1][i], train_info[2][i], train_info[3][i], train_info[4][i]) for i in 1:length(train_info[1])]
	tsti = [(test_info[1][i], test_info[2][i], test_info[3][i], test_info[4][i]) for i in 1:length(test_info[1])]
	@test length(intersect(tri, tsti)) == 0

	Npatches = 50
	training_patches, training_shotnos, training_labels, training_tstarts, training_fstarts = 
		AlfvenDetectors.collect_training_data_oneclass(datapath, Npatches, AlfvenDetectors.readnormlogupsd,
		patchsize; α = 0.8, seed=1)
	@test size(training_patches, 4) == Npatches == length(training_shotnos) == length(training_labels) == length(training_tstarts) == length(training_fstarts)
	@test sum(training_labels) == Npatches

	testing_patches, testing_shotnos, testing_labels, testing_tstarts, testing_fstarts = 
		AlfvenDetectors.collect_testing_data_oneclass(datapath, AlfvenDetectors.readnormlogupsd,
		patchsize; α = 0.8, seed=1)
	@test 0 < size(testing_patches, 4) == length(testing_shotnos) == length(testing_labels) == length(testing_tstarts) == length(testing_fstarts)
	@test 0 < sum(testing_labels) < length(testing_labels)
	println("$(sum(testing_labels))")
	println("$(length(testing_labels))")
	
	
	# msc amplitude + AE
	rawdata = hcat(AlfvenDetectors.collect_signals(shots, AlfvenDetectors.readmscampphase, coils; type="flattop")...)
	if usegpu
		data = rawdata |> gpu
	else
		data = rawdata
	end
	xdim = size(data,1)
	batchsize = 64
	outer_nepochs = 2
	inner_nepochs = 1
	@testset "single column unsupervised - AE" begin
		modelname = "AE"
		model_args = [
			:xdim => xdim,
			:ldim => ldim, 
			:nlayers => nlayers
			]
		model_kwargs = Dict(
			)
		fit_kwargs = Dict(
			)

		model = GenModels.construct_model(modelname, model_args...; model_kwargs...)
		model, history, t = AlfvenDetectors.fitsave_unsupervised(data, model, batchsize, 
			outer_nepochs, inner_nepochs,
			model_args, model_kwargs, fit_kwargs, 
			savepath; filename = "ae_test.bson", verb = verb)
		@test isfile(joinpath(savepath,"ae_test.bson"))
	end

	# msc phase + VAE
	GC.gc()
	rawdata = hcat(AlfvenDetectors.collect_signals(shots, AlfvenDetectors.readnormmscphase, coils; type="valid")...)
	if usegpu
		data = rawdata |> gpu
	else
		data = rawdata
	end
	xdim = size(data,1)
	@testset "single column unsupervised - VAE" begin
		modelname = "VAE"
		model_args = [
			:xdim => xdim,
			:ldim => ldim, 
			:nlayers => nlayers
			]
		model_kwargs = Dict(
			:variant => :unit
			)
		fit_kwargs = Dict(
			:beta => 1.0,
			:L => 1
			)
		model = GenModels.construct_model(modelname, model_args...; model_kwargs...)
		model, history, t = AlfvenDetectors.fitsave_unsupervised(data, model, batchsize, 
			outer_nepochs, inner_nepochs,
			model_args, model_kwargs, fit_kwargs, 
			savepath; optname = "NADAM", eta=0.0005, filename = "vae_test.bson", verb = verb)
		@test isfile(joinpath(savepath,"vae_test.bson"))
	end

	# uprobe psd + TSVAE
	GC.gc()
	rawdata = hcat(AlfvenDetectors.collect_signals(shots, AlfvenDetectors.readnormlogupsd)...)
	if usegpu
		data = rawdata |> gpu
	else
		data = rawdata
	end
	xdim = size(data,1)
	@testset "single column unsupervised - TSVAE" begin
		modelname = "TSVAE"
		model_args = [
			:xdim => xdim,
			:ldim => ldim, 
			:nlayers => nlayers
			]
		model_kwargs = Dict(
			)
		fit_kwargs = Dict(
			:beta => 1.0,
			:L => 1
			)
		model = GenModels.construct_model(modelname, model_args...; model_kwargs...)
		model, history, t = AlfvenDetectors.fitsave_unsupervised(data, model, batchsize, 
			outer_nepochs, inner_nepochs,
			model_args, model_kwargs, fit_kwargs, 
			savepath; filename = "tsvae_test.bson", verb = verb)
		@test isfile(joinpath(savepath,"tsvae_test.bson"))
	end
	
	# uprobe psd + ConvTSVAE
	readfun = AlfvenDetectors.readnormlogupsd
	patchsize = 64
	convdata = AlfvenDetectors.collect_conv_signals(shots, readfun, patchsize)
	GC.gc()
	xdim = size(convdata)
	kernelsize = 3
	channels = (4,8)
	scaling = 4
	batchsize = 16
	outer_nepochs = 2
	inner_nepochs = 1
	@testset "patches of uprobe data - TSVAE" begin
		modelname = "ConvTSVAE"
		model_args = [
			:xdim => xdim[1:3],
			:ldim => ldim, 
			:nlayers => nlayers,
			:kernelsize => kernelsize,
			:channels => channels,
			:scaling => scaling
			]
		model_kwargs = Dict(
			)
		fit_kwargs = Dict(
			:L => 1
			)
		model = GenModels.construct_model(modelname, model_args...; model_kwargs...)
		model, history, t = AlfvenDetectors.fitsave_unsupervised(convdata, model, batchsize, 
			outer_nepochs, inner_nepochs,
			model_args, model_kwargs, fit_kwargs, 
			savepath; usegpu=usegpu, filename = "convtsvae_test.bson", verb = verb)
		@test isfile(joinpath(savepath,"convtsvae_test.bson"))
	end
	
	rm(savepath, recursive = true)


end