using AlfvenDetectors
using Test
using ValueHistories
using Pkg
using Flux
using GenerativeModels

verb = true

hostname = gethostname()
if hostname == "vit-ThinkPad-E470"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/data_sample"
elseif hostname == "tarbik.utia.cas.cz"
	datapath = "/home/skvara/work/alfven/cdb_data/data_sample"
elseif hostname == "soroban-node-03"
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
	shotnos, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()
	@test length(shotnos) == length(labels) == length(tstarts) == length(fstarts) == 371
	
	# get a patch
	ipatch = 10
	patch, t, f = AlfvenDetectors.get_patch(datapath, shotnos[ipatch], tstarts[ipatch],
		fstarts[ipatch], 128, AlfvenDetectors.readnormlogupsd)
	@test size(patch) == (128,128)
	@test length(t) == 128
	@test length(f) == 128

	# test the data preparation functions
	available_shots = readdir(datapath)
	shotlist = AlfvenDetectors.select_training_shots(5, available_shots; seed = 1)
	@test length(shotlist) == 5
	shotlist2 = AlfvenDetectors.select_training_shots(11, available_shots; seed = 1)
	@test length(shotlist2) == 11
	@test shotlist == shotlist2[1:5]
	shotlist3 = AlfvenDetectors.select_training_shots(11, available_shots; seed = 2)
	@test length(shotlist3) == 11
	@test shotlist2 != shotlist3

	# select_training_patches(α::Real; seed = nothing)
	@test AlfvenDetectors.select_training_patches(0.0) == (nothing, nothing, nothing, nothing)
	patchdata = AlfvenDetectors.select_training_patches(0.1)
	@test length(patchdata) == 4
	@test length(patchdata[1]) == length(patchdata[2]) == length(patchdata[3]) == length(patchdata[4]) != 0
	patchdata = AlfvenDetectors.select_positive_training_patches(0.1)
	@test length(patchdata) == 4
	@test length(patchdata[1]) == length(patchdata[2]) == length(patchdata[3]) == length(patchdata[4]) != 0
	
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

		model = GenerativeModels.construct_model(modelname, model_args...; model_kwargs...)
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
		model = GenerativeModels.construct_model(modelname, model_args...; model_kwargs...)
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
		model = GenerativeModels.construct_model(modelname, model_args...; model_kwargs...)
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
		model = GenerativeModels.construct_model(modelname, model_args...; model_kwargs...)
		model, history, t = AlfvenDetectors.fitsave_unsupervised(convdata, model, batchsize, 
			outer_nepochs, inner_nepochs,
			model_args, model_kwargs, fit_kwargs, 
			savepath; usegpu=usegpu, filename = "convtsvae_test.bson", verb = verb)
		@test isfile(joinpath(savepath,"convtsvae_test.bson"))
	end
	
	rm(savepath, recursive = true)


end