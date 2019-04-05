using AlfvenDetectors
using Test
using Flux
using ValueHistories
using Pkg

verb = false

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

	# msc amplitude + AE
	rawdata = hcat(AlfvenDetectors.collect_signals(shots, AlfvenDetectors.readmscampphase, coils; type="flattop")...)
	data = rawdata |> gpu
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

		model, history, t = AlfvenDetectors.fitsave_unsupervised(data, modelname, batchsize, 
			outer_nepochs, inner_nepochs,
			model_args, model_kwargs, fit_kwargs, 
			savepath; filename = "ae_test", verb = verb)
		@test isfile(joinpath(savepath,"ae_test.bson"))
	end

	# msc phase + VAE
	GC.gc()
	rawdata = hcat(AlfvenDetectors.collect_signals(shots, AlfvenDetectors.readnormmscphase, coils; type="valid")...)
	data = rawdata |> gpu
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

		model, history, t = AlfvenDetectors.fitsave_unsupervised(data, modelname, batchsize, 
			outer_nepochs, inner_nepochs,
			model_args, model_kwargs, fit_kwargs, 
			savepath; optname = "NADAM", eta=0.0005, filename = "vae_test", verb = verb)
		@test isfile(joinpath(savepath,"vae_test.bson"))
	end

	# uprobe psd + TSVAE
	GC.gc()
	rawdata = hcat(AlfvenDetectors.collect_signals(shots, AlfvenDetectors.readnormlogupsd)...)
	data = rawdata |> gpu
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

		model, history, t = AlfvenDetectors.fitsave_unsupervised(data, modelname, batchsize, 
			outer_nepochs, inner_nepochs,
			model_args, model_kwargs, fit_kwargs, 
			savepath; filename = "tsvae_test", verb = verb)
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

		model, history, t = AlfvenDetectors.fitsave_unsupervised(convdata, modelname, batchsize, 
			outer_nepochs, inner_nepochs,
			model_args, model_kwargs, fit_kwargs, 
			savepath; usegpu=usegpu, filename = "convtsvae_test", verb = verb)
		@test isfile(joinpath(savepath,"convtsvae_test.bson"))
	end
	
	rm(savepath, recursive = true)
end