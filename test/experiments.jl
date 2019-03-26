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
else 
	datapath = "xyz"
end

if "CuArrays" in keys(Pkg.installed())
	using CuArrays
end

# only run the test if the needed data is present
if isdir(datapath)
	savepath = joinpath(dirname(@__FILE__), "tmp")
	mkpath(savepath)

	usegpu = false
	ldim = 2
	nlayers = 2
	coils = [12,13,14]	
	if usegpu
		using CuArrays
	end

	shots = readdir(datapath)[1:2]
	shots = joinpath.(datapath, shots)
	# msc amplitude + AE
	rawdata = AlfvenDetectors.collect_signals(shots, AlfvenDetectors.readmscampphase, coils; type="flattop") 
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
	rawdata = AlfvenDetectors.collect_signals(shots, AlfvenDetectors.readnormmscphase, coils; type="valid") 
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
			:β => 1.0,
			:L => 1
			)

		model, history, t = AlfvenDetectors.fitsave_unsupervised(data, modelname, batchsize, 
			outer_nepochs, inner_nepochs,
			model_args, model_kwargs, fit_kwargs, 
			savepath; filename = "vae_test", verb = verb)
		@test isfile(joinpath(savepath,"vae_test.bson"))
	end

	# uprobe psd + TSVAE
	GC.gc()
	rawdata = AlfvenDetectors.collect_signals(shots, AlfvenDetectors.readnormlogupsd) 
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
			:β => 1.0,
			:L => 1
			)

		model, history, t = AlfvenDetectors.fitsave_unsupervised(data, modelname, batchsize, 
			outer_nepochs, inner_nepochs,
			model_args, model_kwargs, fit_kwargs, 
			savepath; filename = "tsvae_test", verb = verb)
		@test isfile(joinpath(savepath,"tsvae_test.bson"))
	end
	rm(savepath, recursive = true)
end