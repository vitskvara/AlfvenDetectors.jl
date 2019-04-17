"""
	construct_model(modelname, modelargs...; modelkwargs...)

Returns a model object given by modelname and arguments.
"""
construct_model(modelname, modelargs...; modelkwargs...) =
	eval(Meta.parse("$modelname"))(modelargs...; modelkwargs...)

################################
### UNSUPERVISED EXPERIMENTs ###
################################

"""
	return_signal(signal,ip,type)

Decide whether to return valid or flattop part of signal and
also detect NaNs.
"""
function return_signal(signal,ip,type)
	if any(isnan,ip) || any(isnan,signal)
		return NaN
	elseif type == "flattop"
		return get_ft_section(signal,ip;minlength = 100)
	elseif type == "valid"
		return get_valid_section(signal,ip;ϵ=0.02)
	else
		return NaN
	end
end

"""
	get_signal(filename, readfun, coil; [warns, memorysafe, type])

Returns flattop portion of signal extracted by readfun and coil.
"""
function get_signal(filename, readfun, coil; type="valid", readkwargs...)
	signal = readfun(filename,coil; readkwargs...)
	ip = readip(filename; readkwargs...)
	return_signal(signal,ip,type)
end

"""
	get_signal(filename, readfun; [warns, memorysafe, type])

Returns flattop portion of signal extracted by readfun.
"""
function get_signal(filename, readfun; type="valid", readkwargs...)
	signal = readfun(filename; readkwargs...)
	ip = readip(filename; readkwargs...)
	return_signal(signal,ip,type)
end

"""
	get_signals(filename, readfun, coils; [warns, memorysafe, type])

Colelct signals from all coils.
"""
function get_signals(filename, readfun, coils; kwargs...)
	signals = []
	for coil in coils
		x = get_signal(filename, readfun, coil; kwargs...)
		if !any(isnan,x)
			push!(signals, x)
		end
	end
	return hcat(signals...)
end

"""
	collect_signals(shots,readfun,coils; [warns, memorysafe, type])

Collect signals from multiple files.
"""
collect_signals(shots,readfun,coils; kwargs...) = 
	filter(x->x!=[], map(x->get_signals(x,readfun,coils; kwargs...), shots))

"""
	collect_signals(shots,readfun; [warns, memorysafe, type])

Collect signals from multiple files.
"""
collect_signals(shots,readfun; kwargs...) = 
	filter(x->!any(isnan,x), map(x->get_signal(x,readfun; kwargs...), shots))

"""
	create_filename(modelname, model_args, model_kwargs, fit_kwargs, kwargs...)

Create model filename.
"""
function create_filename(modelname, model_args, model_kwargs, fit_kwargs, kwargs...)
	filename = "$(modelname)"
	for pair in model_args
		filename*="_$(pair[1])-$(pair[2])"
	end
	for (key, val) in model_kwargs
		filename *= "_$(key)-$(val)"
	end
	for (key, val) in fit_kwargs
		filename *= "_$(key)-$(val)"
	end
	for (key, val) in kwargs
		filename *= "_$(key)-$(val)"
	end
	filename *= "_$(now()).bson"

	return filename
end

"""
	fitsave_unsupervised(modelname, batchsize, outer_nepochs, inner_nepochs,
	 model_args, model_kwargs, fit_kwargs, savepath[,optname, eta, usegpu,
	 filename,verb,savepoint)

Create, fit and save a model.
"""
function fitsave_unsupervised(data, modelname, batchsize, outer_nepochs, inner_nepochs,
	 model_args, model_kwargs, fit_kwargs, savepath;
	 optname = "ADAM", eta = 0.001, usegpu = false, filename = "", verb = true,
	 savepoint=1)
	# create the model
	model = AlfvenDetectors.construct_model(modelname, [x[2] for x in model_args]...; model_kwargs...)
	usegpu ? model = model |> gpu : nothing
	if occursin("TSVAE", "$modelname")
		history = (MVHistory(), MVHistory())
		opt = Array{Any,1}([eval(Meta.parse(optname))(eta), eval(Meta.parse(optname))(eta)])
	else
		history = MVHistory()
		opt = eval(Meta.parse(optname))(eta)
	end

	# append time and bson suffix to filename
	if filename == ""
		filename *= "$(now()).bson"
	end
	
	# fit the model
	t = 0.0

	tall = @timed for epoch in 1:outer_nepochs
		verb ? println("outer epoch counter: $epoch/$outer_nepochs") : nothing
		restime = @timed AlfvenDetectors.fit!(model, data, batchsize, inner_nepochs; 
			usegpu = usegpu, verb = verb, history = history, cbit=1, opt=opt, η = eta,
			fit_kwargs...)
		t += restime[2]
		opt = restime[1]

		# save the model structure, history and time of training after each savepoint epoch
		# to load this, you need to load Flux, AlfvenDetectors and ValueHistories
		if epoch%savepoint==0
			# replace the number of epochs in the filename string with the current number of epochs
			if occursin("nepochs",filename)
				fs = split(filename, "_")
				fs[collect(1:length(fs))[map(x->occursin("nepochs",x), fs)][1]] = "nepochs-$epoch"
				filename = join(fs, "_")
			end
			cpumodel = model |> cpu
			bson(joinpath(savepath, filename), model = cpumodel, history = history, time = t)
		end
		GC.gc()
	end
	# save the final version
	cpumodel = model |> cpu
	bson(joinpath(savepath, filename), model = cpumodel, history = history, time = t, timeall=tall[2])
	
	println("model and timing saved to $(joinpath(savepath, filename))")

	return cpumodel, history, t
end

###############################
### stuff for conv networks ###
###############################

"""
	cat_split_reshape(data, heigth, width)

Transforms a list of arrays into a 4D array chunks for convolutional networks.
"""
function split_reshape(data::AbstractArray,heigth::Int,width::Int)
	data = data[1:(end-size(data,1)%heigth),1:(end-size(data,2)%width)]
	permutedims(reshape(permutedims(reshape(data,size(data,1),width,1,:), [2,1,3,4]), 
			width, heigth, 1, :), 
			[2,1,3,4])
end
split_reshape(data,s::Int) = split_reshape(data,s,s)
function cat_split_reshape(data::AbstractVector,heigth::Int,width::Int)
	data = hcat(map(x->x[1:(end-size(x,1)%heigth),1:(end-size(x,2)%width)],data)...)
	permutedims(reshape(permutedims(reshape(data,size(data,1),width,1,:), [2,1,3,4]), 
			width, heigth, 1, :), 
			[2,1,3,4])
end
cat_split_reshape(data,s::Int) = cat_split_reshape(data,s,s)


"""
	collect_conv_signals(shots,readfun,heigth,width,coils [,warns, type, memorysafe])

Returns a 4D array consisting of blocks of given width, extracted by readfun.
"""
function collect_conv_signals(shots,readfun,heigth::Int,width::Int,coils::AbstractVector; kwargs...)
	data = collect_signals(shots, readfun, coils; kwargs...)
	cat_split_reshape(data, heigth, width)
end
collect_conv_signals(shots,readfun,s::Int,coils::AbstractVector; kwargs...) = 
	collect_conv_signals(shots,readfun,s,s,coils; kwargs...)

"""
	collect_conv_signals(shots,readfun,heigth,width [,warns, type, memorysafe])

Returns a 4D array consisting of blocks of given width, extracted by readfun.
"""
function collect_conv_signals(shots,readfun,heigth::Int,width::Int; kwargs...)
	data = collect_signals(shots, readfun; kwargs...)
	cat_split_reshape(data, heigth, width)
end
collect_conv_signals(shots,readfun,s::Int; kwargs...) = 
	collect_conv_signals(shots,readfun,s,s; kwargs...)

### functions for working with labeled data ###
"""
    labeled_data()

Get the information on the few hand-labeled shots.
"""
function labeled_data()
    f = joinpath(dirname(pathof(AlfvenDetectors)), "../experiments/conv/data/labeled_shots.csv")
    labels_shots = readdlm(f, ',', Int32)
    labels = labels_shots[:,2]
    labeled_shots = labels_shots[:,1] 
    return labeled_shots, labels
end

"""
    labeled_patches()

Get the information on the few hand-labeled shots.
"""
function labeled_patches()
    f = joinpath(dirname(pathof(AlfvenDetectors)), "../experiments/conv/data/labeled_patches.csv")
    data,header = readdlm(f, ',', Float32, header=true)
    shots = Int.(data[:,1])
    labels = Int.(data[:,4])
	tstarts = data[:,2]
    fstarts = data[:,3]
    return shots, labels, tstarts, fstarts
end

"""
	get_patch(datapath, shot, tstart, fstart, patchsize, readfun, coil=nothing [, warns, memorysafe, type])

Get a patch of given size starting at fstart and tstart coordinates.
"""
function get_patch(datapath, shot, tstart, fstart, patchsize, readfun, coil=nothing; kwargs...)
	file = joinpath(datapath, filter(x->occursin(string(shot), x), readdir(datapath))[1])
	if coil == nothing
		data = get_signal(file, readfun; kwargs...)
	else
		data = get_signal(file, readfun, coil; kwargs...)
	end
	if readfun == AlfvenDetectors.readnormlogupsd
		t = get_signal(file, AlfvenDetectors.readtupsd; kwargs...)
		f = AlfvenDetectors.readfupsd(file)
	else
		t = get_signal(file, AlfvenDetectors.readtcoh; kwargs...)
		f = AlfvenDetectors.readfcoh(file)
	end
	tinds = tstart .< t
	finds = fstart .< f
	tpatch = t[tinds][1:patchsize]
	fpatch = f[finds][1:patchsize]
	patch = data[finds,tinds][1:patchsize,1:patchsize]
	return patch, tpatch, fpatch
end

"""
	get_patch_from_csv(datapath, shot, label, tstart, fstart)

Get a patch of given size starting at fstart and tstart coordinates saved in a csv file.
"""
function get_patch_from_csv(datapath, shot, label, tstart, fstart)
	file = joinpath(datapath, "$shot-$label-$tstart-$fstart.csv")
	patch = readdlm(file, ',', Float32)
	return patch
end

