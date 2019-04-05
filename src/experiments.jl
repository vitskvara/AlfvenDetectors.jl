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
	get_signal(filename, readfun, coil; warns=true, type="valid")

Returns flattop portion of signal extracted by readfun and coil.
"""
function get_signal(filename, readfun, coil; warns=true, type="valid")
	signal = readfun(filename,coil; warns=warns)
	ip = readip(filename; warns=warns)
	return_signal(signal,ip,type)
end

"""
	get_signal(filename, readfun; warns=true, type="valid")

Returns flattop portion of signal extracted by readfun.
"""
function get_signal(filename, readfun; warns=true, type="valid")
	signal = readfun(filename; warns=warns)
	ip = readip(filename; warns=warns)
	return_signal(signal,ip,type)
end

"""
	get_signals(filename, readfun, coils; warns=true, type="valid")

Colelct signals from all coils.
"""
function get_signals(filename, readfun, coils; warns=true, type="valid")
	signals = []
	for coil in coils
		x = get_signal(filename, readfun, coil; warns=warns, type=type)
		if !any(isnan,x)
			push!(signals, x)
		end
	end
	return hcat(signals...)
end

"""
	collect_signals(shots,readfun,coils; warns=true, type="valid")

Collect signals from multiple files.
"""
collect_signals(shots,readfun,coils; warns=true, type="valid") = 
	filter(x->x!=[], map(x->get_signals(x,readfun,coils; warns=warns, type=type), shots))

"""
	collect_signals(shots,readfun; warns=true)

Collect signals from multiple files.
"""
collect_signals(shots,readfun; warns=true, type="valid") = 
	filter(x->!any(isnan,x), map(x->get_signal(x,readfun; warns=warns, type=type), shots))

"""
	fitsave_unsupervised(modelname, batchsize, outer_nepochs, inner_nepochs,
	 model_args, model_kwargs, fit_kwargs, savepath)

Create, fit and save a model.
"""
function fitsave_unsupervised(data, modelname, batchsize, outer_nepochs, inner_nepochs,
	 model_args, model_kwargs, fit_kwargs, savepath;
	 optname = "ADAM", eta = 0.001, usegpu = false, filename = "", verb = true)
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

	# now create the filename
	if filename == ""
		filename = "$(modelname)"
		for pair in model_args
			filename*="_$(pair[1])-$(pair[2])"
		end
		for (key, val) in model_kwargs
			filename *= "_$(key)-$(val)"
		end
		filename *= "_batchsize-$batchsize"
		filename *= "_nepochs-$(outer_nepochs*inner_nepochs)"
		filename *= "_opt-$optname"
		filename *= "_eta-$eta"
		for (key, val) in fit_kwargs
			filename *= "_$(key)-$(val)"
		end
		filename *= "_$(now())"
	end
	filename = joinpath(savepath, "$(filename).bson")
	
	# fit the model
	t = 0.0

	tall = @timed for epoch in 1:outer_nepochs
		verb ? println("outer epoch counter: $epoch/$outer_nepochs") : nothing
		restime = @timed AlfvenDetectors.fit!(model, data, batchsize, inner_nepochs; 
			usegpu = usegpu, verb = verb, history = history, cbit=1, opt=opt, η = eta,
			fit_kwargs...)
		t += restime[2]
		opt = restime[1]

		# save the model structure, history and time of training after each epoch
		# to load this, you need to load Flux, AlfvenDetectors and ValueHistories
		cpumodel = model |> cpu
		bson(filename, model = cpumodel, history = history, time = t)
		GC.gc()
	end
	cpumodel = model |> cpu
	bson(filename, model = cpumodel, history = history, time = t, timeall=tall[2])
	
	println("model and timing saved to $filename")

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
	collect_conv_signals(shots,readfun,heigth,width,coils [,warns, type])

Returns a 4D array consisting of blocks of given width, extracted by readfun.
"""
function collect_conv_signals(shots,readfun,heigth::Int,width::Int,coils::AbstractVector; warns=true, type="valid")
	data = collect_signals(shots, readfun, coils; warns=warns, type=type)
	cat_split_reshape(data, heigth, width)
end
collect_conv_signals(shots,readfun,s::Int,coils::AbstractVector; warns=true, type="valid") = 
	collect_conv_signals(shots,readfun,s,s,coils; warns=warns, type=type)

"""
	collect_conv_signals(shots,readfun,heigth,width [,warns, type])

Returns a 4D array consisting of blocks of given width, extracted by readfun.
"""
function collect_conv_signals(shots,readfun,heigth::Int,width::Int; warns=true, type="valid")
	data = collect_signals(shots, readfun; warns=warns, type=type)
	cat_split_reshape(data, heigth, width)
end
collect_conv_signals(shots,readfun,s::Int; warns=true, type="valid") = 
	collect_conv_signals(shots,readfun,s,s; warns=warns, type=type)

