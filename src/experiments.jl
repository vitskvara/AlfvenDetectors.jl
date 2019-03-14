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
	get_ft_signal(filename, readfun, coil; warns=true)

Returns flattop portion of signal extracted by readfun and coil.
"""
function get_ft_signal(filename, readfun, coil; warns=true)
	signal = readfun(filename,coil; warns=warns)
	ip = readip(filename; warns=warns)
	if any(isnan,ip) || any(isnan,signal)
		return NaN
	else
		return get_ft_section(signal,ip;minlength = 100)
	end
end

"""
	get_ft_signal(filename, readfun; warns=true)

Returns flattop portion of signal extracted by readfun.
"""
function get_ft_signal(filename, readfun; warns=true)
	signal = readfun(filename; warns=warns)
	ip = readip(filename; warns=warns)
	if any(isnan,ip) || any(isnan,signal)
		return NaN
	else
		return get_ft_section(signal,ip;minlength = 100)
	end
end

"""
	get_ft_signals(filename, readfun, coils; warns=true)

Colelct signals from all coils.
"""
function get_ft_signals(filename, readfun, coils; warns=true)
	signals = []
	for coil in coils
		x = get_ft_signal(filename, readfun, coil; warns=warns)
		if !any(isnan,x)
			push!(signals, x)
		end
	end
	return hcat(signals...)
end

"""
	collect_signals(shots,readfun,coils; warns=true)

Collect signals from multiple files.
"""
collect_signals(shots,readfun,coils; warns=true) = hcat(filter(x->x!=[], map(x->get_ft_signals(x,readfun,coils; warns=warns), shots))...)

"""
	collect_signals(shots,readfun; warns=true)

Collect signals from multiple files.
"""
collect_signals(shots,readfun; warns=true) = hcat(filter(x->!any(isnan,x), map(x->get_ft_signal(x,readfun; warns=warns), shots))...)

"""
	fitsave_unsupervised(modelname, batchsize, outer_nepochs, inner_nepochs,
	 model_args, model_kwargs, fit_kwargs, savepath)

Create, fit and save a model.
"""
function fitsave_unsupervised(data, modelname, batchsize, outer_nepochs, inner_nepochs,
	 model_args, model_kwargs, fit_kwargs, savepath; filename = "", verb = true)
	# create the model
	model = AlfvenDetectors.construct_model(modelname, [x[2] for x in model_args]...; model_kwargs...) |> gpu
	if "$modelname" == "TSVAE"
		history = (MVHistory(), MVHistory())
	else
		history = MVHistory()
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
		timestats = @timed AlfvenDetectors.fit!(model, data, batchsize, inner_nepochs; cbit = 1, verb = verb, history = history, fit_kwargs...)
		t += timestats[2]

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