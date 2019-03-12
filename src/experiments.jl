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
	get_ft_signal(filename, readfun, coil)

Returns flattop portion of signal extracted by readfun and coil.
"""
function get_ft_signal(filename, readfun, coil)
	signal = readfun(filename,coil)
	ip = readip(filename)
	if ip == nothing || signal == nothing
		return nothing
	else
		return get_ft_section(signal,ip;minlength = 100)
	end
end

"""
	get_ft_signal(filename, readfun)

Returns flattop portion of signal extracted by readfun.
"""
function get_ft_signal(filename, readfun)
	signal = readfun(filename)
	ip = readip(filename)
	if ip == nothing || signal == nothing
		return nothing
	else
		return get_ft_section(signal,ip;minlength = 100)
	end
end

"""
	get_ft_signals(filename, readfun, coils)

Colelct signals from all coils.
"""
function get_ft_signals(filename, readfun, coils)
	mscs = []
	for coil in coils
		x = get_ft_signal(filename, readfun, coil)
		if x != nothing
			push!(mscs, x)
		end
	end
	return hcat(mscs...)
end

"""
	collect_signals(shots,readfun,coils)

Collect signals from multiple files.
"""
collect_signals(shots,readfun,coils) = hcat(filter(x->x!=[], map(x->get_ft_signals(x,readfun,coils), shots))...)

"""
	collect_signals(shots,readfun)

Collect signals from multiple files.
"""
collect_signals(shots,readfun) = hcat(filter(x->x!=[], map(x->get_ft_signal(x,readfun), shots))...)

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
	
	# fit the model
	t = @timed for epoch in 1:outer_nepochs
		verb ? println("outer epoch counter: $epoch/$outer_nepochs") : nothing
		AlfvenDetectors.fit!(model, data, batchsize, inner_nepochs; cbit = 1, verb = verb, history = history, fit_kwargs...)
		GC.gc()
	end
	cpumodel = model |> cpu

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
	# save the model structure, history and time of training
	# to load this, you need to load Flux, AlfvenDetectors and ValueHistories
	bson(filename, model = cpumodel, history = history, time = t[2])
	println("model and timing saved to $filename")
	
	return cpumodel, history, t[2]
end