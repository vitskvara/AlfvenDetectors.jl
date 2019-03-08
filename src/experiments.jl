"""
	construct_model(modelname, modelargs...; modelkwargs...)

Returns a model object given by modelname and arguments.
"""
construct_model(modelname, modelargs...; modelkwargs...) =
	eval(Meta.parse("$modelname"))(modelargs...; modelkwargs...)

"""
	get_ft_section(signal, ip; minlength=0)

Get the flattop part of the signal. 
"""
function get_ft_section(signal::AbstractArray, ip::AbstractVector; minlength=0)
	ipftstart, ipftstop = flattopbe(ip,0.6,8e-4;wl=20)
	ls = size(signal,2)
	lip = length(ip)
	ftstart = floor(Int, ipftstart/lip*ls)
	ftstop = floor(Int, ipftstop/lip*ls)
	if ftstop - ftstart > minlength
		return signal[:,ftstart:ftstop]
	else
		return signal[:,2:1] # an empty array of the correct vertical dimension
	end
end
function get_ft_section(signal::AbstractVector, ip::AbstractVector; minlength=0)
	ipftstart, ipftstop = flattopbe(ip,0.6,8e-4;wl=20)
	ls = length(signal)
	lip = length(ip)
	ftstart = floor(Int, ipftstart/lip*ls)
	ftstop = floor(Int, ipftstop/lip*ls)
	if ftstop - ftstart > minlength
		return signal[ftstart:ftstop]
	else
		return signal[2:1] # an empty array of the correct vertical dimension
	end
end

##############################################
### UNSUPERVISED MSC AMPLITUDES EXPERIMENT ###
##############################################

"""
	get_ft_mscamp(filename, coil)

Returns the msc amplitude 
"""
function get_ft_mscamp(filename, coil)
	msc = readmscamp(filename,coil)
	ip = readip(filename)
	return get_ft_section(msc,ip;minlength = 100)
end

"""
	get_ft_mscamps(filename, coils)

Colelct all the data from msc amplitudes
"""
function get_ft_mscamps(filename, coils)
	mscs = []
	for coil in coils
		try
			push!(mscs, get_ft_mscamp(filename, coil))
		catch e
			rethrow(e)
		end
	end
	return hcat(mscs...)
end

"""
	collect_mscamps(shots,coils) = hcat(map(x->get_ft_mscamps(x,coils), shots)...)	

Collect all the data.
"""
collect_mscamps(shots,coils) = hcat(map(x->get_ft_mscamps(x,coils), shots)...)

"""
	fitsave(modelname, batchsize, outer_nepochs, inner_nepochs,
	 model_args, model_kwargs, fit_kwargs, savepath)

Create, fit and save a model.
"""
function fitsave_mscamps(data, modelname, batchsize, outer_nepochs, inner_nepochs,
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
	
	return cpumodel, history, t[2]
end