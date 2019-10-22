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

function prepend_zeros(x::Real, n::Int)
   s = "$x"
   l = length(s)
   if l < n
       s = reduce(*,fill("0",n-l))*s
   end
   return s
end

"""
	fitsave_unsupervised(data, model, batchsize, outer_nepochs, inner_nepochs,
	 model_args, model_kwargs, fit_kwargs, savepath[,optname, eta, usegpu,
	 filename,verb,savepoint,modelname])

Create, fit and save a model.
"""
function fitsave_unsupervised(data, model, batchsize, outer_nepochs, inner_nepochs,
	 model_args, model_kwargs, fit_kwargs, savepath;
	 optname = "ADAM", eta = 0.001, usegpu = false, filename = "", verb = true,
	 savepoint=1, experiment_args=nothing,modelname=nothing)
	usegpu ? model = model |> gpu : nothing
	if occursin("TSVAE", "$model")
		history = (MVHistory(), MVHistory())
		opt = Array{Any,1}([eval(Meta.parse(optname))(eta), eval(Meta.parse(optname))(eta)])
	elseif occursin("WAAE", "$model")
		history = MVHistory()
		opt = eval(Meta.parse(optname))(eta)
	elseif occursin("AAE", "$model")
		history = MVHistory()
		opt = Array{Any,1}([eval(Meta.parse(optname))(eta) for i in 1:3])
	else
		history = MVHistory()
		opt = eval(Meta.parse(optname))(eta)
	end

	# append time and bson suffix to filename
	tstart = now()
	if filename == ""
		filename *= "$tstart.bson"
	end
	
	# fit the model
	t = 0.0

	tall = @timed for epoch in 1:outer_nepochs
		verb ? println("outer epoch counter: $epoch/$outer_nepochs") : nothing
		restime = @timed GenModels.fit!(model, data, batchsize, inner_nepochs; 
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
				epoch_string = prepend_zeros(epoch, length("$outer_nepochs"))
				fs[collect(1:length(fs))[map(x->occursin("nepochs",x), fs)][1]] = "nepochs-$epoch_string"
				filename = join(fs, "_")
			end
			cpumodel = model |> cpu
			GenModels.save_model(joinpath(savepath, filename), cpumodel, 
				modelname=modelname, history = history, time = t, tstart = string(tstart), 
				model_args=model_args, model_kwargs=model_kwargs, 
				experiment_args=experiment_args)
			println("model and timing saved to $(joinpath(savepath, filename))")
		end
		GC.gc()
	end
	# save the final version
	cpumodel = model |> cpu
	GenModels.save_model(joinpath(savepath, filename), cpumodel, 
				modelname=modelname, history = history, time = t, tstart = string(tstart), 
				model_args=model_args, model_kwargs=model_kwargs, 
				experiment_args=experiment_args, timeall=tall[2])
	
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
    labeled_patches([; only_negative, only_positive])

Get the information on the few hand-labeled shots.
"""
function labeled_patches(;only_positive = false, only_negative = false)
    f = joinpath(dirname(pathof(AlfvenDetectors)), "../experiments/conv/data/labeled_patches.csv")
    data,header = readdlm(f, ',', Float32, header=true)
    shots = Int.(data[:,1])
    labels = Int.(data[:,4])
	tstarts = data[:,2]
    fstarts = data[:,3]
    if only_positive
    	return shots[labels.==1], labels[labels.==1], tstarts[labels.==1], fstarts[labels.==1]
    elseif only_negative
    	return shots[labels.==0], labels[labels.==0], tstarts[labels.==0], fstarts[labels.==0]
    else
	    return shots, labels, tstarts, fstarts
	end
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
	if readfun in [AlfvenDetectors.readnormlogupsd, AlfvenDetectors.readlogupsd, AlfvenDetectors.readupsd]
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

"""
	add_noise(patch, δ)

Adds a gaussion noise of selected level δ to the original patch.
"""
add_noise(patch::AbstractArray, δ::Real) = patch + randn(Float, size(patch))*Float(δ).*patch

"""
	split_patches(α, shotnos, labels, tstarts, fstarts[, seed])

Splits the supplied information in a training/testing ratio given by α.
"""
function split_patches(α::Real, shotnos, labels, tstarts, fstarts; seed = nothing)
	Npatches = length(shotnos)
	# set the seed and shuffle the data
	(seed != nothing) ? Random.seed!(seed) : nothing
	used_inds = sample(1:Npatches, Npatches, replace=false)
	shotnos, labels, tstarts, fstarts =
		shotnos[used_inds], labels[used_inds], tstarts[used_inds], fstarts[used_inds]
	# now return the data using given α
	Nused = floor(Int, Npatches*α)
	# restart the seed
	(seed == nothing) ? nothing : Random.seed!()
	if 0 < Nused < Npatches 
		return map(x->x[1:Nused], (shotnos, labels, tstarts, fstarts)), used_inds[1:Nused],
			map(x->x[Nused+1:end], (shotnos, labels, tstarts, fstarts)), used_inds[Nused+1:end]
	else
		@warn "split_patches(...) not returning anything since one of the sets is empty, Nused=$Nused, Npatches=$Npatches"
		return fill(nothing, 4), nothing, fill(nothing, 4), nothing
	end
end

"""
	split_unique_shotnos(shotnos,α[;seed])

Splits unique shot numbers using α ratio.
"""
function split_unique_shotnos(shotnos,α;seed=nothing)
	# set the seed and shuffle the data
	(seed != nothing) ? Random.seed!(seed) : nothing

	# get unique shotnos
	ushotnos = unique(shotnos)
	Nunique = length(ushotnos)
	# shuffle them
	ushotnos = ushotnos[sample(1:Nunique, Nunique, replace=false)]
	Ntrushots = floor(Int,Nunique*α )
	trushots = ushotnos[1:Ntrushots]
	tstushots = ushotnos[Ntrushots+1:end]

	(seed == nothing) ? nothing : Random.seed!()
	return trushots, tstushots
end

"""
	split_unique_patches(α, shotnos, labels, tstarts, fstarts[; seed])

Splits the supplied information on the level of unique shots in a trainig/testing ratio given by α.
"""
function split_unique_patches(α::Real, shotnos, labels, tstarts, fstarts; seed = nothing)
	# set the seed and shuffle the data
	(seed != nothing) ? Random.seed!(seed) : nothing

	# shuffle everything
	Npatches = length(shotnos)
	used_inds = sample(1:Npatches, Npatches, replace=false)
	shotnos, labels, tstarts, fstarts =
		shotnos[used_inds], labels[used_inds], tstarts[used_inds], fstarts[used_inds]

	# in this part, extract the unique shot numbers, then shuffle and split them with α
	trshots1, tstshots1 = split_unique_shotnos(shotnos[labels.==1], α, seed=seed)
	trshots0, tstshots0 = split_unique_shotnos(shotnos[labels.==0], α, seed=seed)
	
	# get indices of training positive and negative samples
	traininds = map(x->any(occursin.(string.(trshots1), string(x))),shotnos) .| 
		map(x->any(occursin.(string.(trshots0), string(x))),shotnos)
	testinds = map(x->any(occursin.(string.(tstshots1), string(x))),shotnos) .|
		map(x->any(occursin.(string.(tstshots0), string(x))),shotnos)
	
	# restart the seed
	(seed == nothing) ? nothing : Random.seed!()
	if 0 < sum(traininds) < Npatches
		return map(x->x[traininds], (shotnos, labels, tstarts, fstarts)), used_inds[traininds],
			map(x->x[testinds], (shotnos, labels, tstarts, fstarts)), used_inds[testinds]
	else
		@warn "split_patches_unique(...) not returning anything since one of the sets is empty, Nused=$(sum(traininds)), Npatches=$Npatches"
		return fill(nothing, 4), nothing, fill(nothing, 4), nothing
	end
end

"""
	split_shots(nshots, available_shots[, testing_patches_shotnos][; seed, use_alfven_shots])

Select a list of training shots.
"""
function split_shots(nshots::Int, available_shots::AbstractVector, test_train_patches_shotnos=nothing;
	seed = nothing, use_alfven_shots=true)
	# get the list of labeled shots
	labeled_shots, labels = AlfvenDetectors.labeled_data()
	# decide whether to use shots with alfven modes or not
	label = Int(use_alfven_shots)
	labeled_shots = labeled_shots[labels.==label]
	labels = labels[labels.==label]
	# also, filter the two lists so there is no intersection
	labeled_shots = filter(x->any(map(y->occursin(string(y), x), labeled_shots)), available_shots)
	available_shots = filter(x->!any(map(y->occursin(string(y),x),labeled_shots)), available_shots)
	# shuffle the available shots	
	Navailable = length(available_shots)
	(seed != nothing) ? Random.seed!(seed) : nothing
	available_shots = available_shots[sample(1:Navailable, Navailable, replace=false)]
	# restart the seed
	(seed == nothing) ? nothing : Random.seed!()
	# now select the training shots from the labeled set
	Nlabeled = length(labeled_shots)
	# initialize the pseudorandom generator so that the training set is fixed
	# and select half of the set at most, then add the remaining 
	if test_train_patches_shotnos == nothing
		Nlabeledout = floor(Int, Nlabeled/2)
		(seed != nothing) ? Random.seed!(seed) : nothing
		train_inds = sample(1:Nlabeled, Nlabeledout, replace=false)
		train_shots = labeled_shots[train_inds]
		# restart the seed
		(seed == nothing) ? nothing : Random.seed!()
	# if test/train patches are used, then do the splitting of the labeled shots according to 
	# this previous split - exclude the testing patch shots from the training set
	else
		trainp_shots = test_train_patches_shotnos[1]
		testp_shots = test_train_patches_shotnos[2]
		train_inds = filter(i->!any(map(x->occursin(string(x), string(labeled_shots[i])),testp_shots)),
			collect(1:Nlabeled))
		train_shots = labeled_shots[train_inds]
		Nlabeledout = length(train_shots)
	end
	# now check if the labeled shots are actually avaiable and then select the requested amount
	# also return the second part of the dataset as testing shots
	if nshots <= Nlabeledout
		outshots = train_shots[1:nshots], 
			vcat(train_shots[nshots+1:end], # unused train shots
				labeled_shots[filter(i->!(i in train_inds),collect(1:Nlabeled))], # rest of the labeled shots
				available_shots) # rest of the available shots
	else 
		# if more than the half of labeled shots are requested, select the rest from the available shots
		# but discard those that are labeled
		if test_train_patches_shotnos == nothing
			train_available_shots = available_shots[1:(nshots-Nlabeledout)]
			test_available_shots = available_shots[(nshots-Nlabeledout+1):end]
		else
			train_available_shots = 
				filter(y->!any(map(x->occursin(string(x), y),testp_shots)), available_shots)[1:(nshots-Nlabeledout)]
			test_available_shots = filter(x->!(x in train_available_shots),available_shots)
		end
		outshots = vcat(
			train_shots,
			train_available_shots
			),
			vcat(labeled_shots[filter(i-> !(i in train_inds), collect(1:length(labeled_shots)))],
				test_available_shots)
				
	end
	return outshots
end

"""
	collect_training_patches(datapath, shotnos, tstarts, fstarts, N, readfun, patchsize[; 
	δ, seed, get_data_kwargs...])

This function will return the 4D tensor of N randomly selected patches with added noise of magnitude δ.
"""
function collect_training_patches(datapath, shotnos, tstarts, fstarts, N, readfun, patchsize; 
	δ = 0.02, seed = nothing, kwargs...)
	# collect all the patch data
	patchdata = map(x->get_patch(datapath, x[1], x[2], x[3], patchsize, readfun; 
		kwargs...)[1], zip(shotnos, tstarts, fstarts))
	# sample patches to be added
	Npatches = length(patchdata)
	(seed != nothing) ? Random.seed!(seed) : nothing
	sample_inds = sample(1:Npatches, N)
	patches = patchdata[sample_inds]
	# finally, add some noise to them
	patches = add_noise.(patches,δ)
	# and return them as a 4D tensor
	patches = cat(patches...,dims=4)
	(seed == nothing) ? nothing : Random.seed!()
	return patches, shotnos[sample_inds], tstarts[sample_inds], fstarts[sample_inds]
end

"""
	collect_training_data(datapath, collect_fun, nshots, readfun,
		positive_patch_ratio, patchsize[; seed, use_alfven_shots])

Returns training data for the convolutional run script.
"""
function collect_training_data(datapath, collect_fun, nshots, readfun,
	positive_patch_ratio, patchsize; seed=nothing, use_alfven_shots=true)
	# load labeled patches information and split them to train/test
	println("\nLoading information on the labeled patches...")
	shotnos, patch_labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()
	train_info, train_inds, test_info, test_inds =  
		AlfvenDetectors.split_unique_patches(0.5, shotnos, patch_labels, tstarts, fstarts; 
			seed = seed)
	println("Done, found $(length(train_inds)) training patches and $(length(test_inds)) testing patches.\n")

	# get the list of training shots
	println("\nLoading basic training data...")
	available_shots = readdir(datapath)
	training_shots, testing_shots = AlfvenDetectors.split_shots(nshots, available_shots, 
		(train_info[1], test_info[1]); seed=seed, use_alfven_shots=use_alfven_shots)
	println("Using $(training_shots)\n")
	shots = joinpath.(datapath, training_shots)

	# get the main portion of the data
	data = collect_fun(shots)
	xdim = size(data)

	# now add some additional positively labeled patches into the training dataset
	if positive_patch_ratio > 0
		println("\nLoading labeled patch data...")
		# get only the positive data
		shotnos = train_info[1][train_info[2].==1]
		tstarts = train_info[3][train_info[2].==1]
		fstarts = train_info[4][train_info[2].==1]
		patch_labels = train_info[2][train_info[2].==1]
		# also, get only the available shot data
		available_inds = 
			filter(i->any(occursin.(string(shotnos[i]), available_shots)),1:length(shotnos))
		shotnos = shotnos[available_inds]
		patch_labels = patch_labels[available_inds]
		tstarts = tstarts[available_inds]
		fstarts = fstarts[available_inds]
		# get the number of patches to be added
		Nadded = floor(Int, xdim[4]*positive_patch_ratio/(1-positive_patch_ratio))
		# now that we know how many samples to add, we can sample the appropriate number of them with some added noise
		added_patches, added_shotnos, added_tstarts, added_fstarts = 
		AlfvenDetectors.collect_training_patches(datapath, shotnos, tstarts, fstarts,
			Nadded, readfun, patchsize; δ = 0.02, seed=seed, memorysafe = true)
		println("Done, loaded additional $(size(added_patches,4)) positively labeled patches.\n")
		data = cat(data, added_patches, dims=4)
	end

	return data, training_shots
end

"""
	filter_available_shots(datapath, shotnos)

Filter only the shots available on the current machine.
"""
function filter_available_shots(datapath, shotnos)
	available_shots = readdir(datapath)
    available_inds = filter(i->any(occursin.(string(shotnos[i]), available_shots)),1:length(shotnos))
end

"""
	function test_train_oneclass(datapath; α=0.8, seed = nothing)

Split the labeled positive patches info into traning and testing parts.
"""
function test_train_oneclass(datapath; α=0.8, seed = nothing)
	println("\nLoading information on the labeled patches...")
	shotnos, patch_labels, tstarts, fstarts = labeled_patches(only_positive=true)

	# iterate only over shot data that are actually available
    available_inds = filter_available_shots(datapath, shotnos)
    shotnos, patch_labels, tstarts, fstarts = map(x->x[available_inds], 
    	(shotnos, patch_labels, tstarts, fstarts))
	
	# do the test/train split
	train_info, train_inds, test_info, test_inds =  
		split_patches(α, shotnos, patch_labels, tstarts, fstarts; 
			seed = seed)
	Ntrain = length(train_inds)
	Ntest = length(test_inds)
	println("Done, found $(Ntrain) training patches and $(Ntest) testing patches.\n")
	return train_info, train_inds, test_info, test_inds
end

"""
	collect_training_data_oneclass(datapath, Npatches, readfun, patchsize; 
		α = 0.8, seed=nothing)

Returns training data for the oneclass run script.
"""
function collect_training_data_oneclass(datapath, Npatches, readfun, patchsize; 
		α = 0.8, seed=nothing)
	# load labeled patches information and split them to train/test
	train_info, train_inds, test_info, test_inds = test_train_oneclass(datapath, α = α, seed = seed)
	Ntrain = length(train_inds)

	# shift the starts
	(seed == nothing) ? nothing : Random.seed!(seed)
	sample_inds = sample(1:Ntrain, Npatches)
	(seed == nothing) ? nothing : Random.seed!()
	starts = map(i -> shift_patch(train_info[3][i], train_info[4][i]; patchsize=patchsize), sample_inds) 
	shotnos = train_info[1][sample_inds]
	tstarts = [x[1] for x in starts]
	fstarts = [x[2] for x in starts]

	# now get the final data and add some noise to them
	patches_out, shotnos_out, tstarts_out, fstarts_out = 
	  AlfvenDetectors.collect_training_patches(datapath, shotnos, tstarts, fstarts,
		Npatches, readfun, patchsize; δ = 0.02, seed=seed, memorysafe = true)

	return patches_out, shotnos_out, ones(Npatches), tstarts_out, fstarts_out
end

"""
	collect_testing_data_oneclass(datapath, readfun, patchsize; α = 0.8, seed=nothing)

Returns teting data for the oneclass run script.
"""
function collect_testing_data_oneclass(datapath, readfun, patchsize; 
		α = 0.8, seed=nothing)
	# load labeled patches information and split them to train/test
	train_info, train_inds, test_info, test_inds = test_train_oneclass(datapath, α = α, seed = seed)
	Ntest = length(test_inds)

	# now load info on the negative patches
	shotnos_0, labels_0, tstarts_0, fstarts_0 = labeled_patches(only_negative=true)
	shotnos_out = vcat(test_info[1], shotnos_0)
	labels_out = vcat(test_info[2], labels_0)
	tstarts_out = vcat(test_info[3], tstarts_0)
	fstarts_out = vcat(test_info[4], fstarts_0)

	# filter available shots
    available_inds = filter_available_shots(datapath, shotnos_out)
    shotnos_out, labels_out, tstarts_out, fstarts_out = map(x->x[available_inds], 
    	(shotnos_out, labels_out, tstarts_out, fstarts_out))

	# get the final data
	patches_out = map(x->get_patch(datapath, x[1], x[2], x[3], patchsize, readfun; 
		memorysafe=true)[1], zip(shotnos_out, tstarts_out, fstarts_out))
	patches_out = cat(patches_out..., dims=4)
	
	return patches_out, shotnos_out, labels_out, tstarts_out, fstarts_out
end

"""
	shift_patch(tstart::Real, fstart::Real; patchsize=128, seed=nothing)

Randomly shift the position of a patch - modifies the f and t coordinates by up to a quarter 
of the ranges given by patchsize.
"""
function shift_patch(tstart::Real, fstart::Real; patchsize=128, seed=nothing)
	# default f nad t ranges for patchsize = 128
	# trange = 0.00650239f0
	# frange = 620117.25f0
	(seed == nothing) ? nothing : Random.seed!(seed)
	tstart = tstart + (rand(typeof(tstart)) - 0.5) * 0.00650239f0 * patchsize/128 * 1/4
	fstart = fstart + (rand(typeof(fstart)) - 0.5) * 620117.25f0 * patchsize/128 * 1/4
	(seed == nothing) ? nothing : Random.seed!() # otherwise this will run out of random seeds soon
	return tstart, fstart
end

function oneclass_training_data_jld(fname, npatches)
	isfile(fname) ? jlddata = load(fname) : error("The requested file $fname does not exist!")
	println("Loading $fname")
	navail = size(jlddata["patches"], 4)
	navail < npatches ? error("not enough patches available, requested $npatches, available $navail") : nothing
	return jlddata["patches"][:,:,:,1:npatches], 
		jlddata["shotnos"][1:npatches],
		jlddata["labels"][1:npatches], 
		jlddata["tstarts"][1:npatches], 
		jlddata["fstarts"][1:npatches]
end

function oneclass_negative_training_data(datapath, nshots, seed, readfun, patchsize)
	testing_shots = unique(AlfvenDetectors.labeled_patches()[1])
	available_shots = readdir(datapath)
	training_shots = filter!(x-> !any(map(y-> occursin(string(y), x), testing_shots)),available_shots)
	# scramble them
	navail = length(training_shots)
	if nshots > navail
		@warn "Requested $nshots shots for training, however only $navail available"
		nshots = navail
	end
	Random.seed!(seed)
	training_shots = sample(training_shots, nshots, replace = false)
	Random.seed!()
	patches = AlfvenDetectors.collect_conv_signals(joinpath.(datapath, training_shots), readfun, patchsize;
		memorysafe = true, type="valid")
	return  patches, nothing, nothing, nothing, nothing
end