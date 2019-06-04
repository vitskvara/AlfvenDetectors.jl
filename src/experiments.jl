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
		restime = @timed GenerativeModels.fit!(model, data, batchsize, inner_nepochs; 
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
			GenerativeModels.save_model(joinpath(savepath, filename), cpumodel, 
				modelname=modelname, history = history, time = t, tstart = string(tstart), 
				model_args=model_args, model_kwargs=model_kwargs, 
				experiment_args=experiment_args)
			println("model and timing saved to $(joinpath(savepath, filename))")
		end
		GC.gc()
	end
	# save the final version
	cpumodel = model |> cpu
	GenerativeModels.save_model(joinpath(savepath, filename), cpumodel, 
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

"""
	add_noise(patch, δ)

Adds a gaussion noise of selected level δ to the original patch.
"""
add_noise(patch::AbstractArray, δ::Real) = patch + randn(Float, size(patch))*Float(δ).*patch

"""
	split_patches(α, shotnos, labels, tstarts, fstarts[, seed])

Splits the supplied information in a trainig/testing ratio given by α.
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
	Random.seed!()
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

	Random.seed!()
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
	Random.seed!()
	if 0 < sum(traininds) < Npatches
		return map(x->x[traininds], (shotnos, labels, tstarts, fstarts)), used_inds[traininds],
			map(x->x[testinds], (shotnos, labels, tstarts, fstarts)), used_inds[testinds]
	else
		@warn "split_patches_unique(...) not returning anything since one of the sets is empty, Nused=$(sum(traininds)), Npatches=$Npatches"
		return fill(nothing, 4), nothing, fill(nothing, 4), nothing
	end
end

"""
	split_shots(nshots, available_shots[; testing_patches_shotnos, seed, use_alfven_shots])

Select a list of training shots.
"""
function split_shots(nshots::Int, available_shots::AbstractVector; test_train_patches_shotnos=nothing,
	seed = nothing, use_alfven_shots=true)
	# get the list of labeled shots
	labeled_shots, labels = AlfvenDetectors.labeled_data()
	# decide whether to use shots with alfven modes or not
	label = Int(use_alfven_shots)
	labeled_shots = labeled_shots[labels.==label]
	labels = labels[labels.==label]
	# also, filter the two lists so there is no intersection
	labeled_shots = filter(x->any(map(y->occursin(string(y), x),labeled_shots)), available_shots)
	available_shots = filter(x->!any(map(y->occursin(string(y),x),labeled_shots)), available_shots)
	Nlabeled = length(labeled_shots)
	# original training subset ["10370", "10514", "10800", "10866", "10870", "10893"]
	# initialize the pseudorandom generator so that the training set is fixed
	# and select half of the set at most, then add the remaining 
	if test_train_patches_shotnos == nothing
		Nlabeledout = floor(Int, Nlabeled/2)
		(seed != nothing) ? Random.seed!(seed) : nothing
		train_inds = sample(1:Nlabeled, Nlabeledout, replace=false)
		train_shots = labeled_shots[train_inds]
		# restart the seed
		Random.seed!()
	# if test/train patches are used, then do the splitting of the labeled shots according to 
	# this previous split
	else
		trainp_shots = test_train_patches_shotnos[1]
		testp_shots = test_train_patches_shotnos[1]
		train_inds = filter(i->any(map(x->occursin(string(x), labeled_shots[i]),trainp_shots)),
			collect(1:Nlabeled))
		train_shots = labeled_shots[train_inds]
		Nlabeledout = length(train_shots)
	end
	Navailable = length(available_shots)
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
		(seed != nothing) ? Random.seed!(seed) : nothing
		outinds = sample(1:Navailable, nshots-Nlabeledout, replace=false)		
		# restart the seed
		Random.seed!()
		outshots = vcat(
			train_shots,
			available_shots[outinds]
			),
			vcat(labeled_shots[filter(i-> !(i in train_inds), collect(1:length(labeled_shots)))],
				available_shots[filter(i-> !(i in outinds), collect(1:Navailable))])
	end
	return outshots
end

"""
	collect_training_patches(datapath, shotnos, tstarts, fstarts, N, readfun, patchsize[; 
	δ, seed, kwargs...])

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
	Random.seed!()
	return patches, shotnos[sample_inds], tstarts[sample_inds], fstarts[sample_inds]
end

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
	training_shots, testing_shots = AlfvenDetectors.split_shots(nshots, available_shots; 
		test_train_patches_shotnos=(train_info[1], test_info[1]) ,seed=seed, use_alfven_shots=use_alfven_shots)
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