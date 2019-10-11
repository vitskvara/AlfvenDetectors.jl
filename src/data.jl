"""
	BaseAlfvenData

A mutable structure representing the basic Alfven eigenmode data.

Fields:

	t = time
	f = frequency
	fnoscale = frequency according to the scaling law ~ I_p/sqrt(n_e)
	tfnoscale = time for fnoscale
	msc = a dictionary containing time evolution of magnitude squared coherence for all pairs of coils
	filepath = adress of the original file
	shot = number of shot
"""
mutable struct BaseAlfvenData
	mscamp::Dict{Any, Matrix}
	mscphase::Dict{Any, Matrix}
	tmsc::Vector
	fmsc::Vector
	upsd::Matrix
	tpsd::Vector
	fpsd::Vector
	fnoscale::Vector
	tfnoscale::Vector
	ip::Vector
	filepath::String
	shot::String
end

"""
	BaseAlfvenData()

Default empty constructor.
"""
BaseAlfvenData() = BaseAlfvenData(
		Dict{Int, Matrix{Float}}(),
		Dict{Int, Matrix{Float}}(),
		Vector{Float}(),
		Vector{Float}(),
		Array{Float,2}(undef,0,0),
		Vector{Float}(),
		Vector{Float}(),
		Vector{Float}(),
		Vector{Float}(),
		Vector{Float}(),
		"",
		""
	)

"""
	BaseAlfvenData(filepath::String; warns=true)

Constructor from a .h5 file.
"""
function BaseAlfvenData(filepath::String; warns=true)
	alfvendata = BaseAlfvenData()
	alfvendata.filepath = filepath
	alfvendata.shot = split(basename(filepath), ".")[1]
	# read the basic signals
	readbasic!(alfvendata, filepath; warns=warns)
	# read the msc data with only some coils
	readmsc!(alfvendata, filepath; warns=warns)
	return alfvendata
end

"""
	BaseAlfvenData(filepath::String, coillist::Vector; warns=true)

Constructor from a .h5 file, coillist specifies a set of coils that are to be extracted.
"""
function BaseAlfvenData(filepath::String, coillist::Vector; warns=true)
	alfvendata = BaseAlfvenData()
	alfvendata.filepath = filepath
	alfvendata.shot = split(basename(filepath), ".")[1]
	# read data
	readbasic!(alfvendata, filepath; warns=warns)
	# read the msc data with only some coils
	readmsc!(alfvendata, filepath; coillist = coillist, warns=warns)
	return alfvendata
end

"""
	readbasic!(alfvendata::BaseAlfvenData, filepath::String; warns=true)

Read the basic signals - time, frequency etc.
"""
function readbasic!(alfvendata::BaseAlfvenData, filepath::String; warns=true)
	alfvendata.tmsc = readsignal(filepath, "t_cohere"; warns=warns)
	alfvendata.fmsc = readsignal(filepath, "f_cohere"; warns=warns)
	alfvendata.fnoscale = readsignal(filepath, "fnoscale"; warns=warns)
	alfvendata.tfnoscale = readsignal(filepath, "t_fnoscale"; warns=warns)
	alfvendata.upsd = readsignal(filepath, "Uprobe_coil_A1pol_psd"; warns=warns)
	alfvendata.tpsd = readsignal(filepath, "t_Uprobe"; warns=warns)
	alfvendata.fpsd = readsignal(filepath, "f_Uprobe"; warns=warns)
	alfvendata.ip = readsignal(filepath, "I_plasma"; warns=warns)
end

"""
	readmsc!(alfvendata::BaseAlfvenData, filepath::String; coillist=nothing, warns=true)

Read the magnitude squared coherence from a .h5 file. If coillist is specified,
only certain coils will be loaded.
"""
function readmsc!(alfvendata::BaseAlfvenData, filepath::String; coillist=nothing, warns=true)
	# if some coil data is missing, save the name in this list and filter them at the end
	_coillist = ((coillist == nothing) ? getcoillist(names(h5open(filepath,"r"))) : coillist)
	#_coillist = String.(_coillist)
	for coil in _coillist
		try 
			@suppress_err begin
				phase = readsignal(filepath, "Mirnov_coil_A&C_theta_$(coil)_cpsdphase"; warns=warns)
				amplitude = readsignal(filepath,  "Mirnov_coil_A&C_theta_$(coil)_coherems"; warns=warns)
				if !any(isnan,phase)
					alfvendata.mscphase[coil] = phase
				end
				if !any(isnan,amplitude)
					alfvendata.mscamp[coil] =  amplitude
				end
			end
		catch e
			if isa(e, ErrorException)
				warns ? @warn("$(alfvendata.filepath): msc data from coil $coil not found") : nothing
			else
				rethrow(e)
			end
		end
	end
end

"""
	readsignal_jl(filepath::String, signal::String; warns=true)

Read a signal from .h5 file using native Julia library. Contains a known memory leak so use with caution.
"""
function readsignal_jl(filepath::String, signal::String; warns=true)
	try 
		@suppress_err begin
			x = h5open(filepath, "r") do file
				read(file, signal)
			end				
			return Float.(x)
		end
	catch e
		if isa(e, ErrorException)
			warns ? @warn("$(filepath): $signal data not found") : nothing
			return NaN
		else
			rethrow(e)
		end
	end
end

"""
	readsignal_py(filepath::String, signal::String; warns=true)

Read a signal from .h5 file using Python h5py library. Is memory safe but slower than the Julia counterpart.
"""
function readsignal_py(filepath::String, signal::String; warns=true)
	# first init the h5py pointer if possible
	(h5py == PyNULL()) ? _init_h5py() : nothing
	(h5py == PyNULL()) ? (return NaN) : nothing # this should happen if h5py is not available
	try
		file = h5py.File(filepath,"r")
		x = Float.(get(file, signal).value)
		file.close()
		if ndims(x) == 2
			return Array(x')
		else
			return x
		end
	catch e
		warns ? @warn("$(filepath): $signal data not found") : nothing
		return NaN
	end
end

readsignal(args...; memorysafe=false, kwargs...) = 
	memorysafe ? readsignal_py(args...; kwargs...) : readsignal_jl(args...; kwargs...)

"""
	normalize(x)

Normalize values of x so that that lie in the interval [0,1].
"""
normalize(x) = (x .- minimum(x))/(maximum(x) - minimum(x))

"""
	readtcoh(filepath::String; warns=true, memorysafe=false)
"""
readtcoh(filepath::String; kwargs...) = readsignal(filepath, "t_cohere";  kwargs...)

"""
	readfcoh(filepath::String; warns=true, memorysafe=false)
"""
readfcoh(filepath::String;  kwargs...) = readsignal(filepath, "f_cohere"; kwargs...)

"""
	readtupsd(filepath::String; warns=true, memorysafe=false)
"""
readtupsd(filepath::String;  kwargs...) = readsignal(filepath, "t_Uprobe"; kwargs...)

"""
	readfupsd(filepath::String; warns=true, memorysafe=false)
"""
readfupsd(filepath::String;  kwargs...) = readsignal(filepath, "f_Uprobe";  kwargs...)

"""
	readfnoscale(filepath::String; warns=true, memorysafe=false)
"""
readfnoscale(filepath::String;  kwargs...) = readsignal(filepath, "fnoscale";  kwargs...)

"""
	readtfnoscale(filepath::String; warns=true, memorysafe=false)
"""
readtfnoscale(filepath::String;  kwargs...) = readsignal(filepath, "t_fnoscale";  kwargs...)

"""
	readmscamp(filepath::String, coil; warns=true, memorysafe=false)
"""
readmscamp(filepath::String, coil;  kwargs...) = readsignal(filepath, "Mirnov_coil_A&C_theta_$(coil)_coherems";  kwargs...)

"""
	readmscphase(filepath::String, coil; warns=true, memorysafe=false)
"""
readmscphase(filepath::String, coil;  kwargs...) = readsignal(filepath, "Mirnov_coil_A&C_theta_$(coil)_cpsdphase";  kwargs...)

"""
	readnormmscphase(filepath::String, coil; warns=true, memorysafe=false)
"""
readnormmscphase(filepath::String, coil;  kwargs...) = normalize(readmscphase(filepath, coil;  kwargs...))

"""
	readmsc(filepath::String, coil; warns=true, memorysafe=false)
"""
readmscampphase(filepath::String, coil; kwargs...) = vcat(readmscamp(filepath, coil; kwargs...), readnormmscphase(filepath, coil; kwargs...))

"""
	readip(filepath::String; warns=true, memorysafe=false)
"""
readip(filepath::String;  kwargs...) = readsignal(filepath, "I_plasma";  kwargs...)

"""
	readupsd(filepath::String; warns=true, memorysafe=false)
"""
readupsd(filepath::String;  kwargs...) = readsignal(filepath, "Uprobe_coil_A1pol_psd";  kwargs...)

"""
	readlogupsd(filepath::String; warns=true, memorysafe=false)
"""
readlogupsd(filepath::String;  kwargs...) = Float(20.0)*log10.(readupsd(filepath;  kwargs...) .+ Float(1e-10))

"""
	readnormlogupsd(filepath::String; warns=true, memorysafe=false)
"""
readnormlogupsd(filepath::String;  kwargs...) = normalize(readlogupsd(filepath;  kwargs...))

"""
	getcoillist(keynames)

Extract all available Mirnov coils from a list of strings (keys of a hdf5 file).
"""
function getcoillist(keynames)
	ks = filter(x->occursin("Mirnov",x), keynames)
	ks = Meta.parse.(unique(vcat(split.(ks, "_")...)))
	ks = filter(x->typeof(x) <: Int, ks)
end

##########################
### flat-top detection ###
##########################
"""
    isnegative(x)

Is x mostly negative?
"""
isnegative(x) = (maximum(x) < abs(minimum(x)))

"""
    makepositive(x)

Make x positive if negative.
"""
makepositive(x) = isnegative(x) ? -x : x

"""
    maxflattop(x,th)

Flattop is identified as being at least th% of maximum of Ip.
"""
maxflattop(x,th=0.7) = (x .>= maximum(x)*th)

"""
    movingmean(x,wl)

Moving average of x with window length wl.
"""
movingmean(x,wl) = map(n->StatsBase.mean(x[max(1,n-wl):n]), 1:length(x))
    
"""
	diffflattop(x, ϵ=1e-5; wl=0)

Flattop is identified as having a derivative smaller than
ϵ*maximum(x), wl is length of window for moving average
of the derivative.
"""
function diffflattop(x, ϵ=1e-5; wl=0)
    dx = diff(x)
    dx = movingmean(dx,wl)
    return vcat([false], abs.(dx) .<= ϵ*maximum(x))
end

"""
	diffmaxflattop(x,th=0.7,ϵ=1e-4;wl=0)

Combination of flattop detection by maximum and derivative.
"""
diffmaxflattop(x,th=0.7,ϵ=1e-4;wl=0) = (maxflattop(x,th) .& diffflattop(x,ϵ;wl=wl))

"""
    maxsection(x::Array{Int,1})

Finds the longest uninterupted section in an array of integers.
Uninterupted means that the difference between two neighboring values
are 1.
"""
function maxsection(x::Array{Int,1})
    inddiffs = diff(x)
    lengths = []
    deltas = []
    k = 0
    for i in 1:length(inddiffs)
        d = inddiffs[i]
        if d > 1
            push!(lengths,k)
            push!(deltas, d)
            k=0
        else
            k += d
        end
    end
    # in case the whole  collection is empty - all the indices are uninterrupted
    push!(lengths,k)
    imx = argmax(lengths)
    if imx == 1
        maxsectioninds = 1:lengths[1]+1
    else
        maxsectioninds = (sum(lengths[1:imx-1])+sum(deltas[1:imx-1])+1):(sum(lengths[1:imx])+sum(deltas[1:imx-1])+1)
    end
    return maxsectioninds
end

"""
    maxsection(mask::AbstractArray{Bool})

For a given boolean vector, obtain the indices of the longest uninterupted section that is true.
"""
function maxsection(mask::AbstractArray{Bool})
    maxinds = maxsection(collect(1:length(mask))[mask])
    offset = 0
    for i in 1:length(mask)
        if mask[i]
            break
        else
            offset += 1
        end
    end
    return maxinds .+ offset
end

"""
    maxsectionbe(mask::Array{Bool,1})

Maximum section beginning and end.
"""
function maxsectionbe(mask::AbstractArray{Bool})
    maxinds = maxsection(mask)
    return maxinds[1], maxinds[end]
end

"""
	flattopbe(x,th=0.7,ϵ=1e-4;wl=0)

Get indices of flattop start and end.
"""
function flattopbe(x,th=0.7,ϵ=1e-4;wl=0)
    _x = makepositive(x)
    mask = diffmaxflattop(_x,th,ϵ;wl=wl)
    return maxsectionbe(mask)
end

"""
	get_ft_section(signal, ip, [minlength, wl, th, ϵ])

Get the flattop part of the signal. 
"""
function get_ft_section(signal::AbstractArray, ip::AbstractVector; minlength=0, wl=20,
		th = 0.6, ϵ = 8e-4)
	ipftstart, ipftstop = flattopbe(ip,th,ϵ;wl=wl)
	M = ndims(signal)
	ls = size(signal,M)
	lip = length(ip)
	ftstart = ceil(Int, ipftstart/lip*ls)
	ftstop = floor(Int, ipftstop/lip*ls)
	inds = [collect(x) for x in axes(signal)]
	if ftstop - ftstart > minlength
		inds[end] = collect(ftstart:ftstop)
	else
		inds[end] = collect(2:1)
		# an empty array of the correct vertical dimension
	end
	return signal[inds...]
end

"""
	valid_ip(x,ϵ=0.02)

Returns boolean indices of valid Ip - Ip is before rampdown and non-zero (larger than ϵ times maximum).
"""
function valid_ip(x,ϵ=0.02)
    _x = makepositive(x)
    mx,imx = findmax(_x)
    inds = _x .>= mx*ϵ
    return [fill(true,imx); inds[imx+1:end]]
end

"""
	get_valid_section(signal, ip; ϵ=0.02)

Get the valid (nonzero) part of the signal. 
"""
function get_valid_section(signal::AbstractArray, ip::AbstractVector; ϵ=0.02)
	valinds = valid_ip(ip,ϵ)
	# valinds are true true true ... true false ... false
	# so we know they begin with true
	lvalip = sum(valinds)
	lip = length(ip)
	ls = size(signal,ndims(signal))
	inds = [collect(x) for x in axes(signal)]
	if lvalip > 0
		inds[end] = collect(1:ceil(Int,ls*lvalip/lip))
	else
		# an empty array of the correct vertical dimension
		inds[end] = collect(2:1)
	end
	return signal[inds...]
end
