import Base.collect

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
	BaseAlfvenData(filepath::String)

Constructor from a .h5 file.
"""
function BaseAlfvenData(filepath::String)
	alfvendata = BaseAlfvenData()
	alfvendata.filepath = filepath
	alfvendata.shot = split(basename(filepath), ".")[1]
	# read the basic signals
	readbasic!(alfvendata, filepath)
	# read the msc data with only some coils
	readmsc!(alfvendata, filepath)
	return alfvendata
end

"""
	BaseAlfvenData(filepath::String, coillist::Vector)

Constructor from a .h5 file, coillist specifies a set of coils that are to be extracted.
"""
function BaseAlfvenData(filepath::String, coillist::Vector)
	alfvendata = BaseAlfvenData()
	alfvendata.filepath = filepath
	alfvendata.shot = split(basename(filepath), ".")[1]
	# read data
	readbasic!(alfvendata, filepath)
	# read the msc data with only some coils
	readmsc!(alfvendata, filepath; coillist = coillist)
	return alfvendata
end

"""
	readbasic!(alfvendata::BaseAlfvenData, filepath::String)

Read the basic signals - time, frequency etc.
"""
function readbasic!(alfvendata::BaseAlfvenData, filepath::String)
	alfvendata.tmsc = readsignal(filepath, "t_cohere")
	alfvendata.fmsc = readsignal(filepath, "f_cohere")
	alfvendata.fnoscale = readsignal(filepath, "fnoscale")
	alfvendata.tfnoscale = readsignal(filepath, "t_fnoscale")
	alfvendata.upsd = readsignal(filepath, "Uprobe_coil_A1pol_psd")
	alfvendata.tpsd = readsignal(filepath, "t_Uprobe")
	alfvendata.fpsd = readsignal(filepath, "f_Uprobe")
	alfvendata.ip = readsignal(filepath, "I_plasma")
end

"""
	readmsc!(alfvendata::BaseAlfvenData, filepath::String; coillist=nothing)

Read the magnitude squared coherence from a .h5 file. If coillist is specified,
only certain coils will be loaded.
"""
function readmsc!(alfvendata::BaseAlfvenData, filepath::String; coillist=nothing)
	# if some coil data is missing, save the name in this list and filter them at the end
	_coillist = ((coillist == nothing) ? getcoillist(names(h5open(filepath,"r"))) : coillist)
	#_coillist = String.(_coillist)
	for coil in _coillist
		try 
			@suppress_err begin
				alfvendata.mscphase[coil] = readsignal(filepath, "Mirnov_coil_A&C_theta_$(coil)_cpsdphase")
				alfvendata.mscamp[coil] = readsignal(filepath,  "Mirnov_coil_A&C_theta_$(coil)_coherems")
			end
		catch e
			if isa(e, ErrorException)
				@warn("$(alfvendata.filepath): msc data from coil $coil not found")
			else
				throw(e)
			end
		end
	end
end

"""
	readsignal(filepath::String, signal::String)
"""
function readsignal(filepath::String, signal::String)
	try 
		@suppress_err begin
			x = h5open(filepath, "r") do file
				read(file, signal)
			end				
			return Float.(x)
		end
	catch e
		if isa(e, ErrorException)
			@warn("$(filepath): $signal data from coil $coil not found")
			return nothing
		else
			throw(e)
		end
	end
end

"""
	readmscamp(filepath::String, coil)
"""
readmscamp(filepath::String, coil) = readsignal(filepath, "Mirnov_coil_A&C_theta_$(coil)_coherems")

"""
	readmscphase(filepath::String, coil)
"""
readmscphase(filepath::String, coil) = readsignal(filepath, "Mirnov_coil_A&C_theta_$(coil)_cpsdphase")

"""
	readip(filepath::String)
"""
readip(filepath::String) = readsignal(filepath, "I_plasma")

"""
	readupsd(filepath::String)
"""
readupsd(filepath::String) = readsignal(filepath, "Uprobe_coil_A1pol_psd")

"""
	getcoillist(keynames)

Extract all availabel Mirnov coils from a list of strings (keys of a hdf5 file).
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