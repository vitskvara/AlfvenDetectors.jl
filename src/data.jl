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
	msc::Dict{Any, Matrix}
	tmsc::Vector
	fmsc::Vector
	psd::Matrix
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
	data = h5open(filepath, "r") do file
		# read the basic signals
		readbasic!(alfvendata, file)
		# read the msc data
		readmsc!(alfvendata, file)
	end
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
	data = h5open(filepath, "r") do file
		# read the basic signals
		readbasic!(alfvendata, file)
		# read the msc data with only some coils
		readmsc!(alfvendata, file; coillist = coillist)
	end
	return alfvendata
end

"""
	readbasic!(alfvendata::BaseAlfvenData, file::HDF5File)

Read the basic signals - time, frequency etc.
"""
function readbasic!(alfvendata::BaseAlfvenData, file::HDF5File)
	alfvendata.tmsc = Float.(read(file, "t_cohere"))
	alfvendata.fmsc = Float.(read(file, "f_cohere"))
	alfvendata.fnoscale = Float.(read(file, "fnoscale"))
	alfvendata.tfnoscale = Float.(read(file, "t_fnoscale"))
	alfvendata.psd = Float.(read(file, "Uprobe_coil_A1pol_psd"))
	alfvendata.tpsd = Float.(read(file, "t_Uprobe"))
	alfvendata.fpsd = Float.(read(file, "f_Uprobe"))
	alfvendata.ip = Float.(read(file, "I_plasma"))
end

"""
	readmsc!(alfvendata::BaseAlfvenData, file::HDF5File; coillist=nothing)

Read the magnitude squared coherence from a .h5 file. If coillist is specified,
only certain coils will be loaded.
"""
function readmsc!(alfvendata::BaseAlfvenData, file::HDF5File; coillist=nothing)
	# if some coil data is missing, save the name in this list and filter them at the end
	_coillist = ((coillist == nothing) ? getcoillist(names(file)) : coillist)
	#_coillist = String.(_coillist)
	for coil in _coillist
		try 
			@suppress_err begin
				alfvendata.msc[coil] = Float.(read(file, "Mirnov_coil_A&C_theta_$(coil)_cpsdphase"))
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
	getcoillist(keynames)

Extract all availabel Mirnov coils from a list of strings (keys of a hdf5 file).
"""
function getcoillist(keynames)
	ks = filter(x->occursin("Mirnov",x), keynames)
	ks = Meta.parse.(unique(vcat(split.(ks, "_")...)))
	ks = filter(x->typeof(x) <: Int, ks)
end
