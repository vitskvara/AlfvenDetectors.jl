"""
	BaseAlfvenData

A mutable structure representing the basic Alfven eigenmode data.

Fields:

	t = time
	f = frequency
	fnoscale = frequency according to the scaling law ~ I_p/sqrt(n_e)
	tfnoscale = time for fnoscale
	coils = names of coil pairs
	msc = time evolution of magnitude squared coherence for all pairs of coils
	filepath = adress of the original file
"""
mutable struct BaseAlfvenData
	t::Vector
	f::Vector
	fnoscale::Vector
	tfnoscale::Vector
	coils::Vector
	msc::Array{Matrix, 1}
	filepath::String
end

"""
	BaseAlfvenData()

Default empty constructor.
"""
BaseAlfvenData() = BaseAlfvenData(
		Vector{Float}(),
		Vector{Float}(),
		Vector{Float}(),
		Vector{Float}(),
		Vector{Int}(),
		Array{Matrix{Float},1}(),
		""
	)

"""
	BaseAlfvenData(filepath::String)

Constructor from a .h5 file.
"""
function BaseAlfvenData(filepath::String)
	alfvendata = BaseAlfvenData()
	alfvendata.filepath = filepath
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
		alfvendata.t = Float.(read(file, "t"))
		alfvendata.f = Float.(read(file, "f"))
		alfvendata.fnoscale = Float.(read(file, "fnoscale"))
		alfvendata.tfnoscale = Float.(read(file, "tfnoscale"))
		alfvendata.coils = read(file, "coils")
end

"""
	readmsc!(alfvendata::BaseAlfvenData, file::HDF5File; coillist=nothing)

Read the magnitude squared coherence from a .h5 file. If coillist is specified,
only certain coils will be loaded.
"""
function readmsc!(alfvendata::BaseAlfvenData, file::HDF5File; coillist=nothing)
	# if some coil data is missing, save the name in this list and filter them at the end
	missingcoils = []
	_coillist = ((coillist == nothing) ? alfvendata.coils : coillist)
	for coil in _coillist
		try 
			@suppress_err begin
				push!(alfvendata.msc, Float.(read(file, "cxy$coil")))
			end
		catch e
			if isa(e, ErrorException)
				@warn("$(alfvendata.filepath): msc data from coil $coil not found")
			else
				throw(e)
			end
			push!(missingcoils, coil)
		end
	end
	filter!(x->!(x in missingcoils), _coillist)
	alfvendata.coils = _coillist
end