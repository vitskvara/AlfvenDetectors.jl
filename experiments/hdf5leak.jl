# try running as `julia hdf5leak.jl 20`
using AlfvenDetectors
using HDF5
using PyCall
h5py = pyimport("h5py")
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
readfun = AlfvenDetectors.readnormlogupsd
batchsize = 10
batchsize = Int(Meta.parse(ARGS[1]))
files = joinpath.(datapath,readdir(datapath))
function readdata(f) 
	data = h5open(f, "r") do file
    	read(file, "Uprobe_coil_A1pol_psd")
	end
	return deepcopy(data)
end
# this does not have the memory leak but it is super slow
function pyreaddata(f)
	file = h5py.File(f,"r")
	data = get(file, "Uprobe_coil_A1pol_psd").value
	file.close()
	return Array(data')
end

@time for i in 1:batchsize
	data = readdata(files[i])	
	GC.gc()
	#data = AlfvenDetectors.get_signal(files[i],readfun)
	#data = AlfvenDetectors.readsignal(files[i], "Uprobe_coil_A1pol_psd")
	#data = AlfvenDetectors.readlogupsd(files[i])
	#data = AlfvenDetectors.readnormlogupsd(files[i])
end

@time for i in 1:batchsize
	data = pyreaddata(files[i])	
	#data = AlfvenDetectors.get_signal(files[i],readfun)
	#data = AlfvenDetectors.readsignal(files[i], "Uprobe_coil_A1pol_psd")
	#data = AlfvenDetectors.readlogupsd(files[i])
	#data = AlfvenDetectors.readnormlogupsd(files[i])
end

i = 1
@time data = readdata(files[i])
@time data = pyreaddata(files[i])