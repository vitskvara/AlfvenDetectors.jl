using Test
using AlfvenDetectors
using Random
using HDF5

fpath = "data/testdata.h5"

@testset "data" begin
	h5data = h5read(fpath,"/")
	@test AlfvenDetectors.getcoillist(keys(h5data)) == [12, 14]

	data = AlfvenDetectors.BaseAlfvenData(fpath)
	vecfields = [:tmsc, :fmsc, :tpsd, :fpsd, :fnoscale, :tfnoscale, :ip]
	matfields = [:psd]
	for field in vecfields
		@test size(getfield(data, field)) == (3,)
	end	 
	for field in matfields
		@test size(getfield(data, field),2) == 3
	end	 	
	@test length(data.msc) == 2

	data = AlfvenDetectors.BaseAlfvenData(fpath, [11,12])
	@test length(data.msc) == 1
end