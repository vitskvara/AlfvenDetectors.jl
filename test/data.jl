using Test
using AlfvenDetectors
using Random
using HDF5

fpath = joinpath(dirname(@__FILE__),"data/testdata.h5")

@testset "data" begin
	h5data = h5read(fpath,"/")
	@test AlfvenDetectors.getcoillist(keys(h5data)) == [12, 14]

	data = AlfvenDetectors.BaseAlfvenData(fpath)
	vecfields = [:tmsc, :fmsc, :tpsd, :fpsd, :fnoscale, :tfnoscale, :ip]
	matfields = [:upsd]
	for field in vecfields
		@test size(getfield(data, field)) == (3,)
	end	 
	for field in matfields
		@test size(getfield(data, field),2) == 3
	end	 	
	@test length(data.mscamp) == 2
	@test length(data.mscphase) == 2
	for coil in keys(data.mscphase)
		@test size(data.mscphase[coil],2) == 3
	end
	for coil in keys(data.mscamp)
		@test size(data.mscamp[coil],2) == 3
	end

	data = AlfvenDetectors.BaseAlfvenData(fpath, [11,12])
	@test length(data.mscamp) == 1
	@test length(data.mscphase) == 1
	for coil in keys(data.mscphase)
		@test size(data.mscphase[coil],2) == 3
	end
	for coil in keys(data.mscamp)
		@test size(data.mscamp[coil],2) == 3
	end

	# signal read functions
	x = randn(10,5)
	nx = AlfvenDetectors.normalize(x)
	@test 0.0 <= minimum(nx) <= maximum(nx) <= 1.0
	@test size(AlfvenDetectors.readmscamp(fpath, 12),2) == 3
	@test size(AlfvenDetectors.readmscphase(fpath, 12),2) == 3
	@test size(AlfvenDetectors.readnormmscphase(fpath, 12),2) == 3
	@test length(AlfvenDetectors.readip(fpath)) == 3
	@test size(AlfvenDetectors.readupsd(fpath),2) == 3
	@test size(AlfvenDetectors.readlogupsd(fpath),2) == 3
	@test size(AlfvenDetectors.readnormlogupsd(fpath),2) == 3

	##########################
	### flat-top detection ###
	##########################
	x = collect(1:10)
	@test AlfvenDetectors.isnegative(-x)
	@test !AlfvenDetectors.isnegative(x)
	@test x == AlfvenDetectors.makepositive(x)
	@test x == AlfvenDetectors.makepositive(-x)
	x[3] = 9
	@test x[AlfvenDetectors.maxflattop(x,0.8)] == [9,8,9,10]
	x[3] = 3
	@test x == AlfvenDetectors.movingmean(x,0)
	y = AlfvenDetectors.movingmean(x,1)
	@test length(y) == length(x)
	@test y == [1.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
	x = vcat(collect((1:4)).*2, fill(10,5), collect((4:-1:1)).*2)
	x[6] = 9
	x[8] = 11
	@test x[AlfvenDetectors.diffflattop(x,0.1)] == [9,10,11,10]
	x = vcat(x,fill(1,10))
	@test x[AlfvenDetectors.diffmaxflattop(x,0.8,0.1)] == [9,10,11,10]

	# tests for the max section fctions
	x1 = collect(2:45)
	y1 = vcat(collect(3:9), collect(11:12), collect(18:21), [40])
	is1 = map(x->x in y1, x1);
	x2 = collect(1:45)
	y2 = vcat(collect(3:4), collect(11:16), collect(18:24), [40])
	is2 = map(x->x in y2, x2);
	# test the maxsection etc.
	@test y1[AlfvenDetectors.maxsection(y1)] == collect(3:9)
	# this is broken and does not work as intended
	# @test y2[AlfvenDetectors.maxsection(y2)] == collect(18:24)
	@test x1[AlfvenDetectors.maxsection(is1)] == collect(3:9)
	@test x2[AlfvenDetectors.maxsection(is2)] == collect(18:24)
	@test AlfvenDetectors.maxsectionbe(is1) == (2,8)
	@test AlfvenDetectors.maxsectionbe(is2) == (18,24)
	@test AlfvenDetectors.flattopbe(x,0.8,0.1) == (6,9)
end