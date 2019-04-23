using AlfvenDetectors
using Test
using Random
using StatsBase
include(joinpath(dirname(pathof(AlfvenDetectors)), "../test/test_utils.jl"))

Random.seed!(12345)

@testset "model utils" begin
    # KL
    @test AlfvenDetectors.KL(0.0, 1.0) == 0.0
    @test AlfvenDetectors.KL(fill(0.0,5), fill(1.0,5)) == 0.0
    @test AlfvenDetectors.KL(fill(0.0,5,5), fill(1.0,5,5)) == 0.0

    # loglikelihood
    @test AlfvenDetectors.loglikelihood(0.0,0.0) == 0.0 - AlfvenDetectors.l2pi/2
    @test AlfvenDetectors.loglikelihood(5,5) == 0.0 - AlfvenDetectors.l2pi/2
    @test AlfvenDetectors.loglikelihood(0.0,0.0,1.0) == 0.0 - AlfvenDetectors.l2pi/2
	@test AlfvenDetectors.loglikelihood(5,5,1) == 0.0 - AlfvenDetectors.l2pi/2

    @test sim(AlfvenDetectors.loglikelihood(fill(0.0,5),fill(0.0,5)), 0.0 - AlfvenDetectors.l2pi/2*5)
    @test sim(AlfvenDetectors.loglikelihood(fill(3,5),fill(3,5)), 0.0 - AlfvenDetectors.l2pi/2*5)
    @test sim(AlfvenDetectors.loglikelihood(fill(0.0,5),fill(0.0,5),fill(1.0,5)), 0.0 - AlfvenDetectors.l2pi/2*5)
	@test sim(AlfvenDetectors.loglikelihood(fill(5,5),fill(5,5),fill(1,5)), 0.0 - AlfvenDetectors.l2pi/2*5)

	# maybe define a different behaviour for vectors and matrices?
	# this would then return a vector
	@test sim(AlfvenDetectors.loglikelihood(fill(0.0,5,5),fill(0.0,5,5)), 0.0 - AlfvenDetectors.l2pi/2*5)
    @test sim(AlfvenDetectors.loglikelihood(fill(3,5,5),fill(3,5,5)), 0.0 - AlfvenDetectors.l2pi/2*5)
    @test sim(AlfvenDetectors.loglikelihood(fill(0.0,5,5),fill(0.0,5,5),fill(1.0,5,5)), 0.0 - AlfvenDetectors.l2pi/2*5)
	@test sim(AlfvenDetectors.loglikelihood(fill(5,5,5),fill(5,5,5),fill(1,5,5)), 0.0 - AlfvenDetectors.l2pi/2*5)
	
	# the version where sigma is a vector (scalar variance)
	@test sim(AlfvenDetectors.loglikelihood(fill(0.0,5,5),fill(0.0,5,5),fill(1.0,5)), 0.0 - AlfvenDetectors.l2pi/2*5)
	@test sim(AlfvenDetectors.loglikelihood(fill(5,5,5),fill(5,5,5),fill(1,5)), 0.0 - AlfvenDetectors.l2pi/2*5)
	
	# loglikelihoodopt
	@test AlfvenDetectors.loglikelihoodopt(0.0,0.0) == 0.0
    @test AlfvenDetectors.loglikelihoodopt(5,5) == 0.0
    @test AlfvenDetectors.loglikelihoodopt(0.0,0.0,1.0) == 0.0
	@test AlfvenDetectors.loglikelihoodopt(5,5,1) == 0.0

    @test sim(AlfvenDetectors.loglikelihoodopt(fill(0.0,5),fill(0.0,5)), 0.0)
    @test sim(AlfvenDetectors.loglikelihoodopt(fill(3,5),fill(3,5)), 0.0)
    @test sim(AlfvenDetectors.loglikelihoodopt(fill(0.0,5),fill(0.0,5),fill(1.0,5)), 0.0)
	@test sim(AlfvenDetectors.loglikelihoodopt(fill(5,5),fill(5,5),fill(1,5)), 0.0)

	# maybe define a different behaviour for vectors and matrices?
	# this would then return a vector
	@test sim(AlfvenDetectors.loglikelihoodopt(fill(0.0,5,5),fill(0.0,5,5)), 0.0)
    @test sim(AlfvenDetectors.loglikelihoodopt(fill(3,5,5),fill(3,5,5)), 0.0)
    @test sim(AlfvenDetectors.loglikelihoodopt(fill(0.0,5,5),fill(0.0,5,5),fill(1.0,5,5)), 0.0)
	@test sim(AlfvenDetectors.loglikelihoodopt(fill(5,5,5),fill(5,5,5),fill(1,5,5)), 0.0)

	# the version where sigma is a vector (scalar variance)
    @test sim(AlfvenDetectors.loglikelihoodopt(fill(0.0,5,5),fill(0.0,5,5),fill(1.0,5)), 0.0)
	@test sim(AlfvenDetectors.loglikelihoodopt(fill(5,5,5),fill(5,5,5),fill(1,5)), 0.0)

	# mu&sigma
	X = randn(4,10)
	@test size(AlfvenDetectors.mu(X)) == (2,10)
	@test size(AlfvenDetectors.sigma2(X)) == (2,10)
	@test all(x->x>0, AlfvenDetectors.sigma2(X)) 

	# mu&sigma scalar
	@test size(AlfvenDetectors.mu_scalarvar(X)) == (3,10)
	@test size(AlfvenDetectors.sigma2_scalarvar(X)) == (10,)
	@test all(x->x>0, AlfvenDetectors.sigma2_scalarvar(X)) 
	
	# sample normal
	M = fill(2,1000)
	sd = fill(0.1,1000)
	X = AlfvenDetectors.samplenormal(M,sd)
	@test size(X) == (1000,)
	@test sim(StatsBase.mean(X), 2, 1e-2)
	
	M = fill(2,10,1000)
	sd = fill(0.1,1000)
	X = AlfvenDetectors.samplenormal(M,sd)
	@test size(X) == (10,1000)
	@test sim(StatsBase.mean(X), 2, 1e-1)
	
	X = randn(4,10)
	y = AlfvenDetectors.samplenormal(X)
	@test size(y) == (2,10)

	# sample normal scalarvar
	X = randn(4,10)
	y = AlfvenDetectors.samplenormal_scalarvar(X)
	@test size(y) == (3,10)
end