using AlfvenDetectors
using Test
using Random

sim(x,y) = abs(x-y) < 1e-6

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

	# mu&sigma
	X = randn(4,10)
	@test size(AlfvenDetectors.mu(X)) == (2,10)
	@test size(AlfvenDetectors.sigma2(X)) == (2,10)
	@test all(x->x>0, AlfvenDetectors.sigma2(X)) 

	# sample normal
	y = AlfvenDetectors.samplenormal(X)
	@test size(y) == (2,10)
end