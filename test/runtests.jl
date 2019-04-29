using AlfvenDetectors
using Test
using Random

@testset "AlfvenDetectors" begin

@testset "utilities" begin
	@info "Testing utilities"
	include("data.jl")
end

@testset "Experiments" begin
	@info "Testing experiments"
	include("experiments.jl")
end

end