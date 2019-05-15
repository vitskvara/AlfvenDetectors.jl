using AlfvenDetectors
using Test
using Random

@testset "AlfvenDetectors" begin

@testset "utilities" begin
	@info "Testing utilities"
	include("data.jl")
	include("umap.jl")
end

include("few_shot_models.jl")

@testset "Experiments" begin
	@info "Testing experiments"
	include("experiments.jl")
end

end