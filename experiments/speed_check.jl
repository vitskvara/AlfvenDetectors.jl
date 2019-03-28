# check the speed of upscale + conv versus convtranspose
using AlfvenDetectors
using Flux
using SparseArrays
using LinearAlgebra

data = randn(Float32,32,32,1,16)
loss(model,x) = Flux.mse(x,model(x))
ldim = 8
model1 = AlfvenDetectors.ConvAE(size(data)[1:3], ldim, 3, 3, (64,64,128),[2,2,1])

model2 = Flux.Chain(
		Flux.Conv((3,3),1=>64,relu,pad=(1,1)),
		x->maxpool(x,(2,2)),
		Flux.Conv((3,3),64=>64,relu,pad=(1,1)),
		x->maxpool(x,(2,2)),
		Flux.Conv((3,3),64=>128,relu,pad=(1,1)),
		x->maxpool(x,(1,1)),
		x->reshape(x,:,size(x,4)),
		Flux.Dense(512*16,ldim),
		Flux.Dense(ldim,512*16,relu),
		x->reshape(x,8,8,128,size(x,2)),
		Flux.ConvTranspose((1,1),128=>64,relu),
		Flux.ConvTranspose((2,2),64=>64,relu,stride=(2,2)),
		Flux.ConvTranspose((2,2),64=>1,stride=(2,2))
)
model2(data)

# precompile
L = loss(model1,data)
Flux.back!(L)
L = loss(model2,data)
Flux.back!(L)

# compare
L = loss(model1,data);
@time Flux.back!(L);

L = loss(model2,data);
@time Flux.back!(L);

# test sparse backpropagation
K=32
function f1(x,K) 
	M = sparse(1:K,1:K,fill(Float32(1),K))
	return M*x
end
data = randn(Float32,4,10)
model = Flux.Chain(
	Dense(4,K,relu),
	x->f1(x,K),
	Dense(K,4)
	)
L=loss(model,data)
Flux.back!(L)
L=loss(model,data)
@time Flux.back!(L)

K=32
function f2(x,K) 
	M = Matrix{Float32}(I,K,K)
	return M*x
end
data = randn(Float32,4,10)
model = Flux.Chain(
	Dense(4,K,relu),
	x->f2(x,K),
	Dense(K,4)
	)
L=loss(model,data)
Flux.back!(L)
L=loss(model,data)
@time Flux.back!(L)

