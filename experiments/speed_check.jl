# check the speed of upscale + conv versus convtranspose
using AlfvenDetectors
using Flux
using SparseArrays
using LinearAlgebra
using CuArrays

data = randn(Float32,32,32,1,16);
gpu_data = data |> gpu;
ldim = 8
cpu_model = AlfvenDetectors.ConvTSVAE(size(data)[1:3], ldim, 3, 3, (64,64,128),[2,2,1]);
gpu_model = AlfvenDetectors.ConvTSVAE(size(data)[1:3], ldim, 3, 3, (64,64,128),[2,2,1]) |> gpu;

# test forward pass
cpu_model(data);
gpu_model(gpu_data);
@info "cpu model forward pass"
@time cpu_model(data);
@info "gpu model forward pass"
@time gpu_model(gpu_data);

# test the backward pass
L = AlfvenDetectors.loss(cpu_model.m1,data,1,1.0)
Flux.back!(L)
@info "cpu model backward pass"
@time L = AlfvenDetectors.loss(cpu_model.m1,data,1,1.0)
@time Flux.back!(L)

L = AlfvenDetectors.loss(gpu_model.m1,gpu_data,1,1.0)
Flux.back!(L)
@info "gpu model backward pass"
@time L = AlfvenDetectors.loss(gpu_model.m1,gpu_data,1,1.0)
@time Flux.back!(L)

