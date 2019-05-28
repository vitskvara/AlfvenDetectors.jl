using AlfvenDetectors
using Test
using Random
using BSON
using Flux
using GenerativeModels
using GaussianMixtures
using EvalCurves

@testset "evaluation" begin
	# setup fake data
	Random.seed!(12345)
	data = randn(Float32,128,128,1,8);
	m,n,c,k = size(data)

	# now setup the convolutional net
	insize = (m,n,c)
	latentdim = 2
	nconv = 3
	kernelsize = 3
	channels = (2,4,6)
	scaling = [(2,2),(2,2),(1,1)]
	batchnorm = true
	s1_model = GenerativeModels.ConvAE(insize, latentdim, nconv, kernelsize, channels, scaling;
		batchnorm = batchnorm)

	# setup valdiation data
	patchsize = 128
	data, shotnos, labels, tstarts, fstarts = AlfvenDetectors.get_validation_data(patchsize);

	# setuo the S2 model
	s2_model_name = "KNN"
    s2_args = [:BruteTree]
    s2_kwargs = Dict()
    s2_model = eval(Meta.parse("AlfvenDetectors."*s2_model_name))(s2_args...; s2_kwargs...);

    fx(m,x) = nothing # there is no point in fitting the unlabeled samples
    fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y);
    #kvec = collect(1:2:31)
    asfs = [AlfvenDetectors.as_mean]
    asf_args = map(x->[x],collect(1:2:3))

    # this contains the fitted aucs and some other data

    df_exp = AlfvenDetectors.fit_fs_model(s1_model, s2_model, fx, fxy, asfs, asf_args, data, shotnos, 
        labels, tstarts, fstarts)


	# setup the S2 model for GMM
	s2_model_name = "GMMModel"
    s2_args = [2]
    s2_kwargs = Dict(
        :kind => :diag,
        :method => :kmeans)
    s2_model = eval(Meta.parse("AlfvenDetectors."*s2_model_name))(s2_args...; s2_kwargs...);

    fx(m,x) = AlfvenDetectors.fit!(m,x) # there is no point in fitting the unlabeled samples
    fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y);
    asfs = [AlfvenDetectors.as_max_ll_mse, AlfvenDetectors.as_mean_ll_mse, 
    	AlfvenDetectors.as_med_ll_mse, AlfvenDetectors.as_ll_maxarg]
    asf_args = [[1]]

    # this contains the fitted aucs and some other data

    df_exp = AlfvenDetectors.fit_fs_model(s1_model, s2_model, fx, fxy, asfs, asf_args, data, shotnos, 
        labels, tstarts, fstarts)

    # S2 model for SVAE
    s2_model_name = "SVAEMem"
    inputdim = 
    hiddenDim = 32
    latentDim = 2
    numLayers = 3
    # params for memory
    memorySize = 128
    α = 0.1 # threshold in the memory that does not matter to us at the moment!
    k = 128
    labelCount = 1
    s2_args = (inputdim, hiddenDim, latentDim, numLayers, memorySize, k, labelCount, α)
    s2_kwargs = Dict()
    s2_model = eval(Meta.parse("AlfvenDetectors."*s2_model_name))(s2_args...; s2_kwargs...);
    β = 0.1 # ratio between reconstruction error and the distance between p(z) and q(z)
    γ = 0.1 # importance ratio between anomalies and normal data in mem_loss
    batchsize = 64
    nbatches = 2 # 200
    sigma = 0.1 # width of imq kernel
    fx(m,x)=AlfvenDetectors.fit!(m, x, batchsize, nbatches, β, sigma, η=0.0001,cbtime=1);
    sigma = 0.01
    batchsize = 10 # this batchsize must be smaller than the size of the labeled training data
    nbatches = 5 # 50
    fxy(m,x,y)=AlfvenDetectors.fit!(m,x,y, batchsize, nbatches, β, sigma, γ, η=0.0001, cbtime=1);
    # finally construct the anomaly score function
    asfs = [AlfvenDetectors.as_logpxgivenz]
    asf_args = [[]]

    # this contains the fitted aucs and some other data
    df_exp = AlfvenDetectors.fit_fs_model(s1_model, s2_model, fx, fxy, asfs, asf_args, data, shotnos, 
        labels, tstarts, fstarts)
end

