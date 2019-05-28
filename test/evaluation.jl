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


	# setup the S2 model
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

end