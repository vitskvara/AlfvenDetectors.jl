using AlfvenDetectors
using Test
using Random
using BSON
using Flux
using GenerativeModels
using GaussianMixtures
using EvalCurves
using ValueHistories

hostname = gethostname()
if hostname == "vit-ThinkPad-E470"
    datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
elseif hostname == "tarbik.utia.cas.cz"
    datapath = "/home/skvara/work/alfven/cdb_data/uprobe_data"
elseif occursin("soroban", hostname)
    datapath = "/compass/home/skvara/no-backup/uprobe_data"
else
    datapath = ""
end

@testset "evaluation" begin
    if datapath != ""
        patchsize = 128
        # data loading
        # get labeled data
        patch_data, shotnos, labels, tstarts, fstarts = AlfvenDetectors.get_labeled_validation_data(patchsize)
        @test length(shotnos) == length(labels) == size(patch_data,4) == length(tstarts) == length(fstarts) > 1
        # split patches    
        train_info, train_inds, test_info, test_inds = 
            AlfvenDetectors.split_unique_patches(0.5, 
                shotnos, labels, tstarts, fstarts);
        train_labeled = (patch_data[:,:,:,train_inds], train_info[2]);
        test = (patch_data[:,:,:,test_inds], test_info[2]);
        # get unlabeled data
        unlabeled_nshots = 30
        measurement_type = "uprobe"
        use_alfven_shots = true
        warns = true
        iptrunc = "valid"
        memorysafe = true
        train_unlabeled, shotnos_unlabeled = AlfvenDetectors.get_unlabeled_validation_data(datapath, 
            unlabeled_nshots, (train_info[1], test_info[1]), patchsize, 
            measurement_type, use_alfven_shots, warns, iptrunc, memorysafe)
        # test for exclusivity of testing labels
        @test !any(map(x->any(occursin.(string(x), shotnos_unlabeled)), test_info[1]))
        @test any(map(x->any(occursin.(string(x), shotnos_unlabeled)), train_info[1]))

        # setup fs model
        mf = "data/conv_ae_model_new.bson"
        s1_model, exp_args, model_args, model_kwargs, history = AlfvenDetectors.load_model(mf)

        # knn model
        s2_model_name = "KNN"
        s2_args = [:BruteTree]
        s2_kwargs = Dict()
        s2_model = eval(Meta.parse("AlfvenDetectors."*s2_model_name))(s2_args...; s2_kwargs...);
        fx(m,x) = nothing # there is no point in fitting the unlabeled samples
        fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y);
        asfs = [AlfvenDetectors.as_mean]
        asf_args = map(x->[x],collect(1:2:31))

        # create the model
        fsmodel = AlfvenDetectors.FewShotModel(s1_model, s2_model, fx, fxy, nothing);
        AlfvenDetectors.fit!(fsmodel, train_unlabeled, train_labeled[1], train_labeled[2];
            encoding_batchsize=128);

        df_exp = fit_fs_model(s1_model, s2_model, fx, fxy, asfs, asf_args, patch_data, shotnos, labels, 
            tstarts, fstarts, datapath, unlabeled_nshots, exp_args)

    end

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

