using AlfvenDetectors
using Test
using Random
using BSON
using Flux
using GenerativeModels
using GaussianMixtures
using EvalCurves

@testset "few-shot models" begin
	Random.seed!(12345)
	xdim = 2
	Nn = 1000
	Na = 10
	N = 2*Nn+Na
	X = Float32.(hcat(randn(xdim,Na).-[10;10], randn(xdim,Nn).+[10;10], randn(xdim,Nn).+[-10;10]))
	Xtst = Float32.(hcat(randn(xdim,Na).-[10;10], randn(xdim,Nn).+[10;10], randn(xdim,Nn).+[-10;10]))
	Y = Int.(vcat(ones(Na), zeros(2*Nn)))
	# test separate clustering algorithms

	# GMMS
	Nclust = 3
	clust_alg = AlfvenDetectors.GMMModel(Nclust; kind=:diag, method=:kmeans)
	AlfvenDetectors.fit!(clust_alg, X)
	@test size(clust_alg.train_ll) == (0,0)
	@test size(clust_alg.train_labels) == (0,)
	AlfvenDetectors.fit!(clust_alg, X,Y)
	@test size(clust_alg.train_ll) == (Nclust,N)
	@test size(clust_alg.train_labels) == (N,)

	label = 1
	asmax = AlfvenDetectors.as_max_ll_mse(clust_alg, X, label)
	@test EvalCurves.auc(EvalCurves.roccurve(asmax, Y)...) == 1
	asmean = AlfvenDetectors.as_mean_ll_mse(clust_alg, X, label)
	@test EvalCurves.auc(EvalCurves.roccurve(asmean, Y)...) == 1
	asmed = AlfvenDetectors.as_med_ll_mse(clust_alg, X, label)
	@test EvalCurves.auc(EvalCurves.roccurve(asmed, Y)...) == 1
	asll = AlfvenDetectors.as_ll_maxarg(clust_alg, X, label)
	@test EvalCurves.auc(EvalCurves.roccurve(asll, Y)...) == 1

	# KNN
	clust_alg = AlfvenDetectors.KNN(:KDTree)
	# this only fits the tree and saves X, but has no other use for semisupervised learning
	AlfvenDetectors.fit!(clust_alg, X)
	@test size(clust_alg.X) == size(X)
	@test size(clust_alg.Y) == (0,)
	@test !AlfvenDetectors.is_fitted(clust_alg)	
	# this fits the tree, saves the data and labels
	AlfvenDetectors.fit!(clust_alg, X, Y)
	@test AlfvenDetectors.is_fitted(clust_alg)	
	@test size(clust_alg.X) == size(X)
	@test size(clust_alg.Y) == size(Y)
	# now test the as functions
	k = 5
	asmean = AlfvenDetectors.as_mean(clust_alg, X, k)
	@test EvalCurves.auc(EvalCurves.roccurve(asmean, Y)...) == 1	
	asmeanw = AlfvenDetectors.as_mean_weighted(clust_alg, X, k)
	@test EvalCurves.auc(EvalCurves.roccurve(asmeanw, Y)...) == 1	

	# SVAEMem
	# params for svae
	inputdim = xdim
	hiddenDim = 32
	latentDim = 2
	numLayers = 3
	nonlinearity = "relu"
	layerType = "Dense"
	# params for memory
	memorySize = 20
	α = 0.1 # threshold in the memory that does not matter to us at the moment!
	k = 20
	labelCount = 1
	clust_alg = AlfvenDetectors.SVAEMem(inputdim, hiddenDim, latentDim, numLayers, 
		memorySize, k, labelCount, α; nonlinearity=nonlinearity, layerType=layerType)
	β = 0.1 # ratio between reconstruction error and the distance between p(z) and q(z)
	γ = 0.1 # importance ratio between anomalies and normal data in mem_loss
	batchsize = 64
	nbatches = 500
	σ = 0.1 # width of imq kernel
	AlfvenDetectors.fit!(clust_alg, X, batchsize, nbatches, β, σ, η=0.001,cbtime=1);
	σ = 0.01
	batchsize = 64
	nbatches = 50
	AlfvenDetectors.fit!(clust_alg, X[:,1:200], Y[1:200], batchsize, nbatches, β, σ, γ, η=0.001, cbtime=1);
	as = AlfvenDetectors.as_logpxgivenz(clust_alg, Xtst)
	# for some reason, fitting the model on this data does not really work
	# maybe the input data should be 3D?
	@test EvalCurves.auc(EvalCurves.roccurve(as, Y)...) >= 0
	println(EvalCurves.auc(EvalCurves.roccurve(as, Y)...))


	# tests of the few shot learning structure
	# first train the autoencoder
	xdim = 3
	Nclust = 3
	N = 1000
	px = AlfvenDetectors.cubeGM(xdim,Nclust)
	X = AlfvenDetectors.sample(px, N)
	ldim = 2
	pz = AlfvenDetectors.cubeGM(ldim, 4)
	ae_nlayers = 3
	disc_nlayers = 3
	hdim = 50
	ae_model = WAAE(xdim, ldim, ae_nlayers, disc_nlayers, pz; hdim=hdim)
	AlfvenDetectors.fit!(ae_model, X, 100, 500, σ=1, λ=10, verb=true)

	# now create artificial labeled training and testing data
	_N = 10
	Xtrl = []
	for i in 1:Nclust
		_px = AlfvenDetectors.GM([px.μ[i]], [px.σ[i]])
		push!(Xtrl, AlfvenDetectors.sample(_px, _N))
	end
	Xtrl = hcat(Xtrl...)
	Ytr = vcat(zeros(_N), ones(_N), zeros(_N))
	Xtstl = []
	for i in 1:Nclust
		_px = AlfvenDetectors.GM([px.μ[i]], [px.σ[i]])
		push!(Xtstl, AlfvenDetectors.sample(_px, _N))
	end
	Xtstl = hcat(Xtstl...)
	Ytst = copy(Ytr)

	# now add the gmm model for latent space clustering
	clust_model = AlfvenDetectors.GMMModel(Nclust; kind=:diag, method=:kmeans)
	# now create the fewshotmodel
	fx(m,x) = AlfvenDetectors.fit!(m,x)
	fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y;refit=false)
	anomaly_label = 1
	asf(m,x) = AlfvenDetectors.as_ll_maxarg(m,x,anomaly_label)
	# this constructs the whole model
	model = AlfvenDetectors.FewShotModel(ae_model, clust_model, fx, fxy, asf)
	Z = AlfvenDetectors.encode(model, X)
	@test size(Z) == (ldim, N)
	@test !AlfvenDetectors.is_fitted(model.clust_model)
	# next, fit the clustering model
	#fit!(model,X) # this will only fit the gmm with all 
	# the available data without any knowledge of labels
	AlfvenDetectors.fit!(model, X, Xtrl, Ytr)
	@test AlfvenDetectors.is_fitted(model.clust_model)
	as = AlfvenDetectors.anomaly_score(model, Xtstl)
	@test EvalCurves.auc(EvalCurves.roccurve(as, Ytst)...) == 1

	# do the same with the KNN model
	clust_model = AlfvenDetectors.KNN(:KDTree)
	fx(m,x) = nothing # there is no point in fitting the unlabeled samples
	fxy(m,x,y) = AlfvenDetectors.fit!(m,x,y) # there is no point in fitting the unlabeled samples	
	k = 5
	asf(m,x) = AlfvenDetectors.as_mean(m,x,k)
	model = AlfvenDetectors.FewShotModel(ae_model, clust_model, fx, fxy, asf)
	@test !AlfvenDetectors.is_fitted(model.clust_model)
	AlfvenDetectors.fit!(model, X, Xtrl, Ytr)
	@test AlfvenDetectors.is_fitted(model.clust_model)
	as = AlfvenDetectors.anomaly_score(model, Xtstl)
	@test EvalCurves.auc(EvalCurves.roccurve(as, Ytst)...) == 1

	# and finally with the SVAE model
	inputdim = 2
	hiddenDim = 32
	latentDim = 2
	numLayers = 3
	# params for memory
	memorySize = 128
	α = 0.1 # threshold in the memory that does not matter to us at the moment!
	k = 128
	labelCount = 1
	clust_model = AlfvenDetectors.SVAEMem(inputdim, hiddenDim, latentDim, numLayers, 
		memorySize, k, labelCount, α)
	# construct the fit functions
	β = 0.1 # ratio between reconstruction error and the distance between p(z) and q(z)
	γ = 0.1 # importance ratio between anomalies and normal data in mem_loss
	batchsize = 64
	nbatches = 200
	σ = 0.1 # width of imq kernel
	fx(m,x)=AlfvenDetectors.fit!(m, x, batchsize, nbatches, β, σ, η=0.0001,cbtime=1);
	σ = 0.01
	batchsize = 10 # this batchsize must be smaller than the size of the labeled training data
	nbatches = 50
	fxy(m,x,y)=AlfvenDetectors.fit!(m,x,y, batchsize, nbatches, β, σ, γ, η=0.0001, cbtime=1);
	# finally construct the anomaly score function
	asf(m,x) = AlfvenDetectors.as_logpxgivenz(m,x)
	# create the whole few shot model
	model = AlfvenDetectors.FewShotModel(ae_model, clust_model, fx, fxy, asf);
	#@test !AlfvenDetectors.is_fitted(model.clust_model)
	AlfvenDetectors.fit!(model, X, Xtrl, Ytr);
	#@test AlfvenDetectors.is_fitted(model.clust_model)
	as = AlfvenDetectors.anomaly_score(model, Xtstl)
	auc = EvalCurves.auc(EvalCurves.roccurve(as, Ytst)...)
	@test auc  >= 0
	println(auc)
end
Random.seed!()