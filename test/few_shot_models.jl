using AlfvenDetectors
using Test
using Random
using BSON
using Flux
using GenerativeModels
using GaussianMixtures
using EvalCurves

@testset "few-shot models" begin
	xdim = 2
	N = 10
	X = hcat(randn(xdim,N).-[10;10], randn(xdim,N).+[10;10], randn(xdim,N).+[-10;10])
	Y = vcat(ones(N), zeros(2*N))
	# test separate clustering algorithms
	Nclust = 3
	clust_alg = AlfvenDetectors.GMMModel(Nclust; kind=:diag, method=:kmeans)
	AlfvenDetectors.fit!(clust_alg, X)
	@test size(clust_alg.train_ll) == (0,0)
	@test size(clust_alg.train_labels) == (0,)
	AlfvenDetectors.fit!(clust_alg, X,Y)
	@test size(clust_alg.train_ll) == (Nclust,3*N)
	@test size(clust_alg.train_labels) == (3*N,)

	label = 1
	asmax = AlfvenDetectors.as_max_ll_mse(clust_alg, X, label)
	@test EvalCurves.auc(EvalCurves.roccurve(asmax, Y)...) == 1
	asmean = AlfvenDetectors.as_mean_ll_mse(clust_alg, X, label)
	@test EvalCurves.auc(EvalCurves.roccurve(asmean, Y)...) == 1
	asmed = AlfvenDetectors.as_med_ll_mse(clust_alg, X, label)
	Y[11:20] .= 1
	Y[1:10] .= 0
	AlfvenDetectors.fit!(clust_alg, X,Y)
	asll = AlfvenDetectors.as_ll_maxarg(clust_alg, X, label)
	@test EvalCurves.auc(EvalCurves.roccurve(asll, Y)...) == 1

	# now train the gmm model together with a neural net for autoencoder
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
	fit!(ae_model, X, 100, 500, σ=1, λ=10, verb=true)
	# now add the gmm model for latent space clustering
	clust_model = AlfvenDetectors.GMMModel(Nclust; kind=:diag, method=:kmeans)
	# now create the fewshotmodel
	anomaly_label = 1
	asf(x,m) = AlfvenDetectors.as_ll_maxarg(x,m,anomaly_label)
	params = nothing
	model = AlfvenDetectors.FewShotModel(ae_model, clust_model, asf, params)
	Z = AlfvenDetectors.encode(model, X)
	@test size(Z) == (ldim, N)
	@test !AlfvenDetectors.is_fitted(model.clust_model)
	# next, fit the clustering model
	fit!(model,X) # this will only fit the gmm with all 
	# the available data without any knowledge of labels
	# now create artificial labeled data
	_N = 10
	Xtrl = []
	for i in 1:Nclust
		_px = AlfvenDetectors.GM([px.μ[i]], [px.σ[i]])
		push!(Xtrl, AlfvenDetectors.sample(_px, _N))
	end
	Xtrl = hcat(Xtrl...)
	Y = vcat(zeros(_N), ones(_N), zeros(_N))
	fit!(model, Xtrl, Y; refit_gmm=false)
	@test AlfvenDetectors.is_fitted(model.clust_model)
	Xtstl = []
	for i in 1:Nclust
		_px = AlfvenDetectors.GM([px.μ[i]], [px.σ[i]])
		push!(Xtstl, AlfvenDetectors.sample(_px, _N))
	end
	Xtstl = hcat(Xtstl...)
	as = AlfvenDetectors.anomaly_score(model, Xtstl)
	@test EvalCurves.auc(EvalCurves.roccurve(as, Y)...) == 1

end