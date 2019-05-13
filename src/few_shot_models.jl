mutable struct FewShotModel{AE, C, AS, P}
	ae::AE
	clust_model::C
	asf::AS
	params::P
end

import GenerativeModels: encode, fit!

"""
	encode(FewShotModel,X,args...)

Produce an encoding of X.
"""
encode(m::FewShotModel,X,args...) = Flux.Tracker.data(encode(m.ae,X,args...))

"""
	fit!(FewShotModel,X[,Y,args...];[kwargs...])	

Fit the clustering model. If Y is not specified, the clustering model will
be fit unsupervisedly.
"""
function fit!(m::FewShotModel,X,args...;encoding_batchsize=128,kwargs...)
	Z = encode(m, X, encoding_batchsize)
	fit!(m.clust_model, Z, args...; kwargs...)
end

"""
	anomaly_score(FewShotModel,X,args...;kwargs...)	

Produce the anomaly score given X.
"""
function anomaly_score(m::FewShotModel,X,args...;encoding_batchsize=128,kwargs...)
	Z = encode(m, X, encoding_batchsize)
	return m.asf(m.clust_model, Z, args...; kwargs...)
end

#####################
####### GMMS ########
#####################

mutable struct GMMModel{G,N,KW,TL,L}
	GMM::G
	n::N
	kwargs::KW
	train_ll::TL
	train_labels::L
end

"""
	GMMModel(n[,kind];[kwargs...])

Initialize a Gaussian-Mixture model for further use. 

	n = number of clusters
	kind = one of :diag, :full
	method (:kmeans)
    nInit (50)
    nIter (10)
    nFinal (nIter)
    sparse (0)
"""
GMMModel(n::Int; kwargs...) = GMMModel(convert(GMM{Float32},GaussianMixtures.GMM(3,1)), 
	n, kwargs, Array{Float32,2}(undef,0,0), Array{Int64,1}(undef,0))

"""
	fit!(GMMModel, X)

Observations are columns of X.
"""
function fit!(m::GMMModel, X)
	# the data has to be transposed
	m.GMM = GaussianMixtures.GMM(m.n, Array(X'); m.kwargs...)
end

"""
	fit!(GMMModel, X, Y[; refit_gmm])

Observations are columns of X.
"""
function fit!(m::GMMModel, X::AbstractArray, Y::AbstractVector; refit_gmm=true)
	@assert length(Y) == size(X,2)
	if refit_gmm
		# the data has to be transposed
		fit!(m, X)
	end
	# now record the labels and loglikelihoods
	m.train_labels = Y
	m.train_ll = loglikelihood(m, X)
end

"""
	loglikelihood(GMMModel, X)	

Computes the log-likelihood of X in respect to components of the model.
"""
loglikelihood(m::GMMModel, X) = Array(llpg(m.GMM,Array(X'))')

# anomaly scores
"""
	check_fit(GMMModel) 

Check if model was fitted with labels.
"""
is_fitted(m::GMMModel) = !(size(m.train_ll,2) == 0 || length(m.train_labels) == 0)
	
"""
	as_max_ll_mse(GMMModel, X,label)

Given the label, computes the anomaly score for samples in X using distance from the 
maximum of loglikelihoods of training samples with the same label.
"""
function as_max_ll_mse(m::GMMModel, X,label)
	is_fitted(m) ? nothing : error("The model has not been fitted with labels!")
	agg_vec = maximum(m.train_ll[:,m.train_labels.==label], dims=2)
	return vec(-StatsBase.mean((loglikelihood(m,X) .- agg_vec).^2, dims=1))
end

"""
	as_mean_ll_mse(GMMModel, X,label)

Given the label, computes the anomaly score for samples in X using distance from the 
mean of loglikelihoods of training samples with the same label.
"""
function as_mean_ll_mse(m::GMMModel, X,label)
	is_fitted(m) ? nothing : error("The model has not been fitted with labels!")
	agg_vec = StatsBase.mean(m.train_ll[:,m.train_labels.==label], dims=2)
	return vec(-StatsBase.mean((loglikelihood(m,X) .- agg_vec).^2, dims=1))
end

"""
	as_med_ll_mse(GMMModel, X,label)

Given the label, computes the anomaly score for samples in X using distance from the 
median of loglikelihoods of training samples with the same label.
"""
function as_med_ll_mse(m::GMMModel, X,label)
	is_fitted(m) ? nothing : error("The model has not been fitted with labels!")
	agg_vec = StatsBase.median(m.train_ll[:,m.train_labels.==label], dims=2)
	return vec(-StatsBase.mean((loglikelihood(m,X) .- agg_vec).^2, dims=1))
end

"""
	as_ll_maxarg(GMMModel, X, label)

Anomaly score is the loglikelihood of belonging to the component that
other samples with the same label are most likely to belong to.
"""
function as_ll_maxarg(m::GMMModel, X, label)
	is_fitted(m) ? nothing : error("The model has not been fitted with labels!")
	# first get the index of the component that the samples with label are most likely to belong to
	inds = map(x->x[1],argmax(m.train_ll[:,m.train_labels.==label],dims=1))
	suminds = map(x->length(inds[inds.==x]), unique(inds))
	maxind = unique(inds)[argmax(suminds)]
	return vec(loglikelihood(m, X)[maxind,:])
end

#####################
####### KNNS ########
#####################
