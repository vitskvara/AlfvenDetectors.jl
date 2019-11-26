"""
	FewShotModel(autoencoder, clustering_model, anomaly_score_function, fit_function)

A structure for general semi-supervised training via clustering model on latent
space produced by the autoencoder.
"""
mutable struct FewShotModel{AE, C, FX, FXY, AS}
	ae::AE
	clust_model::C
	fitx::FX
	fitxy::FXY
	asf::AS
end

import GenerativeModels: encode_untracked, fit!

"""
	encode(FewShotModel,X,args...)

Produce an encoding of X.
"""
encode(m::FewShotModel,X,args...) = encode_untracked(m.ae,X,args...)
# normalize the data here?

"""
	fit!(FewShotModel,ff, X[, Y, args...][; encoding_batchsize, kwargs...])	

Fit the clustering model. If Y is not specified, the clustering model will
be fit unsupervisedly.
"""
function fit!(m::FewShotModel,ff,X::AbstractArray, args...;encoding_batchsize::Int=128)
	Z = encode(m, X, encoding_batchsize)
	# normalize the data here?
	ff(m.clust_model, Z, args...)
end
fitx!(m::FewShotModel,X::AbstractArray;kwargs...) = fit!(m,m.fitx,X;kwargs...)
fitxy!(m::FewShotModel,X::AbstractArray, Y::AbstractVector;kwargs...) = fit!(m,m.fitxy,X,Y;kwargs...)
function fit!(m::FewShotModel,X_unlabeled::AbstractArray,X_labeled::AbstractArray,Y::AbstractVector;kwargs...)
	fitx!(m,X_unlabeled;kwargs...)
	fitxy!(m,X_labeled,Y;kwargs...)
end

"""
	anomaly_score(FewShotModel[,anomaly_score_function],X,args...;kwargs...)	

Produce the anomaly score given X. If anomaly_score_function is not given,
the internal one is going to be used.
"""
function anomaly_score(m::FewShotModel,X::AbstractArray,args...;encoding_batchsize=128,kwargs...)
	Z = encode(m, X, encoding_batchsize)
	# normalize the data here?
	return m.asf(m.clust_model, Z, args...; kwargs...)
end
function anomaly_score(m::FewShotModel,asf,X::AbstractArray,args...;encoding_batchsize=128,kwargs...)
	Z = encode(m, X, encoding_batchsize)
	return asf(m.clust_model, Z, args...; kwargs...)
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
function fit!(m::GMMModel, X; refit=nothing)
	# the data has to be transposed
	m.GMM = GaussianMixtures.GMM(m.n, Array(Float32.(X)'); m.kwargs...)
end

"""
	fit!(GMMModel, X, Y[; refit])

Observations are columns of X.
"""
function fit!(m::GMMModel, X::AbstractArray, Y::AbstractVector; refit=false)
	@assert length(Y) == size(X,2)
	if refit
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
	# first get the index of the component that the samples with label are most likely to 
	# belong to
	inds = map(x->x[1],argmax(m.train_ll[:,m.train_labels.==label],dims=1))
	suminds = map(x->length(inds[inds.==x]), unique(inds))
	maxind = unique(inds)[argmax(suminds)]
	return vec(loglikelihood(m, X)[maxind,:])
end

#####################
####### KNNS ########
#####################

"""
    KNN

Implements k-nearest-neighbor algorithm for semi-supervised anomaly detection. 
"""
mutable struct KNN
    tree::NNTree
    X::Matrix
    Y::Vector
    tree_type::Symbol
end

"""
    KNN(type)

Create the KNN anomaly detector with tree of type T.
"""
KNN(tree_type::Symbol = :BruteTree) = KNN(eval(tree_type)(Array{Float32,2}(undef,1,0)), 
	Array{Float32,2}(undef,1,0), Array{Int64,1}(undef,0), tree_type)


"""
	fit!(KNN, X)

Observations are columns of X.
"""
function fit!(m::KNN, X)
	# the data has to be transposed
	m.tree = eval(m.tree_type)(X)
	m.X = copy(X)
end

"""
	fit!(KNN, X, Y)

Observations are columns of X.
"""
function fit!(m::KNN, X::AbstractArray, Y::AbstractVector)
	@assert length(Y) == size(X,2)
	fit!(m, X)
	# now record the labels
	m.Y = copy(Y)
end

"""
	check_fit(KNN) 

Check if model was fitted with labels.
"""
is_fitted(m::KNN) = !(size(m.X,2) == 0 || length(m.Y) == 0)


"""
	as_mean(KNN, X, k)

Anomaly score is the average labels of the k--nearest neighbors. This works under 
the condition that anomalous samples are labeled with a higher number 
(e.g. 0 for normal and 1 for anomalous).
"""
function as_mean(m::KNN, X, k::Int)
	is_fitted(m) ? nothing : error("The model has not been fitted with labels!")
	inds, dists = NearestNeighbors.knn(m.tree, X, k,true)
	# now get the labels of the nearest neighbors 
	ys = map(x->m.Y[x],inds)
	return map(x->StatsBase.mean(x), ys)
end


"""
	as_mean_weighted(KNN, X, k)

Anomaly score is the average labels of the k--nearest neighbors weighted by their distance.
This works under the condition that anomalous samples are labeled with a higher number 
(e.g. 0 for normal and 1 for anomalous).
"""
function as_mean_weighted(m::KNN, X, k::Int)
	is_fitted(m) ? nothing : error("The model has not been fitted with labels!")
	inds, dists = NearestNeighbors.knn(m.tree, X, k,true)
	# now get the labels of the nearest neighbors 
	ys = map(x->m.Y[x],inds)
	return map(x->StatsBase.mean(x[1],Weights(x[2]/sum(x[2]))), zip(ys,dists))
end
