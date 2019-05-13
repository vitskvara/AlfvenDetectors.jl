"""
    KNNAnomaly(V)

Implements all three variants of the anomaly detectors based on k-nearest neighbors from
From outliers to prototypes: Ordering data,
Stefan Harmelinga and Guido Dornhegea and David Tax and Frank Meinecke and Klaus-Robert Muller, 2005

    V = :kappa - anomaly score is the distance between x and its kth nearest neighbor
    V = :gamma - anomaly score is the average distance of x and its k-nearest neighbors
    V = :delta - anomaly score is the norm of the mean of vectors pointing from x to its k-nearest neighbors
"""
struct KNNAnomaly{V<:Val}
    t::NNTree
    X::Matrix
    v::V
end

"""
    function KNNAnomaly(X::Matrix, v::Symbol, tree_type::Symbol = :BruteTree)

    create the k-nn anomaly detector with variant v::Symbol

"""
KNNAnomaly(X::Matrix, v::Symbol, tree_type::Symbol = :BruteTree) = KNNAnomaly(eval(tree_type)(X), X, Val(v))

"""
    anomaly score is the distance between x and its kth nearest neighbor

"""
function StatsBase.predict(model::KNNAnomaly, x, k, v::V) where {V<:Val{:kappa}}
    inds, dists = NearestNeighbors.knn(model.t, x, k,true)
    map(d -> d[end],dists)
end

"""
    anomaly score is the average distance of x and its k-nearest neighbors

"""
function StatsBase.predict(model::KNNAnomaly, x, k, v::V) where {V<:Val{:gamma}}
    inds, dists = NearestNeighbors.knn(model.t, x, k)
    map(d -> Statistics.mean(d),dists)
end

"""
    anomaly score is the norm of the mean of vectors pointing from x to its k-nearest neighbors

"""
function StatsBase.predict(model::KNNAnomaly, x, k, v::V) where {V<:Val{:delta}}
    inds, dists = NearestNeighbors.knn(model.t, x, k)
    map(i -> LinearAlgebra.norm(x[:,i[1]] - Statistics.mean(model.X[:,i[2]],dims=2)) ,enumerate(inds))
end

StatsBase.predict(model::KNNAnomaly, x, k, v::Symbol) = StatsBase.predict(model, x, k, Val(v))
StatsBase.predict(model::KNNAnomaly, x, k) = StatsBase.predict(model, x, k, model.v)
