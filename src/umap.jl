"""
	UMAP(dim[; n_neighbors, min_dist, metric, kwargs...])

Initialize the UMAP object.

	n_neighbors [15] - low values focus on local neighborhood and vice versa 
	mind_dist [0.1] - how tightly packed together is the output space
"""
function UMAP(dim, args...; kwargs...)
	_init_umap()
	(umap == PyNULL()) ? (return nothing) : nothing 
	return umap.UMAP(args...; n_components=dim, kwargs...)
end

"""
	fit!(UMAP, X)

Fit the UMAP model and return transformed X.	
"""
fit!(m::PyObject, X) = Array(m.fit_transform(X')')

"""
	transform(UMAP, X)

Transform X using a fittted UMAP model.
"""
transform(m::PyObject, X) =  Array(m.transform(X')')