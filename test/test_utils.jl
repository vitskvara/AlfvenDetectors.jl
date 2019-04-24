paramchange(frozen_params, m) = 
	map(x-> x[1] != x[2], zip(frozen_params, collect(params(m))))
getparams(m) =  map(x->copy(Flux.Tracker.data(x)), collect(params(m)))
sim(x,y,δ=1e-6) = abs(x-y) < δ
