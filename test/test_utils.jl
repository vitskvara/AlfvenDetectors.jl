paramchange(frozen_params, params) = 
	map(x-> x[1] != x[2], zip(frozen_params, params))
sim(x,y,δ=1e-6) = abs(x-y) < δ
