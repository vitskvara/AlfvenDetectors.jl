include("eval.jl")

path = ARGS[1]
usegpu = (length(ARGS) > 1)

# get the paths
hostname = gethostname()
if hostname == "gpu-node"
	evaldatapath = "/compass/home/skvara/no-backup" 
	datapath = joinpath("/compass/home/skvara/alfven/experiments/oneclass", path)
elseif hostname == "tarbik.utia.cas.cz"
	evaldatapath = "/home/skvara/work/alfven/cdb_data"
	datapath = joinpath("/home/skvara/work/alfven/experiments/oneclass", path)
else
	evaldatapath = "/home/vit/vyzkum/alfven/cdb_data" 
	datapath = joinpath("/home/vit/vyzkum/alfven/experiments/oneclass", path)
end
modelpath = joinpath(datapath, "models")
evalpath = joinpath(datapath, "eval")
mkpath(evalpath)

pz(n)=GenModels.randn_gpu(Float32,8,n)
# now iterate over all models
models = readdir(modelpath)
data = []

for (i,mf) in enumerate(models)
	println("processing $i")
	_data = eval_model(joinpath(modelpath,mf), evaldatapath, usegpu)
	push!(data, _data)
end

df = DataFrame(
	:model=>Any[],
	:channels=>Any[],
	:ldim=>Int[],
	:nepochs=>Int[],
	:seed=>Int[],
	:normalized=>Bool[],
	:neg=>Bool[],
	:λ=>Float64[],
	:γ=>Float64[],
	:σ=>Float64[],
	:β=>Float64[],
	:train_mse=>Float64[],
	:test_mse=>Float64[],
	:test1_mse=>Float64[],
	:test0_mse=>Float64[],
	:test_var=>Float64[],
	:auc_mse=>Float64[],
	:auc_mse_pos=>Float64[],
	:prec_10_mse=>Float64[],
	:prec_10_mse_pos=>Float64[],
	:prec_20_mse=>Float64[],
	:prec_20_mse_pos=>Float64[],
	:prec_50_mse=>Float64[],
	:prec_50_mse_pos=>Float64[],
	:file=>String[]
	)
for row in data
	push!(df, [row[2][:model], row[2][:channels], row[3]["ldimsize"], row[2][:nepochs], row[3]["seed"], !(row[3]["unnormalized"]),
		get(row[3], "normal-negative", false), get(row[3], "lambda", 0), get(row[3], "gamma", 0), get(row[3], "sigma", 0),
		get(row[3], "beta", 0), row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11],
		row[12], row[13], row[14], row[15], row[16], row[1]])
end

# write/read the results
csvf = joinpath(evalpath, "models_eval.csv") 
CSV.write(csvf,df)
