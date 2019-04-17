using AlfvenDetectors
using DelimitedFiles

patchsize = Int(Meta.parse(ARGS[1]))
i = Int(Meta.parse(ARGS[2]))
hostname = gethostname()
if hostname == "vit-ThinkPad-E470"
	datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
	savepath = "/home/vit/vyzkum/alfven/experiments/conv/labeled_patches/$patchsize"
elseif hostname == "tarbik.utia.cas.cz"
	datapath = "/home/skvara/work/alfven/cdb_data/uprobe_data"
	savepath = "/home/skvara/work/alfven/experiments/conv/labeled_patches/$patchsize"
elseif hostname == "soroban-node-03"
#	datapath = "/compass/Shared/Exchange/Havranek/Link to Alfven"
	datapath = "/compass/home/skvara/no-backup/uprobe_data"
	savepath = "/compass/home/skvara/alfven/experiments/conv/labeled_patches/$patchsize"
end
mkpath(savepath)
shots, labels, tstarts, fstarts = AlfvenDetectors.labeled_patches()

function load_save_patch(shotno, label, tstart, fstart, patchsize, inpath, outpath)
	data,t,f = AlfvenDetectors.get_patch(inpath, shotno, tstart, fstart, patchsize, AlfvenDetectors.readnormlogupsd)
	filename = "$shotno-$label-$tstart-$fstart.csv"
	filename = joinpath(outpath, filename)
	open(filename, "w") do file
		writedlm(file,data,',')
	end
	println("written file $filename")
end
batchsize = 50
L = length(shots)
map(x->load_save_patch(x[1], x[2], x[3], x[4], patchsize, datapath, savepath), 
	zip(shots[(i-1)*batchsize+1:min(i*batchsize,L)], labels[(i-1)*batchsize+1:min(i*batchsize,L)], 
		tstarts[(i-1)*batchsize+1:min(i*batchsize,L)], fstarts[(i-1)*batchsize+1:min(i*batchsize,L)]))
