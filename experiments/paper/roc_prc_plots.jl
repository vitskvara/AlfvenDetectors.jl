using PyPlot
using DataFrames
using CSV
using EvalCurves
using PyCall

outpath = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/paper/figs_tables/"
oneclass_f = joinpath(outpath, "roc_prc_oneclass.csv")

oneclass_data = CSV.read(oneclass_f)

decode(x) = eval(Meta.parse(x))

matplotlib.rc("font", family = "normal",
    weight = "bold",
    size = 14
)
PyCall.PyDict(matplotlib."rcParams")["text.usetex"] = true
PyCall.PyDict(matplotlib."rcParams")["font.family"] = "serif"

# now finally do the plot
labels = [
	"oneclass MMD Alfvén",
	"oneclass MMD+GAN non-Alfvén",
]
figure(figsize=(10,4))
for (i, label) in enumerate(labels)
	subplot(121)
	roc = decode(oneclass_data[i,:roc])
	plot(roc..., label=label)
	xlabel("FPR")
	ylabel("TPR")

	subplot(122)
	prc = decode(oneclass_data[i,:prc])
	plot(prc..., label=label)
	xlabel("precision")
	ylabel("recall")
end
legend(frameon=false)
#legend(frameon = false, bbox_to_anchor=[1.05,0.9])
#legend(frameon = false, bbox_to_anchor=[0.4,1.28])
tight_layout()
savefig(joinpath(outpath, "roc_prc.pdf"))