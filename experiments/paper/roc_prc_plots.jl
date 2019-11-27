using PyPlot
using DataFrames
using CSV
using EvalCurves
using PyCall

outpath = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/paper/figs_tables/"

oneclass_f = joinpath(outpath, "roc_prc_oneclass.csv")
oneclass_data = CSV.read(oneclass_f)

twostage_f = joinpath(outpath, "roc_prc_two_stage.csv")
twostage_data = CSV.read(twostage_f)

data = vcat(oneclass_data, twostage_data)

decode(x) = eval(Meta.parse(x))

matplotlib.rc("font", family = "normal",
    weight = "bold",
    size = 14
)
PyCall.PyDict(matplotlib."rcParams")["text.usetex"] = true
PyCall.PyDict(matplotlib."rcParams")["font.family"] = "serif"

# now finally do the plot
labels = [
	"one class Alfvén",
	"one class non-Alfvén",
	"two stage kNN",
	"two stage GMM"
]
figure(figsize=(10,4))
for (i, label) in enumerate(labels)
	subplot(121)
	roc = decode(data[i,:roc])
	plot(roc..., label=label)
	xlabel("FPR")
	ylabel("TPR")

	subplot(122)
	prc = decode(data[i,:prc])
	plot(prc..., label=label)
	xlabel("precision")
	ylabel("recall")
end
legend(frameon=false)
#legend(frameon = false, bbox_to_anchor=[1.05,0.9])
#legend(frameon = false, bbox_to_anchor=[0.4,1.28])
tight_layout()
savefig(joinpath(outpath, "roc_prc_both_models.pdf"))
