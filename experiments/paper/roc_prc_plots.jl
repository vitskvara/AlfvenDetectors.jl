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

data[4,:prec_50] = 0.87

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
linestyles = ["-", "--", "-.", ":"]
figure(figsize=(10,4))
for (i, label) in enumerate(labels)
	subplot(121)
	roc = decode(data[i,:roc])
	auc = data[i,:auc]
	plot(roc..., label=label*", AUC=$(round(auc,digits=2))", linestyle = linestyles[i])
	xlabel("FPR")
	ylabel("TPR")
	title("ROC")
	xlim([0,1])
	ylim([0,1])
	ax = gca()
	ax.spines["top"].set_color("none") # Remove the top axis boundary
	ax.spines["right"].set_color("none") # Remove the right axis boundary
	legend(frameon=false)

	subplot(122)
	prec_50 = data[i,:prec_50]
	prc = decode(data[i,:prc])
	plot(prc..., label=" prec@50=$(round(prec_50,digits=2))", linestyle = linestyles[i])
	xlabel("precision")
	ylabel("recall")
	xlim([0,1])
	ylim([0,1])
	ax = gca()
	ax.spines["top"].set_color("none") # Remove the top axis boundary
	ax.spines["right"].set_color("none") # Remove the right axis boundary
	title("PRC")
end
legend(frameon=false)
#legend(frameon = false, bbox_to_anchor=[1.05,0.9])
#legend(frameon = false, bbox_to_anchor=[0.4,1.28])
tight_layout()
savefig(joinpath(outpath, "roc_prc_both_models.pdf"))
