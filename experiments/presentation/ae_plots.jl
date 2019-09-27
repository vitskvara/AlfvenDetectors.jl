using PyPlot
using AlfvenDetectors
using PyCall

# setup
outpath = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/presentation/images"
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
shots = readdir(datapath)
shotno = "10870"
shotf = joinpath(datapath, filter(x->occursin(shotno, x),shots)[1])

# load the data
psd = AlfvenDetectors.readnormlogupsd(shotf,memorysafe=true);
t = AlfvenDetectors.readtupsd(shotf,memorysafe=true);
f = AlfvenDetectors.readfupsd(shotf,memorysafe=true);

# plot params
cmap = "plasma" # colormap
matplotlib.rc("font", family = "normal",
    weight = "bold",
    size = 16
)
PyCall.PyDict(matplotlib["rcParams"])["text.usetex"] = true
PyCall.PyDict(matplotlib["rcParams"])["font.family"] = "serif"

# plot the whole shot
fname = "whole_psd_$(shotno).png"
figure(figsize=(8,4))
pcolormesh(t,f/1e6,psd,cmap=cmap)
xlabel("t [s]")
ylabel("f [MHz]")
title("U-probe PSD")
tight_layout()
savefig(joinpath(outpath, fname),dpi=500)

# now just a small patch
x0 = 1.096
y0 = 0.9
patchsize = 128
xsize=patchsize*4
ysize=patchsize
xinds=t.>=x0
yinds=(f/1e6).>y0
patchpsd=psd[yinds,xinds][1:ysize,1:xsize]
patcht = t[xinds][1:xsize]
patchf = f[yinds][1:ysize]/1e6

# mark it on the original image
fname = "whole_psd_$(shotno)_patchframe.png"
figure(figsize=(8,4))
pcolormesh(t,f/1e6,psd,cmap=cmap)
xlabel("t [s]")
ylabel("f [MHz]")
plot([patcht[1], patcht[end], patcht[end], patcht[1], patcht[1]], 
	[patchf[1], patchf[1], patchf[end], patchf[end], patchf[1]],c="r")
title("U-probe PSD")
tight_layout()
savefig(joinpath(outpath, fname),dpi=500)

# now plot just the patch
fname = "patch_psd_$(shotno).png"
figure(figsize=(8,4))
pcolormesh(patcht,patchf,patchpsd,cmap=cmap)
xlabel("t [s]")
ylabel("f [MHz]")
title("U-probe PSD")
tight_layout()
savefig(joinpath(outpath, fname),dpi=500)

