using PyPlot
using AlfvenDetectors
using PyCall
using HDF5

# setup
outpath = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/paper/figs"
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
shots = readdir(datapath)
shotno = "10870"
shotf = joinpath(datapath, filter(x->occursin(shotno, x),shots)[1])
rawf = joinpath(dirname(datapath), "raw_signals/$(shotno).h5")

# load the data
psd = AlfvenDetectors.readnormlogupsd(shotf,memorysafe=true);
t = AlfvenDetectors.readtupsd(shotf,memorysafe=true);
f = AlfvenDetectors.readfupsd(shotf,memorysafe=true);
signal = h5read(rawf, "f")
tsignal = h5read(rawf, "t")/1000
sinds = minimum(t) .<= tsignal .<= maximum(t)
signal = signal[sinds]
tsignal = tsignal[sinds]

# plot params
cmap = "plasma" # colormap
matplotlib.rc("font", family = "normal",
    weight = "bold",
    size = 16
)
PyCall.PyDict(matplotlib."rcParams")["text.usetex"] = true
PyCall.PyDict(matplotlib."rcParams")["font.family"] = "serif"

# plot the raw signal, the whole spectrogram and a patch
fname = "uprobe_data.png"
figure(figsize=(8,4))
subplot(311)
#title("U probe signal - DUMMY PLOT, REPLACE WITH REAL DATA")
plot(tsignal, signal, lw=0.4)
xlim([minimum(tsignal), maximum(tsignal)])
#xlabel("t [s]")
ylabel("I [A]")
# we dont have the raw data - find it

# the spectrogram
subplot(312)
pcolormesh(t,f/1e6,psd,cmap=cmap)
#xlabel("t [s]")
ylabel("f [MHz]")
#title("U-probe PSD")

# a patch
subplot(313)
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

pcolormesh(patcht,patchf,patchpsd,cmap=cmap)
xlabel("t [s]")
ylabel("f [MHz]")

tight_layout(h_pad = 0.1)

savefig(joinpath(outpath, fname),dpi=500)
