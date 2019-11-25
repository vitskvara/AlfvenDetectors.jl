using PyPlot
using AlfvenDetectors
using PyCall
using HDF5
using JLD2
using FileIO

# setup
outpath = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/paper/figs_tables"
datapath = "/home/vit/vyzkum/alfven/cdb_data/uprobe_data"
shots = readdir(datapath)
shotno = "10870"
shotf = joinpath(datapath, filter(x->occursin(shotno, x),shots)[1])
rawf = joinpath(dirname(datapath), "raw_signals/$(shotno).h5")

# load the data
psd = AlfvenDetectors.readnormlogupsd(shotf,memorysafe=true);
t = AlfvenDetectors.readtupsd(shotf,memorysafe=true);
f = AlfvenDetectors.readfupsd(shotf,memorysafe=true);
I = AlfvenDetectors.readip(shotf,memorysafe=true);
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

# limit the time and frequency axis
ts = [0.98, 1.3]

# load the algorithm output data
#detector_f = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/paper/figs_tables/detector_data_2019-11-17T19:50:08.jld2"
detector_f = "/home/vit/Dropbox/vyzkum/alfven/iaea2019/paper/figs_tables/detector_data_2019-11-18T15:40:18.jld2"
plot_data = load(detector_f)
scores = plot_data["scores"]
f0 = plot_data["f0"]
plot_t = plot_data["plot_t"]
mf = plot_data["mf"]

# and filter it with a median filter
using Statistics
function median_filter(x,wl)
	y = similar(x, length(x)-wl+1)
	for i in 1:length(y)
		y[i] = median(x[i:i+wl-1])
	end
	y
end

wl = 20
wl2 = Int(wl/2)
med_scores = median_filter(scores,wl)

# plot the raw signal, the whole spectrogram and a patch
fname = "uprobe_data.png"
figure(figsize=(8,6))
subplot(411)
signal_inds = ts[1] .< tsignal .< ts[2] 
plot(tsignal[signal_inds][1:2000:end], signal[signal_inds][1:2000:end], lw=1)
xlim(ts)
ylim([-30,35])
#xlabel("t [s]")
ylabel("I [A]")

# the spectrogram
subplot(412)
tpsd_inds = ts[1] .< t .< ts[2] 
pcolormesh(t[tpsd_inds],f/1e6,psd[:,tpsd_inds],cmap=cmap)
#xlabel("t [s]")
ylabel("f [MHz]")
#title("U-probe PSD")

# the detector signal
subplot(413)
plot(plot_t[wl2:end-wl2], med_scores, label="\$f_0 = 0.9\$ MHz")
xlim(ts)
ylabel("score")
legend(frameon=false)

# a patch
subplot(414)
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


# plot the raw signal, the whole spectrogram and a patch
fname = "uprobe_data_threshold.png"
figure(figsize=(8,6))
subplot(411)
signal_inds = ts[1] .< tsignal .< ts[2] 
plot(tsignal[signal_inds][1:2000:end], signal[signal_inds][1:2000:end], lw=1)
xlim(ts)
ylim([-30,35])
#xlabel("t [s]")
ylabel("I [A]")

# the spectrogram
subplot(412)
tpsd_inds = ts[1] .< t .< ts[2] 
pcolormesh(t[tpsd_inds],f/1e6,psd[:,tpsd_inds],cmap=cmap)
#xlabel("t [s]")
ylabel("f [MHz]")
#title("U-probe PSD")

# the detector signal
subplot(413)
plot(plot_t[wl2:end-wl2], med_scores, label="\$f_0 = 0.9\$ MHz")
xlim(ts)
ylabel("score")
legend(frameon=false)
#thresh = 194
thresh = 281
over_inds = med_scores .> thresh
plot(plot_t[wl2:end-wl2][over_inds], med_scores[over_inds], c = "r")
plot([0.99, 1.17], [thresh, thresh], "k--", lw = 1, alpha=1)

# a patch
subplot(414)
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

