import os
import sys
import math
import pickle

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from scipy.optimize import curve_fit
from scipy import stats
from scipy.integrate import quad
from scipy.interpolate import interp1d

from sklearn.neighbors import KernelDensity

from utils import *

from tqdm.notebook import tqdm, trange



def get_burst_info(GBMData = "./fermi/GBM.fits", grbType="short", plotting=False):
    GRBinfo = fits.open(GBMData)
    bursts = []
    for grbinfo in GRBinfo[1].data:
        if int(grbinfo["TRIGGER_NAME"][2:])<180714086 and grbinfo["Pflx_Best_Fitting_Model"] !='' and grbinfo["T90"]>0:
            bursts.append([grbinfo["TRIGGER_NAME"], grbinfo["T90"], grbinfo["Pflx_Best_Fitting_Model"]])
    bursts=np.asarray(bursts)

    selected_bursts = []
    for n, du, mo in bursts:
        if float(du)<=2 and grbType=="short":
            selected_bursts.append([n, mo])
        elif float(du)>2 and grbType=="long":
            selected_bursts.append([n, mo])
    selected_bursts = np.asarray(selected_bursts)

    if plotting:
        # Duration histogram
        plt.hist(bursts[:,1].astype("float"), bins=np.logspace(np.log10(0.002), np.log10(2), 10)+np.logspace(np.log10(2), np.log10(2000), 10))
        plt.hist(bursts[:,1].astype("float")[bursts[:,1].astype("float")<2], bins=np.logspace(np.log10(0.002), np.log10(2), 10))
        plt.xscale("log")
        plt.axvline(2, color='r')
        plt.xlabel("GBM T$_{90}$")
        plt.ylabel("Counts")
    return selected_bursts

def get_GBM_flux(GBMData="./fermi/GBM.fits", grbType="short", eRange=[50, 300], plotting = False, **kwargs):    
    GRBinfo = fits.open(GBMData)
    flx = []
    for grbinfo in GRBinfo[1].data:
        if int(grbinfo["TRIGGER_NAME"][2:])<180714086 and grbinfo["Pflx_Best_Fitting_Model"] !='' and grbinfo["T90"]>0:
            if grbType=="short" and grbinfo["T90"]>2 : continue
            if grbType=="long"  and grbinfo["T90"]<=2: continue

            if grbinfo["Pflx_Best_Fitting_Model"] == 'PFLX_PLAW':
                flx.append(quad(PL, eRange[0], eRange[1], args=(grbinfo.field("PFLX_PLAW_ampl"),grbinfo.field("PFLX_PLAW_index")))[0])
            elif grbinfo["Pflx_Best_Fitting_Model"] == 'PFLX_COMP':
                flx.append(quad(CUTOFFPL, eRange[0], eRange[1], args=(grbinfo.field("PFLX_COMP_ampl"),grbinfo.field("PFLX_COMP_index"),grbinfo.field("PFLX_COMP_epeak")))[0])
            elif grbinfo["Pflx_Best_Fitting_Model"] == 'PFLX_BAND':
                flx.append(quad(BAND, eRange[0], eRange[1], args=(grbinfo.field("PFLX_BAND_ampl"),grbinfo.field("PFLX_BAND_alpha"),grbinfo.field("PFLX_BAND_beta"), grbinfo.field("PFLX_BAND_epeak")))[0])
            elif grbinfo["Pflx_Best_Fitting_Model"] == 'PFLX_SBPL':
                flx.append(quad(SBPL, eRange[0], eRange[1], args=(grbinfo.field("PFLX_SBPL_ampl"),grbinfo.field("PFLX_SBPL_indx1"),grbinfo.field("PFLX_SBPL_indx2"), grbinfo.field("PFLX_SBPL_brken")))[0])
    
    if plotting:
        distM = lambda x, n: -3/2.*x + n
        y, x, etc = plt.hist(flx, bins=np.logspace(-1, 3, 30), cumulative=-1, histtype='step')
        x = center_pt(x)
        p, cov = curve_fit(distM, np.log10(x[(y>0) * (y<max(y*0.8))]), np.log10(y[(y>0) * (y<max(y*0.8))]), p0=4)
        plt.plot(x, 10**p[0]*x**(-3/2.))
        plt.ylim(0.8,500)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Photon flux", fontsize=12)
        plt.ylabel("Number of GRBs", fontsize=12)
        plt.text(30, 30, r"p$^{-3/2}$", fontsize=12)
    
    return np.asarray(flx)

def get_GBM_params(GBMData="./fermi/GBM.fits", grbType="short", **kwargs):
    # Get parameter correlation
    GRBinfo = fits.open(GBMData)

    params = []
    bursts = get_burst_info(GBMData=GBMData, grbType=grbType)
    for n, mo in bursts:
        if grbType == "short" and mo == 'PFLX_COMP':
            GRBnum = GRBinfo[1].data["TRIGGER_NAME"] == n
            params.append([GRBinfo[1].data["PFLX_COMP_ampl"][GRBnum][0],GRBinfo[1].data["PFLX_COMP_index"][GRBnum][0], GRBinfo[1].data["PFLX_COMP_epeak"][GRBnum][0],
                            GRBinfo[1].data["PFLX_PLAW_ampl"][GRBnum][0],GRBinfo[1].data["PFLX_PLAW_index"][GRBnum][0]])
        elif grbType=="long" and mo == "PFLX_BAND":
            GRBnum = GRBinfo[1].data["TRIGGER_NAME"] == n
            params.append([GRBinfo[1].data["PFLX_BAND_ampl"][GRBnum][0],GRBinfo[1].data["PFLX_BAND_ALPHA"][GRBnum][0], GRBinfo[1].data["PFLX_COMP_epeak"][GRBnum][0],
                        GRBinfo[1].data["PFLX_BAND_BETA"][GRBnum][0]])

    params = np.asarray(params)
    return params

def get_swift_flux(inputDir = "./swift/", grbType="short", **kwargs):
    flx = np.load(inputDir+"Swift_flux_{}.npy".format(grbType))
    return flx

def fit_logNlogS_curve(grbType="short", minFlux = None, factor = 1./0.7, wbins=[18, 19, 20], plotting=False, **kwargs):
    GBMflx = get_GBM_flux(grbType=grbType, **kwargs)
    Swiftflx = get_swift_flux(grbType=grbType, **kwargs)

    b = np.logspace(-3, 3, 30)
    gy, gx = np.histogram(GBMflx, bins=b)
    gx = center_pt(gx)
    gy = np.cumsum(gy[::-1])[::-1]

    sy, sx = np.histogram(Swiftflx, bins=b)
    sx = center_pt(sx)
    sy = np.cumsum(sy[::-1])[::-1]

    weight_GS = []
    for w in wbins:
        weight_GS.append(np.log10(gy[w]/sy[w]))
    weight_GS = np.average(weight_GS)
    
    p0, cov = curve_fit(distM, np.log10(gx[(gy>0)*(gy<100)]), np.log10(gy[(gy>0)*(gy<100)]), p0=4)
    
    if minFlux is None:
        minFlux = min(Swiftflx)
        
    N = int(10**p0[0]*minFlux**(-3/2.)*factor)
    N_max = int(max(sy*10**weight_GS))

    if plotting:
        plt.hist(GBMflx, bins=b, cumulative=-1, histtype='step', label="GBM GRBs")
        etc = plt.hist(Swiftflx, bins=b, cumulative=-1, histtype='step', label="Swift GRBs")
        plt.step(sx, sy*10**weight_GS, color=etc[2][0].get_edgecolor(), lw=1, ls="--", label="Scaled Swift", where="mid")
        
        plt.plot(gx, 10**p0[0]*gx**(-3/2.), label="Linear-fit")
        plt.plot(gx, 10**p0[0]*gx**(-3/2.)*factor, label="Linear-fit (All sky, x{:.2f})".format(factor))
        #plt.axvline(b[wbins[0]], ls=":")
        #plt.axvline(b[wbins[-1]+1], ls=":")
        if grbType == "long":
            plt.ylim(0.8,2000000)
        else:
            plt.ylim(0.8,10000)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Photon flux (50 - 300 keV) [ph. cm$^{-2}$ s$^{-1}$]", fontsize=12)
        plt.ylabel("Number of GRBs (in 10 yrs)", fontsize=12)
        plt.axhline(N, label="Minimum GRBs", ls=":", color="r")
        plt.text(1e3, N*0.9, "N = {}".format(N), va="top", ha="right")
        plt.text(1e3, N_max*0.9, "N = {}".format(N_max), va="top", ha="right")
        plt.axvline(minFlux, label="Minimum flux", ls="--", color="k")
        plt.text(minFlux*2, 10, r"Flux$_{{min}}$ = {:.1e}".format(minFlux), va="top", ha="left")
        plt.axhline(N_max, label="Maximum GRBs", ls=":", color="orange")
        plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

        
    output = {"N_0": p0[0], "N_max": N, "N_min":N_max, "w": weight_GS}
    return output

def generate_params(N, GBMData="./fermi/GBM.fits", grbType="short"):

    gbm_params = get_GBM_params(GBMData=GBMData, grbType=grbType)
    X = []
    if grbType == "short":
        for alp, ep in zip(gbm_params[:,1],np.log10(gbm_params[:,2])):
            X.append([alp, ep])
    elif grbType == "long":    
        for alp, ep, beta in zip(gbm_params[:,1],np.log10(gbm_params[:,2]), gbm_params[:,3]):
            X.append([alp, ep, beta])
    
    # Note that bandwidth from alpha, beta, and Ep are similar.
    kde = KernelDensity(kernel='gaussian', bandwidth=np.std(gbm_params[:,1])*1.06*len(gbm_params[:,1])**(-1/5.)).fit(X)
    params = kde.sample(N)

    return params

def generate_bursts(logNlogS=None, grbType="short", plotting=False, verbose=True, **kwargs):
    if logNlogS is None:
        logNlogS = fit_logNlogS_curve(grbType=grbType, **kwargs)

    p0 = logNlogS["N_0"]
    if grbType == "short":
        N = logNlogS["N_max"]
    elif grbType == "long":
        N = logNlogS["N_max"]

    rawflx = 1./np.random.power(1.5, N)
    y, x = np.histogram(rawflx/100., np.logspace(np.log10(min(rawflx))-3, np.log10(max(rawflx))-2, 100))
    x = center_pt(x)
    y = np.cumsum(y[::-1])[::-1]
    p_raw, cov = curve_fit(distM, np.log10(x[(y>0)*(y<max(y))]), np.log10(y[(y>0)*(y<max(y))]), p0=4)
    p_raw = p_raw[0]
    flux = (rawflx/100.)*10**(-2/3.*(p_raw-p0))    
    params = generate_params(N, grbType=grbType, **kwargs)

    norm = []
    for i, (f, par) in enumerate(tqdm(zip(flux, params), total=len(flux)) if verbose else zip(flux, params)):
        alpha = par[0]
        ep = 10**par[1]
        if grbType == "short":
            norm.append(f/quad(CUTOFFPL, 50, 300, args=(1, alpha, ep))[0])
        elif grbType == "long":
            beta = par[2]
            norm.append(f/quad(BAND, 50, 300, args=(1, alpha, beta, ep))[0])

    norm = np.asarray(norm)

    if grbType == "short":
        table = Table([flux, np.log10(norm), params[:,0], params[:,1]], names=("flux", "log10(N)", "alpha", "log10(Ep)"))
    elif grbType == "long":
        table = Table([flux, np.log10(norm), params[:,0], params[:,1], params[:,2]], names=("flux", "log10(N)", "alpha", "log10(Ep)", "beta"))

    if plotting:
    
        GBMflx = get_GBM_flux(grbType=grbType, **kwargs)
        Swiftflx = get_swift_flux(grbType=grbType, **kwargs)
        gbm_params = get_GBM_params(grbType=grbType, **kwargs)

        f, ax = plt.subplots(1, 3 if grbType == "short" else 4, figsize=(15, 4))
        b = np.logspace(-3, 3, 40)
        ax[0].hist(GBMflx, bins=b, cumulative=-1, histtype='step', label="GBM GRBs")
        sy, sx, etc = ax[0].hist(Swiftflx, bins=b, cumulative=-1, histtype='step', label="Swift GRBs")
        sx = center_pt(sx)
        ax[0].step(sx, sy*10**logNlogS["w"], color=etc[0].get_edgecolor(), lw=1, ls="--", label="Scaled Swift", where="mid")
        checker, unused, etc = ax[0].hist(table["flux"], bins=b, cumulative=-1, histtype='stepfilled', lw=2, label="Synthesized", zorder=-1, alpha=0.5)
        if checker[0] < N: print("Error")
            
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("Photon flux (50 - 300 keV) [ph. cm$^{-2}$ s$^{-1}$]", fontsize=12)
        ax[0].set_ylabel("Number of GRBs", fontsize=12)
        ax[0].legend(fontsize=12)

        ax[1].hist(gbm_params[:,1], bins=np.linspace(-1.5, 1, 20), density=True, histtype="step", label="GBM GRBs")
        ax[1].hist(table["alpha"], bins=np.linspace(-1.5, 1, 50), density=True, zorder=-1, color=etc[0].get_facecolor(), label="Synthesized", alpha=0.5)
        ax[1].set_xlabel("Low energy index", fontsize=12)
        ax[1].set_ylabel("Occurance rate", fontsize=12)
        ax[1].axvline(np.average(table["alpha"]), label="Ave: {:.2f}".format(np.average(table["alpha"])), color="r")
        ax[1].legend(fontsize=12)

        ax[2].hist(gbm_params[:,2], bins=np.logspace(1.5, 4, 20), density=True, histtype="step", label="GBM GRBs" )
        ax[2].hist(10**table["log10(Ep)"], bins=np.logspace(1.5, 4, 50), density=True, zorder=-1, color=etc[0].get_facecolor(), label="Synthesized", alpha=0.5)
        ax[2].set_xscale("log")
        ax[2].set_xlabel("Peak energy [keV]", fontsize=12)
        ax[2].set_ylabel("Occurance rate", fontsize=12)
        ax[2].axvline(10**np.average(table["log10(Ep)"]), label="Ave: {:.0f}".format(10**np.average(table["log10(Ep)"])), color='r')
        ax[2].legend(fontsize=12)

        if grbType == "long":
            ax[3].hist(gbm_params[:,3], bins=np.linspace(-3, -1.5, 20), density=True, histtype="step", label="GBM GRBs" )
            ax[3].hist(table["beta"], bins=np.linspace(-3, -1.5, 20), density=True, zorder=-1, color=etc[0].get_facecolor(), label="Synthesized", alpha=0.5)
            ax[3].set_xlabel("High energy index", fontsize=12)
            ax[3].set_ylabel("Occurance rate", fontsize=12)
            ax[3].axvline(np.average(table["beta"]), label="Ave: {:.2f}".format(np.average(table["beta"])), color="r")
            ax[3].legend(fontsize=12)
    
    return table

def effective_area(file, plotting = False, **kwargs):

    eff = np.genfromtxt(file)
    eff = eff[np.isfinite(eff[:,1])]
    temp = center_pt(eff[:,1])
    ave_eff = np.asarray([eff[:,0][:-1][temp>0], eff[:,0][1:][temp>0], temp[temp>0]]).T
        
    if plotting:
        label = file.split("/")
        plt.plot(eff[:,0], eff[:,1], label=label[-1])
        plt.xscale("log")
        plt.xlabel("Energy [GeV]")
        plt.yscale("log")
        plt.ylabel(r"Effective area [cm$^2$]")
        plt.legend()
        
    return ave_eff

def bkg_from_ROOT(file, eLowEdges, eHighEdges, evttype = "UC", plotting = False, **kwargs):
    try:
        import ROOT
    except:
        print("pyROOT should be installed.")
        return
    f = ROOT.TFile(file)
    if evttype not in ["UC", "TC", "P"]:
        print("[Error] The entered event type is wrong. options: 'evttype' = 'UC', 'TC', or 'P'.")
        return

    bkg = []
    tree = f.Get("totalBG_"+evttype)
    for l, h in zip(eLowEdges, eHighEdges):
        li = tree.FindBin(l*1000)
        hi = tree.FindBin(h*1000)
        bkg.append(tree.Integral(li, hi, "width"))
    bkg = np.asarray(bkg)
    
    if plotting:
        c = ROOT.TCanvas()
        f.totalBG_UC.DrawClone("hist")
        f.totalBG_TC.SetLineColor(2)
        f.totalBG_TC.DrawClone("hist same")
        f.totalBG_P.SetLineColor(3)
        f.totalBG_P.DrawClone("hist same")
        c.SetLogx()
        c.SetLogy()
        c.SetRightMargin(0.09);
        c.SetLeftMargin(0.15);
        c.SetBottomMargin(0.15);
        c.Draw()
    
    if plotting:
        return bkg, c, f
    else:
        return bkg

def make_table(eff, bkg, **kwargs):
    if type(eff) == str:
        eff = effective_area(eff, **kwargs)
    if type(bkg) == str:
        bkg = bkg_from_ROOT(bkg, eff[:,0], eff[:,1], **kwargs)
    table = Table(np.vstack((eff.T, bkg)).T,  names = ("E_low", "E_high", "EA", "background"))
    table["E_low"].unit = u.MeV
    table["E_high"].unit = u.MeV
    table["EA"].unit = u.cm**2
    table["background"].unit = u.ph/u.s
    return table

def dict_to_table(input_dict):
    rows = []
    for key in input_dict.keys():
        if np.size(input_dict[key].value) == 1:
            rows.append([input_dict[key].value])
        else:
            rows.append(input_dict[key].value)
    table = Table(rows, names = input_dict.keys())
    for key in input_dict.keys():
        table[key].unit = input_dict[key].unit
    return table

def get_flux_dist(output, snr_cut = 6.5, detected=False):
    y_tot = []
    for i, out in enumerate(output):
        if detected:
            y, x = np.histogram(out["flux"]*(out["snr"]>snr_cut), np.logspace(-5, 5, 200))
        else:
            y, x = np.histogram(out["flux"], np.logspace(-5, 5, 200))
        cumy = y[::-1].cumsum()[::-1]
        y_tot.append(cumy)
    x_ctr = center_pt(x)
    y_tot = np.asarray(y_tot).T
    y_out = []
    FoV = output[0]["FoV"]
    for i, y in enumerate(y_tot):
        y_out.append([np.percentile(y, 50)*FoV, 
                      np.percentile(y, 16)*FoV, 
                      np.percentile(y, 84)*FoV])
    y_out = np.asarray(y_out)
    return y_out, x_ctr


def plotRateHist(rate, ax = None, label = None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if label is None:
        label = "{:.1f} +/- {:.1f}".format(np.average(rate), np.std(rate))

    y, x, etc = ax.hist(rate, histtype="step", label=label)
    x = center_pt(x)
    p, cov = curve_fit(gaus, x, y, p0=(20, np.average(rate), 4))
    fitx = np.linspace(min(rate)/2, max(rate)*2, 100)
    ax.plot(fitx, gaus(fitx, *p), color=etc[0].get_edgecolor(), ls=":")
    ax.set_xlabel("GRB rate (GRBs/yr)", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.axvline(p[1], color='k')
    ax.axvline(np.percentile(rate, 16), color=etc[0].get_edgecolor(), ls='--')
    ax.axvline(np.percentile(rate, 84), color=etc[0].get_edgecolor(), ls='--')
    ax.legend()


def plotFluxHist(output, ax=None, snr_cut = 6.5, snr_cuts = None, **kwargs):
    if ax is None:
        ax = plt.gca()

    if snr_cuts is None:
        snr_cuts = [snr_cut]

    if np.size(output) == 1:
        y, x = np.histogram(output["flux"], bins=np.logspace(-3, 3, 100))
        rate = np.cumsum(y[::-1])[::-1]*output["FoV"]
        ax.step(center_pt(x), rate, where="mid", label="All sample")
        ax.set_ylim(1e-2, max(y[:,2])*2)

        for snr_cut in snr_cuts:
            y, x = np.histogram(output["flux"][output["snr"]>snr_cut], bins=np.logspace(-3, 3, 100))
            rate = np.cumsum(y[::-1])[::-1]*output["FoV"]
            etc = ax.step(center_pt(x), rate, where="mid", label="Apply SNR cut (SNR>{})".format(snr_cut))
        
            dist_line = interp1d(rate, center_pt(x), kind='linear')
            cont90 = dist_line(max(rate)*0.9)
            ax.axvline(cont90, ls=":", color=etc[0].get_color(), label = r"90% cont. ({:.1f} ph/cm$^2$/s) ".format(cont90))
    else:
        y, x = get_flux_dist(output)
        ax.step(x, y[:,0], where="mid", label="All sample")
        ax.set_ylim(1e-2, max(y[:,2])*2)

        for snr_cut in snr_cuts:
            y, x = get_flux_dist(output, detected=True, snr_cut=snr_cut)
            etc = ax.step(x, y[:,0], where="mid", label="Apply SNR cut (SNR>{})".format(snr_cut))
            ax.fill_between(x, y[:,1], y[:,2], alpha=0.3, color=etc[0].get_color())
        
            dist_line = interp1d(y[:,0], x, kind='linear')
            cont90 = dist_line(max(y[:,0])*0.9)
            ax.axvline(cont90, ls=":", color=etc[0].get_color(), label = r"90% cont. ({:.1f} ph/cm$^2$/s) ".format(cont90))
            
        output=output[0]
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Photon flux ({:.1f} - {:.1f} MeV) [ph/cm$^2$/s]".format(output["eRange"][0]/1000., output["eRange"][1]/1000.), fontsize=12)
    ax.set_ylabel("Rate [GRBs/yr]", fontsize=12)
    ax.grid()
    if np.size(snr_cuts) > 1:
        ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend(fontsize=12)

def plotEfficiency(output, snr_cut = 6.5):
    flux = output["flux"]
    snr = output["snr"]
    eran = output["eRange"]

    f, ax = plt.subplots(2,1, figsize=(7, 7), gridspec_kw={'height_ratios':[5,1]})
    ys, xs, etc = ax[0].hist(flux, np.logspace(-5, 5, 200), histtype="step", label="All sample")
    yd, xd, etc = ax[0].hist(flux[snr > snr_cut], np.logspace(-5, 5, 200), color=etc[0].get_edgecolor(), label="Apply SNR cut (SNR>6.5)")
    ax[0].set_ylabel("The number of events", fontsize=12)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].grid()
    ax[0].legend(fontsize=12)

    x_ctr = center_pt(xs)
    ys[ys==0] = 1e-20
    eff = yd/ys
    eff = np.nan_to_num(eff)
    eff = np.hstack((eff[x_ctr < 10], np.ones(sum(x_ctr >= 10))))

    ax[1].step(x_ctr, eff, color=etc[0].get_facecolor())
    ax[1].set_xscale("log")
    ax[1].set_ylim(5e-3, 2)
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"$\eta$", fontsize=12)
    ax[1].set_yticks([1e-2, 1e-1, 1])
    ax[1].set_yticklabels(["1%", "10%", "100%"])
    ax[1].grid()
    ax[1].set_xlabel(r"Photon flux ({:.1f} - {:.1f} MeV) [ph/cm$^2$/s]".format(eran[0]/1000., eran[1]/1000.), fontsize=12)



class EstimateRate:

    def __init__(self, inst=None, grbType="short", snr_cut = 6.5, FoV = None, load_status = False, verbose=False, **kwargs):

        if load_status:
            filename = "./output/{}.pickle".format(load_status)
            with open(filename, 'rb') as file:
                self.__dict__.update(pickle.load(file).__dict__)
        else:
            self._grbType = grbType
            self._inst = inst
            
            if FoV is None:
                FoV = 0.25 # defalut (3.14 steradian)

            self._FoV = obs_factor = 1./10.*FoV
            self._verbose = verbose
            self._snr_cut = snr_cut
            self._eRange = kwargs.get("eRange", [200, 1000])

    def create_spectra(self, **kwargs):
        verbose = kwargs.get("verbose", self.verbose)
        if verbose:
            print("[Log] Generating burst spectra")
        return generate_bursts(grbType=self.grbType, verbose=verbose)

    def forward_folding(self, burst_table=None, inst_table=None, calc_flux = True, return_output=False, save_result=False, filename=None, **kwargs):
        verbose = kwargs.get("verbose", self.verbose)
        self._eRange = kwargs.get("eRange", self.eRange)

        if burst_table is None:
            self.burst_table = self.create_spectra(verbose=verbose)
        else:
            self.burst_table = burst_table

        if inst_table is None:
            if self.inst is None:
                print("[Error] The EA and background cannot be loaded.")
                return
            else:
                inst_table = self.inst
            
        conv_factor = [(inst_table["E_low"].unit).to(u.keV),
                       (inst_table["E_high"].unit).to(u.keV),
                       (inst_table["EA"].unit).to(u.cm**2)]
            
        cnts_tot = []
        flx_tot = []

        if verbose:
            print("[Log] Forward folding the burst spectra")
        for burst in tqdm(self.burst_table) if verbose else self.burst_table:
            cnts = []
            norm = 10**burst["log10(N)"]
            alpha = burst["alpha"]
            ep = 10**burst["log10(Ep)"]
            for channel in inst_table:
                elow = channel["E_low"]*conv_factor[0]
                ehigh = channel["E_high"]*conv_factor[1]
                eff = channel["EA"]*conv_factor[2]
                if self.grbType == "short":
                    cnts_per_cm2 = quad(CUTOFFPL, elow, ehigh, args=(norm, alpha, ep))[0]
                elif self.grbType == "long":
                    beta = burst["beta"]
                    cnts_per_cm2 = quad(BAND, elow, ehigh, args=(norm, alpha, beta, ep))[0]
                cnts.append(cnts_per_cm2*eff)
            cnts_tot.append(cnts)
            
            if calc_flux:
                if self.grbType == "short":
                    flx_tot.append(quad(CUTOFFPL, self.eRange[0], self.eRange[1], args=(norm, alpha, ep))[0])
                elif self.grbType == "long":
                    flx_tot.append(quad(BAND, self.eRange[0], self.eRange[1], args=(norm, alpha, beta, ep))[0])

        self.counts = np.asarray(cnts_tot)
        self.background = inst_table["background"]
        self.flux = np.asarray(flx_tot)
        self.snr = self.calc_snr(self.counts, self.background)
        self.rate = self.calc_rate(self.snr, **kwargs)
        self.output = {"counts": self.counts, 
                       "background": self.background, 
                       "flux": self.flux, 
                       "snr": self.snr, 
                       "parameters": self.burst_table, 
                       "rate": self.rate, 
                       "FoV": self.FoV,
                       "eRange": self.eRange}

        if verbose:
            snr_cut = kwargs.get("snr_cut", self.snr_cut)
            print("[Log] Analysis done. The estimated rate is {:.1f} GRBs/yr (SNR > {})".format(self.rate, snr_cut))

        if save_result:
            if filename is None:
                filename = "current.npy"
            elif ".npy" not in filename:
                filename = filename+".npy"
            filename = "./output/"+filename
            np.save(filename, self.output)
            print("[Log] The result is saved in {}".format(filename))

        if return_output:
            return self.output

    def calc_rate(self, snr, **kwargs):
        snr_cut = kwargs.get("snr_cut", self.snr_cut)
        FoV = kwargs.get("FoV", self.FoV)
        if len(np.shape(snr)) == 1:
            return sum(snr>snr_cut)*FoV
        elif len(np.shape(snr)) == 2:
            return sum((snr>snr_cut).T)*FoV

    def calc_snr(self, cnts, bkg):
        return np.asarray([max(s) for s in np.nan_to_num(cnts/np.sqrt(cnts+bkg))])

    def simulation(self, save_status="current", runs=100, inst_table=None, **kwargs):
        if inst_table is None:
            if self.inst is None:
                print("[Error] The EA and background cannot be loaded.")
                return
            else:
                inst_table = self.inst

        output = []
        rate_sim = []
        for i in trange(runs):
            out = self.forward_folding(inst_table=inst_table, return_output=True, **kwargs)
            output.append(out)
            rate_sim.append(out["rate"])

        self.output_sim = np.asarray(output)
        self.rate_sim = np.asarray(rate_sim)

        filename = "./output/{}.pickle".format(save_status)
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print("[Log] Analysis Done (status: '{}'). The estimated rate is {:.1f} +/- {:.1f} GRBs/yr".format(save_status, np.average(self.rate_sim), np.std(self.rate_sim)))

    @property
    def grbType(self):
        return self._grbType
    
    @property
    def inst(self):
        return self._inst
    
    @property
    def FoV(self):
        return self._FoV
    
    @property
    def snr_cut(self):
        return self._snr_cut

    @property
    def eRange(self):
        return self._eRange

    @property
    def verbose(self):
        return self._verbose
    




