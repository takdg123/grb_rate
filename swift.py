import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from utils import *

def readSwiftData(modelData    = "./swift/best_model.txt",
				  durationData = "./swift/summary_general.txt", 
				  fluxData     = {"PL": "./swift/summary_pow_photon_flux.txt", "CPL": "./swift/summary_cutpow_photon_flux.txt"}, 
				  paraData     = {"PL": "./swift/summary_pow_parameters.txt", "CPL": "./swift/summary_cutpow_parameters.txt"}, 
				  verbose=False):
	
	output = {}
	
	with open(modelData) as swift_m:
		model = []
		i=0
		for m in swift_m:
		    if i < 7:
		        i+=1
		    else:
		        temp = m.split("|")
		        if 'N/A' not in temp[1] and 'N/A' not in temp[2] and '000' not in temp[1]:
		            model.append([temp[1],temp[2][:-1]])
		model = np.asarray(model)        
		output["Model"] = model
		if verbose: print("[1] Data size= {}".format(len(model)))

	with open(durationData) as swift_T90:
		T90 = []
		i=0
		for t90 in swift_T90:
		    if i < 23:
		        i+=1
		    else:
		        temp = t90.split("|")
		        if "N/A" not in temp[1]:
		            if float(temp[1]) in model[:,0].astype("float"):
		                checker = np.asarray(T90)
		                if len(checker)>=1:
		                    if float(temp[1]) not in checker[:,0].astype("float"):
		                        if 'N/A' in temp[8]:
		                            T90.append([temp[1], 1e9, 1e9])
		                        else:
		                            T90.append([temp[1], temp[8], temp[10]])
		                else:
		                    if 'N/A' in temp[8]:
		                        T90.append([temp[1], 1e9, 1e9])
		                    else:
		                        T90.append([temp[1], temp[8], temp[10]])
		
		T90 = np.asarray(T90)        
		output["T90"] = T90
		if verbose: print("[2] Data size= {} ({})".format(len(T90), len(T90)==len(model)))

	with open(fluxData["PL"]) as swift_PL_flx:
		PL_flx = []
		i=0
		for pl_flx in swift_PL_flx:
		    if i < 13:
		        i+=1
		    else:
		        temp = pl_flx.split("|")
		        if "N/A" not in temp[1]:
		            if float(temp[1]) in model[:,0].astype("float"):
		                checker = np.asarray(PL_flx)
		                if len(checker)>=1:
		                    if float(temp[1]) not in checker[:,1].astype("float"):
		                        PL_flx.append(temp)
		                else:
		                    PL_flx.append(temp)
		
		PL_flx = np.asarray(PL_flx)
		output["PL_flux"] = PL_flx
		if verbose: print("[3] Data size= {} ({})".format(len(PL_flx), len(PL_flx)==len(model)))
	
	with open(paraData["PL"]) as swift_PL_para:
		PL_para = []
		i=0
		for pl in swift_PL_para:
		    if i < 21:
		        i+=1
		    else:
		        temp = pl.split("|")
		        if "N/A" not in temp[1]:
		            if float(temp[1]) in model[:,0].astype("float"):
		                checker = np.asarray(PL_para)
		                if len(checker)>=1:
		                    if float(temp[1]) not in checker[:,1].astype("float"):
		                        PL_para.append(temp)
		                else:
		                    PL_para.append(temp)
		        
		PL_para = np.asarray(PL_para)
		output["PL_params"] = PL_para
		if verbose: print("[4] Data size= {} ({})".format(len(PL_para), len(PL_para)==len(model)))


	with open(fluxData["CPL"]) as swift_CPL_flx:
		CPL_flx = []
		i=0
		for cpl_flx in swift_CPL_flx:
		    if i < 13:
		        i+=1
		    else:
		        temp = cpl_flx.split("|")
		        if "N/A" not in temp[1]:
		            if float(temp[1]) in model[:,0].astype("float"):
		                checker = np.asarray(CPL_flx)
		                if len(checker)>=1:
		                    if float(temp[1]) not in checker[:,1].astype("float"):
		                        CPL_flx.append(temp)
		                else:
		                    CPL_flx.append(temp)
		
		CPL_flx = np.asarray(CPL_flx)		        
		output["CPL_flux"] = CPL_flx
		if verbose: print("[5] Data size= {} ({})".format(len(CPL_flx), len(CPL_flx)==len(model)))

	with open(paraData["CPL"]) as swift_CPL_para:
		CPL_para = []
		i=0
		for cpl in swift_CPL_para:
		    if i < 25:
		        i+=1
		    else:
		        temp = cpl.split("|")
		        if "N/A" not in temp[1]:
		            if float(temp[1]) in model[:,0].astype("float"):
		                checker = np.asarray(CPL_para)
		                if len(checker)>=1:
		                    if float(temp[1]) not in checker[:,1].astype("float"):
		                        CPL_para.append(temp)
		                else:
		                    CPL_para.append(temp)
		CPL_para = np.asarray(CPL_para)        
		output["CPL_params"] = CPL_para
		if verbose: print("[6] Data size= {} ({})".format(len(CPL_para), len(CPL_para)==len(model)))
	return output
	
def getSwiftflx(info, eRange = [100, 1000], grbType="short", outDir="./swift/", plotting=False):

	flx = []

	for m, plf, cplf, t90 in zip(info["Model"], info["PL_flux"], info["CPL_flux"], info["T90"]):
	    if float(t90[1])<=2 and grbType=="short":
	        if 'CPL' in m[1]:
	            flx.append([plf[2], plf[5], plf[8], plf[11],
	                        plf[14], plf[17], plf[20]])
	        elif 'PL' in m[1]:
	            flx.append([cplf[2], cplf[5], cplf[8], cplf[11],
	                        cplf[14], cplf[17], cplf[20]])
	    elif float(t90[1])>2 and grbType=="long":
	        if 'CPL' in m[1]:
	            flx.append([plf[2], plf[5], plf[8], plf[11],
	                        plf[14], plf[17], plf[20]])
	        elif 'PL' in m[1]:
	            flx.append([cplf[2], cplf[5], cplf[8], cplf[11],
	                        cplf[14], cplf[17], cplf[20]])
	flx = np.asarray(flx)

	flx_reproduced = []
	params = []
	for m, pl, cpl, t90 in zip(info["Model"], info["PL_params"], info["CPL_params"], info["T90"]):
	    if float(t90[1])<=2 and grbType=="short":
	        if 'CPL' in m[1] and 'N/A' not in cpl[2] and 'N/A' not in cpl[5] and 'N/A' not in cpl[8]:
	            flx_reproduced.append(quad(CUTOFFPL, eRange[0], eRange[1], args=(float(cpl[8]),float(cpl[2]),float(cpl[5])))[0])
	            params.append([float(cpl[2]),float(cpl[5])])
	        elif' PL' in m[1] and 'N/A' not in pl[2] and 'N/A' not in pl[5]:
	            flx_reproduced.append(quad(PL, eRange[0], eRange[1], args=(float(pl[5]), float(pl[2])))[0])
	    elif float(t90[1])>2 and grbType=="long":
	        if 'CPL' in m[1] and 'N/A' not in cpl[2] and 'N/A' not in cpl[5] and 'N/A' not in cpl[8]:
	            flx_reproduced.append(quad(CUTOFFPL, eRange[0], eRange[1], args=(float(cpl[8]),float(cpl[2]),float(cpl[5])))[0])
	            params.append([float(cpl[2]),float(cpl[5])])
	        elif' PL' in m[1] and 'N/A' not in pl[2] and 'N/A' not in pl[5]:
	            flx_reproduced.append(quad(PL, eRange[0], eRange[1], args=(float(pl[5]), float(pl[2])))[0])


	flx_reproduced = np.asarray(flx_reproduced)
	
	if plotting:
		f, ax = plt.subplots(1,1)
		y, x, etc = ax.hist(flx_reproduced, bins = np.logspace(-2, 3, 30), histtype='step', cumulative=-1, label=grbType)
		x = center_pt(x)
		p, cov = curve_fit(distM, np.log10(x[(y>0) * (y<max(y*0.5))]), np.log10(y[(y>0) * (y<max(y*0.5))]), p0=4)
		ax.plot(x, 10**p[0]*x**(-3/2.))
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_ylim(0.9, 200)
		ax.set_xlim(0.01, 30)
		ax.text(0.7, 0.5, r"p$^{-3/2}$", fontsize=12, transform = ax.transAxes)
		ax.set_xlabel(r"Flux ({} - {} keV) [ph. cm$^{{-2}}$ s$^{{-1}}$]".format(int(eRange[0]), int(eRange[1])), fontsize=12)
		ax.set_ylabel("Number of GRBs", fontsize=12)


	if grbType == "short":
		filename = outDir+"Swift_flux_short.npy"
	else:
		filename = outDir+"Swift_flux_long.npy"


	np.save(filename, flx_reproduced)
	print("Swift flux data is saved in {}.".format(filename))