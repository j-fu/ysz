import numpy as np
import matplotlib.pyplot as plt
import time   
import sys           
import os
import csv
from scipy.optimize import curve_fit
import scipy	
import math


####### BOOOLS
results_plot_bool = True
debug_plot_bool = False
plot_save_bool = True
data_save_bool = True
print_bool = False
#############

#file_label = "i-e_BP100.0_gA0.0_gO-1.60217662e-20_C10.0_MA"+sys.argv[1]+"_MO"+sys.argv[2]+"_1D-v100-m01-2049"
#file_label = "i-e_BP100.0_gA0.0_gO-1.60217662e-20_C10.0_MA"+sys.argv[1]+"_MO"+sys.argv[2]+"_jR"+sys.argv[3]+"_1D-v100-467"
#file_label = "EIS"
file_label = sys.argv[1]

input_directory = "results/EIS_aux_data"
fig_dir 	= "results/EIS_FIGURES"
data_dir 	= "results/EIS_DATA"


#freq = "1.0E+0", "2.0E+0", "3.0E+0", "4.0E+0", "5.0E+0", "6.0E+0", "7.0E+0", "8.0E+0", "9.0E+0", "1.0E+1", "2.0E+1", "3.0E+1", "4.0E+1", "5.0E+1", "6.0E+1", "7.0E+1", "8.0E+1", "9.0E+1", "1.0E+2", "2.0E+2", "3.0E+2", "4.0E+2", "5.0E+2", "6.0E+2", "7.0E+2", "8.0E+2", "9.0E+2", "1.0E+3", "2.0E+3", "3.0E+3", "4.0E+3", "5.0E+3", "6.0E+3", "7.0E+3", "8.0E+3", "9.0E+3", "1.0E+4", "2.0E+4", "3.0E+4", "4.0E+4", "5.0E+4", "6.0E+4", "7.0E+4", "8.0E+4", "9.0E+4", "1.0E+5", "2.0E+5", "3.0E+5", "4.0E+5", "5.0E+5", "6.0E+5", "7.0E+5", "8.0E+5", "9.0E+5", "1.0E+6", "2.0E+6", "3.0E+6", "4.0E+6", "5.0E+6", "6.0E+6", "7.0E+6", "8.0E+6", "9.0E+6"


#freq = "1.0E-6", "2.0E-6", "3.0E-6", "4.0E-6", "5.0E-6", "6.0E-6", "7.0E-6", "8.0E-6", "9.0E-6", "1.0E-5", "2.0E-5", "3.0E-5", "4.0E-5", "5.0E-5", "6.0E-5", "7.0E-5", "8.0E-5", "9.0E-5", "1.0E-4", "2.0E-4", "3.0E-4", "4.0E-4", "5.0E-4", "6.0E-4", "7.0E-4", "8.0E-4", "9.0E-4", "1.0E-3", "2.0E-3", "3.0E-3", "4.0E-3", "5.0E-3", "6.0E-3", "7.0E-3", "8.0E-3", "9.0E-3", "1.0E-2", "2.0E-2", "3.0E-2", "4.0E-2", "5.0E-2", "6.0E-2", "7.0E-2", "8.0E-2", "9.0E-2", "1.0E-1", "2.0E-1", "3.0E-1", "4.0E-1", "5.0E-1", "6.0E-1", "7.0E-1", "8.0E-1", "9.0E-1", "1.0E+0", "2.0E+0", "3.0E+0", "4.0E+0", "5.0E+0", "6.0E+0", "7.0E+0", "8.0E+0", "9.0E+0", "1.0E+1", "2.0E+1", "3.0E+1", "4.0E+1", "5.0E+1", "6.0E+1", "7.0E+1", "8.0E+1", "9.0E+1", "1.0E+2", "2.0E+2", "3.0E+2", "4.0E+2", "5.0E+2", "6.0E+2", "7.0E+2", "8.0E+2", "9.0E+2", "1.0E+3", "2.0E+3", "3.0E+3", "4.0E+3", "5.0E+3", "6.0E+3", "7.0E+3", "8.0E+3", "9.0E+3", "1.0E+4", "2.0E+4", "3.0E+4", "4.0E+4", "5.0E+4", "6.0E+4", "7.0E+4", "8.0E+4", "9.0E+4", "1.0E+5", "2.0E+5", "3.0E+5", "4.0E+5", "5.0E+5", "6.0E+5", "7.0E+5", "8.0E+5", "9.0E+5", "1.0E+6", "2.0E+6", "3.0E+6", "4.0E+6", "5.0E+6", "6.0E+6", "7.0E+6", "8.0E+6", "9.0E+6"

#freq = "1.0E+0", "2.0E+0", "3.0E+0", "4.0E+0"

freq = "1.0e-06","1.3e-06","1.6e-06","2.0e-06","2.5e-06","3.2e-06","4.0e-06","5.0e-06","6.3e-06","7.9e-06","1.0e-05","1.3e-05","1.6e-05","2.0e-05","2.5e-05","3.2e-05","4.0e-05","5.0e-05","6.3e-05","7.9e-05","1.0e-04","1.3e-04","1.6e-04","2.0e-04","2.5e-04","3.2e-04","4.0e-04","5.0e-04","6.3e-04","7.9e-04","1.0e-03","1.3e-03","1.6e-03","2.0e-03","2.5e-03","3.2e-03","4.0e-03","5.0e-03","6.3e-03","7.9e-03","1.0e-02","1.3e-02","1.6e-02","2.0e-02","2.5e-02","3.2e-02","4.0e-02","5.0e-02","6.3e-02","7.9e-02","1.0e-01","1.3e-01","1.6e-01","2.0e-01","2.5e-01","3.2e-01","4.0e-01","5.0e-01","6.3e-01","7.9e-01","1.0e+00","1.3e+00","1.6e+00","2.0e+00","2.5e+00","3.2e+00","4.0e+00","5.0e+00","6.3e+00","7.9e+00","1.0e+01","1.3e+01","1.6e+01","2.0e+01","2.5e+01","3.2e+01","4.0e+01","5.0e+01","6.3e+01","7.9e+01","1.0e+02","1.3e+02","1.6e+02","2.0e+02","2.5e+02","3.2e+02","4.0e+02","5.0e+02","6.3e+02","7.9e+02","1.0e+03","1.3e+03","1.6e+03","2.0e+03","2.5e+03","3.2e+03","4.0e+03","5.0e+03","6.3e+03","7.9e+03","1.0e+04","1.3e+04","1.6e+04","2.0e+04","2.5e+04","3.2e+04","4.0e+04","5.0e+04","6.3e+04","7.9e+04","1.0e+05","1.3e+05","1.6e+05","2.0e+05","2.5e+05","3.2e+05","4.0e+05","5.0e+05","6.3e+05","7.9e+05","1.0e+06"

freq = freq[0:]




NYQ_RE = []
NYQ_IM = []
NYQ_PH = []
NYQ_AM = []
NYQ_FR = []



for ff in freq:
	print (ff)

	try: 
		path = input_directory+"/"+file_label+"/"+file_label+"_f"+ff+".csv"
		#print(path)
		with open(path, 'r') as file:
			
			reader = csv.reader(file)
			genList = [r for r in reader]
			genList.pop(0)
			elpoList = [float(p[0]) for p in genList]
			fluxList = [-float(p[1]) for p in genList]
			#print(elpoList)
			#print(fluxList)

	except:
		print ("file not found ... so ein pech!")
		continue
	
	take_the_last_points = 274

	edata=elpoList[-take_the_last_points:]
	fdata=fluxList[-take_the_last_points:]


	#plt.plot(edata)
	#plt.show()
	#plt.plot(fdata)
	#plt.show()

	#########################
	# https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy #

	def fit_sin(tt, yy):
	    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
	    tt = np.array(tt)
	    yy = np.array(yy)
	    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
	    Fyy = abs(np.fft.fft(yy))
	    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
	    guess_amp = np.std(yy) * 2.**0.5
	    guess_offset = np.mean(yy)
	    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

	    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
	    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
	    A, w, p, c = popt
	    f = w/(2.*np.pi)
	    fitfunc = lambda t: A * np.sin(w*t + p) + c
	    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

	#########################

	

	def normalize(li):
		m = max(li)
		return [r/float(m) for r in li]

	
	xdata = np.linspace(0,300,len(edata))

	#popt, pcov = curve_fit(func, xdata, edata)
	res1 = fit_sin(xdata,edata)

	#plt.plot(xdata, res1["fitfunc"](xdata), "-g")
	a1 = res1["amp"]
	b1 = res1["omega"]
	c1 = res1["phase"]
	d1 = res1["offset"]

	if debug_plot_bool:
		plt.plot(xdata, res1["fitfunc"](xdata), "-r", label="fit volt")
		#plt.show()
		plt.plot(xdata, edata, "x", label="data")
		plt.legend(loc="best")
		plt.show()

	res2 = fit_sin(xdata,fdata)
	a2 = res2["amp"]
	b2 = res2["omega"]
	c2 = res2["phase"]
	d2 = res2["offset"]
	
	if debug_plot_bool:
		plt.plot(xdata, res2["fitfunc"](xdata), "-r", label="fit flux")
		#plt.show()
		plt.plot(xdata, fdata, "x", label="data")
		plt.legend(loc="best")
		plt.show()
		
	if a1 < 0:
		kor1 = 1
	else: 
		kor1 = 0


	if a2 < 0:
		kor2 = 1
	else:
		kor2 = 0

	phDIFF 	= ((c2 - c1) + (kor1 - kor2)*np.pi) % (2*np.pi)
	mD 	= np.abs(a1/a2)

	T = np.tan(phDIFF)
	A = mD

	RE = A*math.sqrt(1/(1+T*T))
	IM = A*T*math.sqrt(1/(1+T*T))
	
	# tady by melo byt -IM, nicmene implicitne predpokladam 1. kvadrant
	NYQ_RE.append(RE)
	NYQ_IM.append(IM)
	NYQ_PH.append(phDIFF)
	NYQ_AM.append(mD)
	NYQ_FR.append(math.log(float(ff)))


#	if print_bool:
#		print phDIFF," // ",mD
#		print RE," //////",IM
#		print a1," ",b1," ",c1," ",d1
#		print a2," ",b2," ",c2," ",d2

	if debug_plot_bool:
		#plt.plot(xdata, res2["fitfunc"](xdata), "-r", label="fit")
		plt.plot(xdata, normalize(res1["fitfunc"](xdata)), "-r", label="elpo")
		plt.plot(xdata, normalize(res2["fitfunc"](xdata)), "-g", label="flux")
		plt.show()
		#exit()

	#print popt," // "
	#print pcov
	#plt.plot(xdata, func(xdata, *popt), "-r")



# results processing ...
def plot_NYQ():
	plt.title("Nyquist diagram")
	plt.plot(NYQ_RE,NYQ_IM,"x")
	plt.xlabel("Re(Z)")	
	plt.ylabel("-Im(Z)")
	
def plot_BODE():
	plt.title("Bode diagram")
	plt.plot(NYQ_FR,NYQ_PH)
	plt.xlabel("log(f)")
	plt.ylabel("PhDiff")
	
if len(NYQ_RE)==0:
	print ("nothing here to display")
	exit()
	
if results_plot_bool:
	plt.figure(figsize=(12,5))
	
	plt.subplot(121)
	plot_NYQ()
	
	plt.subplot(122)
	plot_BODE()
	
	plt.show()



if debug_plot_bool:
	plt.plot(NYQ_RE,NYQ_IM)
	plt.show()

if plot_save_bool:
	if not os.path.exists(fig_dir):
	    os.makedirs(fig_dir)
	    
	plot_BODE()
	#plt.show()
	plt.savefig(fig_dir+"/BODE_"+file_label+".png")
	
	plt.clf()
	plot_NYQ()
	#plt.show()
	plt.savefig(fig_dir+"/NYQ_"+file_label+".png")

if data_save_bool:
	if not os.path.exists(data_dir):
	    os.makedirs(data_dir)
	
	with open(data_dir+"/BODE_"+file_label+".csv", 'w') as file:
		pom = 0
		while pom < len(NYQ_FR):
			file.write("{},{}\n".format(NYQ_FR[pom],NYQ_PH[pom]) )
			pom = pom + 1

	with open(data_dir+"/NYQ_"+file_label+".csv", 'w') as file:
                pom = 0
                while pom < len(NYQ_RE):
                        file.write("{},{}\n".format(NYQ_RE[pom],NYQ_IM[pom]) )
                        pom = pom + 1

