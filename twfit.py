from astropy.io import fits 
import numpy as num 
from extinction import calzetti00,apply,remove,fitzpatrick99# 
from vdisp_gconv_original import vdisp_gconv
from line_info3 import line_info3
import sfdmap 
from reject_bad import reject_bad
from scipy.interpolate import interp1d
from scipy.optimize import nnls
import math 
import matplotlib.pyplot as plt 
from create_fits import create_fits 
from spectres import spectres 
import os 
import mpfit as mpfit



PATH_IC=r"C:\Users\32924\Desktop\twfit\17_ic.fit" # the path to the stellar templates, change to your own path before running the procedure 
PATH_dustmap=r"D:\sfddata-master"  #the path to the SFD dust map data 
#raise Exception('Please correct the PATH_IC and PATH_dustmap to your own ones, and then mask this line')
def sdssfit(path_sdss,starlight=None,
					  and_mask=True,extra_mask=None,line_mask=True,bw=1200/3e5,
					  poly_correct=True,poly_deg=3,
					  fit_BB=False,fit_POW=False,
					  varyTorIndex=False,add_pow=True, add_BB=False,add_FeII=False,
					  BB_temp=10000, BB_temps=num.arange(5000,50000,1000),powlaw_index=2, powlaw_indexs=num.arange(-2,4,0.5),
					  path_savefile=None,path_savefig=None,show=True,save_fmt='default'):
	#fitting for sdss spectrum 
	#The Input parameters: 
	#starlight,  provide your own stellar template using the format {'wave':wave, 'flux': flux} 
	#and_mask,   mask bad points using the mask array in the sdss file 
	#extra_mask, mask the specified region with the format [[region_left1,region_right1], [left2,right2], ....] 
	#line_mask,  mask the common emission lines 
	#bw,         the width used to mask the emission lines 
	#poly_correct, correct the final fitting with a low-order polynoimal to get a better  results 
	#poly_deg,   the deg of polynominal used to do the poly-correction 
	#fit_BB,     if True, we fit the continuum with  only a Blackbody component 
	#fit_POW,    if True, we fit the continuum with  only a polwer-law component 
	#add_pow,    if True, we include a pow-law component in the fitting 
	#add_BB,     if True, we include a blackbody component in the fitting 
	#varyTorIndex, if True, we travel the range indicated by the parameters BB_temps or powlaw_index,to find the best 
	#              temperature or power-law index used in the fitting procedure 
	#              if False,we fixed the temperature of the blacbody included (or pow-law index) as the value given 
	#              by the parameter BB_temp (or powlaw_index) 
	#path_savefile, if it's not None, we will save the results at this path 
	#path_savefig,  the path used to save the figure 
	#show, if True, we will display the fitting results in an image 
	# 


	#read the spectrum and do the redshift correction 
	hdu=fits.open(path_sdss)
	wave   =hdu[1].data['loglam']*num.log(10)-num.log(1+hdu[2].data['Z'][0])
	flux   =hdu[1].data['flux']*(1+hdu[2].data['Z'][0])
	ivar   =hdu[1].data['ivar']/(1+hdu[2].data['Z'][0])**2
	loglam =hdu[1].data['loglam']
	sdss_and_mask=hdu[1].data['and_mask']
	redshift=hdu[2].data['Z'][0]
	
	ind= (ivar==0) | num.isnan(flux) | num.isinf(flux) | num.isnan(ivar) | (num.isinf(ivar))
	mask=num.ones_like(wave,dtype=int)
	mask[ind]==0 #mask the nan value, otherwise it may raise exceptions 

	# Galactic extinction correction 
	ra,dec=hdu[0].header['plug_ra'],hdu[0].header['plug_dec']
	mw_ebv=sfdmap.ebv((ra,dec),unit='degree',mapdir=PATH_dustmap)
	fcorr=remove(fitzpatrick99(num.array(10**loglam,dtype='float64'),3.1*mw_ebv), num.ones_like(flux)) #loglam, the lg10 value of the observed wavelength 
	flux=flux*fcorr
	ivar=ivar/fcorr**2 

	if and_mask:
		mask=mask & ( 1-reject_bad(sdss_and_mask) )

	if extra_mask is not None:
		for region in extra_mask:
			ind_mask=(num.e**wave>region[0]) & (num.e**wave<region[1])
			mask[ind_mask]=0

	if varyTorIndex & (add_pow | add_BB): 
		if add_BB==True:
			chi2min=[]  
			for BB_temp in BB_temps: 
				tf=twfit(wave,flux,ivar,starlight_temp=starlight, mask=mask,show=False,path_ic=PATH_IC,mw_ebv=mw_ebv,redshift=redshift)
				tf.running(save_fmt=save_fmt,BB_temp=BB_temp, line_mask=line_mask,polycorr=poly_correct,polycorr_deg=poly_deg,powlaw=add_pow,bw=bw,add_FeII=add_FeII,add_BB=add_BB)
				chi2min.append(tf.chi2) 
			for BB_temp,tf_chi2 in zip(BB_temps, chi2min):
					print('Finding best BB_temp', BB_temp, tf_chi2) 
			ind_chi2=num.argmin(chi2min)
			BB_temp=BB_temps[ind_chi2]  
		#	print(chi2min)
			print('varyTemp-chi2min', BB_temp,  chi2min[ind_chi2]) 
		else: 
			chi2min=[]  
			for powlaw_index in powlaw_indexs: 
				tf=twfit(wave,flux,ivar,starlight_temp=starlight, mask=mask,show=False,path_ic=PATH_IC,mw_ebv=mw_ebv,redshift=redshift)
				tf.running(save_fmt=save_fmt,powlaw_index=powlaw_index, line_mask=line_mask,polycorr=poly_correct,polycorr_deg=poly_deg,powlaw=add_pow,bw=bw,add_FeII=add_FeII,add_BB=add_BB)
				chi2min.append(tf.chi2)   
			for powlaw_index,tf_chi2 in zip(powlaw_indexs, chi2min):
				print('Finding Best Powelaw Index', powlaw_index, tf_chi2)
			ind_chi2=num.argmin(chi2min) 
			powlaw_index=powlaw_indexs[ind_chi2]  
			print('varyIndex-chi2min', powlaw_index, chi2min[ind_chi2])

	tf=twfit(wave,flux,ivar,starlight_temp=starlight, mask=mask,path_savefile=path_savefile,path_savefig=path_savefig,show=show,path_ic=PATH_IC,mw_ebv=mw_ebv,redshift=redshift)
	if fit_BB:
		tf.fit_BB() # fit  the continuum with only blackbody component 
		return ':)' 
	if fit_POW:
		wave,flux,yf=tf.fit_pow() # fit the continuum with only powlaw component 
		return wave,flux,yf
	tf.running(save_fmt=save_fmt,powlaw_index=powlaw_index, BB_temp=BB_temp, line_mask=line_mask,polycorr=poly_correct,polycorr_deg=poly_deg,powlaw=add_pow,bw=bw,add_FeII=add_FeII,add_BB=add_BB)

	print('Best ired: %s'%tf.ired_best)
	print('Best sigma: %s'%tf.best_sigma)
	print('Min reduced-chi2: %s'%tf.chi2)# 
	print('Best_weights')
	print(tf.ic_weights)

	return num.e**tf.wic,tf.f_int,tf.msp,tf.ivar_int 
	
def spec_fit(wave,flux,ivar,starlight=None,mw_ebv=None,ra_dec=None,redshift=0, 
							extra_mask=None,line_mask=True,bw=1200/3e5,
							poly_correct=True,poly_deg=3,
							fit_BB=False,fit_POW=False,log_rebin=True,
							varyTorIndex=False,add_BB=False,add_pow=True,add_FeII=False,	
							BB_temp=10000, BB_temps=num.arange(5000,50000,1000),powlaw_index=2, powlaw_indexs=num.arange(-2,4,0.5),  
							path_savefile=None,path_savefig=None,show=True, save_fmt='default',):
	#fitting for spectra 
	#wave, the observed wavelength of the target 
	#flux, the obserced flux 
	#ivar, 1/err**2,in which  err is the 1-sigma error of the flux measurements 
	#ra_dec=(ra,dec), in unit 'deg', which  can be used to get the galactic extinction ebv from dust map
	#                 if mw_ebv was supplied, then  it can be None 
	#log_rebin, if True, we will resample the input spectra to a log-uniformed  wavelength grid 
	#           it is neccessary for those with the spectra wavelength grid not uniformed in log-space 
	if (mw_ebv is None ) and (ra_dec is None): 
		raise Exception('Must give one of the mw_ebv or ra_dec to do the Galactic extinction ')
	if mw_ebv is None: mw_ebv= sfdmap.ebv(ra_dec,unit='degree',mapdir=PATH_dustmap)
	fcorr=remove(fitzpatrick99(wave,3.1*mw_ebv), num.ones_like(flux))
	flux=flux*fcorr
	ivar=ivar/fcorr**2 
	
	wave=wave/(1+redshift);flux=flux*(1+redshift); ivar=ivar/(1+redshift)**2 #NOTE ferr redshift? 
	
	if log_rebin:
		wave_log=num.linspace(num.log(wave[1]),num.log(wave[len(wave)-2]),num=len(wave))
		flux_log,err_log=spectres(num.e**wave_log,wave,flux,spec_errs=ivar**-0.5)
		wave,flux,ivar=wave_log,flux_log,1/err_log**2
		#wave_gap=wave[1]-wave[0]
	
	if extra_mask is not None:
		mask=num.ones_like(wave,dtype=int)
		for region in extra_mask:
			ind_mask=(num.e**wave>region[0]) & (num.e**wave<region[1])
			mask[ind_mask]=0
	else:
		mask=num.ones_like(flux,dtype=int)
 
	if varyTorIndex & (add_BB | add_pow): 
		if add_BB==True:
			chi2min=[]  
			for BB_temp in BB_temps: 
				tf=twfit(wave,flux,ivar,starlight_temp=starlight, mask=mask,show=False,path_ic=PATH_IC,mw_ebv=mw_ebv,redshift=redshift)
				tf.running(save_fmt=save_fmt,BB_temp=BB_temp, line_mask=line_mask,polycorr=poly_correct,polycorr_deg=poly_deg,powlaw=add_pow,bw=bw,add_FeII=add_FeII,add_BB=add_BB)
				chi2min.append(tf.chi2) 
			for BB_temp,tf_chi2 in zip(BB_temps, chi2min):
					print('Finding Best Temperature', BB_temp, tf_chi2) 
			ind_chi2=num.argmin(chi2min)
			BB_temp=BB_temps[ind_chi2]  
			print('varyTemp-chi2min', BB_temp,  chi2min[ind_chi2]) 
		else: 
			chi2min=[]  
			for powlaw_index in powlaw_indexs: 
				tf=twfit(wave,flux,ivar,starlight_temp=starlight, mask=mask,show=False,path_ic=PATH_IC,mw_ebv=mw_ebv,redshift=redshift)
				tf.running(save_fmt=save_fmt,powlaw_index=powlaw_index, line_mask=line_mask,polycorr=poly_correct,polycorr_deg=poly_deg,powlaw=add_pow,bw=bw,add_FeII=add_FeII,add_BB=add_BB)
				chi2min.append(tf.chi2)   
			for powlaw_index,tf_chi2 in zip(powlaw_indexs, chi2min):
				print('Finding Best Powlaw Index', powlaw_index, tf_chi2)
			ind_chi2=num.argmin(chi2min) 
			powlaw_index=powlaw_indexs[ind_chi2]  
			print('varyIndex-chi2min', powlaw_index, chi2min[ind_chi2])

	tf=twfit(wave,flux,ivar,starlight_temp=starlight, mask=mask,path_savefile=path_savefile,path_savefig=path_savefig,show=show,path_ic=PATH_IC,mw_ebv=mw_ebv,redshift=redshift)
	if fit_BB:
		tf.fit_BB() # fit  the continuum with only blackbody component 
		return ':)' 
	if fit_POW:
		wave,flux,yf=tf.fit_pow() # fit the continuum with only powlaw component 
		return wave,flux,yf
	tf.running(save_fmt=save_fmt,powlaw_index=powlaw_index, BB_temp=BB_temp, line_mask=line_mask,polycorr=poly_correct,polycorr_deg=poly_deg,powlaw=add_pow,bw=bw,add_FeII=add_FeII,add_BB=add_BB)


	print('Best ired: %s'%tf.ired_best)
	print('Best sigma: %s'%tf.best_sigma)
	print('Min reduced-chi2: %s'%tf.chi2) 
	print('Best_weights')
	print(tf.ic_weights)
	
	return num.e**tf.wic,tf.f_int,tf.msp,tf.ivar_int

class twfit():
	def __init__(self,wave,flux,ivar,starlight_temp=None, path_savefile=None,path_savefig=None,show=False,mask=None,mw_ebv=0,redshift=None,path_ic=PATH_IC):
		self.wave=wave # wave must have uniform grid in ln space 
		self.dw=wave[1]-wave[0]
		self.flux=flux 
		self.ivar=ivar 
		self.path_ic=PATH_IC
		self.ic=num.array([])
		self.path_savefile=path_savefile 
		self.show=show
		self.path_savefig=path_savefig
		self.npow=0
		self.redshift=redshift
		self.mw_ebv=mw_ebv
		self.starlight_temp=starlight_temp 
		if mask is None : self.mask=num.ones_like(wave) # 
		else: self.mask=mask # Extra pixels needed to mask ,except for the emission lines
	#	plt.plot(self.wave[self.mask==0],self.flux[self.mask==0])
	#	plt.show()
	

	def get_ic(self,sel=[0,1,2,3,5,6,8,11,13,15,16],add_FeII=False):
		# sel, the serial number of the ics to use
		self.nni=len(sel) # number of ics to use 
		self.nsl=self.nni 
		with fits.open(self.path_ic) as hdu:
			nic=hdu[0].header['NAXIS1']
			wic=hdu[0].header['coeff0']+hdu[0].header['coeff1']*num.arange(nic) # wavelength in log space
			wic=wic*num.log(10) # lg(wave)--->ln(wave)
			ind_ic=(wic>self.wave[0])& (wic<max(self.wave)) #mask, the region not overlapped with the spectrum to fit
			self.wic=wic[ind_ic]
			self.ic=hdu[0].data[sel[0]][ind_ic]
			for i in range(1,self.nni):		
				self.ic=num.vstack([self.ic , hdu[0].data[sel[i]][ind_ic] ]) # spectra of each templates 

		#add FeII,
		if add_FeII:
			hdu=fits.open(r"C:\Users\32924\Downloads\newfe2.fits")
			wFe=10**hdu[1].data['wave'][0]
			nfe=hdu[1].data['nfe'][0] 
			bfe=hdu[1].data['bfe'][0]
			nfe=spectres(num.e**self.wic,wFe,nfe,fill=0)
			bfe=spectres(num.e**self.wic,wFe,bfe,fill=0)
			self.ic=num.vstack([self.ic,nfe]) 
			self.ic=num.vstack([self.ic,bfe]) 
			self.nni=self.nni+2 
			self.add_FeII=True 
		else: 
			self.add_FeII=False
	def use_providestarlight(self): 
		self.nni=1
		self.nsl=1 
		ind_ic=(self.starlight_temp['wave']>self.wave[0])&(self.starlight_temp['wave']<self.wave[-1]) 
		self.wic=self.starlight_temp['wave'][ind_ic] 
		self.ic=num.array([self.starlight_temp['flux'][ind_ic]])
		self.add_FeII=False

	def add_pow(self,index=2):
		#index, list of powlaw_index to add 
		if not index.__class__==list: index=[index]
		self.powlaw_index=index
		self.npow=len(index)
		for indx in index:
			powlaw=num.array([ 0.02*num.e**(-indx*(self.wic-num.log(6000))) ])
			self.ic=num.vstack([self.ic, powlaw])
	def add_BB(self,Temp=10000):
		if not Temp.__class__==list: Temp=[Temp] 
		self.BB_temp=Temp
		self.npow =len(Temp) 
		lam=num.e**self.wic*1e-10
		h=6.62607015e-34; c=299792458.0
		k=1.380649e-23 # International system of units 
		
		for temp in Temp: 
			bb_lam= 2*h/lam**5 * 1/( num.e**(h*c/lam/k/temp) -1 )
			bb_lam= num.array( bb_lam*1000 ) # 
			self.ic=num.vstack( [self.ic, bb_lam])
 
	def mask_line(self,nw=300./3e5,bw=1200./3e5):
		#extra_mask_ind, supplement to typical mask results 
		l_info=line_info3()

		ll=12
		maskline=num.recarray(ll,dtype={'names':('id','wc','bn'),'formats':('<U16','float','int')} )
		maskline.id=['O3727','Hd','Hg','O4363','Hb','O4959','O5007','Ha','N6548','N6583','S6718','S6732']
		maskline.wc=num.log([3728.3,4100.0,4341.7,4364.4,4862.7,4960.3,5008.2,6564.6,6549.9, 6585.3,6718.3,6732.7])
		maskline.bn=[0,1,1,0,1,0,0,1,0,0,0,0]
		ws=maskline.wc-nw*(1-maskline.bn)-maskline.bn*bw
		wl=maskline.wc+nw*(1-maskline.bn)+maskline.bn*bw

	#	print('dw',self.dw)
		ms=(ws-self.wave[0])/self.dw ; ml=(wl-self.wave[0])/self.dw 
		gg=num.where( (ms>0) & (ms<len(self.wave)-1) & (ml<len(self.wave)-1)) # gg= (ms>0) &(ml<nw-1) ? or gg=ws>w[0] & wl<w[len(w)-1]? or ind_mask for very line(maybe better)?, In this way, some lines located a little exceed the wavelength space will not be mask 
		ngg=len(ms[gg]) # line number, need to be masked 
	#	print(ngg)

		for k in range(ngg):
			self.mask[round(ms[gg][k]):round(ml[gg][k])+1]=0
	#		print(round(ms[gg][k]),round(ml[gg][k])+1)
	#	a=input()
	def match_grids(self):
		#match the spectrum to the same grid of templates
		#Here, interp1d method was used
 			
		self.f_int=interp1d(self.wave,self.flux,fill_value='extrapolate')(self.wic)
		self.ivar_int=interp1d(self.wave,self.ivar,fill_value='extrapolate')(self.wic)
		self.mask_int=interp1d(self.wave,self.mask,kind='nearest',fill_value='extrapolate')(self.wic)

	
	def solve_reddening(self,redden_ebvs=num.arange(21)*0.025):
		#sigma is not important, hence only reddening was applied to the templates and to find the best one 
        #TODO, the nuclei powlaw component may need use different extinction curve and ebv value
		nred=len(redden_ebvs)
		rnorm=num.zeros(nred) # resnorm 
		xx=num.zeros((nred,self.nni+self.npow))

		for i in range(nred):
			f_red=apply(calzetti00(num.e**self.wic,3.1*redden_ebvs[i],3.1),num.ones_like(self.wic))
			A=self.ic*f_red*self.mask_int
			x,resnorm=nnls(A.T, self.f_int*self.mask_int)
			xx[i]=x
			rnorm[i]=resnorm
		self.ired_best=redden_ebvs[num.argmin(rnorm)] # index to best reddening
	
	def solve_sigmas(self,sigmas=num.arange(9)*30/2.998e5/1e-4):
		#solving the sigma
		nsigma=len(sigmas) 
		rnorm=num.zeros(nsigma)
		xx1=num.zeros((nsigma,self.nni+self.npow))

	#	self.ired_best=0.125
		f_red=apply(calzetti00(num.e**self.wic,3.1*self.ired_best,3.1),num.ones_like(self.wic))
	#	print(f_red)
		nni=self.nni;n=self.nni+self.npow;m=len(self.wic)
		for i in range(nsigma):
		#	print(sigmas[i])
			A=num.zeros((n,m))
			for j in range(nni): A[j]=vdisp_gconv(self.ic[j],sigmas[i])*f_red*self.mask_int
			for j in range(nni,n): A[j]=self.ic[j]*f_red*self.mask_int
			x,resnorm=nnls(A.T,self.f_int*self.mask_int)
	#		plt.plot(A[0])
			xx1[i]=x
			rnorm[i]=resnorm
	#		print(sigmas[i],resnorm)
	#	plt.show()
		indx=num.argmin(rnorm)
		self.ic_weights=xx1[indx]
		self.best_sigma=sigmas[indx]

		A=num.zeros((n,m))
		for j in range(nni): A[j]=vdisp_gconv(self.ic[j],self.best_sigma)*f_red
		for j in range(nni,n): A[j]=self.ic[j]*f_red
		self.msp=num.dot(A.T,self.ic_weights)  #mix of starlight and powlaw
		self.fsl=num.dot(A[0:self.nsl].T, self.ic_weights[0:self.nsl]) # galaxy starlight, changed by Yibo at 2022-0722
		self.fp =num.dot(A[nni:n].T, self.ic_weights[nni:n]) # powlaw
		if self.add_FeII:
			self.ffe=num.dot(A[self.nsl:nni].T,self.ic_weights[self.nsl:nni])  #FeII 

	@staticmethod
	def smooth(flux,N):
		if N%2==0:
			raise Exception('The width must be an odd number')
		if N==1: return flux 
		fluxsm=num.zeros_like(flux)
		fluxsm[0:(N-1)//2]=flux[0:(N-1)//2]
		fluxsm[len(flux)-(N-1)//2:len(flux)]=flux[len(flux)-(N-1)//2:len(flux)]
		for i in range((N-1)//2,len(flux)-(N-1)//2):
			fluxsm[i]=sum(flux[i-(N-1)//2:i+(N-1)//2+1])/N
		return fluxsm

	def poly_correct(self,deg=3):
		# poly will be multipied to the best mix of different templates and powlaw to improve the final results 
		ssl= (self.mask_int==1)
		ratio=self.smooth(self.f_int[ssl]/self.msp[ssl],21)-1

		ierr=self.ivar_int[ssl]**0.5*self.msp[ssl] # Do we need to consider the error propagation of the function smooth?

		ras=num.polyfit(self.wic[ssl],ratio,deg,w=ierr) # w is 'weights', hence 1/sigma is needed
		
		print('poly',ras)
		f_correct=(1+num.polyval(ras,self.wic))
		# plt.plot(num.e**self.wic[ssl],ratio)
		# plt.plot(num.e**self.wic,f_correct-1)
		# plt.show()
		self.msp=self.msp*f_correct
		self.fsl=self.fsl*f_correct
		self.fp =self.fp *f_correct
		if self.add_FeII:
			self.ffe=self.ffe*f_correct 
		
		

	def save_results(self,save_fmt='default'):
		self.chi2=sum( (self.msp-self.f_int)**2*self.ivar_int*self.mask_int )/( len(self.mask_int[self.mask_int==1])-self.nni-self.npow )
		if save_fmt=='default':
			params={0:{'redshift':self.redshift,'ired':self.ired_best,'sigma':self.best_sigma,'chi2':self.chi2},1:None,2:None}
			datas={0:None,1:{'wave':{'data':num.e**self.wic,'fmt':'D','unit':'none'},\
							'spec':{'data':self.msp,'fmt':'D','unit':'none'},\
							'starlight':{'data':self.fsl,'fmt':'D','unit':'none'},\
							'powlaw':{'data':self.fp,'fmt':'D','unit':'none'},\
							'mask':{'data':self.mask_int,'fmt':'bool','unit':'none'},\
							'flux':{'data':self.f_int,'fmt':'D','unit':'none'},\
							'ivar':{'data':self.ivar_int,'fmt': 'D','unit':'none'}},\
						 2: {'ic_weights':{'data':self.ic_weights, 'fmt':'D', 'unit':'none'}} } 
		elif save_fmt=='dbsp':
			params={0:{'redshift':self.redshift},1:None}
			datas={0:None,1:{'wave_dbsp':{'data':num.e**self.wic,'fmt':'D','unit':'Angstrom'},\
						'flux_dbsp':{'data':self.f_int,'fmt':'D','unit':'none'},\
						'err_dbsp':{'data':self.ivar_int**-0.5,'fmt':'D','unit':'none'},\
						'continuum':{'data':self.msp,'fmt':'D','unit':'none'},\
						'conti_err':{'data':num.zeros_like(self.msp),'fmt':'D','unit':'none'},\
						'stellar':{'data':self.fsl,'fmt':'D','unit':'none'},\
						'stellar_err':{'data':num.zeros_like(self.fsl),'fmt':'D','unit':'none'}}} # we didn't save the information of line mask, since it has different array length 
		elif save_fmt=='ppxf':
		#	print('redshift',self.redshift)
		#	a=input()
			params={0:{'redshift':self.redshift,'ebv':self.mw_ebv},1:None}
			datas={0:None,1:{'wave':{'data':num.e**self.wic,'fmt':'D','unit':'Angstrom'},\
							'original_data':{'data':self.f_int,'fmt':'D','unit':'none'},\
							'stellar':{'data':self.fsl,'fmt':'D','unit':'none'},\
							'spec_fit':{'data':self.msp,'fmt':'D','unit':'none'},\
							'err_stellar':{'data':num.zeros_like(self.fsl),'fmt':'D','unit':'none'},\
							'err_sdss':{'data':self.ivar_int**-0.5,'fmt':'D','unit':'none'}}}
		else:
			raise Exception('No such savefile format %s'%save_fmt)
		if not self.path_savefile==None:
			create_fits(datas,params,self.path_savefile)


		fig=plt.figure(figsize=(15,8))
		ax=plt.axes()
		ax.tick_params(labelsize=20)
		ax.tick_params(which='both',direction='in')
		ax.tick_params(which='major',length=7)
		ax.tick_params(which='major',width=1.3) 
		ax.tick_params(which='minor',length=3) 
		ax.tick_params(which='minor',width=1.3) 
		linewidth=1.5
		ax.spines['bottom'].set_linewidth(linewidth);ax.spines['left'].set_linewidth(linewidth) ;ax.spines['right'].set_linewidth(linewidth); ax.spines['top'].set_linewidth(linewidth)

		plt.plot(num.e**self.wic,self.f_int)
#		print(len(self.wic[self.mask_int==0]))
		plt.plot(num.e**self.wic,self.msp,'r',label='bestfit')
		plt.plot(num.e**self.wic,self.fsl,'pink',label='starlight')
		plt.plot(num.e**self.wic,self.f_int-self.msp,'y',label='lineSpec')
		if self.BB:
			if self.add_FeII:plt.plot(num.e**self.wic,self.msp-self.fsl-self.ffe,label='blackbody-%sK'%self.BB_temp[0])
			else: plt.plot(num.e**self.wic,self.msp-self.fsl,label='blackbody-%sK'%self.BB_temp[0])

		elif self.include_powlaw:
			if self.add_FeII: plt.plot(num.e**self.wic,self.msp-self.fsl-self.ffe,label='powlaw-index=%s'%self.powlaw_index[0])
			else: plt.plot(num.e**self.wic,self.msp-self.fsl,label='powlaw-index=%s'%self.powlaw_index[0])
		plt.scatter(num.e**self.wic[self.mask_int==0],self.f_int[self.mask_int==0],c='g',marker='+')
		if self.add_FeII: plt.plot(num.e**self.wic,self.ffe,'brown',label='FeII')
		# ylim=plt.ylim()
		# top=1
		# for lname, linew in zip(['Ha','Hb', 'HeII', 'NIII','OIII','OIII'],[6563,4861,4686,4640,5007,4959]):
		# 	plt.plot([linew,linew],ylim,linestyle='--',alpha=0.8) 
		# 	plt.text(linew-50,60-top*10,lname,fontsize=10) 
		# 	top=abs(top-1)
		plt.xlabel('Restframe wavelength',fontsize=20)
		plt.ylabel('Flux ',fontsize=20)
		plt.legend(fontsize=20)
		if not self.path_savefig==None: plt.savefig(self.path_savefig)
		if self.show: plt.show()

		plt.close()

	def running(self,BB_temp=10000, powlaw_index=2, powlaw=True,polycorr=True,add_BB=False,add_FeII=False,polycorr_deg=3,line_mask=True,bw=1200/3e5,save_fmt='default'):
		# running fitting  procedure
		if self.starlight_temp is None:
			self.get_ic(add_FeII=add_FeII) #get the starlight templates
		else:
			self.use_providestarlight()

		self.BB=False #NOTE,now we only use single blackbody or powlaw as the nuclei transient component 
		self.include_powlaw=False
		if add_BB:
			self.BB=True
			self.add_BB(Temp=BB_temp)  # add the blackbody component 
		elif powlaw:
			self.include_powlaw=True
			self.add_pow(index=powlaw_index) # add the powlaw

		if line_mask:
			self.mask_line(bw=bw) # mask the emission lines
		self.match_grids() # interpolate to the templates grids 
		self.solve_reddening() # find the best intrinsic ebv, without brodening
		self.solve_sigmas() # find the best sigmas to broadening 
		if polycorr:
			self.poly_correct(deg=polycorr_deg) # poly correct to improve the fitting results
		self.save_results(save_fmt=save_fmt) # save the results
	

	def fit_BB(self):
		h=6.62607015e-34; c=299792458.0
		k=1.380649e-23 # International system of units 
	#	print('Hello')
		def residuals0(p,fjac=None, xval=num.e**self.wave[self.mask==1]*1e-10, yval=self.flux[self.mask==1], errval=(self.ivar[self.mask==1])**-0.5):
			bb_lam= 2*h/xval**5 * 1/( num.e**(h*c/xval/k/p[1]) -1 )
			bb_lam= num.array( bb_lam*1000 )*p[0] # 
	#		print(bb_lam)
			return [0,(yval - bb_lam)/errval]

		par=[{'value':1   ,'limited':[1,0],'limits':[0,0],'parname':'scale'},
			 {'value':15000,'limited':[1,0],'limits':[5000,0],'parname':'Temperature'}] 


		res=mpfit.mpfit(residuals0, parinfo=par,quiet=False)
		print(res.errmsg)

		p=res.params
		xval=num.arange(num.e**self.wave[0],num.e**self.wave[-1],0.1)*1e-10
		bb_lam= 2*h/xval**5 * 1/( num.e**(h*c/xval/k/p[1]) -1 )
		bb_lam= num.array( bb_lam*1000 )*p[0]

		print(p)
		plt.plot(num.e**self.wave,self.flux)
		plt.plot(xval*1e10,bb_lam)
		plt.scatter(num.e**self.wave[self.mask==0],self.flux[self.mask==0],c='g',marker='+')
		plt.show()

	def fit_pow(self):
		def residuals0(p,fjac=None, xval=num.e**self.wave[self.mask==1], yval=self.flux[self.mask==1], errval=(self.ivar[self.mask==1])**-0.5):
			yf= p[0]* (xval/1000)**-p[1]
			return [0,(yval - yf)/errval]

		par=[{'value':1   ,'limited':[1,0],'limits':[0,0],'parname':'scale'},
			 {'value':4,'limited':[1,0],'limits':[0,0],'parname':'index'}] 
		
		res=mpfit.mpfit(residuals0, parinfo=par,quiet=False)
		p=res.params 
		xval=num.arange(num.e**self.wave[0],num.e**self.wave[-1],0.1)
		yf= p[0]* (xval/1000)**-p[1] 

		plt.plot(num.e**self.wave,self.flux)
		plt.plot(xval,yf)
		plt.scatter(num.e**self.wave[self.mask==0],self.flux[self.mask==0],c='g',marker='+')
		plt.show()
		return num.e**self.wave,self.flux, p[0]* (num.e**self.wave/1000)**-p[1]
