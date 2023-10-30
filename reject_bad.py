def reject_bad(objMask):
#mask the bad pixels for sdss spectrum 
#can mask the bad pixels use ind_bad = objMask>0?
	ind_NoPlug=          (objMask & 2**0 ) >= 1 
	ind_BadTrace=        (objMask & 2**1 ) > 1 
	ind_BadFlat=         (objMask & 2**2 ) > 1 
	ind_BadArc=          (objMask & 2**3 ) > 1 
	ind_ManyBadCol=      (objMask & 2**4 ) > 1 
	ind_ManyReject=      (objMask & 2**5 ) > 1 
	ind_LargeShift=      (objMask & 2**6 ) > 1 
	ind_NearBadPixel=    (objMask & 2**16) > 1 
	ind_LowFlat=         (objMask & 2**17) > 1 
	ind_FullReject=      (objMask & 2**18) > 1 
	ind_PartialRej=      (objMask & 2**19) > 1 
	ind_ScatLight=       (objMask & 2**20) > 1 
	ind_CrossTalk=       (objMask & 2**21) > 1 
	ind_NoSky=           (objMask & 2**22) > 1 
	ind_BrightSky=       (objMask & 2**23) > 1 
	ind_NoData=          (objMask & 2**24) > 1 
	ind_CombineRej=      (objMask & 2**25) > 1 
#	print(ind_FullReject)
	ind_bad= ind_FullReject | ind_CombineRej | ind_NoData  #or ind_NearBadPixel $ ;or ind_BrightSky $ 
			#;or ind_ManyBadCol ind_BadArc or ind_LargeShift or ind_ManyReject  or ind_ScatLight 
	return ind_bad

