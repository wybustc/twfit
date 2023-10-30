import numpy as num
def vdisp_gconv(x,sigma):
	if sigma==0:
		return x
	khalfsz = round(4*sigma+1)
	xx = num.arange(0,khalfsz*2+1,1.0) - khalfsz
	kernel = num.exp(-xx**2 / (2*sigma**2))
	kernel = kernel / sum(kernel)
	return num.convolve(x, kernel,'same')
