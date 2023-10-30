from astropy.io import fits
import os

def create_fits(datas,params,path_savefile):
	# create the primary HDU
	prihdr=fits.Header()
	if params[0]==None:
		prihdu=fits.PrimaryHDU(data=datas[0])
	else:
		for key in params[0].keys():
			prihdr[key]=params[0][key]
		prihdu=fits.PrimaryHDU(data=datas[0],header=prihdr)

#   create other hdus....
	tbhdus=[]
	for i in range(1,len(datas)):
		if datas[i]==None:
			raise Exception('when not primary HDU, datas[i] should not be None')
		datacols=[]
		for key in datas[i].keys():
			if 'disp' in datas[i][key].keys():
				datacol=fits.Column(name=key,format=datas[i][key]['fmt'],unit=datas[i][key]['unit'],array=datas[i][key]['data'],disp=datas[i][key]['disp'])
			else:
				datacol=fits.Column(name=key,format=datas[i][key]['fmt'],unit=datas[i][key]['unit'],array=datas[i][key]['data'])
			datacols.append(datacol)
		cols=fits.ColDefs(datacols)
		tbhdu=fits.BinTableHDU.from_columns(cols)

		if not params[i]==None:
			for key in params[i].keys():
				tbhdu.header[key]=params[i][key]

		tbhdus.append(tbhdu)

	tbhdulist=fits.HDUList([prihdu]+tbhdus)
	if os.path.exists(path_savefile):
		os.remove(path_savefile)
	tbhdulist.writeto(path_savefile)
	return ':)'
