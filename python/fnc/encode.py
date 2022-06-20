##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import numpy as np


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def encode(polar_array, noise_array, minWaveLength, mult, sigmaOnf):
	
	filterbank = gaborconvolve(polar_array, minWaveLength, mult, sigmaOnf)

	length = polar_array.shape[1]
	template = np.zeros([polar_array.shape[0], 2 * length])
	h = np.arange(polar_array.shape[0])

	mask = np.zeros(template.shape)
	eleFilt = filterbank[:, :]

	H1 = np.real(eleFilt) > 0
	H2 = np.imag(eleFilt) > 0

	H3 = np.abs(eleFilt) < 0.0001
	for i in range(length):
		ja = 2 * i

		template[:, ja] = H1[:, i]
		template[:, ja + 1] = H2[:, i]

		mask[:, ja] = noise_array[:, i] | H3[:, i]
		mask[:, ja + 1] = noise_array[:, i] | H3[:, i]

	# Return
	return template, mask


#------------------------------------------------------------------------------
def gaborconvolve(im, minWaveLength, mult, sigmaOnf):
	
	rows, ndata = im.shape					# Size
	logGabor = np.zeros(ndata)				# Log-Gabor
	filterbank = np.zeros([rows, ndata], dtype=complex)

	radius = np.arange(ndata/2 + 1) / (ndata/2) / 2
	radius[0] = 1

	wavelength = minWaveLength

	fo = 1 / wavelength 		# Centre frequency of filter
	logGabor[0 : int(ndata/2) + 1] = np.exp((-(np.log(radius/fo))**2) / (2 * np.log(sigmaOnf)**2))
	logGabor[0] = 0

	for r in range(rows):
		signal = im[r, 0:ndata]
		imagefft = np.fft.fft(signal)
		filterbank[r , :] = np.fft.ifft(imagefft * logGabor)

	# Return
	return filterbank