##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
from cv2 import imread

from fnc.segment import segment
from fnc.normalize import normalize
from fnc.encode import encode


##-----------------------------------------------------------------------------
##  Parameters for extracting feature
##-----------------------------------------------------------------------------
# Parameter Segmentation
eyelashes_thres = 80

# Parameter Normalisation
radial_res = 20
angular_res = 240

# Parameter Fitur encoding
minWaveLength = 18
mult = 1
sigmaOnf = 0.5


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def extractFeature(im_filename, eyelashes_thres=80, use_multiprocess=True):
	"""
	Deskripsi:
		Extraksi fitur dari citra iris 
	Input:
		im_filename			- Input citra iris
	Output:
		template			- Template yang diekstraksi
		mask				- Mask yang diekstraksi
		im_filename			- Input citra iris
	"""
	# Segmentation
	im = imread(im_filename, 0)
	ciriris, cirpupil, imwithnoise = segment(im, eyelashes_thres, use_multiprocess)

	# Normalization
	polar_array, noise_array = normalize(imwithnoise, ciriris[1], ciriris[0], ciriris[2],
										 cirpupil[1], cirpupil[0], cirpupil[2],
										 radial_res, angular_res)

	# Fitur encoding
	template, mask = encode(polar_array, noise_array, minWaveLength, mult, sigmaOnf)

	# Return
	return template, mask, im_filename