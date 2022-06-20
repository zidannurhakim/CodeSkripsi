##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import numpy as np
from os import listdir
from fnmatch import filter
import scipy.io as sio
from multiprocessing import Pool, cpu_count
from itertools import repeat

import warnings
warnings.filterwarnings("ignore")


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def matching(template_extr, mask_extr, temp_dir, threshold=0.38):
	
	n_files = len(filter(listdir(temp_dir), '*.mat'))
	if n_files == 0:
		return -1

	# Use all cores to calculate Hamming distances & WED
	args = zip(
		sorted(listdir(temp_dir)),
		repeat(template_extr),
		repeat(mask_extr),
		repeat(temp_dir),
	)
	with Pool(processes=cpu_count()) as pools:
		result_list = pools.starmap(matchingPool, args)

	filenames = [result_list[i][0] for i in range(len(result_list))]
	hm_dists = np.array([result_list[i][1] for i in range(len(result_list))])

	ind_valid = np.where(hm_dists>0)[0]
	hm_dists = hm_dists[ind_valid]
	filenames = [filenames[idx] for idx in ind_valid]

	ind_thres = np.where(hm_dists<=threshold)[0]

	# Return
	if len(ind_thres)==0:
		return 0
	else:
		hm_dists = hm_dists[ind_thres]
		filenames = [filenames[idx] for idx in ind_thres]
		ind_sort = np.argsort(hm_dists)
		return [filenames[idx] for idx in ind_sort]


#------------------------------------------------------------------------------
def calHammingDist(template1, mask1, template2, mask2):
	
	hd = np.nan
	for shifts in range(-8,9):
		template1s = shiftbits(template1, shifts)
		mask1s = shiftbits(mask1, shifts)

		mask = np.logical_or(mask1s, mask2)
		nummaskbits = np.sum(mask==1)
		totalbits = template1s.size - nummaskbits

		C = np.logical_xor(template1s, template2)
		C = np.logical_and(C, np.logical_not(mask))
		bitsdiff = np.sum(C==1)

		if totalbits==0:
			hd = np.nan
		else:
			hd1 = bitsdiff / totalbits
			if hd1 < hd or np.isnan(hd):
				hd = hd1

	# Return
	return hd


#------------------------------------------------------------------------------
def shiftbits(template, noshifts):
	
	templatenew = np.zeros(template.shape)
	width = template.shape[1]
	s = 2 * np.abs(noshifts)
	p = width - s

	
	if noshifts == 0:
		templatenew = template

	elif noshifts < 0:
		x = np.arange(p)
		templatenew[:, x] = template[:, s + x]
		x = np.arange(p, width)
		templatenew[:, x] = template[:, x - p]

	else:
		x = np.arange(s, width)
		templatenew[:, x] = template[:, x - s]
		x = np.arange(s)
		templatenew[:, x] = template[:, p + x]

	# Return
	return templatenew


#------------------------------------------------------------------------------
def matchingPool(file_temp_name, template_extr, mask_extr, temp_dir):
	
	data_template = sio.loadmat('%s%s'% (temp_dir, file_temp_name))
	template = data_template['template']
	mask = data_template['mask']
	hm_dist = calHammingDist(template_extr, mask_extr, template, mask)
	return (file_temp_name, hm_dist)