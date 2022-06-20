##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import argparse, os
from glob import glob
from tqdm import tqdm
from time import time
from scipy.io import savemat
from multiprocessing import cpu_count, Pool

from fnc.extractFeature import extractFeature


#------------------------------------------------------------------------------
#	Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="../CASIA1/",
					help="Path direktori citra iris CASIA1.")

parser.add_argument("--temp_dir", type=str, default="./templates/CASIA1/",
					help="Path direktori template.")

parser.add_argument("--n_cores", type=int, default=cpu_count(),
					help="Jumlah core yang digunakan pada proses enrolling template.")

args = parser.parse_args()


##-----------------------------------------------------------------------------
##  Fungsi .mat
##-----------------------------------------------------------------------------
def pool_func(file):
	template, mask, _ = extractFeature(file, use_multiprocess=False)
	basename = os.path.basename(file)
	out_file = os.path.join(args.temp_dir, "%s.mat" % (basename))
	savemat(out_file, mdict={'template': template, 'mask': mask})


##-----------------------------------------------------------------------------
##  Proses
##-----------------------------------------------------------------------------
start = time()

# Cek temp_dir
if not os.path.exists(args.temp_dir):
	print("makedirs", args.temp_dir)
	os.makedirs(args.temp_dir)

# List file enrolling template
files = glob(os.path.join(args.data_dir, "*_*_*.jpg"))
n_files = len(files)
print("Number of files for enrolling:", n_files)

# Enroll templates
print("Start enrolling...")
pools = Pool(processes=args.n_cores)
for _ in tqdm(pools.imap_unordered(pool_func, files), total=n_files):
	pass

end = time()
print('\n>>> Enrollment time: {} [s]\n'.format(end-start))