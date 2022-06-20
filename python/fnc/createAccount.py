##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import os
import scipy.io as sio
from path import temp_database_path


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def createAccount(template, mask, name, exinfo):
	
	# Get file name for the account
	files = []
	for file in os.listdir(temp_database_path):
	    if file.endswith(".mat"):
	        files.append(file)
	filename = str(len(files) + 1)

	# Save the file
	sio.savemat(temp_database_path + filename + '.mat',	\
		mdict={'template':template, 'mask':mask,\
		'name':name, 'exinfo':exinfo})

