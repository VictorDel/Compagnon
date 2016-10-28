# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:47:49 2015

@author: vd239549
"""
#create a 4d nifti file from a 3D nifti with different labels (integers): all those labels become 1 but are differentiated in the 4th dimension
#usefull to create atlases

 

import numpy as np
from copy import deepcopy
import glob
import os
import nibabel as nb

label_map_dir = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/Labelmaps_clean_watershed_thresh01_d35'
out_dir = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/3Drois_clean_watershed_thresh01_d35_200vox'
numvox_tresh = 200
roi_num=0

for in_file in glob.glob(os.path.join(label_map_dir,'map*.nii')):
	

	image = nb.load(in_file)
	data = image.get_data()
	dim = image.shape	

	for i in range(int(np.min(data)),int(np.max(data)+1)):
		out_data = deepcopy(data)    
		out_data [out_data<>float(i)]=0.
		out_data [out_data<>0.]=1. #binarize
		numvox = np.sum(np.asarray(np.ravel(out_data)))
		if numvox > numvox_tresh:
			name = os.path.join(out_dir,'roi_'+str(roi_num)+str(i)+'.nii')
			out_img = nb.Nifti1Image(out_data, image.affine)
			nb.save(out_img,name)
			roi_num += 1

		
	


