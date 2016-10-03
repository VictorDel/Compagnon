


#ne marche que si il n'y a pas d'overlap entre les regions


import numpy as np
from copy import deepcopy
import glob
import os
import nibabel as nb


roi_dir = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/3Drois_clean_watershed_thresh01_d35_200vox'
out_dir = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/Labelmaps_clean_watershed_thresh01_d35'
roi_list=glob.glob(os.path.join(roi_dir,'roi*.nii'))
name=os.path.join(out_dir,'3Dlabels_200vox.nii')
new_dat = np.zeros(nb.load(roi_list[0]).shape)
for i in range(len(roi_list)):
    roi_obj = nb.load(roi_list[i])
    roi_dat = roi_obj.get_data()
    print(str((i+1)*np.max(roi_dat)))
    new_dat = new_dat + (i+1)*roi_dat

outimage = nb.Nifti1Image(new_dat, roi_obj.affine)
nb.save(outimage,name)
    	
