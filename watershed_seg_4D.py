
# coding: utf-8


# TO EXPLAIN



#get_ipython().magic(u'pylab inline')
#import nibabel as nb
#from nipy.labs.viz_tools.activation_maps import plolandscape
import nibabel as nb
from nilearn import image
from soma import aims
import numpy as np
#import matplotlib.pyplot as plt
from scipy import ndimage
#from nipy.io.api import load_image
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import os
import glob


#parameters and paths
i=0
dist = 3.5
t=0.1
landscape_4D = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/AVCnn/resultats/ICA/controls_all_DictionaryLearning_alpha15_20_t_auto.nii'
mask_file = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/AVCnn/masks/ext_filled_voronoi_resampled.nii'
out_dir ='/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/Labelmaps_clean_watershed_thresh01_d35'

noise_maps = [11,15] #unwanted maps (noise, white...)
#loop on map list
for landscape in image.iter_img(landscape_4D):

	name = 'map_'+str(i)
	
	
	if i not in noise_maps:
		landscape_img = landscape
		voronoi_mask_img = nb.load(mask_file)
		voronoi_mask_r = image.resample_img(voronoi_mask_img, target_affine=landscape_img.affine,
										  target_shape = landscape_img.shape,
										  interpolation='nearest')

		landscape_data=landscape_img.get_data()
		thresh = t*np.max(landscape_data)
		landscape_data[landscape_data<=thresh] = 0

		voronoi_mask_data = voronoi_mask_r.get_data()
		voronoi_mask_data[landscape_data<=thresh] = 0

		mask_LH = np.array(voronoi_mask_data == 2,dtype=np.bool)
		mask_RH = np.array(voronoi_mask_data == 1,dtype=np.bool)
		mask_cereb = np.array(voronoi_mask_data == 3,dtype=np.bool)

		local_maxi = peak_local_max(landscape_data, indices=False, min_distance=dist, labels=mask_LH, )
		markers = ndimage.label(local_maxi)[0]
		flooded = watershed(-landscape_data, markers, mask=mask_LH)
		segmented = os.path.join(out_dir,name+'_LH_segmented_dist_'+str(dist)+'.nii')
		flooded_img = nb.Nifti1Image(flooded, landscape_img.affine)
		nb.save(flooded_img,segmented)

		local_maxi = peak_local_max(landscape_data, indices=False, min_distance=dist, labels=mask_RH, )
		markers = ndimage.label(local_maxi)[0]
		flooded = watershed(-landscape_data, markers, mask=mask_RH)
		segmented = os.path.join(out_dir,name+'_RH_segmented_dist_'+str(dist)+'.nii')
		flooded_img = nb.Nifti1Image(flooded, landscape_img.affine)
		nb.save(flooded_img,segmented)

		local_maxi = peak_local_max(landscape_data, indices=False, min_distance=dist, labels=mask_cereb, )
		markers = ndimage.label(local_maxi)[0]
		flooded = watershed(-landscape_data, markers, mask=mask_cereb)
		segmented = os.path.join(out_dir,name+'_cereb_segmented_dist_'+str(dist)+'.nii')
		flooded_img = nb.Nifti1Image(flooded, landscape_img.affine)
		nb.save(flooded_img,segmented)
	i += 1
        
	





