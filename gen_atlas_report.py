
#create individual atlases fitting to individual brains based on a reference atlas and generating a report for all created atlases 

import nibabel as nb
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
from nilearn import plotting
from nilearn import image
from nilearn.masking import apply_mask
from nilearn.image import math_img
from soma import aims
from copy import deepcopy
import numpy as np
import os
import csv
import glob


### initializes paths and set parameters
atlas_template = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/AVCnn.nii'  #reference atlas on which all individual atlases will be based
labels = np.recfromcsv('/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/AVCnnlabels.csv')#reference roi labels on which all individual atlases will be based
labels_names_ref =labels['name'].T
root = '/media/vd239549/LaCie/victor/AVCnn/AVCnn_2016_DARTEL/patients'
subjects = open('/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/ressources_txt/AVCnn_patients.txt','r').read().split()
#AVCnn_cont_all.txt
#AVCnn_patients.txt
basename = 'func_atlas'
func_type= 'RS1'
save_report = '/media/vd239549/LaCie/victor/AVCnn/AVCnn_2016_DARTEL/patients/docs/Atlas/atlas_func_report_patients.pdf'
at_dir = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/AVCnn/resultats/atlas_indiv_func' #dir where to save the atlases 
roi_thresh = 0.5 #Threshold of voxel ratio for roi to be considered 
#if number of voxel in individual brain mask and roi of reference atlas / number of voxel in roi of reference atlas < threshold then roi will be empty
brain_mask_thresh = 0.1 #Threshold for mask binarization (times average intensity of mask : 0.1 means 10% of average mask intensity)

#gathering data from reference atlas
template_img = nb.load(atlas_template) #image oject
template_data =  np.asarray(template_img.get_data()) #array of data
dim_template =template_data.shape # dimensions of array
template_hdr = template_img #header (not used actually...)


#generate atlases and pdf report
with backend_pdf.PdfPages(save_report) as pdf:
	#loop over subject
	for s in subjects :
		
		anat =  glob.glob(root+'/'+ s+'/'+func_type+'/anatstar/'+ 'wnobias_anat' + '*.nii') #individual brain mask of subject
		brain_mask = glob.glob(root+'/'+ s+'/'+func_type+'/anatstar/'+ 'wbrain' + '*.nii') #individual brain mask of subject
		
		labels_names_new = map(str,deepcopy(labels['name'].T))
		
		if anat and brain_mask :
			print ('Generating atlas for '+s)
			anat =  anat[0]
			brain_mask = brain_mask [0]
			### resample brain mask to atlas template
			brain_mask_img = nb.load(brain_mask)
			brain_mask_r = image.resample_img(brain_mask_img, target_affine=template_img.affine,
											  target_shape = template_img.shape[0:-1],
											  interpolation='nearest')
			
			### binarize brain mask  
			avg_mask = np.mean(brain_mask_r.get_data())
			bin_thresh = "img>"+str(brain_mask_thresh*avg_mask)
			brainbin_img = math_img(bin_thresh, img=brain_mask_r) #binarize above thresh 			
			brain_bin_data = np.asarray(brainbin_img.get_data()) #get data in array

			### check atlas template on brain mask
			display_atlas= plotting.plot_prob_atlas(atlas_template,anat_img=brainbin_img,
													title='ref_test_'+s,
													cut_coords = (0,0,0),threshold=0.)
			pdf.savefig()
			
			### initialize data for individual atlas
			at_indiv_data =np.zeros(dim_template) #initialisation of data for individual atlas 
			n_roi = dim_template[3] #number of rois

			### loop over rois
			for r in range(n_roi):
				#gathering data for given roi from reference atlas
				roi_img = image.index_img(atlas_template,r)
				roi_data = np.asarray(roi_img.get_data())
				#computing intersection between roi of reference and brain mask 
				masked_data = np.multiply(roi_data,brain_bin_data)
				
				#counting voxels
				common_vox = np.sum(masked_data) #voxels in intersection
				roi_vox = np.sum(roi_data) #voxels of reference atlas roi
				#print(common_vox/roi_vox)
				if float(common_vox)/float(roi_vox)>roi_thresh:		
					at_indiv_data[:,:,:,r]= masked_data #add new intersection roi to individual atlas  
				else :
					print('roi number '+ str(r) + ' ('+ str(labels_names_ref[r]) + ') will be empty (ratio ='+ str(float(common_vox)/float(roi_vox))+')') 
					labels_names_new[r]='void' #prepare void rois labels for individual atlas csv
				
			#at_dir = os.path.join(root,s) #dir where individual atlas will be saved
			at_name =   s+'_'+basename+'.nii'  #name of individual atlas 
			nb.save(image.new_img_like(template_img,at_indiv_data),
					os.path.join(at_dir,at_name))
			
			#### create csv file for labels
			#with open(roi_label_file):
			#with open(os.path.join(at_dir,s+'_atlas.csv'), 'wb') as csvfile:	
				#labelwriter = csv.writer(csvfile, delimiter=' ')
				#for r in range(n_roi):
					#labelwriter.writerow(labels_names_new[r])			
			roi_label_file = open(os.path.join(at_dir,s+basename+'.csv'),'w')
			for r in range(n_roi):
					roi_label_file.write(str(labels_names_new[r]))
					roi_label_file.write('\n')
			roi_label_file.close()		
			### visual check of individual atlas on an image of your choice 
			visu_check = anat   
			display_atlas= plotting.plot_prob_atlas(os.path.join(at_dir,at_name),anat_img=visu_check,
													title=s,
													cut_coords = (0,0,0),threshold=0.)
			pdf.savefig()
			#plt.show()
		else :
			print('Impossible to generate atlas for '+s+' (missing anat or brain mask)') 
				

