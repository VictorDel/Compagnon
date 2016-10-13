import nibabel as nb
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
from nilearn import plotting
from nilearn import image
from nilearn.masking import apply_mask
from nilearn.image import math_img

from copy import deepcopy
import numpy as np
import os
import csv
import glob
import sys


def gen_atlas_indiv(atlas_template,labels_file_dir,root,basename,savedir,at_dir
                   ,roi_thresh,brain_mask_thresh,func_type,save_report,list_subjects,namelog):
    
    
    saveout = sys.stdout
    fsock = open(os.path.join(root,namelog+'.log'), 'w')
    sys.stdout = fsock
     

    print('Rois Thresold:',roi_thresh,
          'Brain Mask Treshold:',brain_mask_thresh,
          'Sequences:',func_type,
          'Saving dir:',savedir,
         )
    labels = np.recfromcsv(labels_file_dir)
    subjects = open(str(list_subjects),'r').read().split()
    labels_names_ref =labels['name'].T
    #gathering data from reference atlas
    template_img = nb.load(atlas_template) #image oject
    template_data =  np.asarray(template_img.get_data()) #array of data
    dim_template =template_data.shape # dimensions of array
    template_hdr = template_img #header (not used actually...)

    #generate atlases and pdf report
    with backend_pdf.PdfPages(save_report) as pdf:
        #loop over subject
        for s in subjects[0:2]:

            anat =  glob.glob(root+'/'+ s+'/'+func_type+'/anatstar/'+ 'wnobias_anat' + '*.nii') #individual brain mask of subject
            brain_mask = glob.glob(root+'/'+ s+'/'+func_type+'/anatstar/'+ 'wbrain' + '*.nii') #individual brain mask of subject

            labels_names_new = list(map(str,deepcopy(labels['name'].T)))

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

                if savedir == 'in_subjdir':

                    at_dir = os.path.join(root,s) #dir where individual atlas will be saved
                    at_name =   s+'_'+basename+'.nii'  #name of individual atlas
                    #at_name = os.path.join(at_dir,s+'_'+basename+'.nii')
                    nb.save(image.new_img_like(template_img,at_indiv_data),
                            os.path.join(at_dir,at_name))

                    #### create csv file for labels        
                    roi_label_file = open(os.path.join(at_dir,s+basename+'.csv'),'w+')
                    for r in range(n_roi):
                        roi_label_file.write(labels_names_new[r])
                        roi_label_file.write('\n')
                    roi_label_file.close()        
                    ### visual check of individual atlas on an image of your choice 
                    visu_check = anat   
                    display_atlas= plotting.plot_prob_atlas(os.path.join(at_dir,at_name),anat_img=visu_check,
                                                            title=s,
                                                            cut_coords = (0,0,0),threshold=0.)
                    pdf.savefig()
                    #plt.show()

                elif savedir == 'other':

                    at_name = os.path.join(at_dir,s+'_'+basename+'.nii')
                    nb.save(image.new_img_like(template_img,at_indiv_data),
                            os.path.join(at_dir,at_name))

                    #### create csv file for labels        
                    roi_label_file = open(os.path.join(at_dir,s+basename+'.csv'),'w+')
                    for r in range(n_roi):
                        roi_label_file.write(labels_names_new[r])
                        roi_label_file.write('\n')
                    roi_label_file.close()        
                    ### visual check of individual atlas on an image of your choice 
                    visu_check = anat   
                    display_atlas= plotting.plot_prob_atlas(os.path.join(at_dir,at_name),anat_img=visu_check,
                                                            title=s,
                                                            cut_coords = (0,0,0),threshold=0.)
                    pdf.savefig()

            else :
                print('Impossible to generate atlas for '+s+' (missing anat or brain mask)') 
    
    sys.stdout = saveout
    fsock.close()
    print('Le fichier log de la génération d\'atlas est enregistrés dans',os.path.join(root))
