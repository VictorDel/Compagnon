import gen_atlas_indiv_function


atlas_template = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/AVCnn.nii'  #reference atlas on which all individual atlases will be based
labels = np.recfromcsv('/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/AVCnnlabels.csv')#reference roi labels on which all individual atlases will be based
root = '/media/db242421/db242421_data/AVCnn_2016_DARTEL/AVCnn_data/patients'
list_subjects = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/ressources_txt/AVCnn_patients.txt'
basename = 'func_atlas'
func_type= 'RS1'
save_report = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/test_gen_atlas/atlas_func_report_patients.pdf'
at_dir = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/test_gen_atlas' #dir where to save the atlases 
roi_thresh = 0.5 #Threshold of voxel ratio for roi to be considered 
#if number of voxel in individual brain mask and roi of reference atlas / number of voxel in roi of reference atlas < threshold then roi will be empty
brain_mask_thresh = 0.1 #Threshold for mask binarization (times average intensity of mask : 0.1 means 10% of average mask intensity)
savedir = 'in_subjdir'# where will the atlases be save 'in_subjdir' means default dir where anats are if 'other' they are saved in at_dir
namelog='patients' #log will be saved in root


gen_atlas_indiv_function.gen_atlas_indiv(atlas_template,labels,root,basename,savedir,at_dir
                ,roi_thresh,brain_mask_thresh,func_type,save_report,list_subjects,namelog)