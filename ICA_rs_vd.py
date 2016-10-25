
#compute ICA and dictionary learning maps from functional data 


import glob
import nilearn.decomposition
import joblib
import os
import nibabel
from nilearn.plotting import (plot_prob_atlas, find_xyz_cut_coords, show,plot_stat_map)
from nilearn.image import index_img



mem_dir = '/media/vd239549/LaCie/victor/mmx/nilearn_cache'

#func_template = nibabel.load('/media/vd239549/LaCie/victor/mmx/comp_cont_pat/cont_all/swrarsc12_sub01_eb110475.nii')
func_template = nibabel.load('/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/AVCnn/resultats/controls_all/art_mv_fmv_wm_vent_ext_beat_hv_RSc12_sub03_ct110201.nii.gz')
# Numeric parameters for signal processing and ICA computation
#(http://nilearn.github.io/modules/generated/nilearn.decomposition.CanICA.html#nilearn.decomposition.CanICA)


n_components = 20 # number of independent components

TR = 2.4 #interscan time
cca = True # perform Canonical Correlation Analysis or not
thresh = 'auto' # intensity thresholding of components: auto keeps a number of voxel equal to the number of voxels (n) in the brain volume; a float f keeps f*n  
red_rat = 'auto' #reduction ratio for dictionary learning
n_epochs = 1 #
init = 10 # number of times the fastICA algorithm is restarted
mask = None #mask to apply to the functional images
smooth = None #spatial smoothing in mm or None
std = True #standardization of data : True or False
rand_state = 0 # Pseudo number generator state used for random sampling
dtd = True #detrend
aff = func_template.affine # If specified, the image is resampled corresponding to this new affine. 
target = None # If specified, the image will be resized to match this new shape.  
LP_filt = None #low pass filtering : value in Hz or None
HP_filt = None #High pass filtering : value in Hz or None
alpha = 15 #sparsity controlling parameter 
msk_strat = 'background' #masking strategy
msk_a = None #mask arg
m_level = 2
n_jobs = -1
verbose = 0

####ICA and dictionary learning for resting state
#(http://nilearn.github.io/auto_examples/03_connectivity/plot_compare_resting_state_decomposition.html#sphx-glr-auto-examples-03-connectivity-plot-compare-resting-state-decomposition-py)

root='/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/AVCnn/resultats/'
save_dir = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/AVCnn/resultats/ICA'
func_type_list = [ 'controls_all']#,'LG_RSc','LD_RSc','MD','MG','patients_all','Bac_RSc','Sup_RSc']# 'all', 'median','young','old''topi_p', 'wotopi_p','cont_p'  #name of each group's directory for functional images
save_dir =os.path.join( save_dir)

try:
    os.makedirs(save_dir)
except:
    print('Warning could not make dir '+save_dir)
    pass



for func_type in func_type_list :
    # select all functional images files 
    func_imgs =  glob.glob(root+func_type+'/*.nii*')
         
    if not func_imgs:
         print('No functional files for '+func_type+' !')
	
    # Create CanICA object (http://nilearn.github.io/modules/generated/nilearn.decomposition.CanICA.html#nilearn.decomposition.CanICA)
    
    canica = nilearn.decomposition.CanICA(mask=None, n_components=n_components, smoothing_fwhm=smooth, do_cca=cca,
                                          threshold=thresh, n_init=init, random_state=rand_state, standardize=std,
                                          detrend=dtd, low_pass=LP_filt, high_pass=HP_filt, t_r=TR, target_affine=aff,
                                          target_shape=target, mask_strategy=msk_strat, mask_args=msk_a, memory=mem_dir,
                                          memory_level=m_level, n_jobs=n_jobs, verbose=verbose)
                                          
    dict_learning = nilearn.decomposition.DictLearning(n_components=n_components, n_epochs=n_epochs,
                                                       alpha=alpha, reduction_ratio=red_rat,
                                                       dict_init=None, random_state=rand_state,
                                                       mask=mask, smoothing_fwhm=smooth, standardize=std,
                                                       detrend=dtd, low_pass=LP_filt, high_pass=HP_filt,
                                                       t_r=TR, target_affine=aff, target_shape=target, 
                                                       mask_strategy=msk_strat, mask_args=msk_a, memory=mem_dir,
                                                       memory_level=m_level, n_jobs=n_jobs, verbose=verbose)    
        
    
    estimators = [dict_learning, canica]
    names = {dict_learning: 'DictionaryLearning', canica: 'CanICA'}
    components_imgs = []
    for estimator in estimators:
        print('Learning maps using %s model' % names[estimator])
        estimator.fit(func_imgs)
        print('Saving results')
        # Decomposition estimator embeds their own masker
        masker = estimator.masker_
        # Drop output maps to a Nifti   file
        components_img = masker.inverse_transform(estimator.components_)
        components_img.to_filename(os.path.join(save_dir,func_type+'_'+ names[estimator]+'_alpha15_'+str(n_components)+'_t_'+str(thresh)+'.nii'))
        components_imgs.append(components_img)


#### visu maps
ica_map =   '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/AVCnn/resultats/ICA/controls_all_DictionaryLearning_alpha15_20_t_auto.nii'
for i in range( n_components):
	img=  image.index_img(ica_map,i)   
	display = nilearn.plotting.plot_glass_brain(img,title = 'comp number ' + str(i))        

###
plotting.plot_prob_atlas(ica_map, view_type='filled_contours',
                         title='Dictionary Learning maps')


##### build regions
from nilearn.regions import RegionExtractor

extractor = RegionExtractor(ica_map, threshold=0.5,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True, min_region_size=1350)
# Just call fit() to process for regions extraction
extractor.fit()
# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Each region index is stored in index_
regions_index = extractor.index_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title='atlas de ouf')

####

## Selecting specific maps to display: maps were manually chosen to be similar
#indices = {dict_learning: 1, canica: 19}
## We select relevant cut coordinates for displaying
#cut_component = index_img(components_imgs[0], indices[dict_learning])
#cut_coords = find_xyz_cut_coords(cut_component)
#for estimator, components in zip(estimators, components_imgs):
#    # 4D plotting
#    plot_prob_atlas(components, view_type="filled_contours",
#                    title="%s" % names[estimator],cut_coords=cut_coords, colorbar=False)
#    # 3D plotting
#    plot_stat_map(index_img(components, indices[estimator]),title="%s" % names[estimator],
#                  cut_coords=cut_coords, colorbar=False)
#show()

















#    # Compute ICA across subjects
##    print('computing ICA')
##    ICA = canica.fit(func_imgs)
##    
##    # Retrieve the independent components in brain space and save them as a nii image to savedir
##    
##    components_img = canica.masker_.inverse_transform(canica.components_)
##    components_img.to_filename(os.path.join(save_dir,func_type+'_canica_'+str(n_components)+'_t_'+str(thresh)+'.nii'))
#
#    
#    ## (http://nilearn.github.io/auto_examples/03_connectivity/plot_compare_resting_state_decomposition.html#sphx-glr-auto-examples-03-connectivity-plot-compare-resting-state-decomposition-py)
#    # Dictionarry learning    
#    dict_learning = nilearn.decomposition.DictLearning(n_components=n_components,memory="nilearn_cache",
#                                                       memory_level=2,verbose=1,random_state=0,n_epochs=1)
#    D_L = dict_learning.fit(func_imgs)
#    D_L_components_img=D_L.masker_.inverse_transform(D_L.components_)
#    D_L_components_img.to_filename(os.path.join(save_dir,func_type+'_D_L_'+str(n_components)+'_t_'+str(thresh)+'.nii'))
 




	

		


