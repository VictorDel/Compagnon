# coding: utf-8


### imports ###
import os
import numpy as np
import nibabel as nb
import glob
import matplotlib
import re
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import nilearn.decomposition
import scipy
import nilearn.connectome
from matplotlib.backends import backend_pdf
from copy import deepcopy
from nilearn import image
from sklearn import covariance 
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker
from nilearn import regions
from nilearn import signal
from nilearn import plotting
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold, cross_val_score, StratifiedShuffleSplit


### home made functions ###
import compagnon_statistique as cp_stats
import compagnon_toolbox as cp_tools
import compagnon_visualize as cp_plt


### memory cache ###
mem_dir = '/media/vd239549/LaCie/victor/nilearn_cache'



### Numeric parameters for initial signal processing ###
TR = 2.4 #volume acquisition time
mask = None #mask to apply to the functional images
smooth = None #spatial smoothing in mm or None
LP_filt = None #low pass filtering : value in Hz or None
HP_filt = None #High pass filtering : value in Hz or None
stdz = True #standardize time series
detr = True #detrend time series



### chose estimator ###
estimator = covariance.LedoitWolf(assume_centered=True)   
#GroupSparseCovarianceCV(n_jobs=-1,assume_centered=True)
#GraphLassoCV(n_jobs=-2,assume_centered=True)
#EmpiricalCovariance(n_jobs=-1,assume_centered=True)



### chose metrics to compute ttest on:  'partial correlation', 'correlation','covariance','precision','tangent' ###
kinds = ['partial correlation','correlation', 'tangent'] 
kind_comps=['partial correlation','correlation','tangent'] #metric for classification
p=0.05 #significativity for display
MC_correction = 'FDR' #chose correction for multiple comparisons 'Bonferoni' or 'FDR'
stat_type = 'p' #choose parametric or non parametric ('np') test, see _NPtest and _ttest2 to see which test are implemented 
classif = True #reform svm classification
Paired = False #should the ttests be paired or not 
MatReorg = False #should correlogram be reorganized according to correlation clusters 



### set up names and filters ###
atlas_name = 'atlas_indiv_func'               
atlas_indiv_dir = atlas_name
root= '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/AVCnn/resultats/'
atlas_indiv_dir = os.path.join(root,atlas_indiv_dir)
func_type_list = [ 'controls_all','patients_all']#  #name of each group's directory for functional images
reg_dirs = [ ]#name of each group's directory for regressors (regressor have to be .txt files)
reg_suffix='.txt'
atlas_dirs = [ atlas_indiv_dir,atlas_indiv_dir]#directory containing individual atlases
atlas_suffix = '.nii'
label_suffix = '.csv' #suffix for atlas labels



### choose report directory and name (default location is in root, default name is atlas_name ###
main_title ='AVCnn_c_p_' 
save_dir = os.path.join(root,'reports_new_atlas')
try:
    os.makedirs(save_dir)
except:
    print('Warning could not make dir '+save_dir)
    pass
save_report=os.path.join(save_dir, main_title+'_'+atlas_name+'_LW.pdf')
if not save_report:
    save_report = os.path.join(save_dir, main_title+'_'+atlas_name+'_defaults.pdf')


### reference files for atlas checks, atlas labels and target affine ###
ref_dir = os.path.join(root,'references')
anat_ref_file=glob.glob(os.path.join(ref_dir,'*anat*.nii*'))
if len(anat_ref_file)>1:
    print('Warning: several anat reference files: '+anat_ref_file[0]+' will be used')
anat_ref = nb.load(anat_ref_file[0])
func_ref_file=list(set(glob.glob(os.path.join(ref_dir,'art*.nii*'))) - set(glob.glob(os.path.join(ref_dir,'*anat*.nii*'))))
if len(func_ref_file)>1:
    print('Warning: several func reference files: '+func_ref_file[0]+' will be used')
func_ref = nb.load(func_ref_file[0])
ref_atlas =  '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/AVCnn_4D_networks.nii' 
display_atlas= nilearn.plotting.plot_prob_atlas(ref_atlas, anat_img=anat_ref_file[0],
                                                title=atlas_name+'_anat',
                                                cut_coords = (5,0,0),threshold=0.)
atlas_ref_labels = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/AVCnn_roi_labels_networks.csv'
labels_ref = open( atlas_ref_labels).read().split()



### set up roi colors and compute roi coordonates ###
networks = [3,9,6,8,3,11,5,7,7,6] 
networks_colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,0.5],[0,1,1],[0,0,0],[0.5,1,0],[1,0,0.5],[0,1,0.5]]
if not networks :
    label_colors = np.random.rand(len(labels_ref),3)
else :
    if not networks_colors :
        networks_colors = np.random.rand(len(networks),3)
    label_colors = np.zeros([len(labels_ref),3])
    label_colors[0:networks[0]]= networks_colors[0]
    for i in range(len(networks)-1):
        label_colors[np.sum(networks[0:i]):np.sum(networks[0:i+1])]= networks_colors[i+1]                  
coords_ref  =[plotting.find_xyz_cut_coords(roi) for roi in image.iter_img(ref_atlas)] 
rois_ref = np.asarray(labels_ref)
n_r = len(rois_ref)
l=300./n_r#roi label size in figures     
visu_ref = ref_atlas
at_check =[plt.gcf()] #figure handle for atlas check




### prepare pairs for pairwise comparisons between groups ###   
comps = [] 
for func_index in range(len(func_type_list)-1) :
    if func_index != len(func_type_list)-func_index:    
        for i in range(func_index,len(func_type_list)-func_index):
            if i+1<len(func_type_list):                
                comps.append([func_type_list[func_index],func_type_list[i+1]])
    else:
        comps.append([func_type_list[func_index],func_type_list[func_index+1]])
Bonf = len(comps)


### initialize dictionaries ###
all_time_series_r = {} #regressed time series
t_s_r_file = {} #names for saving file regressed time series
all_time_series = {} #detrended and standardized time series 
t_s_file={} #names for saving file detrended and standardized time series
all_regressors ={} #regressors
r_file={} #names for saving file regressors
non_void_file= {} #names for saving non void inices 
non_void_indices_all={} #indices of void regions in individual atlases
non_void_list_all={}


### start loop over groups ###
for func_type in func_type_list :
    func_index=func_type_list.index(func_type)
    # initialize variables 
    time_series_r = []
    time_series = []
    regressors = []
    non_void_indices=[]
    # select all functional images files 
    func_imgs =  glob.glob(os.path.join(root,func_type+'/*.nii*'))     
    if not func_imgs:
         print('No functional files for '+func_type+' !')
    
    # choose random subject to check atlas and functional file normalization on random functional file 
    random_sub =  np.random.randint(0,len(func_imgs)) 
    
    # select matching regressor files
    for f_name in func_imgs:            
        f=func_imgs.index(f_name)
        nipObj =  re.search(r'..\d{6}',f_name)
        nip = nipObj.group(0)
        if reg_dirs :
            reg_file = glob.glob(os.path.join(reg_dirs[func_index],'*'+nip+'*'+reg_suffix))
            print('could not find matching regressor for file '+f_name+' in '+ reg_dirs[func_index])
        else:
            reg_file=[]         
                 

        #select matching atlas file
        atlas_filename = glob.glob(os.path.join(atlas_dirs[func_index],'*'+nip+'*'+atlas_suffix))[0]   
        labels = open(glob.glob(os.path.join(atlas_dirs[func_index],'*'+nip+'*'+label_suffix))[0]).read().split() 
        coords =[plotting.find_xyz_cut_coords(roi) for roi in image.iter_img(atlas_filename)] 
        rois = np.asarray(labels)     
        visu = atlas_filename        
        non_void_indices.append(np.where(rois != 'void')[0])
        print('void roi: '+str(np.where(rois == 'void')[0]))
        #check atlas and functional file normalization on random functional file
        if f == random_sub:
                display_atlas= nilearn.plotting.plot_prob_atlas(atlas_filename,anat_img=nilearn.image.index_img(
                                                    func_imgs[f], 0),title=atlas_name+
                                                    '_'+func_type,cut_coords = (5,0,0),
                                                    threshold=0.)        
                at_check.append(plt.gcf())
                plt.close()

        # extracting time series according to atlas    
        if func_imgs[f]:                        
            time_series_subj_raw = regions.img_to_signals_maps(func_imgs[f],atlas_filename)[0]
            time_series.append(time_series_subj_raw)
            if not reg_file:               
                time_serie_r=signal.clean(time_series_subj_raw,standardize=stdz,
                                          detrend=detr,low_pass=LP_filt,
                                          high_pass=HP_filt, t_r=TR)
                print('no confounds removed')                
            else:
                
                time_serie_r = signal.clean(time_series_subj_raw,standardize=stdz,
                                          detrend=detr,low_pass=LP_filt,
                                          high_pass=HP_filt, t_r=TR, confounds=reg_file)
                regressors.append(np.loadtxt(reg_file[0]))
                               
            time_series_r.append(time_serie_r)    
            progress = np.round(100*( (float(f)+1.)/len(func_imgs)))            
            print(str(progress) + '% done in computing time series for '+func_type)
        
    
    
            
    # update dictionary containing all data
    non_void_indices_all[func_type] = non_void_indices
    
    all_time_series_r[func_type] = time_series_r
    all_time_series[func_type] = time_series    
    all_regressors[func_type] = regressors
    # save time series and regressors_
    t_s_r_file[func_type] = save_dir+'/'+func_type+'_'+atlas_name+'_time_series_r.npy'
    t_s_file[func_type] = save_dir+'/'+func_type+'_'+atlas_name+'_time_series.npy'
    r_file[func_type] = save_dir+'/'+func_type+'_'+atlas_name+'_regressors.npy'
    non_void_file[func_type] = save_dir+'/'+func_type+'_'+atlas_name+'_non_void.npy'    
    
    np.save(t_s_r_file[func_type],np.asarray(all_time_series_r[func_type]))
    np.save(t_s_file[func_type],np.asarray(all_time_series[func_type]))
    np.save(r_file[func_type],np.asarray(all_regressors))
    
    
    # build list of roi couples to consider in connectivity measures
    non_void_list=[[[]*n_r]*n_r]*n_r
    l_unravelled=[]
    for i in range(n_r):
        for j in range(n_r):
            l_tmp=[]        
            for s in range(len(non_void_indices_all[func_type])):
                if i in non_void_indices_all[func_type][s] and j in non_void_indices_all[func_type][s]:                                        
                    l_tmp.append(s)    
            
            l_unravelled.append(l_tmp)
            
    if len(np.shape(l_unravelled))>1:
        non_void_list = np.reshape(l_unravelled,(n_r,n_r,np.shape(l_unravelled)[-1]))
    else:
        non_void_list = np.reshape(l_unravelled,(n_r,n_r))
                    
    non_void_list_all[func_type] =     non_void_list    
    non_void_file[func_type] = save_dir+'/'+func_type+'_'+atlas_name+'_non_void.npy'
    np.save(non_void_file[func_type],np.asarray(non_void_list_all[func_type]))
    
    

# Compute connectivity metrics

individual_connectivity_matrices = {}
mean_connectivity_matrix = {}
p_ind = 0.
for func_type in func_type_list:
    subjects_connectivity = {}
    mean_connectivity = {}     
    for kind in kinds:
        try:
            conn_measure = nilearn.connectome.ConnectivityMeasure(cov_estimator =estimator, kind=kind)
            t_s=np.asarray(all_time_series_r[func_type])
            #np.delete(t_s,void_indices_all[func_type])                                
            subjects_connectivity[kind] = conn_measure.fit_transform(t_s)
            
            if kind == 'tangent':
                mean_connectivity[kind] = conn_measure.mean_
                ###look into how the mean is computed for tangent space to account for missing regions
            else:
                moy = np.zeros([n_r,n_r])
                for i in range(n_r):
                    for j in range(n_r):
                        nvi = [subjects_connectivity[kind][k] [i][j] for k in non_void_list_all[func_type][i][j]]
                        moy[i][j]=np.mean(nvi,axis=0)
                mean_connectivity[kind] = moy

        except:
            print('estimation failed for '+ kind + ' in group ' + func_type)            
            pass
        p_ind = p_ind + 1.            
        progress = str(100*p_ind/(len(kinds)*len(func_type_list)))
        
        print(str(progress) + '% done in computing metrics ('+kind+' '+func_type+')')
        
    individual_connectivity_matrices[func_type] = subjects_connectivity
    mean_connectivity_matrix[func_type] = mean_connectivity
    
comp_list=cp_tools.partperm(func_type_list)

##compute stats and write report   
with backend_pdf.PdfPages(save_report) as pdf:
    for g_index in range(len(func_type_list)+1):    
        pdf.savefig(at_check[g_index])
        plt.close()
    #tstats for comparison of metrics  accross groups
    for kind in kinds:
        print('saving report: '+kind)
        for func_type in func_type_list :
            #average across all subjects            
            Mean_mat = mean_connectivity_matrix[func_type][kind]
            Mean_tot = np.mean(Mean_mat)            
            #plot connectomes                    
            plotting.plot_connectome(Mean_mat, coords_ref,node_color=label_colors,
                                     title= func_type +' '+ kind+' connectome',
                                     edge_threshold='90%')                                
            pdf.savefig()    
            plt.close()            
            if kind in ['correlation','partial correlation']:
                span = [-1,1]
            else:
                m_span = np.max(np.abs(Mean_mat))                
                span = [-m_span,m_span]
            #reorganize matrix with hierarchical clustering
            if MatReorg :
                Mean_mat_r,rois_r,I = cp_tools.matReorg(Mean_mat,labels_ref)
            else:
                Mean_mat_r = Mean_mat
                rois_r = labels_ref
                I = range(n_r)
                    
            new_label_colors=[label_colors[i] for i in I] 
            cp_plt.plot_matrices(Mean_mat_r,span ,rois_r,new_label_colors, 'Average '+func_type + ' ' + kind+ ' across subjects\nTotal average for '+kind+' = '+str(Mean_tot) ,colmap ="bwr",labelsize=l) 
            pdf.savefig()
            plt.close()
            
        for comp in comps :
            paired = Paired
            if kind in ['correlation', 'partial correlation']:
                # Z-Fisher transform
                g1 = cp_stats.corr_to_Z(np.asarray(individual_connectivity_matrices [comp[0]] [kind]))    
                g2 = cp_stats.corr_to_Z(np.asarray(individual_connectivity_matrices [comp[1]] [kind]))
            else :
                
                g1 = np.asarray(individual_connectivity_matrices[comp[0]][kind] )    
                g2 = np.asarray(individual_connectivity_matrices [comp[1]][kind] )
            # set nans and inf to 1
            g1[np.isinf(g1)] = 1.
            g2[np.isinf(g2)] = 1.
            g1[np.isnan(g1)] = 1.
            g2[np.isnan(g2)] = 1.        


                
            # statistical test between g1 and g2
            testies_1 = np.zeros(n_r)
            testies_2 = np.zeros(n_r)
            t2 = np.zeros([2,n_r,n_r])
            if stat_type =='np':

                if kind == 'tangent':
                    for i in range(n_r):
                        for j in range(n_r):
                            testies_1 = np.asarray([g1[k] [i][j] for k in non_void_list_all[comp[0]][i][j]])
                            testies_2 = np.asarray([g2[k] [i][j] for k in non_void_list_all[comp[1]][i][j]])
                            # check if paired ttest is possible
                            if len(non_void_list_all[comp[0]][i][j])!= len(non_void_list_all[comp[1]][i][j]) and Paired==True:
                                print ('Warning: number of subjects different between ' + comp[0] + ' and ' +  comp[1] + ' for '+ kind)
                                print('Ttest cant be paired')
                                paired = False

                            t2[0][i][j]=cp_stats._ttest2(testies_1, testies_2, axis = 0, paired = paired)[0]                
                            t2[1][i][j]=cp_stats._ttest2(testies_1, testies_2, axis = 0, paired = paired)[1]
                            t2[0]=mean_connectivity_matrix[comp[0]][kind]-mean_connectivity_matrix[comp[1]][kind] 
                else:   
                    for i in range(n_r):
                        for j in range(n_r):
                            testies_1 = np.asarray([g1[k] [i][j] for k in non_void_list_all[comp[0]][i][j]])
                            testies_2 = np.asarray([g2[k] [i][j] for k in non_void_list_all[comp[1]][i][j]])
                            # check if paired ttest is possible
                            if len(non_void_list_all[comp[0]][i][j])!= len(non_void_list_all[comp[1]][i][j]) and Paired==True:
                                print ('Warning: number of subjects different between ' + comp[0] + ' and ' +  comp[1] + ' for '+ kind)
                                print('Ttest cant be paired')
                                paired = False

                            t2[0][i][j]=cp_stats._ttest2(testies_1, testies_2, axis = 0, paired = paired)[0]                
                            t2[1][i][j]=cp_stats._ttest2(testies_1, testies_2, axis = 0, paired = paired)[1]
                
                #t2 = _ttest2(g1, g2, axis = 0, paired = paired)
            if MC_correction == 'FDR':            
                fdr_correction = cp_stats.fdr(t2[1][:][:])
                fdr_correction[np.isnan(fdr_correction)] =1.            
                t2_corrected = cp_stats.sym_fdr(fdr_correction) #fdr multiple comparison correction                                            
            elif MC_correction == 'Bonferoni':
                b_factor = (n_r*n_r)/2. #bonferoni correction factor = number of pairs of regions
                t2_corrected = t2[1][:][:]*b_factor 
            else :
                t2_corrected = t2[1][:][:]
                
            thresholded =  deepcopy(t2_corrected)
            thresholded[np.isnan(thresholded)] = 1.           
            thresholded[thresholded>p] = 1.            
            logarized = -np.log(thresholded)            
            sym = cp_tools.symetrize(logarized)            
            sym_mask = deepcopy(sym) 
            sym_mask[np.nonzero(sym)] = 1.
            sig_effect = np.multiply(sym_mask,t2[0][:][:])
            sig_value =  np.multiply(sym_mask,t2_corrected)
            sig_value[sig_value==0.]=1.
            plotting.plot_connectome(sig_effect, coords_ref,node_color=label_colors,
                                     title= comp[1] + ' VS '+ comp[0] + ' ' + kind + ' ' +' Paired = ' + str(paired) +' '+ MC_correction+' corrected p='+ str(p))                                
            pdf.savefig()    
            plt.close()
            if MatReorg:
                sig_value_r,labels_r,I_=cp_tools.matReorg(sig_value,labels,I)
            else:
                sig_value_r = sig_value
                labels_r = labels
                new_label_colors = label_colors
            cp_plt.plot_matrices(sig_value_r,[0,2*p] ,rois_r,new_label_colors,
                                 comp[1] + ' VS '+ comp[0] + ' ' + kind + ' ' + ' Paired = ' + str(paired)+' ' + MC_correction+' corrected',colmap ="hot",labelsize=l)                                           
            cp_plt.siglines(sig_value_r,p,new_label_colors,style = 'solid')
            pdf.savefig()
            plt.close()
            if MatReorg:
                t2_r = cp_tools.matReorg(t2[0][:][:],labels,I)
                labels_r = cp_tools.matReorg(t2[0][:][:],labels,I)[1]
                
            else:
                t2_r = t2
                labels_r = labels
                new_label_colors = label_colors
            cp_plt.plot_matrices(t2_r[0],[-np.max(np.abs(t2[0][:][:])),np.max(np.abs(t2[0][:][:]))],
                                 rois_r,new_label_colors, comp[1] + ' - '+ comp[0] + ' ' + kind + ' effect',
                                 colmap ="bwr",labelsize=l)                                           
            pdf.savefig()
            plt.close()            
                                   
###classifier    
    if classif == True:
        for kind_comp in kind_comps:
            ## Use the connectivity coefficients to classify different groups
            classes = func_type_list
            mean_scores = []
            save_classif=open(os.path.join(save_dir,main_title+'_'+kind_comp+'_classif.txt'),'w') #txt report of classification scores
            save_classif.write('Classification accuracy scores\n\n')
            for comp in comps:    
                individual_connectivity_matrices_all=np.vstack((individual_connectivity_matrices[comp[0]][kind_comp],individual_connectivity_matrices[comp[1]][kind_comp]))
                block1 = np.hstack((np.zeros(len(individual_connectivity_matrices[comp[0]][kind_comp])),np.ones(len(individual_connectivity_matrices[comp[1]][kind_comp])))) 
                cv = StratifiedShuffleSplit(block1, n_iter=1000)
                svc = LinearSVC()
                #Transform the connectivity matrices to 1D arrays
                conectivity_coefs = nilearn.connectome.sym_to_vec(np.concatenate((individual_connectivity_matrices[comp[0]][kind_comp],
                                                                    individual_connectivity_matrices[comp[1]][kind_comp]),axis=0))              
                cv_scores = cross_val_score(svc, conectivity_coefs,block1, cv=cv, scoring='accuracy')
                mean_scores.append(cv_scores.mean())
                save_classif.write('using '+kind_comp + ',' + comp[0]+' VS '+comp[1]+ ':%20s score: %1.2f +- %1.2f' % (kind_comp, cv_scores.mean(),cv_scores.std()))
                save_classif.write('\n')

            save_classif.close()

            ### Display the classification scores
            plt.figure()
            ypos = np.arange(len(comps)) * .1 + .1

            plt.barh(ypos, mean_scores, align='center', height=.05)
            yticks = [comp[0] + '\nVS\n '+comp[1] +'\n' for comp in comps]
            plt.yticks(ypos, yticks)
            plt.xlabel('Classification accuracy')
            plt.grid(True)
            plt.title( 'pairwise classifications '+kind_comp)
            for acc in range(len(mean_scores)):
                score = str(np.round(mean_scores[acc],2))
                plt.figtext(mean_scores[acc],ypos[acc],score,weight='bold')
            pdf.savefig() 