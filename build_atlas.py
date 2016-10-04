# coding: utf-8

import nibabel as nb
from nilearn import image
import numpy as np
from scipy import ndimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from copy import deepcopy
import os
import sys
import errno
import glob

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def ICAwatershed(landscape_4D,mask_file,studydir,nameMaps,noise_maps,distance=3.5,threshold=0.1,cervelet='non'):
    
    """
    Cette fonction permet de seuiller les cartes issues du Dictionnary Learning, puis d'executer l'algortithme de montée
    des eaux sur ces cartes.
    Parametres:
    - distance, float: pic à pic en mm, par défault 3.5mm. C'est la distance entre deux maxima d'intensités classiquement utilisé
    dans l'algorithme de segmentation 
    - threshold, float: seuil des cartes d'intensite, par défault 0.1
    - landscape_4D, string : chemin d'acces du fichier 4D contenant les cartes
    - mask_file, string : chemin d'acces contenant le masque du cerveau
    - studydir, string: chemin ou sera crée le dossier ou seront écrit les cartes
    - nameMaps, string : nom des cartes seuillés
    - noise_maps, list : les cartes d'ICA interprété comme du bruit ne seront pas pris en compte
    - cervelet, string : converser les rois avec des voxels de cervelet (default = yes)
   
    """
    i=0
    out_dir = studydir+'/'+nameMaps+'_'+'thresh'+str(threshold)+'_'+'dist'+str(distance)
  #  noise_maps = [11,15] #unwanted maps (noise, white...)
    mkdir_p(out_dir)
    for landscape in image.iter_img(landscape_4D):
        name = 'map_'+str(i)
        if i not in noise_maps:

            landscape_img = landscape
            voronoi_mask_img = nb.load(mask_file)
            voronoi_mask_r = image.resample_img(voronoi_mask_img, target_affine=landscape_img.affine,
                                              target_shape = landscape_img.shape,
                                              interpolation='nearest')

            landscape_data=landscape_img.get_data()
            thresh = threshold*np.max(landscape_data)
            landscape_data[landscape_data<=thresh] = 0

            voronoi_mask_data = voronoi_mask_r.get_data()
            voronoi_mask_data[landscape_data<=thresh] = 0

            mask_LH = np.array(voronoi_mask_data == 2,dtype=np.bool)
            mask_RH = np.array(voronoi_mask_data == 1,dtype=np.bool)
            mask_cereb = np.array(voronoi_mask_data == 3,dtype=np.bool)

            local_maxi = peak_local_max(landscape_data, indices=False, min_distance=distance, labels=mask_LH, )
            markers = ndimage.label(local_maxi)[0]
            flooded = watershed(-landscape_data, markers, mask=mask_LH)
            segmented = os.path.join(out_dir,name+'_LH_segmented_dist_'+str(distance)+'.nii')
            flooded_img = nb.Nifti1Image(flooded, landscape_img.affine)
            nb.save(flooded_img,segmented)

            local_maxi = peak_local_max(landscape_data, indices=False, min_distance=distance, labels=mask_RH, )
            markers = ndimage.label(local_maxi)[0]
            flooded = watershed(-landscape_data, markers, mask=mask_RH)
            segmented = os.path.join(out_dir,name+'_RH_segmented_dist_'+str(distance)+'.nii')
            flooded_img = nb.Nifti1Image(flooded, landscape_img.affine)
            nb.save(flooded_img,segmented)

            local_maxi = peak_local_max(landscape_data, indices=False, min_distance=distance, labels=mask_cereb, )
            markers = ndimage.label(local_maxi)[0]
            flooded = watershed(-landscape_data, markers, mask=mask_cereb)
            segmented = os.path.join(out_dir,name+'_cereb_segmented_dist_'+str(distance)+'.nii')
            flooded_img = nb.Nifti1Image(flooded, landscape_img.affine)
            nb.save(flooded_img,segmented)
        i += 1
    
    
    if cervelet == 'non':
        Cereb = glob.glob(out_dir+'/*cereb*.nii')
        reponse = input('Effacer les rois contenant une partie de cervelet? yes or no \n')
        if reponse == 'yes':
            for cerebs in Cereb:
                os.remove(cerebs)
        elif reponse == 'no':
            print('Les ROIs avec du cervelet sont conservées, il y en a',len(Cereb))
    else:
        print('Les ROIs avec des voxels appartenant au label du cervelet seront conservées')

def Labels_split(studydir,nameRois,nameMaps_dir,threshold_vox = 200):
    """
    Cette fonction éclate en rois 3D les cartes de ROIs issue de la fonction ICAwatershed.
    Paramètres:
    - studydir, string: nomm du dossier de travail
    - nameRois, string: prefixe du dossier ou seront écrits les ROIs
    - treshold_vox, float: seuil en voxels en deca duquel la région ne sera pas conservés. Par défautls 200 voxels.
    - nameMaps_dir, string: nom du dossier ou sont les cartes isssue de ICAwatershed.py 
    """
    
    label_map_dir = studydir + '/' + nameMaps_dir
    out_dir = studydir + '/' + nameRois + '_' + nameMaps_dir + '_' + str(threshold_vox) + 'vox'
    mkdir_p(out_dir)
    #numvox_tresh = 200
    roi_num=0

    for in_file in glob.glob(os.path.join(label_map_dir,'map*.nii')):


        image = nb.load(in_file)
        data = image.get_data()
        dim = image.shape

        for i in range(int(np.min(data)),int(np.max(data)+1)):
            out_data = deepcopy(data)    
            out_data [out_data!=float(i)]=0.
            out_data [out_data!=0.]=1. #binarize
            numvox = np.sum(np.asarray(np.ravel(out_data)))
            if numvox > threshold_vox:
                name = os.path.join(out_dir,'roi_'+str(roi_num)+str(i)+'.nii')
                out_img = nb.Nifti1Image(out_data, image.affine)
                nb.save(out_img,name)
                roi_num += 1

def create_3D_labels_file(studydir,roisDirName,name3Dlabelfile):
    

    #roi_dir = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/3Drois_clean_watershed_thresh01_d35_200vox'
    #out_dir = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/Labelmaps_clean_watershed_thresh01_d35'
    roi_list=glob.glob(os.path.join(studydir,roisDirName,'roi*.nii'))
    name=os.path.join(studydir,name3Dlabelfile+'.nii')
    new_dat = np.zeros(nb.load(roi_list[0]).shape)
    for i in range(len(roi_list)):
        roi_obj = nb.load(roi_list[i])
        roi_dat = roi_obj.get_data()
        print(str((i+1)*np.max(roi_dat)))
        new_dat = new_dat + (i+1)*roi_dat

    outimage = nb.Nifti1Image(new_dat, roi_obj.affine)
    nb.save(outimage,name)                
                
threshold_vox = 200               
landscape_4D = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/AVCnn/resultats/ICA/controls_all_DictionaryLearning_alpha15_20_t_auto.nii'
mask_file = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/AVCnn/masks/ext_filled_voronoi_resampled.nii'
studydir = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/TEST_ATLAS'
nameMaps = 'test'
noise_maps = [11,15]
nameRois = 'ROIs'
nameMaps_dir = 'test_thresh0.1_dist3.5'
roisDirName = nameRois + '_' + nameMaps_dir + '_' + str(threshold_vox) + 'vox'


ICAwatershed(landscape_4D,mask_file,studydir,nameMaps,noise_maps,cervelet='non')

Labels_split(studydir,nameRois,nameMaps_dir)

create_3D_labels_file(studydir,roisDirName,'test')
