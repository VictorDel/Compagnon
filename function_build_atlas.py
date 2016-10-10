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
from nilearn.plotting import (plot_prob_atlas, find_xyz_cut_coords, show,plot_stat_map)
from nilearn.image import *
from nilearn.regions import RegionExtractor
from nilearn.image import concat_imgs, load_img
import csv


def mkdir_p(path):
    """
    Création d'un dossier.
    Parametres:
    - path, string: chemin complet du dossier a crée
    Cette fonction IGNORE l'erreur 17 qui se produit lorsque le dossier existe deja.
    """
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
    out_dir = os.path.join(studydir,nameMaps+'_'+'thresh'+str(threshold)+'_'+'dist'+str(distance))
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
        Cereb = glob.glob(os.path.join(out_dir,'*cereb*.nii'))
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
    
    label_map_dir = os.path.join(studydir,nameMaps_dir)
    out_dir = os.path.join(studydir,nameRois + '_' + nameMaps_dir + '_' + str(threshold_vox) + 'vox')
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
    """
    Cette fonction crée un crée un fichier 3D de label a partir des ROIs 3D issue de Labels_Split.
    Paramètres:
    studydir, string: nom du dossier d'étude de création de l'atlas
    roisDirName, string: nom du dossier ou se trouve les Rois issue de Labels_Split
    name3Dlabelfile, string: nom du fichier 3D qui sera crée au format NifTi.
    
    """
    
    roi_list=glob.glob(os.path.join(studydir,roisDirName,'roi*.nii'))
    name=os.path.join(studydir,name3Dlabelfile+'.nii')
    new_dat = np.zeros(nb.load(roi_list[0]).shape)
    
    for i in range(len(roi_list)):
        roi_obj = nb.load(roi_list[i])
        roi_dat = roi_obj.get_data()
        #print(str((i+1)*np.max(roi_dat)))
        new_dat = new_dat + (i+1)*roi_dat
    
    outimage = nb.Nifti1Image(new_dat, roi_obj.affine)
    nb.save(outimage,name)
    print('Vous avez généré un atlas de',len(roi_list),'ROIs, enregistrées dans le fichier 3D de labels',name3Dlabelfile+'.nii','dans le dossier',studydir)
    
def concatenate_Nifti(studydir,roisdirs_name,name_file):
    """
    Cette fonction crée un fichier 4D à partir de la liste des ROIs
    Paramètres:
    studydir, string: nom du dossier d'étude de création de l'atlas
    roisdirs_Name, string: nom du dossier ou se trouve les Rois issue de Labels_Split
    name_file, string: nom du fichier 4D généré qui constituera l'atlas
    
    """
    rois_files = glob.glob(os.path.join(studydir,roisdirs_name,'roi*.nii'))
    rois_concatenate = concat_imgs(rois_files)
    nb.save(rois_concatenate,os.path.join(studydir,name_file+'.nii'))
    print('Le fichier 4D contenant l\'atlas se trouve dans le dossier',studydir)
    
def basics_info_atlas(atlas_filedir,atlas_name,threshold_carte,threshold_vox,distance_watershed,presence_cereb):
    """
    Cette fonction affiche et enregistre dans un .csv les informations basiques concernant la taille de l'atlas. Elle reprend les
    paramètres des fonctions précedentes.
    """
    
    atlas4D = os.path.join(atlas_filedir,atlas_name+'.nii')
    
    atlas_img = load_img(atlas4D)
    atlas_img_data = atlas_img.get_data()
    dim_atlas_img = atlas_img_data.shape
    n_roi = dim_atlas_img[3]

    numvox_array = np.zeros(n_roi)

    for r in range(n_roi):
        roi_img = index_img(atlas_img,r)
        roi_data = roi_img.get_data()
        sum_vox = np.sum(roi_data)
        numvox_array[r] = sum_vox
        
    moy_vox = numvox_array.mean()
    min_roi_vox = numvox_array.min()
    max_roi_vox = numvox_array.max()
    data_dict = {"Nom de l'atlas":atlas_name,"Adresse de l'atlas":atlas4D,"Nombre de Roi":n_roi,
                 "Threshold des cartes DictionnaryLearning":threshold_carte,"Threhold nombre de voxels":threshold_vox,
                 "Distance pic à pic watershed":distance_watershed,"Prise en compte du cervelet":presence_cereb,
                 "Moyenne en voxel":moy_vox, "Région la plus petite":min_roi_vox, "Région la plus grande":max_roi_vox}
    
    with open(atlas_filedir+'/'+atlas_name+'.csv','w+') as basic_info_csv:
        writer = csv.writer(basic_info_csv)
        for key, value in data_dict.items():
            writer.writerow([key, value])

    