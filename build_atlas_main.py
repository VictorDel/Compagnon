# -*- coding: utf-8 -*-
import function_build_atlas

"""
Fichier main de la création d'atlas.
Les fonction a importer sont dans le fichier function_build_atlas.py

"""

#distance entre deux maximas locaux pour initier l'algorithme de montée des eaux
distance = 3.5
#seuil pour les cartes du Dictionnay learning
threshold = 0.1
#taille de la plus petite région de l'atlas
threshold_vox = 200 
#fichier ou se trouve les cartes brutes isssue du dictionnary
landscape_4D = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/AVCnn/resultats/ICA/controls_all_DictionaryLearning_alpha15_20_t_auto.nii'
#Masque utilisé au cours de la segmenation par du Watershed
mask_file = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/AVCnn/masks/ext_filled_voronoi_resampled.nii'
#Dossier d'etude ou seront stockés les différents fichiers necessaire à la création de l'atlas
studydir = '/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/atlases/atlas_fonctionel_control_AVCnn/TEST_ATLAS'
#suffixe du dossier ou seront crées les cartes issue de ICAwatershed
nameMaps = 'AVCnn'
#numéro des cartes non pertinentes isssue du dictionnary learning
noise_maps = [11,15]
#suffixe du dossier ou seront les ROIs sortie de Labels_split
nameRois = 'ROIs'
#nom du dossier ou sont stockés les cartes sortie de ICAwatershed
nameMaps_dir = nameMaps + '_' + 'thresh' + str(threshold) + '_' + 'dist' + str(distance)                           
#nom du dossier ou sont stockés les ROIs sortie de Labels_split
roisDirName = nameRois + '_' + nameMaps_dir + '_' + str(threshold_vox) + 'vox'
#nom du fichier 3D de labels
name3Dlabelfile = 'AVCnn'
#presence du cervelet dans l'atlas ?
cervelet = 'non'
atlas_4D_name = 'AVCnn4D'

#Seuil et segmentation des cartes avec un algorithme montée des eaux
#pour voir l'aide de la fonction : ICAwatershed?
function_build_atlas.ICAwatershed(landscape_4D,mask_file,studydir,nameMaps,noise_maps,distance,threshold,cervelet)

#Eclate en ROIs les cartes seuillé et segmentés isssue de ICAwatershed
#pour voi l'aide de la fonction: Labels_split?
function_build_atlas.Labels_split(studydir,nameRois,nameMaps_dir)

#Crée un fichier 3D de Labels avec les ROIs issue de Labels_split
#pour voir l'aide de la fonction: create_3D_labels_file?
function_build_atlas.create_3D_labels_file(studydir,roisDirName,name3Dlabelfile)

#Crée un fichier 4D a partir des ROIs issue de Labels_split au format nii.
function_build_atlas.concatenate_Nifti(studydir,roisDirName,name_file=atlas_4D_name)

#Enregistre les informations basiques concernant l'atlas généré dans le dossier studydir dans un .csv
function_build_atlas.basics_info_atlas(atlas_filedir=studydir,atlas_name=atlas_4D_name,threshold_carte=threshold
                                       ,distance_watershed=distance,presence_cereb=cervelet,threshold_vox=threshold_vox)
