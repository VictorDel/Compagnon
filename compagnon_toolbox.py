
# coding: utf-8

# In[ ]:

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
from nilearn import plotting
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold, cross_val_score, StratifiedShuffleSplit


def partperm(List):
    #returns list of all possible partial permutations in lists of common first elements
    #for instance partperm[0,1,2,] = [[[0, 1], [0, 2]], [[1, 0], [1, 2]], [[2, 0], [2, 1]]]
    L=[]
    for f in List : 
        List_tmp = List[:]
        del(List_tmp[List.index(f)])
        L_tmp=[]        
        for g in List_tmp :
            L_tmp.append([f,g])
        L.append(L_tmp)    
    return L
    
def matReorg(mat,label,indices='none'):

    # Compute hierarchical clustering to reorganize matrix
    #mat matrix to reorganize
    #label : nom des indices de la matrice (dans l'ordre)
    if indices == 'none' :   
        Y = sch.linkage(mat, method='centroid')
        Z = sch.dendrogram(Y, orientation='right')
        indices = Z['leaves']
    new_mat=deepcopy(mat)
    new_mat = new_mat[indices,:]
    new_mat = new_mat[:,indices]
    new_label=[label[i] for i in indices]
    return new_mat,new_label,indices
    
def symetrize(M,tri = 'upper', diag = 1):
    #symetrize M according to 'upper' or 'lower' triangle and set diag to diag
    M_sym=deepcopy(M)             
    for i in range(np.shape(M)[0]):
        for j in range(np.shape(M)[1]):
            if i==j:
                M_sym[i][j] = diag
            else:
                if tri == 'lower':
                    if i<j :                    
                        M_sym[i][j] = M[j][i]
                elif tri == 'upper':
                    if i<j :
                        M_sym[np.shape(M)[0]-i-1][np.shape(M)[1]-j-1] = M[np.shape(M)[1]-j-1][np.shape(M)[0]-i-1]
    return M_sym    

def scaling_PSC(time_serie,time_dim):
    T=100*time_serie/np.mean(time_serie,axis=time_dim)
    return T

