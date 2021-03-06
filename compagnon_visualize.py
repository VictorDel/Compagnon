
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

    
def plot_matrices(mat,span ,labels,label_colors, title,colmap="bwr",labelsize=4 ):
    #plot matrices
    #define titles and colors
    fig = plt.figure()
    ax = plt.subplot(111)
    cax = ax.imshow(mat, interpolation="nearest",vmin=span[0], vmax=span[1],cmap=plt.cm.get_cmap(colmap))
    plt.title("%s" % title)
    # Add labels and adjust margins
    ax.set_xticks([i for i in range(0, len(labels))])
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks([i for i in range(0, len(labels))])
    ax.set_yticklabels(labels)
    ax.yaxis.tick_left()
    for xtick, color in zip(ax.get_xticklabels(), label_colors):
        xtick.set_color(color)
        xtick.set_fontsize(labelsize)
    for ytick, color in zip(ax.get_yticklabels(), label_colors):
        ytick.set_color(color)
        ytick.set_fontsize(labelsize)
    #plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)    
    cbar=fig.colorbar(cax,ticks = span)
    cbar.ax.set_yticklabels(span)
    #plt.colorbar.make_axes(location="left")    
    
    
def siglines(mat,sig,colors,style = 'solid'):
    mat=np.asarray(mat)
    ind_mat=np.asarray(np.where((mat<=sig) &(mat>0.))).T
    for c in ind_mat:  
        if c[0]>=c[1]:
            plt.plot([0,c[1]],[c[0],c[0]],linestyle =style,color = colors[c[0]])
            plt.plot([c[1],c[1]],[c[0],len(mat)],linestyle =style,color = colors[c[1]])

