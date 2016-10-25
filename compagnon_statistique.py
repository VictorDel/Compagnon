
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


def _NPtest(m1, m2, axis, paired):
    """non parametric tests

    Parameters
    ==========
    baseline : 3D numpy.ndarray
        Baseline matrices over subjects.
    follow_up : 3D numpy.ndarray
        Follow-up matrices over subjects.
    axis : int
        Subjects axis.
    paired : bool
        If True, the wilcoxonrank test is computed on two related samples of scores.
        else the mannwhitneyu test is computed
    Returns
    ======
    effect : array
        The mean effect, difference between follow_up and baseline.
    pval : array
        The p-values of the test.
    """   
        
    #scipy.stats.ks_2samp(data1, data2)
        
        
    #pval[i,j] = scipy.stats.ks_2samp(m1[:,i,j], m2[:,i,j]).pvalue
    effect = m2.mean(axis=axis) - m1.mean(axis=axis)
        
    pval = np.ones(m1[0].shape)
    
    for i in range(m1[0].shape[0]):
        for j in range(m1[0].shape[0]):
            if i!=j:
                if paired:
                    pval[i,j] = scipy.stats.wilcoxon(m1[:,i,j], m2[:,i,j],
                            zero_method='wilcox', correction=True).pvalue
                else:
                    #pval[i,j] = scipy.stats.mannwhitneyu(m1[:,i,j], m2[:,i,j],
                    #        use_continuity=True).pvalue
                    pval[i,j] = scipy.stats.ranksums(m1[:,i,j], m2[:,i,j]).pvalue        


    return effect, pval

def _ttest2(baseline, follow_up, axis, paired):
    """Paired or independent two samples Student t-test.

    Parameters
    ==========
    baseline : 3D numpy.ndarray
        Baseline matrices over subjects.
    follow_up : 3D numpy.ndarray
        Follow-up matrices over subjects.
    axis : int
        Subjects axis.
    paired : bool
        If True, the t-test is computed on two related samples of scores.

    Returs
    ======
    effect : array
        The mean effect, difference between follow_up and baseline.
    pval : array
        The p-values of the test.
    """
    effect = follow_up.mean(axis=axis) - baseline.mean(axis=axis)
    if paired:
        test = scipy.stats.ttest_rel
    else:
        test = scipy.stats.ttest_ind

    _, pval = test(baseline, follow_up, axis=axis)
    return effect, pval


def fdr(p):
    # TODO: rewrite in python style
    """ FDR correction for multiple comparisons.

    Computes FDR corrected p-values from an array of multiple-test false
    positive levels (uncorrected p-values) a set after removing nan values,
    following Benjamin & Hockenberg procedure.

    Parameters
    ==========
    p : numpy.ndarray
        Uncorrected p-values.

    Returns
    =======
    p_fdr : numpy.ndarray
        Corrected p-values.
    """
    if p.ndim == 1:
        n = p.shape[0]
        p_fdr = np.nan + np.ones(p.shape)
        idx = p.argsort()
        p_sorted = p[idx]
        n = np.sum(np.logical_not(np.isnan(p)))
        if n > 0:
            qt = np.minimum(1, n * p_sorted[0:n] / (np.arange(n)+1))
            min1 = np.inf
            for k in range(n - 1, -1, -1):
                min1 = min(min1, qt[k])
                p_fdr[idx[k]] = min1
    else:
        p_fdr = np.array([fdr(p[j]) for j in range(p.shape[1])])
    return p_fdr

def corr_to_Z(corr, tol=1e-7):
    """Applies Z-Fisher transform. """
    Z = corr.copy()  # avoid side effects
    corr_is_one = 1.0 - abs(corr) < tol
    Z[corr_is_one] = np.inf * np.sign(Z[corr_is_one])
    Z[np.logical_not(corr_is_one)] = np.arctanh(corr[np.logical_not(corr_is_one)])
    return Z

def LogL(M_1,M_2):
    #calculate log likelihood between two matrices  
    LL= -np.trace(np.dot(scipy.linalg.inv(M_1),M_2))/2-np.log(np.linalg.det(M_1))/2-np.shape(M_1)[0]/2*np.log(2*np.pi)
    return LL

def sym_fdr(M):
    M_sym_fdr=deepcopy(M)
    for i in range(np.shape(M)[0]):
        for j in range(i+1,np.shape(M)[1]):
            M_sym_fdr[i][j]=max(M[i][j],M[j][i])
            M_sym_fdr[j][i]=max(M[i][j],M[j][i])            
    return M_sym_fdr

