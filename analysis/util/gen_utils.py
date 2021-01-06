import pandas as pd
import numpy as np
import scipy.stats
import pickle as pkl
import mygene
from gprofiler import GProfiler

from sklearn import cluster, preprocessing
from kneed import KneeLocator

from statsmodels.stats import multitest
from util.class_def.obj_classes import *

def read_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        return pkl.load(f)

def write_pkl(obj, pkl_file):
    with open(pkl_file, "wb") as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def read_sample_meta_table(file_path, index_col = 0):
    return pd.read_csv(file_path, index_col = index_col)

def multitest_corr(pvals, method = "bonferroni"):
    return multitest.multipletests(pvals, method = method)

def test_and_adj(test_combos, data, label_col, col_to_test, alternative, use_ttest = False, corr_method = "bonferroni"):
    pvals = {}
    for label_i, label_j in test_combos:
        x_data_mask = (data.loc[:, label_col] == label_i) 
        y_data_mask = (data.loc[:, label_col] == label_j) 
        
        x_data = data.loc[x_data_mask, col_to_test].to_numpy()
        y_data = data.loc[y_data_mask, col_to_test].to_numpy()

        #Order here matters - per source code: We interpret one-sided tests as asking whether y is 'test (greater or less)' than x
        #For ttest see https://stackoverflow.com/questions/15984221/how-to-perform-two-sample-one-tailed-t-test-with-numpy-scipy
        if use_ttest:
            tstat, twoside_pval = scipy.stats.ttest_ind(x_data, y_data, equal_var = False) 
            if alternative == 'less':
                tstat, twoside_pval = scipy.stats.ttest_ind(y_data, x_data, equal_var = False) #Flipped for less
            pval = twoside_pval
            if alternative != 'two-sided':
                pval = (twoside_pval / 2) if (tstat > 0) else (1 - (twoside_pval / 2)) #Get 1-sided pval from 2-sided
        else:
            pval = scipy.stats.mannwhitneyu(x = x_data, y = y_data, alternative = alternative).pvalue 
        
        pvals[(label_i, label_j)] = pval

    pvals_adj = multitest_corr(np.array(list(pvals.values())), method = corr_method)
    return pvals, pvals_adj