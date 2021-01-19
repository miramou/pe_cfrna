## General utilities used throughout analyses

#Import packages used throughout
import pandas as pd
import numpy as np
import scipy.stats
import pickle as pkl
import mygene
import os
from gprofiler import GProfiler

from sklearn import cluster, preprocessing
from kneed import KneeLocator

from statsmodels.stats import multitest
from util.class_def.obj_classes import *

def check_make_folder(path):
    '''
    Checks if a provided path exists. If it does not, creates it.
    '''
    if not (os.path.exists(path)):
        os.makedirs(path)
    return

def read_pkl(pkl_file):
    '''
    Reads and unpickles a file that has been saved with pickle
    Input: pkl_file - the path to a pickled file
    Return: An unpickled object
    '''
    with open(pkl_file, "rb") as f:
        return pkl.load(f)

def write_pkl(obj, pkl_file):
    '''
    Takes an obj and pickles it, saving it to the path specificied with pkl_file
    Input: 
        obj - an object you wish to pickle
        pkl_file - the path at which you want to save the file
    '''
    with open(pkl_file, "wb") as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def read_sample_meta_table(file_path, index_col = 0):
    '''
    Utility fxn to read a metadata csv file with pandas. Assumes that first column is index unless otherwise specificied
    Input: 
        file_path - path to metadata csv file to read
        index_col - the column number associated with the index
    Return: an indexed pandas df corresponding to csv at file_path
    '''
    return pd.read_csv(file_path, index_col = index_col)

def multitest_corr(pvals, method = "bonferroni", **kwargs):
    '''
    Utility fxn to perform multiple hypothesis correction. Assumes alpha = 0.05
    Input: 
        pvals - array-like, 1D of pvalues to correct
        method - the method you wish to use. Default = 'bonferroni', For accepted values, see https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    Return: Dictionary, for details see https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    '''
    return multitest.multipletests(pvals, method = method, **kwargs)

def test_and_adj(test_combos, data, label_col, col_to_test, alternative, use_ttest = False, corr_method = "bonferroni"):
    '''
    Utility fxn to perform hypothesis testing and multiple hypothesis correction. Assumes alpha = 0.05. Default test is Mann-Whitney U
    Input: 
        test_combos - list of tuples specifying the row values data.loc[:, label_col] to compare. 
                        Can be the result of itertools.product
                        For example, if label_col contains 3 unique IDs then test_combos = [(1, 2), (2, 3), (1, 3)]
        data - pandas df (long form) containing a label_column and a col_to_test
        label_col - the column name that corresponds to the variable
        col_to_test - the column name that corresponds to the values
        alternative - string specifying whether desired test is 'two-sided', 'less', or 'greater'
        use_ttest - boolean specifying whether to use ttest instead of MW
        corr_method - string specifying multiple hypothesis method. Default = 'bonferroni'. For accepted values, see https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    Return: Tuple
        pvals - original p-values
        pvals_adj - multiple-hypothesis adjusted p-values
    '''
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