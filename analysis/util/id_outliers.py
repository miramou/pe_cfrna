## Utility script that can be run to identify samples for which QC metrics fall outside expected range
## Expected range based on empirically derived 95th percentile of ~700 samples from 3 different experiments

import argparse
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import datetime

def argparser():
    '''
    Argument parser to run pipeline from command line
    '''
    parser = argparse.ArgumentParser(description = "ID Outliers")
    parser.add_argument("ribo_paths", type = str, help = "Comma separated paths for all ribosomal QC files")
    parser.add_argument("deg_paths", type = str, help = "Comma separated paths for all intron exon ratio QC files")
    parser.add_argument("intron_exon_paths", type = str, help = "Comma separated paths for all RNA degradataion QC files")
    parser.add_argument("out_folder", type = str, help = "Path to parent folder within to save output")
    parser.add_argument("--cutoffs_file", type = str, default = '', help = "Path to cutoff file to use")
    parser.add_argument("--outlier_cutoff", "-oc", default = 95, type = int, help = "Specifies cutoff for calling samples outliers. Default = 95")
    parser.add_argument("--sample_regex", "-sr", default = None, type = str, help = "Sample regex to select outlier info for some of samples passed above")
    parser.add_argument("--outliers_to_include", '-out_incl', default = '', type = str, help = "Comma separated list of outlier types to include in final analysis. Accepted values are ribo,intron,deg")
    return parser.parse_args()

def read_qc_file(path):
    '''
    Utility to read QC files. As pipeline is written, assumes 2nd column contains QC value of interest
    '''
    return pd.read_csv(path, sep = "\t", index_col = 0) #2 col file where first is sample name

def read_merge_qc_files(paths):
    '''
    If passing QC files from more than 1 experiment, this will merge them.
    '''
    df_made = 0

    for path in paths:
        df = read_qc_file(path)
        df_full = df if df_made == 0 else pd.concat((df_full, df), axis = 0)
        df_made += 1

    return df_full.round(2)

def get_percentiles(df, p):
    '''
    Finds the requested percentile for df's QC data
    '''
    return np.round(np.percentile(df.iloc[:, 0].dropna(), q=[p]), 2)[0]

def plot_dist(df, percentile, figure_name):
    '''
    Quick visualization of data
    '''
    to_plot = df.iloc[:,0]
    to_plot.loc[to_plot > 1000] = 30 #Readjust big outliers
    plt.figure()
    plt.hist(to_plot, bins=20, color='c', edgecolor='k')
    plt.axvline(percentile, color='k', linestyle='dashed', linewidth=1)
    plt.savefig(figure_name, bbox_inches='tight')

def check_make_folder(path):
    '''
    Utility to make a folder if it does not exist
    '''
    if not (os.path.exists(path)):
        os.makedirs(path)

def main():
    '''
    Fxn that puts it all together
    '''

    args = argparser()
    #Split paths for each QC metric
    paths = {"ribo" : args.ribo_paths.split(","),
            "deg" : args.deg_paths.split(","),
            "intron" : args.intron_exon_paths.split(",")
            }

    plot_folder = args.out_folder + "plots/"
    check_make_folder(plot_folder)

    outlier_df_exists = False

    #Get outlier percentile cutoffs if cutoffs file included. Otherwise make and keep blank df for now
    p_row = str("percentile_" + str(args.outlier_cutoff))
    outlier_cutoffs = pd.DataFrame(columns = list(paths.keys()), index = [p_row])
    
    if args.cutoffs_file != '':
        outlier_cutoffs = pd.read_csv(args.cutoffs_file, sep = "\t", index_col = 0)
        print(outlier_cutoffs)
        p_row = outlier_cutoffs.index.to_list()[0]

    is_outlier_keys = []

    #Iterate through each QC metric and corresponding files
    for qc_type, files in paths.items():

        #Load and merge files, get cutoff corresponding to that QC
        qc_df = read_merge_qc_files(files)
        if args.cutoffs_file == '':
            outlier_cutoffs.loc[p_row, qc_type] = get_percentiles(qc_df, args.outlier_cutoff)

        #Label QC outliers
        is_outlier = qc_df.iloc[:, 0] >= outlier_cutoffs.loc[p_row, qc_type]
        
        #QC plots
        plot_dist(qc_df, outlier_cutoffs.loc[p_row, qc_type], plot_folder + qc_type + ".png")

        #Make df
        outlier_df = qc_df if not outlier_df_exists else outlier_df.join(qc_df)
        is_outlier_keys.append(qc_type + "_outlier")
        outlier_df.insert(outlier_df.shape[1], is_outlier_keys[-1], is_outlier)
        outlier_df_exists = True

    #If one would only like to call outliers on a subset of the QC params
    is_outlier_keys_include = is_outlier_keys 
    if args.outliers_to_include != '': 
        outliers_to_include = args.outliers_to_include.split(',') 
        is_outlier_keys_include = [key for key in is_outlier_keys if key.split("_outlier")[0] in outliers_to_include]

    #Any outlier flag = outlier sample
    outlier_df.insert(outlier_df.shape[1], "is_outlier", (np.sum(outlier_df.loc[:, is_outlier_keys_include], axis = 1) > 0))
    
    #To save some subset of the total passed in files
    if args.sample_regex is not None:
        idx_matches_regex = [bool(re.search(args.sample_regex, idx))for idx in outlier_df.index.to_list()]
        outlier_df = outlier_df[idx_matches_regex]

    #Save tsv file
    outlier_df.to_csv(args.out_folder + "outlier_data.txt", sep = "\t")

    #If identified percentile cutoffs from files provided, save it. Otherwise move on
    if args.cutoffs_file == '':
        outlier_cutoffs.to_csv(args.out_folder + "outlier_cutoffs" + str(datetime.datetime.now()) + ".txt", sep = "\t")

    return

if __name__ == "__main__":
    main()

    
