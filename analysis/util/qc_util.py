## QC specific utilities

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from util.plot_utils import *

def filter_genes_CPMcutoff(CPM_df, CPM_cutoff = 0.5, frac_samples_exp = 0.75):
	exp_genes = np.sum((CPM_df > CPM_cutoff), axis = 1) / CPM_df.shape[1]
	keep = (exp_genes > frac_samples_exp)
	print("%d genes (Fraction = %f) passed cutoff" % (np.sum(keep), np.sum(keep)/CPM_df.shape[0]))
	return keep

def mod_qc_data_to_plot(qc):
	'''
	Make qc data easier to visualize via plot
	Input:
		qc - df containing QC data
	Return:
		qc - cleaned df that contains
			(1) Values within plotting range [no NA and rescaled very large values]
			(2) renamed columns that work for plotting
	'''
	qc.loc[qc.bias_frac.isna(), "bias_frac"] = 1.0 #Sometimes cannot estimate
	qc.loc[qc.intron_exon_ratio>1000, "intron_exon_ratio"] = 10 #rescale for standard scaling below to make sense
	qc.sort_values(by="is_outlier", inplace=True)
	qc.rename(index=str, columns={"ribo_frac":"Ribosomal fraction", 
									"intron_exon_ratio":"DNA contamination", 
									"bias_frac":"RNA degradation",
									"is_outlier" : "Outlier"}, inplace=True)
	return qc

def read_qc_data(qc_path):
	'''
	Read QC data and modify for plotting
	'''
	init_df = pd.read_csv(qc_path, sep = "\t", index_col = 0)
	return mod_qc_data_to_plot(init_df)

def get_term_labels(metadata, ga_col, has_pp = True, pp_col = 'is_pp', cutoffs = np.array([40, 23, 13])):
	'''
	Get time group for all samples

	Inputs:
		metadata - sample_metadata df
		ga_col - column name that contains gestational_week info
		has_pp - whether sample_metadata has PP samples
		pp_col - column name that contains binary info about whether sample is PP
		cutoffs - cutoffs based on sampling strategy used to define time groups

	Returns
		np.array of same length as metadata.shape[0] with corresponding term for every sample in df
	'''
	terms = np.ones(metadata.shape[0], dtype=np.int)*-1
	term_cutoffs = np.array(cutoffs)
	term_i = np.arange(len(term_cutoffs), 0, -1)

	for i in range(len(term_cutoffs)):
		mask = metadata.loc[:, ga_col] < term_cutoffs[i]
		mask = np.logical_and(mask, metadata.loc[:, pp_col] == 0) if has_pp else mask
		terms[mask] = term_i[i]

	if has_pp:
		terms[metadata.loc[:, pp_col] == 1] = max(terms) + 1
	
	return terms

def scale_data(df, scaler):
	'''
	Scale df and return as pandas df instead of array 
	'''
	scaler.fit(df)
	return pd.DataFrame(scaler.transform(df), index = df.index, columns = df.columns)

def calc_pca(logCPM_df, meta_df, rm_na = True, n_components = 2):
	if rm_na:
		logCPM_df = logCPM_df.loc[:, ~(logCPM_df.isna().sum() > 0)]

	zscore_data = scale_data(logCPM_df.T, StandardScaler())
	pca = PCA(n_components)
	pcs = pd.DataFrame(pca.fit_transform(zscore_data), index = zscore_data.index, columns = ['PC' + str(i) for i in np.arange(1, (n_components+1))])

	return pcs.join(meta_df)

def pca_and_viz_qc(qc_data, logCPM_df, gene_qc_mask, pca_plot_title):
	qc_df_no_outliers = qc_data.loc[~qc_data.Outlier]
	logCPM_df_no_sample_outliers = logCPM_df.loc[:, qc_df_no_outliers.index.to_list()]

	pcas_dict = {'All samples' : calc_pca(logCPM_df, qc_data.loc[:, 'Outlier'].to_frame()), 
				'Samples that pass QC' : calc_pca(logCPM_df_no_sample_outliers, 
													qc_df_no_outliers.loc[:, 'Outlier'].to_frame(), rm_na = False),
				'Samples and genes that pass QC' : calc_pca(logCPM_df_no_sample_outliers.loc[gene_qc_mask, :], 
					  										qc_df_no_outliers.loc[:, 'Outlier'].to_frame(), rm_na = False)
				}

	heatmap, _ = nhm_plot_heatmap(scale_data(qc_data.loc[:, ['Ribosomal fraction', 'RNA degradation', 'DNA contamination']], RobustScaler()).T, 
						  dfc = qc_data.loc[:, 'Outlier'].to_frame(),
						 cmaps = {'Outlier' : make_color_map("",outlier_palette)},
						 center_args = {'mid' : 0}
						 )

	pca, ax = plt.subplots(1, 3, figsize = (10, 5))
	ax_i = 0
	for pc_name, pc_df in pcas_dict.items():
		palette_to_use = outlier_palette if ax_i == 0 else [outlier_palette[0]]
		sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'Outlier', data = pc_df, ax = ax[ax_i], color = outlier_palette[0], palette = palette_to_use)
		ax[ax_i].set_title(pc_name)

		if ax_i > 0:
			ax[ax_i].get_legend().remove()

		ax_i += 1
		
	plt.suptitle(pca_plot_title, y = 1.05)
	plt.tight_layout()
	return heatmap, pca
