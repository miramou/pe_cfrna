## QC specific utilities

import pandas as pd
import numpy as np
import re

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from util.gen_utils import read_sample_meta_table
from util.plot_utils import *

def filter_genes_CPMcutoff(CPM_df, CPM_cutoff = 0.5, frac_samples_exp = 0.75):
	'''
	Utility fxn to filter poorly detected genes.
	For DE, to reduce burden of MH testing, want to remove poorly detected genes from annotated counts matrix
	Input:
		CPM_df - df containing CPM (counts per million reads) data [n_genes, n_samples]
		CPM_cutoff - per sample. Minimum detected level to pass threshold. 0.5 = 25 unique molecules per 50 mil reads
		frac_samples_exp - per gene. Fraction of samples that must meet CPM_cutoff threshold
	Return:
		keep - boolean mask of size [n_genes,] that denotes whether gene in CPM_df at same idx passed filter threshold 
	'''
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

def get_pe_type(metadata, pe_onset_col = 'pe_onset_ga_wk', case_col = 'case'):
	'''
	Get PE onset group for all samples

	Inputs:
		metadata - sample_metadata df
		pe_onset_col - column name that contains gestational_week at PE onset info
		case_col - column name that contains binary info [0,1] about whether person developed PE

	Returns
		np.array of same length as metadata.shape[0] with corresponding pe_type [early, late or control] for every sample in df
	'''
	pe_type = pd.Series(index = metadata.index, name = 'pe_type', dtype = str)
	is_cntrl = (metadata.loc[:, case_col] == 0)
	is_early = metadata.loc[:, pe_onset_col] < 34
	is_late = np.logical_and(~is_early, ~is_cntrl)

	pe_type[is_cntrl] = 'control'
	pe_type[np.logical_and(is_early, ~is_cntrl)] = 'early'
	pe_type[is_late] = 'late'

	return pe_type
	
def scale_data(df, scaler):
	'''
	Scale df and return as pandas df instead of array 
	Input:
		df - pandas df to start
		scaler - instance of scaler to use (unfitted, just initialized)
	Return:
		transformed df with data in df transformed using fitted scaler
	'''
	scaler.fit(df)
	return pd.DataFrame(scaler.transform(df), index = df.index, columns = df.columns)

def calc_pca(logCPM_df, meta_df, rm_na = True, n_components = 2):
	'''
	Utility fxn to perform PCA, Used in pca_and_viz_qc
	Input:
		logCPM_df - df containing logCPM (log(counts per million reads + 1)) data [n_genes, n_samples]
		meta_df - df containing metadata for samples in logCPM_df [n_samples, n_features]
		rm_na - whether to remove samples where no genes were detected and so logCPM is NA
		n_components - number of PC to solve for 
	Return:
		pcs - df with pc value for every sample in meta_df along with meta_df values
	'''
	if rm_na:
		logCPM_df = logCPM_df.loc[:, ~(logCPM_df.isna().sum() > 0)]

	zscore_data = scale_data(logCPM_df.T, StandardScaler())
	pca = PCA(n_components)
	pcs = pd.DataFrame(pca.fit_transform(zscore_data), index = zscore_data.index, columns = ['PC' + str(i) for i in np.arange(1, (n_components+1))])

	return pcs.join(meta_df)

def pca_and_viz_qc(qc_data, logCPM_df, gene_qc_mask, pca_plot_title):
	'''
	Wrapper fxn to visualize QC data in 2 ways (1) heatmap of QC metrics (2) PCA pre all QC, post sample QC, post sample + gene QC
	Want to make sure that (1) For the most part samples that fail QC cluster together in heatmap  and (2) Final PCA isnt driven by a few leverage pts for most part
	Important: 
		Visualization helps in understanding decision making but decisions on what samples or genes pass QC should **not** be made based on these plots. 
		They only help understand whether the QC metrics you have are sufficient
	Input:
		qc_data - df containing QC data for samples in logCPM_df [n_samples, n_qc_metrics]
		logCPM_df - df containing logCPM (log(counts per million reads + 1)) data [n_genes, n_samples]
		gene_qc_mask - boolean mask from filter_genes_CPMcutoff
		pca_plot_title - title for PCA plot
	Return:
		heatmap - plt.figure() with heatmap
		pca - plt.figure() with PCA
 	'''
	qc_df_no_outliers = qc_data.loc[~qc_data.Outlier]
	print('%d samples (%.2f) passed QC' % (qc_df_no_outliers.shape[0], (qc_df_no_outliers.shape[0]/qc_data.shape[0])))
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

def read_save_delvecchio_meta(biosample_results_path, sra_results_path, out_path):
	'''
	Utility to read biosample_result.txt and sra_results files and pull out relevant information. 
	Specific to the DelVecchio et al files

	Input:
		biosample_results_path - filepath to file that contains Biosample results
		sra_results_path - filepath to file that contains SRA Run Table 
		out_path - filepath to save metadata asso with Delvecchio et al 
	Returns
		pandas df with relevant sample specific metadata
	'''
	with open(biosample_results_path) as fp:
		curr_line = fp.readline()
		
		is_sample_line = True
		data_dict = {key : [] for key in ['subj_id', 'sample_type', 'sample_id']}
		
		while curr_line:
			if is_sample_line:
				elems = re.split(": |, |\n| \(|\)", curr_line)
				data_dict['subj_id'].append(elems[1].replace(" ", "_"))
				data_dict['sample_type'].append(elems[2].replace(" ", "_"))
			if 'BioSample: SAMN' in curr_line :
				data_dict['sample_id'].append(re.search('SAMN\\d+', curr_line).group())
			
			is_sample_line = True if curr_line == "\n" else False
			curr_line = fp.readline()

	meta = pd.DataFrame(data_dict).set_index('sample_id').sort_values('subj_id')
	sra_run_table = read_sample_meta_table(sra_results_path)

	meta = meta.merge(sra_run_table.loc[:, ['BioSample', 'complication_during_pregnancy']], left_index = True, right_on = 'BioSample', sort = True)
	meta.reset_index(inplace = True)
	meta.set_index('Run', inplace = True)

	#Add term col
	meta.insert(meta.shape[1], 'term', np.nan) 
	for label, term in {'1st_Trimester' : 1, '2nd_Trimester' : 2, '3rd_Trimester' : 3}.items():
		meta.loc[meta.sample_type == label, 'term'] = term

	meta.insert(meta.shape[1], 'case', 0) 
	meta.loc[meta.complication_during_pregnancy.str.contains('Preeclampsia'), 'case'] = 1

	meta.index.rename('sample', inplace = True)
	meta.to_csv(out_path)
	return meta

def read_save_munchel_data(filepath, cnts_out_path, meta_out_path, mult_samples_per_person, is_all_nt = False):
	'''
	Utility to read Munchel supp file and pull out relevant information. 
	Specific to the Munchel et al 

	Input:
		filepath - filepath to counts table with metadata
		cnts_out_path - filepath to save counts table alone [no metadata]
		meta_out_path - filepath to save metadata alone [no cnts]
		mult_samples_per_person - bool, True if more than 1 sample per person
 		is_all_nt - bool, True if all samples included in df are normotensive, False if should pull case group from sample name 
	Returns 
		meta - pandas df with relevant sample specific metadata
	'''
	df = pd.read_excel(filepath, usecols = "A,G:FB", skiprows = 3)
	df.rename(columns = {df.columns[0] : 'gene_name'}, inplace = True)

	#Pull GA metadata from 1st row
	meta = pd.DataFrame({'ga_at_collection' : df.iloc[0, :].astype(float).round(1)}, index = df.columns).dropna()

	#Remove metadata from cnts table and add to list
	df = df.iloc[1:, :].set_index('gene_name')

	#Add term, subject, and case data to meta
	meta.insert(meta.shape[1], 'term', get_term_labels(meta, ga_col = 'ga_at_collection', has_pp = False))

	subject = [subj[0] for subj in df.columns.str.split('.').to_list()] if mult_samples_per_person else df.columns
	meta.insert(meta.shape[1], 'subject', subject)

	case_grp = np.repeat('CTRL', meta.shape[0]) if is_all_nt else np.array([case_grp[0] for case_grp in df.columns.str.split('.').to_list()])
	meta.insert(meta.shape[1], 'pe_type', case_grp)

	case = np.repeat(0, meta.shape[0])
	case[meta.loc[:, 'pe_type'].str.contains('PE')] = 1
	meta.insert(meta.shape[1], 'case', case)

	#Save full meta and full cnts
	meta.to_csv(meta_out_path)
	df.to_csv(cnts_out_path)

	return meta
