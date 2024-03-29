### General class def used to analyze RNAseq data and called by multiple analyses

import pandas as pd
import numpy as np
from copy import deepcopy
import mygene


def read_sample_meta_table(file_path, index_col = 0):
    '''
    Utility fxn to read a metadata csv file with pandas. Assumes that first column is index unless otherwise specificied
    Input: 
        file_path - path to metadata csv file to read
        index_col - the column number associated with the index
    Return: an indexed pandas df corresponding to csv at file_path
    '''
    return pd.read_csv(file_path, index_col = index_col)

class rnaseq_data():
	"""
	Class to easily read and manipulate raw counts data from RNAseq
	Will import RNAseq data (counts table), normalize using TMM (calculated using EdgeR), and incorporate annotations from mygene_db
	"""
	def __init__(self, counts_df_path, tmm_scaling_df_path = None, is_logCPM_path_R = False, counts_index_cols = [0,1], mygene_db = None, mL_plasma = 1.0):
		"""
		Init fxn for rnaseq_data
		Input:
			counts_df_path - path to counts table or logCPM with batch corr from R
			tmm_scaling_df_path - Optional, path to TMM norm csv saved after running TMM in R
			is_logCPM_path_R = Optional, bool to indicate reading in an R file
			counts_index_cols - columns that include indices to use. Assumes multiindex of form [gene_name, gene_num] hence [0,1] idx
			mygene_db - instance of mygene_db to use when getting annotations
			mL_plasma - mL of plasma cfRNA was extracted from, specific to cfRNA. Default is 1.0 which does not affect normalization.
		Attributes:
			counts_path - points to counts path used for a specific instance
			tmm_path - points to TMM path used for a specific instance
			mL_plasma - float, mL plasma from which cfRNA extracted from. Default is 1.0 which does not affect normalization
			tmm - df with TMM data
			counts - df with counts data [genes x samples]
			CPM - df with counts per million reads data, same shape as counts
			logCPM - df with logCPM, same shape as counts
			is_anno - bool denoting whether gene data is annotated
			anno - df with gene annotations
		"""
		self.counts_path = counts_df_path
		self.tmm_path = tmm_scaling_df_path
		self.mL_plasma = mL_plasma
		self.is_logCPM_R = is_logCPM_path_R

		self.tmm = None
		if self.tmm_path is not None:
			self.tmm = self._read_tmm_scaling_file()
		self.counts = self._read_counts_csv(counts_index_cols)

		self.CPM = self.get_CPM()

		if self.is_logCPM_R:
			self.CPM = None
			self.logCPM = self.counts.copy()
			self.split_gene_name_num_indices()
		else:
			self.logCPM = self.get_logCPM()


		self.has_features = self.counts.sum(axis = 0) > 0
		self.is_anno = False
		self.anno = None

		if mygene_db is not None:
			self.get_anno(mygene_db)

	def _read_tmm_scaling_file(self):
		'''
		Private class method to read TMM file
		'''
		#TMM file generated via R so rename columns to work with python - namely having a . in the col name isn't conducive to object oriented calls which also use .
		return pd.read_csv(self.tmm_path, index_col = 0).rename(columns = {'lib.size' : "lib_size", "norm.factors" : "norm_scaling"})

	def _read_counts_csv(self, idx_cols):
		init_df = pd.read_csv(self.counts_path, index_col = idx_cols)
		if self.tmm is not None:
			init_df = init_df.loc[:, self.tmm.index.to_list()] #Ensure that the TMM and counts table match

		return init_df
	
	def get_CPM(self):
		'''
		Method to calculate counts per million reads (CPM) using TMM normalization if included.
		Else sum sample col to obtain 'lib_size'
		'''
		lib_sizes = self.counts.sum(axis = 0)

		if self.tmm is not None:
			lib_sizes = (self.tmm.lib_size * self.tmm.norm_scaling)

		lib_sizes *= self.mL_plasma
		return self.counts / lib_sizes * 10**6

	def get_logCPM(self):
		'''
		Method to calculate log(CPM) where CPM = counts per million reads
		'''
		return np.log2(self.CPM + 1)

	def combine_gene_name_num_indices(self):
		'''
		Method to combine gene_name and gene_number into 1 index
		'''
		return self.logCPM.index.get_level_values('gene_name') + '_' + self.logCPM.index.get_level_values('gene_num')

	def split_gene_name_num_indices(self):
		'''
		Method to split gene_name_gene_number into 2 index
		Convenience when reading back from R
		'''

		for attr in [self.counts, self.logCPM]:
			split_idx = attr.index.str.split('_ENSG', n = 1, expand = True).rename(['gene_name', 'gene_num_split'])
			attr['gene_name'] = split_idx.get_level_values('gene_name')
			attr['gene_num'] = 'ENSG' + split_idx.get_level_values('gene_num_split')
			attr.reset_index(drop = True, inplace = True)		
			attr.set_index(['gene_name', 'gene_num'], inplace = True)
		return

	def _sift_through_gene_anno_query(self, mygeneinfo_query, query_type, verbose = False):
		'''
		Private class method to identify gene annotations
		'''
		gene_ensgs = pd.DataFrame(columns = ['name', 'gene_type', 'summary', 'GO_BP', 'GO_MF'], index = self.logCPM.index)
		notfound = []

		for entry in mygeneinfo_query:
			if verbose:
				print(entry)

			if 'notfound' in entry.keys() or 'ensembl' not in entry.keys():
				notfound.append(entry['query']) 
				continue

			if type(entry['ensembl']) == list:
				correct_entry_idx = 0 #If querying gene names and get DDX1 and DDX10 -> DDX1 appears first
				if query_type == 'gene_num':
					correct_entry_idx = np.where(np.array([entry['query'] == entry['ensembl'][i]['gene'] for i in range(len(entry['ensembl']))]))[0][0]
				entry['ensembl'] = entry['ensembl'][correct_entry_idx]

			#Other than gene_type other fields may not be available. Check prior to entering. Otherwise leave as nan
			query_match = gene_ensgs.index.get_level_values(query_type).isin([entry['query']])
			gene_ensgs.loc[query_match, 'gene_type'] = entry['ensembl']['type_of_gene']
			if 'name' in entry.keys():
				gene_ensgs.loc[query_match, 'name'] = entry['name']

			if 'summary' in entry.keys():
				gene_ensgs.loc[query_match, 'summary'] = entry['summary']

			if 'go' in entry.keys():
				for GO_term in ['BP', 'MF']:
					if GO_term in entry['go'].keys():
						all_terms = ', '.join(list(set([bp_i['term'] for bp_i in entry['go'][GO_term]]))) if isinstance(entry['go'][GO_term], list) else entry['go'][GO_term]['term']
						gene_ensgs.loc[query_match, 'GO_' + GO_term] = all_terms

		return gene_ensgs, notfound

	def get_anno(self, gene_db):
		'''
		Method to identify gene annotations
		'''
		idx_to_pull = 'gene_num'
		scope = ['ensembl.gene']

		if 'gene_num' not in self.logCPM.index.names:
			idx_to_pull = 0
			scope = ['symbol', 'ensembl.gene']

		gene_names = self.logCPM.index.get_level_values(idx_to_pull).to_list()
		fields_to_include = 'name,symbol,alias,ensembl.gene,ensembl.type_of_gene,summary,go.BP.term,go.MF.term'
		init_query = gene_db.querymany(gene_names, scopes = scope, species = 'human', fields = fields_to_include,
										returnall = True)['out']
	
		gene_ensgs, notfound = self._sift_through_gene_anno_query(init_query, idx_to_pull)
		
		if idx_to_pull == 'gene_num': #Apparently ENSG IDs can change even within the same build. See TBCE which has both ENSG00000116957 and ENSG00000285053 both for GrCH38
			new_idx_to_pull = 'gene_name'
			not_found_names = self.logCPM.loc[self.logCPM.index.get_level_values(idx_to_pull).isin(notfound)].index.get_level_values(new_idx_to_pull)
			notfound_requery = gene_db.querymany(not_found_names, scopes = ['symbol', 'ensembl.gene', 'alias'], species = 'human', fields = fields_to_include,
											returnall = True)['out']
			newlyfound_ensgs, still_notfound = self._sift_through_gene_anno_query(notfound_requery, new_idx_to_pull)
			gene_ensgs.loc[newlyfound_ensgs.dropna().index, 'gene_type'] = newlyfound_ensgs.dropna().gene_type

		self.anno = gene_ensgs
		self.is_anno = True
		return

	def _check_if_annotated(self):
		'''
		Private method to check if data is annotated
		'''
		if not self.is_anno:
			raise(ValueError)

	def filter_to_gene_types(self, gene_types):
		'''
		Method to filter rnaseq data instance to genes of specific types
		Input:
			gene_types - list of gene_types to include like ['protein_coding', 'lncRNA']
		Modified attributes: 
			counts, CPM, and logCPM now all take shape [n_genes_in_gene_types, n_samples]
			where n_genes_in_gene_types are all genes of types included in gene_types list
		'''
		self._check_if_annotated()

		gene_type_idx = self.anno.loc[self.anno.gene_type.isin(gene_types)].index
		self.anno = self.anno.loc[gene_type_idx]
		self.counts = self.counts.loc[gene_type_idx]
		if self.CPM is not None:
			self.CPM = self.CPM.loc[gene_type_idx]
		self.logCPM = self.logCPM.loc[gene_type_idx]

		return

	def filter_to_samples(self, sample_idx):
		'''
		Method to filter rnaseq data instance to genes of specific types
		Input:
			sample_idx - indices for samples to keep
		Modified attributes: 
			counts, CPM, and logCPM now all take shape [n_genes, n_samples_in_mask]
			where n_samples_in_mask are all samples in sample_idx
		'''
		self.counts = self.counts.loc[:, sample_idx]
		if self.CPM is not None:
			self.CPM = self.CPM.loc[:, sample_idx]
		self.logCPM = self.logCPM.loc[:, sample_idx]

		return

class rnaseq_and_meta_data():
	"""
	Class to process and keep together asso rnaseq data instance and meta table
	"""
	def __init__(self, meta_df_path, counts_df_path, tmm_scaling_df_path = None, **kwargs):
		"""
		Init fxn for rnaseq_and_meta_data
		Input:
			meta_df_path - path to metadata df
			counts_df_path - path to counts table
			tmm_scaling_df_path - Optional, path to TMM norm csv saved after running TMM in R
			**kwargs - Any additional kwargs to rnaseq_data call
		"""
		self.meta = read_sample_meta_table(meta_df_path)
		self.rnaseq = rnaseq_data(counts_df_path, tmm_scaling_df_path = tmm_scaling_df_path, **kwargs)
		self.meta = self.meta.loc[self.rnaseq.logCPM.columns]
	def filter_to_gene_types(self, gene_types):
		"""
		Method to call corresponding method from rnaseq_data
		"""
		self.rnaseq.filter_to_gene_types(gene_types)

	def filter_samples(self, sample_mask, inplace = True):
		"""
		Method to filter samples to those in sample_mask
		Inputs:
			sample_mask - boolean mask
			inplace - if False, create and return modified obj copy
		"""
		if inplace:
			self.meta = self.meta.loc[sample_mask]
			self.rnaseq.filter_to_samples(self.meta.index)
			return

		obj_copy = deepcopy(self)
		obj_copy.meta = obj_copy.meta.loc[sample_mask]
		obj_copy.rnaseq.filter_to_samples(obj_copy.meta.index)
		return obj_copy

	def join(self, rnaseq_and_meta_data_obj_2, how = 'inner', on = 'gene_name', rescale_rnaseq_2 = False):
		common_meta_columns = self.meta.columns.intersection(rnaseq_and_meta_data_obj_2.meta.columns)
		self.meta = pd.concat((self.meta.loc[:, common_meta_columns], rnaseq_and_meta_data_obj_2.meta.loc[:, common_meta_columns]), axis = 0)

		self.rnaseq.counts = self.rnaseq.counts.join(rnaseq_and_meta_data_obj_2.rnaseq.counts, how  = how)
		if self.rnaseq.CPM is not None and rnaseq_and_meta_data_obj_2.rnaseq.CPM is not None:
			self.rnaseq.CPM = self.rnaseq.CPM.join(rnaseq_and_meta_data_obj_2.rnaseq.CPM, how = how)
		self.rnaseq.logCPM = self.rnaseq.logCPM.join(rnaseq_and_meta_data_obj_2.rnaseq.logCPM, how = how)

		
		

class logFC_data_by_group():
	"""
	Class to easily calculate logFC grouped by some attribute (e.g. time)
	Also calculates confidence interval around logFC
	"""
	def __init__(self, logCPM_df_idx, group_labels, group_col = 'term', CV_cutoff = 0.5, logFC_cutoff = 0,
				lfc_col = 'case', logFC_num = 1, logFC_denom = 0):
		"""
		Init fxn for logFC_data_by_group. logCPM_df_idx and group_labels are used to create empty dfs of correct size and labels
		Input:
			logCPM_df_idx - pd indices, correspond to logCPM (e.g. from an instance of rnaseq_data.logCPM)
			group_labels - dictionary, keys = values in group_col, values = labels. all group labels for which logFC should be calculated
			group_col - str, column name by which to group by
			CV_cutoff - float, defines the max acceptable coefficient of variation (CV)
			logFC_cutoff - float, defines the min acceptable |logFC|
			lfc_col - str, column name that contains labels of 2 groups to be used to calculate logFC
			logFC_num - [str, int, float], value that corresponds to label for numerator in FC calculation
			logFC_denom - [str, int, float], value that corresponds to label for denominator in FC calculation
		Attributes:
			calculated_logFC - bool denoting whether logFC has been calculated
			logFC - df containing calculated logFCs once calc, [n_genes, n_group_labels]
			logFC_CI - df containing one-sided delta [that approaches logFC = 0] for 95% confidence interval, [n_genes, n_group_labels]
			CV - df containing coefficient of variation [logFC_CI / logFC] for all genes where logFC > 0.1 [n_genes, n_group_labels]
			CV_mask - bool mask denoting where CV <= CV_cutoff
			CV_cutoff - see above
			is_neg_logFC_mask - Mask denoting where logFC < 0
			lfc_col - see above
			logFC_num - see above
			logFC_denom - see above
			group_labels - see above
			group_col - see above
		"""
		self.calculated_logFC = False
		self.lfc_col = lfc_col
		self.logFC_num = logFC_num
		self.logFC_denom = logFC_denom

		self.group_labels = group_labels
		self.group_col = group_col
		self.logFC = pd.DataFrame(columns = list(group_labels.values()), index = logCPM_df_idx, dtype = float)
		self.is_neg_logFC_mask = None

		#Want 95% CI when varying both num and denom of FC
		self.logFC_CI = pd.DataFrame(columns = list(group_labels.values()), index = logCPM_df_idx, dtype = float) 

		#Want to know when during gestation logFC is stable
		self.CV = pd.DataFrame(columns = list(group_labels.values()), index = logCPM_df_idx, dtype = float)
		self.CV_cutoff = CV_cutoff #By how much can the CI vary (e.g. 0.2 = 20% around avg FC)
		
		self.CV_mask = pd.DataFrame(columns = self.logFC.columns, index = self.logFC.index, dtype = bool)

		#logFC cutoff mask
		self.logFC_cutoff = logFC_cutoff
		self.logFC_mask = self.CV_mask.copy()
	
	def _check_if_calculated_logFC(self):
		'''
		Private method to check if logFC has been calculated
		'''
		if not self.calculated_logFC:
			raise(NameError('logFC has not been calculated yet'))

	def _get_avg_logCPM(self, logCPM, meta):
		'''
		Private method to calculate median logCPM for each unique value in lfc_col
		Called by get_grp_avg_logCPM
		'''
		avgs = pd.DataFrame(index = logCPM.index)

		for name, data in meta.groupby(self.lfc_col):
			avgs.insert(avgs.shape[1], name, np.median(logCPM.loc[:, data.index], axis = 1))

		return avgs

	def _get_logFC(self, avgs):
		'''
		Private method to calculate logFC using average logCPM values
		Called by get_grp_avg_logCPM
		'''
		return np.round((avgs.loc[:, self.logFC_num] - avgs.loc[:, self.logFC_denom]), 2)

	def get_grp_avgs_logFC(self, logCPM, meta):
		'''
		Method to calculate group averages and corresponding logFC 
		Called by get_logFC_and_CI_by_group
		Input:
			logCPM - df containing logCPM values [n_genes, n_samples]
			meta - metadata for samples in logCPM. Must contain lfc_col and group_col [n_samples, n_attributes]
		Output: Dictionary
			'avg_logCPM' - average logCPM values
			'logFC' - calculated logFC
		'''

		#Sometimes a person will have 2 samples in same time period - ensures only using one
		one_sample_p_subj_mask = meta.subject.drop_duplicates(keep = 'first').index

		avgs = self._get_avg_logCPM(logCPM, meta.loc[one_sample_p_subj_mask, :])
		logFC = self._get_logFC(avgs)
		return {'avg_logCPM' : avgs, 'logFC' : logFC}

	def _get_neg_mask(self):
		'''
		Private method to see which logFC are negative
		'''
		self._check_if_calculated_logFC()
		self.is_neg_logFC_mask = (self.logFC < 0)

	def _sample_values(self, meta_df, grp_to_resample, seed, with_replacement = True):
		'''
		Private method to randomly sample values from one logFC group with replacement for bootstrap
		Called by _get_CI_delta
		Input:
			meta_df - df of sample metadata to use for resampling
			grp_to_resample - lfc_col group to resample [Typically lfc_num or lfc_denom]
			seed - Seed to use during random sampling
			with_replacement - Bool, Default = True for bootstrap
		Return: Resampled df using pandas sample
		'''
		resampling_mask = (meta_df.loc[:, self.lfc_col] == grp_to_resample)
		return meta_df.loc[resampling_mask].sample(frac = 1.0, replace = with_replacement, random_state = seed, axis = 0)

	def _get_CI_delta(self, logCPM_df, meta_df_group, corresponding_group, ci_interval, n_iters = 2000):
		'''
		Private method to calculate confidence interval delta [avg FC - delta = bound]
		Only want one-sided confidence interval (the side that approaches 0)
		Recall logFC = 0 means no change
		For negative logFC, want delta for upper bound
		For positive logFC, want delta for lower bound

		Input:
			logCPM_df - df of logCPM values to use during bootsrap
			meta_df - df of sample metadata to use for resampling
			corresponding_group - group_label from group_labels for which delta is being calculated
			ci_interval - float, defines confidence interval eg 0.025 = 95% CI
			n_iters - int, Default = 2000, n iterations for bootstrap
		Return: Relevant one-sided delta values for CI
		'''
		self._check_if_calculated_logFC()

		iters = np.arange(0, n_iters)
		logFC_sampled = pd.DataFrame(index = logCPM_df.index, columns = iters)

		for sampling_i in iters:
			resampled_meta = pd.concat((self._sample_values(meta_df_group, self.logFC_num, sampling_i),
										self._sample_values(meta_df_group, self.logFC_denom, sampling_i)),
									axis = 0)

			logFC_sampled.loc[:, sampling_i] = self.get_grp_avgs_logFC(logCPM_df, resampled_meta)['logFC']

			if (sampling_i + 1) % 1000 == 0:
				print('%d resampling iterations completed' % (sampling_i + 1))

		delta_star = logFC_sampled.to_numpy() - self.logFC.loc[:, corresponding_group].to_numpy()[:, np.newaxis]

		#Calculate CI
		delta_star.sort(axis = 1)
		delta_upper = delta_star[:, int((ci_interval * n_iters))]
		delta_lower = delta_star[:, int(((1 - ci_interval) * n_iters))]

		#Only care about CI that is closer to 0 - So want delta_upper for negative FC and delta_lower for positive FC
		neg_group_mask = self.is_neg_logFC_mask.loc[:, corresponding_group]
		delta_approaching_0 = delta_lower
		delta_approaching_0[neg_group_mask] = delta_upper[neg_group_mask]

		return delta_approaching_0
	
	def mod_CV_mask(self, new_CV_cutoff):
		'''
		Method to modify CV mask
		
		Input:
			new_CV_cutoff - new cutoff val for CV
		
		Modified attributes: 
			CV_cutoff - now = new_CV_cutoff
			CV_mask - modified to reflect which CV pass threshold
		'''
		self._check_if_calculated_logFC()
		self.CV_cutoff = new_CV_cutoff
		self.CV_mask.loc[:, :] = False
		self.CV_mask[self.CV < self.CV_cutoff] = True

	def _id_stable_logFC(self, min_mean = 1e-1):
		'''
		Private method to calculate (CV) 'coefficient of variation' for each gene and identify genes for which logFC is reasonably stable
		CV = delta_CI / logFC (traditionally standard_deviation / mean). Roughly a non-parametric equivalent

		Input:
			min_mean - float, default = 0.1. Given that CV is a ratio, it will explode as the denominator approaches 0. This prevents calculating 
			
		Modified attributes: 
			CV - Calculated and added
			CV_mask - Bool mask of same shape as CV, Whether CV < CV_cutoff
		'''
		self._check_if_calculated_logFC()

		#Want to see where FC is stable and not noisy
		#For this, use relevant 95% CI bound 
		#If for a given gene, the CI 'coefficient of variation' <= CI cutoff then we believe the FC difference between case and control at that time point
		#Filter by min mean to avoid dividing by 0 + explosion of CV
		
		mean_mask = np.round(np.abs(self.logFC), 1) > min_mean
		self.CV[mean_mask] = np.round((self.logFC_CI[mean_mask] / self.logFC[mean_mask]), 2)		
		self.mod_CV_mask(self.CV_cutoff)

	def mod_logFC_mask(self, new_logFC_cutoff):
		'''
		Method to modify logFC mask
		
		Input:
			new_logFC_cutoff - new cutoff val for logFC
		
		Modified attributes: 
			logFC_cutoff - now = new_logFC_cutoff
			logFC_mask - modified to reflect which logFC pass threshold
		'''
		self._check_if_calculated_logFC()
		self.logFC_cutoff = new_logFC_cutoff
		self.logFC_mask.loc[:, :] = False
		self.logFC_mask[self.logFC.abs() >= self.logFC_cutoff] = True

	def get_logFC_and_CI_by_group(self, logCPM_df, meta_df, ci_interval = 0.025):
		'''
		Method to calculate logFC and corresponding confidence interval for every group
		
		Input:
			logCPM_df - df of logCPM values to use during bootstrap
			meta_df - df of sample metadata to use for resampling
			ci_interval - float, defines confidence interval eg 0.025 = 95% CI
		
		Modified attributes: 
			logFC - calculated and added
			calculated_logFC - changed to True
			neg_mask - modified to reflect which logFC are negative
			logFC_CI - calculated and added
			CV - calculated and added
			CV_mask - modified to reflect which CV pass threshold
		'''
		for group, group_label in self.group_labels.items():
			print('Now calculating logFC for %s' % group_label)

			meta_group = meta_df.loc[meta_df.loc[:, self.group_col] == group]
			self.logFC.loc[:, group_label] = self.get_grp_avgs_logFC(logCPM_df, meta_group)['logFC']

		self.calculated_logFC = True
		self.mod_logFC_mask(self.logFC_cutoff)
		self._get_neg_mask()

		for group, group_label in self.group_labels.items():
			print('Now estimating logFC confidence interval for %s' % group_label)

			meta_group = meta_df.loc[meta_df.loc[:, self.group_col] == group]
			self.logFC_CI.loc[:, group_label] = self._get_CI_delta(logCPM_df, meta_group, group_label, ci_interval)

		print('Identifying when during gestation we observe changes')
		self._id_stable_logFC()

	def isec_same_sign_lfc(self, logFC_2, colnames):
		isec_lfc = self.logFC.join(logFC_2, rsuffix = '2')
		sign_check = np.sign(isec_lfc.loc[:, colnames + [colname + '2' for colname in colnames]]).sum(axis = 1).abs() == len(colnames)*2
		same_sign_idx = isec_lfc.loc[sign_check].index

		self.logFC = self.logFC.loc[same_sign_idx]
		self.CV = self.CV.loc[same_sign_idx]
		self.logFC_CI = self.logFC_CI.loc[same_sign_idx]

		self.mod_CV_mask(self.CV_cutoff)
		self.mod_logFC_mask(self.logFC_cutoff)

		return
