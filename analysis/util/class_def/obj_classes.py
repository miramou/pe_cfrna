'''
Utiilty classes used often to analyze RNAseq data
Written by: Mira Moufarrej

To dos - Add more documentation
'''

import pandas as pd
import numpy as np

class rnaseq_data():
	def __init__(self, counts_df_path, tmm_scaling_df_path, counts_index_cols = [0,1], mygene_db = None):
		self.counts_path = counts_df_path
		self.tmm_path = tmm_scaling_df_path

		self.tmm = self._read_tmm_scaling_file()
		self.counts = self._read_counts_csv(counts_index_cols)
		self.CPM = self.get_CPM()
		self.logCPM = self.get_logCPM()

		self.is_anno = False
		self.anno = None

		if mygene_db is not None:
			self.get_anno(mygene_db)

	def _read_tmm_scaling_file(self):
		#TMM file generated via R so rename columns to work with python - namely having a . in the col name isn't conducive to object oriented calls which also use .
		return pd.read_csv(self.tmm_path, index_col = 0).rename(columns = {'lib.size' : "lib_size", "norm.factors" : "norm_scaling"})

	def _read_counts_csv(self, idx_cols):
		init_df = pd.read_csv(self.counts_path, index_col = idx_cols)
		return init_df.loc[:, self.tmm.index.to_list()] #Ensure that the TMM and counts table match
	
	def get_CPM(self):
		lib_sizes = (self.tmm.lib_size * self.tmm.norm_scaling)
		return self.counts / lib_sizes * 10**6

	def get_logCPM(self):
		return np.log2(self.CPM + 1)

	def combine_gene_name_num_indices(self):
		return self.logCPM.index.get_level_values('gene_name') + '_' + self.logCPM.index.get_level_values('gene_num')

	def _sift_through_gene_anno_query(self, mygeneinfo_query, query_type, verbose = False):
		gene_ensgs = pd.DataFrame(columns = ['gene_type'], index = self.logCPM.index)
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

			gene_ensgs.loc[gene_ensgs.index.get_level_values(query_type).isin([entry['query']]), :] = entry['ensembl']['type_of_gene']

		return gene_ensgs, notfound

	def get_anno(self, gene_db):
		idx_to_pull = 'gene_num'
		scope = ['ensembl.gene']

		if 'gene_num' not in self.logCPM.index.names:
			idx_to_pull = 0
			scope = ['symbol', 'ensembl.gene']
		gene_names = self.logCPM.index.get_level_values(idx_to_pull).to_list()

		init_query = gene_db.querymany(gene_names, scopes = scope, species = 'human', fields = 'name,symbol,alias,ensembl.gene,ensembl.type_of_gene',
										returnall = True)['out']
	
		gene_ensgs, notfound = self._sift_through_gene_anno_query(init_query, idx_to_pull)
		
		if idx_to_pull == 'gene_num': #Apparently ENSG IDs can change even within the same build. See TBCE which has both ENSG00000116957 and ENSG00000285053 both for GrCH38
			new_idx_to_pull = 'gene_name'
			not_found_names = self.logCPM.loc[self.logCPM.index.get_level_values(idx_to_pull).isin(notfound)].index.get_level_values(new_idx_to_pull)
			notfound_requery = gene_db.querymany(not_found_names, scopes = ['symbol', 'ensembl.gene', 'alias'], species = 'human', fields = 'name,symbol,alias,ensembl.gene,ensembl.type_of_gene',
											returnall = True)['out']
			newlyfound_ensgs, still_notfound = self._sift_through_gene_anno_query(notfound_requery, new_idx_to_pull)
			gene_ensgs.loc[newlyfound_ensgs.dropna().index, 'gene_type'] = newlyfound_ensgs.dropna().gene_type

		self.anno = gene_ensgs
		self.is_anno = True
		return

	def _check_if_annotated(self):
		if not self.is_anno:
			raise(ValueError)

	def filter_to_gene_types(self, gene_types):
		self._check_if_annotated()

		gene_type_idx = self.anno.loc[self.anno.gene_type.isin(gene_types)].index
		self.counts = self.counts.loc[gene_type_idx]
		self.CPM = self.CPM.loc[gene_type_idx]
		self.logCPM = self.logCPM.loc[gene_type_idx]

		return

class logFC_data_by_group():
	def __init__(self, logCPM_df_idx, group_labels, group_col = 'term', CI_cutoff = 0.5, 
				lfc_col = 'case', logFC_num = 1, logFC_denom = 0):
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
		self.CI_cutoff = CI_cutoff #By how much can the CI vary (e.g. 0.2 = 20% around avg FC)
		
		self.CI_mask = pd.DataFrame(columns = self.logFC.columns, index = self.logFC.index, dtype = bool)
		self.CI_mask.loc[:, :] = False
	
	def _check_if_calculated_logFC(self):
		if not self.calculated_logFC:
			raise(NameError('logFC has not been calculated yet'))

	def _get_avg_logCPM(self, logCPM, meta):
		avgs = pd.DataFrame(index = logCPM.index)

		for name, data in meta.groupby(self.lfc_col):
			avgs.insert(avgs.shape[1], name, np.median(logCPM.loc[:, data.index], axis = 1))

		return avgs

	def _get_logFC(self, avgs):
		return np.round((avgs.loc[:, self.logFC_num] - avgs.loc[:, self.logFC_denom]), 4)

	def get_grp_avgs_logFC(self, logCPM, meta):
		#Sometimes a person will have 2 samples in same time period - ensures only using one
		one_sample_p_subj_mask = meta.subject.drop_duplicates(keep = 'first').index

		avgs = self._get_avg_logCPM(logCPM, meta.loc[one_sample_p_subj_mask, :])
		logFC = self._get_logFC(avgs)
		return {'avg_logCPM' : avgs, 'logFC' : logFC}

	def _get_neg_mask(self):
		self._check_if_calculated_logFC()
		self.is_neg_logFC_mask = (self.logFC < 0)

	def _sample_values(self, meta_df, grp_to_resample, seed, with_replacement = True):
		resampling_mask = (meta_df.loc[:, self.lfc_col] == grp_to_resample)
		return meta_df.loc[resampling_mask].sample(frac = 1.0, replace = with_replacement, random_state = seed, axis = 0)

	def _get_CI_delta(self, logCPM_df, meta_df_group, corresponding_group, ci_interval, n_iters = 2000):
		self._check_if_calculated_logFC()

		iters = np.arange(0, n_iters)
		logFC_sampled = pd.DataFrame(index = logCPM_df.index, columns = iters)

		for sampling_i in iters:
			resampled_meta = pd.concat((self._sample_values(meta_df_group, self.logFC_num, sampling_i),
										self._sample_values(meta_df_group, self.logFC_denom, sampling_i)),
									axis = 0)

			logFC_sampled.loc[:, sampling_i] = self.get_grp_avgs_logFC(logCPM_df, resampled_meta)['logFC']

			if (sampling_i +1) % 1000 == 0:
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
	
	def _id_stable_logFC(self, min_mean = 1e-1):
		self._check_if_calculated_logFC()

		#Want to see where FC is stable and not noisy
		#For this, use relevant 95% CI bound 
		#If for a given gene, the CI 'coefficient of variation' <= CI cutoff then we believe the FC difference between case and control at that time point
		#Filter by min mean to avoid dividing by 0 + explosion of CV
		
		mean_mask = np.round(np.abs(self.logFC), 1) > min_mean
		self.CV[mean_mask] = np.round((self.logFC_CI[mean_mask] / self.logFC[mean_mask]), 2)		
		self.CI_mask[self.CV <= self.CI_cutoff] = True

	def get_logFC_and_CI_by_group(self, logCPM_df, meta_df, ci_interval = 0.025):
		for group, group_label in self.group_labels.items():
			print('Now calculating logFC for %s' % group_label)

			meta_group = meta_df.loc[meta_df.loc[:, self.group_col] == group]
			self.logFC.loc[:, group_label] = self.get_grp_avgs_logFC(logCPM_df, meta_group)['logFC']

		self.calculated_logFC = True
		self._get_neg_mask()

		for group, group_label in self.group_labels.items():
			print('Now estimating logFC confidence interval for %s' % group_label)

			meta_group = meta_df.loc[meta_df.loc[:, self.group_col] == group]
			self.logFC_CI.loc[:, group_label] = self._get_CI_delta(logCPM_df, meta_group, group_label, ci_interval)

		print('Identifying when during gestation we observe changes')
		self._id_stable_logFC()

