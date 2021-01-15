### Machine learning specific class def

import pandas as pd
import numpy as np
from util.class_def.obj_classes import *

class data_masks():
	def __init__(self, train_frac, seed, label_col):
		self.train_frac = train_frac
		self.seed = seed
		self.label_col = label_col

		self.masks = {}

	def _get_sample_split(self, meta):
		train_data = []
		for label in np.sort(meta.loc[:, self.label_col].unique()):
			label_mask = (meta.loc[:, self.label_col] == label)
			train_data.append(meta.loc[label_mask].sample(frac = self.train_frac, replace = False, random_state = self.seed))
		return pd.concat(train_data, axis = 0)

	def get_sampled_mask(self, meta, addtnl_mask_label = None, blocking_col = None):
		meta_to_sample = meta.loc[self.masks[addtnl_mask_label], :] if addtnl_mask_label is not None else meta

		if blocking_col is None:
			idx_sampled = self._get_sample_split(meta_to_sample)
			return meta.index.isin(idx_sampled.index.to_list())

		meta_w_blocking = meta_to_sample.loc[:, [self.label_col, blocking_col]].reset_index(drop = True).drop_duplicates().set_index(blocking_col)
		idx_sampled = self._get_sample_split(meta_w_blocking)
		return meta.loc[:, blocking_col].isin(idx_sampled.index.to_list())

	def add_mask(self, mask_label, mask):
		self.masks[mask_label] = mask

	def add_mask_logical_and_combinations(self, mask_label_i, mask_label_j):
		self.add_mask(mask_label_i + "_and_" + mask_label_j, np.logical_and(self.masks[mask_label_i], self.masks[mask_label_j]))
		self.add_mask("not_" + mask_label_i + "_and_not_" + mask_label_j, np.logical_and(~self.masks[mask_label_i], ~self.masks[mask_label_j]))
		self.add_mask(mask_label_i + "_and_not_" + mask_label_j, np.logical_and(self.masks[mask_label_i], ~self.masks[mask_label_j]))
		self.add_mask("not_" + mask_label_i + "_and_" + mask_label_j, np.logical_and(~self.masks[mask_label_i], self.masks[mask_label_j]))
		
	def remove_mask(self, mask_label):
		val = self.idx_masks.pop(mask_label, None)

		if val is None: #check if key exists
			return 'Mask label does not exist. Please pass valid label.'

class ML_data():
	def __init__(self, meta, rnaseq_inst, y_col, group_col = None, features = None):
		self.y = meta.loc[:, y_col]

		self.groups = None
		if group_col is not None:
			self.groups = meta.loc[:, group_col]
		
		self.X = None
		self.get_X(rnaseq_inst, features)

	def shrink_X_filter_genes(self, gene_mask):
		new_X = self.X.loc[:, gene_mask]
		self.X = new_X
		return

	def get_masked_feats(self, mask):
		return self.X.loc[:, mask].columns

	def get_X(self, rnaseq, features):
		logCPM = rnaseq.logCPM.copy()
		if features is not None:
			logCPM = logCPM.reindex(features)
			print("%d features missing" % np.sum(logCPM.iloc[:, 0].isna()))
			logCPM = logCPM.fillna(0)

		self.X = logCPM.loc[:, self.y.index.to_list()].T

	def get_no_zero_skew_mask(self, sample_idx, is_iloc, no_zero_skew_cutoff = 0.01):
		#Filter features selected where there is a skewed 0 distribution. 
		#Depending on data quality, unclear if these are biological or technical so apply conservative filter

		X_to_use, y_to_use = self.filter_samples(sample_idx, is_iloc)
		logfrac_zeros_per_feat_by_y_label = pd.DataFrame(columns = y_to_use.unique(), index = X_to_use.columns)
		
		for label in y_to_use.unique():
			X_label = X_to_use.loc[y_to_use == label]
			frac_zeros = (np.sum(X_label == 0, axis = 0) / X_label.shape[0])
			logfrac_zeros_per_feat_by_y_label.loc[:, label] = np.log2((frac_zeros + 1))

		#Assumes binary data - Should put in warning
		logFC = np.abs((logfrac_zeros_per_feat_by_y_label.iloc[:, 0] - logfrac_zeros_per_feat_by_y_label.iloc[:, 1]))
		return (logFC <= no_zero_skew_cutoff).to_numpy()

	def filter_samples(self, sample_idx, is_iloc = False):
		if is_iloc:
			return (self.X.iloc[sample_idx, :], self.y.iloc[sample_idx])
		
		return (self.X.loc[sample_idx, :], self.y.loc[sample_idx])		