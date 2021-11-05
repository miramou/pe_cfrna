### Machine learning specific class def

import pandas as pd
import numpy as np
from copy import copy

from sklearn.preprocessing import StandardScaler, RobustScaler
from util.gen_utils import preprocess_data
from util.class_def.obj_classes import *


import matplotlib.pyplot as plt
class data_masks():
	"""
	Class to create data masks
	"""
	def __init__(self, train_frac, seed, label_col):
		"""
		Init fxn for data_masks
		Input:
			train_frac - Fraction of data reserved for training
			seed - Seed for reproducibility
			label_col - Column name asso with future prediction task
		Attributes:
			train_frac - see above
			seed - see above
			label_col - see above
			masks - dict with masks
		"""
		self.train_frac = train_frac
		self.seed = seed
		self.label_col = label_col

		self.masks = {}

	def _get_sample_split(self, meta):
		"""
		Private class - Samples training fraction from data without replacement
		Input:
			meta - df with metadata
		Returns:
			Sampled train_data
		"""
		train_data = []
		for label in np.sort(meta.loc[:, self.label_col].unique()):
			label_mask = (meta.loc[:, self.label_col] == label)
			train_data.append(meta.loc[label_mask].sample(frac = self.train_frac, replace = False, random_state = self.seed))
		return pd.concat(train_data, axis = 0)

	def get_sampled_mask(self, meta, addtnl_mask_label = None, blocking_col = None):
		"""
		Samples training metadata with train_frac from init 
		Input:
			meta - df with metadata
			addtnl_mask_label - mask key in self.masks used as an additional mask prior to sampling
			blocking_col - column on which to block prior to sampling training frac [e.g. block on subject to avoid samples in train and val from same subj]
		Returns:
			Sampled metadata with train metadata
		"""
		meta_to_sample = meta.loc[self.masks[addtnl_mask_label], :] if addtnl_mask_label is not None else meta

		if blocking_col is None:
			idx_sampled = self._get_sample_split(meta_to_sample)
			return meta.index.isin(idx_sampled.index.to_list())

		meta_w_blocking = meta_to_sample.loc[:, [self.label_col, blocking_col]].reset_index(drop = True).drop_duplicates().set_index(blocking_col)
		idx_sampled = self._get_sample_split(meta_w_blocking)
		return meta.loc[:, blocking_col].isin(idx_sampled.index.to_list())

	def add_mask(self, mask_label, mask):
		"""
		Adds mask to self.masks
		Input:
			mask_label - key for self.masks
			mask - val, bool mask for self.masks
		Modified attributes:
			self.masks
		"""
		self.masks[mask_label] = mask

	def add_mask_logical_and_combinations(self, mask_label_i, mask_label_j):
		"""
		Adds masks to self.masks that are logical AND combinations of i and j
		Note that no error checking is done to ensure that i and j are in the dict
		Input:
			mask_label_i - key for mask_i to use in combinations
			mask_label_j - key for mask_j to use in combinations
		Modified attributes:
			self.masks - now contains masks for 
				(1) (i) and (j)
				(2) not(i) and not (j)
				(3) (i) and not (j)
				(4) not (i) and (j)
		"""
		self.add_mask(mask_label_i + "_and_" + mask_label_j, np.logical_and(self.masks[mask_label_i], self.masks[mask_label_j]))
		self.add_mask("not_" + mask_label_i + "_and_not_" + mask_label_j, np.logical_and(~self.masks[mask_label_i], ~self.masks[mask_label_j]))
		self.add_mask(mask_label_i + "_and_not_" + mask_label_j, np.logical_and(self.masks[mask_label_i], ~self.masks[mask_label_j]))
		self.add_mask("not_" + mask_label_i + "_and_" + mask_label_j, np.logical_and(~self.masks[mask_label_i], self.masks[mask_label_j]))
		
	def remove_mask(self, mask_label):
		"""
		Removes mask from self.masks
		Input:
			mask_label - key for self.masks to remove
		Modified attributes:
			self.masks
		"""
		val = self.idx_masks.pop(mask_label, None)

		if val is None: #check if key exists
			return 'Mask label does not exist. Please pass valid label.'

class ML_data():
	"""
	Class for ML_data
	"""
	def __init__(self, rnaseq_meta, y_col, 
				sample_weights = None,
				to_norm_to_stable_genes = False, stable_genes = None, 
				to_center = False, fitted_center = None, 
				to_scale = False, fitted_scaler = None,
				impute_dropout = False,
				group_col = None, 
				features = None, 
				only_gene_name = False,
				only_gene_num = False):
		"""
		Init fxn for ML_data
		Input:
			rnaseq_meta - rnaseq_and_meta_data associated with data, see rnaseq_and_meta_data in obj_classes for more info
			y_col - Column name asso with prediction task and y_var
			to_norm_to_stable_genes - Whether to subtract the median of genes that are invariant across PE/NT (bool)
			stable_genes - List of genes to use for to_norm_to_stable_genes
			to_center - Whether to center (RobustScaler) data (bool)
			fitted_center - fitted RobustScaler to use for centering, if none - fit on data
			to_scale - Whether to scale (RobustScaler) data (bool)
			fitted_scaler - fitted RobustScaler to use for to_scale, if none - fit on data
			group_col - Blocking column name (e.g. subject)
			features - Indices to filter ML_data to contain. Will impute to 0 if missing (Indices should match rnaseq_inst indices)
			only_gene_name - bool to indicate whether genes should be referred to just by name or by name and ENSG ID
			only_gene_num - bool to indicate whether genes should be referred to just by ENSG ID or by name and ENSG ID
		Attributes:
			y - column in meta that corresponds to y_col
			groups - column in meta that corresponds to group_col if group_col included
			X - rnaseq.logCPM that contains only features if included [n_samples, n_features]	
		"""
		self.y = rnaseq_meta.meta.loc[:, y_col]
		self.sample_weights = sample_weights.loc[rnaseq_meta.meta.index].to_numpy()[:, 0] if sample_weights is not None else sample_weights

		self.groups = None
		if group_col is not None:
			self.groups = rnaseq_meta.meta.loc[:, group_col]

		self.X = None
		self.is_dropout = None
		self.only_gene_name = only_gene_name
		self.only_gene_num = only_gene_num
		self._get_X(rnaseq_meta.rnaseq, features)
		self.frac_dropout = ((self.X == 0).sum(axis = 1)) / self.X.shape[1]
		self.frac_dropout.rename('frac_dropout', inplace = True)

		self.zc_scaler = fitted_center
		self.fitted_scaler = fitted_scaler

		if to_center:
			if self.zc_scaler is None:
				self.zc_scaler = RobustScaler(with_centering = True, with_scaling = False).fit(self.X)
			X_zc = preprocess_data(self.zc_scaler, self.X) 
			self.X = X_zc
		
		if to_scale:
			if self.fitted_scaler is None:
				self.fitted_scaler = RobustScaler(with_centering = False, with_scaling = True, unit_variance = True).fit(self.X)
			X_sc = preprocess_data(self.fitted_scaler, self.X) 
			self.X = X_sc

		if to_norm_to_stable_genes:
			self.stable_genes = stable_genes

			X_stable_genes, stable_dropout = self._get_X(rnaseq_meta.rnaseq, stable_genes, set_X = False)
			X_stable = RobustScaler(with_centering = to_center, with_scaling = to_scale, unit_variance = True).fit_transform(X_stable_genes.to_numpy())
			X_stable[stable_dropout] = np.nan
			X_stable = np.nanmedian(X_stable, axis = 1)[:, np.newaxis]
			
			X_adj = pd.DataFrame(data = (self.X.to_numpy() - X_stable), index = self.X.index, columns = self.X.columns)
			self.X = X_adj

		if impute_dropout:
			self.X[self.is_dropout] = 0 

	def shrink_X_filter_genes(self, gene_mask):
		"""
		Modify X to only include genes in gene_mask

		Inputs:
			gene_mask - bool mask of same shape as jth dimension of X [n_features,]

		Modified attributes:
			X - Now of shape [n_samples, n_filtered_features]
		"""
		new_X = self.X.loc[:, gene_mask]
		self.X = new_X
		return

	def get_masked_feats(self, mask):
		"""
		Get names of X features that correspond to gene mask 

		Inputs:
			gene_mask - bool mask of same shape as jth dimension of X [n_features,]

		Returns:
			gene_names
		"""
		return self.X.loc[:, mask].columns

	def _get_X(self, rnaseq, features, set_X = True):
		"""
		Get X from rnaseq and filter to features if not none

		Inputs:
			rnaseq - rnaseq instance
			features - optional, used to filter X to desired feature set
			set_X - optional, bool. If 

		Modified or returned attributes:
			Modified if set_X else returned
			X - [n_samples, n_features] with imputed 0 features for any missing gene 
		"""
		logCPM = rnaseq.logCPM.copy()
		if features is not None:

			if logCPM.index.nlevels == 2 and features.nlevels == 1:
				to_drop = 'gene_num' if self.only_gene_name else 'gene_name'
				logCPM.reset_index(to_drop, drop = True, inplace = True)
				logCPM = logCPM.loc[~logCPM.index.duplicated(keep='first')]

			logCPM = logCPM.reindex(features)
			n_feat_missing = np.sum(logCPM.iloc[:, 0].isna())
			logCPM = logCPM.fillna(0)

			if n_feat_missing > 0:
				print("%d features missing" % n_feat_missing)
		

		if isinstance(logCPM.index, pd.MultiIndex) and self.only_gene_name:
			logCPM.reset_index('gene_num', drop = True, inplace = True)

		elif isinstance(logCPM.index, pd.MultiIndex) and self.only_gene_num:
			logCPM.reset_index('gene_name', drop = True, inplace = True)

		logCPM_T = logCPM.T
		is_dropout = (logCPM_T == 0)

		if set_X:
			self.X = logCPM_T
			self.is_dropout = is_dropout
			return

		return logCPM_T, is_dropout

	def get_no_zero_skew_mask(self, sample_idx, is_iloc, no_zero_skew_cutoff = 0.01):
		"""
		Utility fxn to filter features selected where there is a skewed 0 distribution (more 0's in one group than another). 
		Depending on data quality, unclear if these are biological or technical so apply conservative filter

		Inputs:
			sample_idx - sample indices
			is_iloc - whether sample_idx is loc or iloc
			no_zero_skew_cutoff - cutoff value where we consider a feature not to be zero skewed

		Returns:
			mask - Bool mask with same shape as [n_features] indicating which features did not contain an uneven distribution of 0 values
		"""
		
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
		'''
		Filters samples in X to only include those listed in sample_idx

		Inputs:
			sample_idx - sample indices
			is_iloc - whether sample_idx is loc or iloc

		Returns:
			New ML_data instance with only samples in sample_idx s.t. X shape is [n_samples_in_sample_idx, n_features]
		'''
		new_obj = copy(self)

		if is_iloc:
			new_obj.X = self.X.iloc[sample_idx, :]
			new_obj.y = self.y.iloc[sample_idx]
		else:
			new_obj.X = self.X.loc[sample_idx, :]
			new_obj.y = self.y.loc[sample_idx]
		
		return new_obj		

	def join(self, other_ml_data):
		'''
		Join to ml_data objects

		'''
		self.X = pd.concat((self.X, other_ml_data.X), axis = 0)
		self.y = pd.concat((self.y, other_ml_data.y), axis = 0)
		if self.groups is not None and other_ml_data.groups is not None:
			self.groups = pd.concat((self.groups, other_ml_data.groups), axis = 0)
			self.groups = self.groups.astype(str)
