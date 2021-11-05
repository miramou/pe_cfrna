## ML specific utilities

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from util.class_def.ml_obj_classes import *
from util.class_def.de_obj_classes import de_data
from util.gen_utils import get_stats

from sklearn.model_selection import GroupKFold
from sklearn.utils import resample 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, classification_report, roc_curve, average_precision_score, precision_recall_curve
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

def get_min_max_ga_collection(ML_data_dict, rnaseq_meta_obj_dict, coll_name, col_name = 'ga_at_collection'):
	'''
	Utility fxn to find GA at collection range
	Inputs:
		ML_data_dict: keys = dataset_name, val = ML_data obj
		rnaseq_meta_obj_dict: keys = dataset_name, val = rnaseq_meta_obj
		coll_name: Dataset collection name
		col_name: col name for GA at collection

	Prints min/max GA at collection
	'''
	all_ga_coll = [rnaseq_meta_obj_dict[dataset_name].meta.loc[ML_data_obj.y.index, col_name] 
					for dataset_name, ML_data_obj in ML_data_dict.items() if col_name in rnaseq_meta_obj_dict[dataset_name].meta.columns]

	all_ga_df = pd.concat(all_ga_coll, axis = 0)
	print('%s: Min = %d, Max = %d' % (coll_name, all_ga_df.min(axis = 0), all_ga_df.max(axis = 0)))
	return

def get_class_label(ML_data_obj, model, threshold):
	'''
	Utility fxn to get predicted class
	Input:
		ML_data_obj - ML_data obj instance
		model - fitted model obj
		threshold - decision at which >= is class 1
	Return pd.Series with labels
	'''

	model_prob = model.predict_proba(ML_data_obj.X)[:, 1]
	model_pred = pd.Series(np.repeat(0, ML_data_obj.y.shape), index = ML_data_obj.y.index, name = 'prediction')
	model_pred.loc[model_prob >= threshold] = 1
	return model_pred

def get_model_stats(dataset_name, ML_data_obj, model, threshold = 0.5, to_print = True, **kwargs):
	'''
	Utility fxn to get PPV, NPV, Spec, Sens from 2x2 confusion matrix and AUC
	Input:
		dataset_name - str, dataset label
		ML_data_obj - ML_data obj instance
		model - fitted model obj
		**kwargs - key word args to pass to get_stats
	Return dictionary with following keys
	    Each key has dict value with 3 keys 'val', 'ci_lb', 'ci_ub' corresponding to the value, CI lower bound, and CI upper bound

		PPV = TP / (TP + FP)
		NPV = TN / (TN + FN)
		Sensitivity = TP / (TP + FN)
		Specificity = TN / (TN + FP)
		ROC AUC
	'''

	model_pred = get_class_label(ML_data_obj, model, threshold)

	curr_confusion_matrix = confusion_matrix(ML_data_obj.y, model_pred)
	
	specs = get_stats(curr_confusion_matrix, **kwargs)
	specs['AUC'] = get_auc_roc_CI(model, ML_data_obj)

	if to_print:
		all_spec_vals = [(prop, specs[prop]['val'], specs[prop]['ci_lb'], specs[prop]['ci_ub']) for prop in ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'AUC']]

		print('%s: %s = %d%% [%d-%d%%], %s = %d%% [%d-%d%%], %s = %d%% [%d-%d%%], %s = %d%% [%d-%d%%], %s = %.2f [%.3f-%.3f]' % (dataset_name, 
																											*all_spec_vals[0],
																											*all_spec_vals[1],
																											*all_spec_vals[2],
																											*all_spec_vals[3],
																											*all_spec_vals[4]
																											))

	return specs

def get_calibration_curve_slope_intercept(y_true, y_prob, dataset_name, n_bins = 10):
	'''
	Utility fxn to assess weak calibration (slope, intercept) of calibration curve
	Input:
		y_true - observed y values
		y_prob - prob of positive prediction
		dataset_name - name of dataset
		n_bins - n_bins to use in making curve
	Return:
		slope - cal curve slope
		intercept - cal curve intercept
	'''
	fraction_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins = n_bins)
	slope, intercept = np.polyfit(mean_predicted_value, fraction_positives, deg = 1)

	print('%s: Calibration slope = %.2f, intercept = %.2f' % (dataset_name, slope, intercept))
	return {'slope' : slope, 'intercept' : intercept}

def make_LR_model_from_fitted_CV_model(og_model, seed = 37, keep_zero_coef = False): 
	'''
	Utility fxn to allow one to easily save fitted model
	Cannot save LogisticRegressionCV obj as pkl so once fitted use params from CV obj to fit LR 
	Input: 
		og_model - LogisticRegressionCV fitted model instance that you'd like to save
		seed - Optional, seed value. Doesnt really matter since we are just creating a model copy
		keep_zero_coef - Optional bool, Whether to keep or remove coefficients = 0, Default = False
	Return: Tuple
		new_model - LogisticRegression model with identical params and coefs
		feat_mask - np boolean array indicating which coef were kept. Only important if keep_zero_coef = True
	'''
	params = {key:val for key,val in og_model.get_params().items() if key not in ['Cs', 'cv', 'l1_ratios', 'refit', 'random_state', 'scoring']}
	new_model = LogisticRegression(**params, random_state = seed)

	#Set fitted params from other model
	new_model.classes_ = og_model.classes_

	#Get relevant feature mask
	feat_mask = np.ones(og_model.coef_.shape, dtype = bool)
	if not keep_zero_coef: 
		feat_mask = og_model.coef_ != 0
	new_model.coef_ = og_model.coef_[feat_mask][np.newaxis, :]

	new_model.intercept_ = og_model.intercept_
	new_model.n_iter_ = og_model.n_iter_[:, 0, 0, 0]

	return new_model, feat_mask[0, :]

def LR_train_w_sklearnCV(train_data, n_cv_folds, inv_reg_strength_arr, penalty = 'elasticnet', scoring = 'roc_auc', seed = 37, l1_ratio_arr = None):
	'''
	Single training loop for logistic regression model with sklearn default CV 
	Input: 
		train_data - instance of ML_data that contains training data. See class_def for details of ML_data
		n_cv_folds - Number of CV folds to run
		inv_reg_strength_arr - Inverse regularization strengths to try
		penalty - Optional string, default = elasticnet. Options include 'l1','l2', or 'elasticnet'
		scoring - Optional string, default = 'roc_auc'. Scoring metrics. See sklearn for details
		seed - Optional int, default = 37. Random seed
		l1_ratio_arr - Optional array-like, default = None, L1 ratios to try [relevant for elastic net]
	Return: fitted model
		LogisticRegressionCV model
	'''

	lr = LogisticRegressionCV(cv = GroupKFold(n_splits = n_cv_folds).split(X = train_data.X, y = train_data.y, groups = train_data.groups),
								Cs = inv_reg_strength_arr, penalty = penalty, l1_ratios = l1_ratio_arr, scoring = scoring, class_weight = 'balanced',
								solver = 'saga', max_iter = 10e6, random_state = seed
								)

	lr.fit(train_data.X, train_data.y, sample_weight = train_data.sample_weights)

	return lr

def RF_train_w_sklearn(train_data, n_estimators, max_features, ccp_alpha, seed = 37):
	'''
	Single training loop for random forest model with sklearn default CV 
	Input: 
		train_data - instance of ML_data that contains training data. See class_def for details of ML_data
		n_estimators - Number of trees to build
		max_features - max features to consider per split
		ccp_alpha - Pruning coefficient
		scoring - Optional string, default = 'roc_auc'. Scoring metrics. See sklearn for details
		seed - Optional int, default = 37. Random seed
	Return: fitted model
		Random Forest model
	'''

	rf = RandomForestClassifier(n_estimators = n_estimators, criterion = 'gini', max_features = max_features, max_depth = 5, 
								bootstrap = True, oob_score = True,
								random_state = seed, class_weight = 'balanced', ccp_alpha = ccp_alpha,
								max_samples = (train_data.y == 1).sum()*2)

	rf.fit(train_data.X, train_data.y)

	return rf

def get_all_combos_2_arrays(arr_1, arr_2, arr_1_name, arr_2_name):
	'''
	Helper fxn to get combinations of 2 arrays

	Return:
		dict with combinations for all 3 arrays
	'''
	arr_1_2_combos = list(product(arr_1, arr_2))
	
	n_combos = len(arr_1_2_combos)
	cutoff_combos = {'cutoff_combo_' + str(i + 1) : {arr_1_name : arr_1_2_combos[i][0], 
													arr_2_name : arr_1_2_combos[i][1]
													}
					for i in np.arange(n_combos)
					}

	return cutoff_combos

def make_train_val_ML_data_obj(train_rnaseq_meta_obj, val_rnaseq_meta_obj, y_col, group_col, feats, sample_weights = None, **kwargs):
	'''
	Utility fxn to make train and val ML objs
	Only called from training_pipeline
	'''
	train_data = ML_data(train_rnaseq_meta_obj, y_col = y_col, group_col = group_col, features = feats, 
						sample_weights = sample_weights, **kwargs)

	val_data = None
	if val_rnaseq_meta_obj is not None:
		val_data = ML_data(val_rnaseq_meta_obj, y_col = y_col, group_col = group_col, features = feats,
							**kwargs, 
							fitted_center = train_data.zc_scaler, fitted_scaler = train_data.fitted_scaler
							)

	return train_data, val_data

def training_pipeline(train_rnaseq_meta, 
						train_logFC_per_group, relevant_group_names_in_logFC, 
						cv_cutoffs_to_try, logFC_cutoffs_to_try, 
						val_rnaseq_meta = None,
						sample_weights = None,
						**kwargs
						):
	'''
	Full training pipeline with feature selection for logistic regression model with sklearn default CV 
	Input: 
		train_rnaseq_meta - rnaseq_and_meta_data associated with training data, see rnaseq_and_meta_data in obj_classes for more info
		train_logFC_per_group - logFC_data_by_group obj instance asso with feature selection
		relevant_group_names_in_logFC - list with relevant column names from train_logFC_per_group to be used during feature selection
		cv_cutoffs_to_try - array of possible cutoff values for CV in train_logFC_per_group
		logFC_cutoffs_to_try - array of possible cutoff values for logFC in train_logFC_per_group
		val_rnaseq_meta - rnaseq_and_meta_data split from training data, used for validatiion
		kwargs - All args to pass to ML_data when making obj
	Return: 
		dict with {'model' : CalibratedClassifierCV fitted model, 'score' : best score, 'combo' : best combo of cutoffs}
	'''

	cutoff_combos = get_all_combos_2_arrays(cv_cutoffs_to_try, logFC_cutoffs_to_try, 'cv', 'logFC')
					
	#Holder vars for best model
	best_score = 0
	best_val_score = 0
	best_combo = None
	best_model = None
	best_model_feats = None

	#Init train_data
	y_col = 'case'
	group_col = 'subject'

	
	#Set up other LR hyperparams
	Cs = np.logspace(start = -3, stop = 1, num = 30) #Inv reg strengths to test
	l1_ratios = np.linspace(0, 1, num = 5) #L1 ratios to test, Want some sparsity. 0 = L2, 1 = L1
	
	i = 0
	for cutoff_combo_name, cutoff_combo in cutoff_combos.items():
		train_logFC_per_group.mod_CV_mask(cutoff_combo['cv'])
		train_logFC_per_group.mod_logFC_mask(cutoff_combo['logFC'])
		
		init_feat_mask = np.logical_and(train_logFC_per_group.CV_mask, train_logFC_per_group.logFC_mask).loc[:, relevant_group_names_in_logFC].sum(axis = 1) > 0
		
		init_feats = train_logFC_per_group.logFC.loc[init_feat_mask].index
		
		if init_feats.shape[0] == 0: #Sometimes no genes come up
			continue

		train_data, val_data = make_train_val_ML_data_obj(train_rnaseq_meta, val_rnaseq_meta, y_col, group_col, init_feats, sample_weights, **kwargs)
		n_cv = 5

		curr_lr = LR_train_w_sklearnCV(train_data, n_cv_folds = n_cv, inv_reg_strength_arr = Cs, scoring = 'roc_auc',
										 penalty = 'elasticnet', l1_ratio_arr = l1_ratios)
		
		curr_score = roc_auc_score(y_true = train_data.y, y_score = curr_lr.predict_proba(train_data.X)[:, 1])
		curr_val_score = roc_auc_score(y_true = val_data.y, y_score = curr_lr.predict_proba(val_data.X)[:, 1]) if val_data is not None else 0

		meets_val_score_req = True if val_data is None else curr_val_score >= best_val_score		

		n_curr_coef = (curr_lr.coef_[0, :] != 0).sum()
		if n_curr_coef > 25:
			continue

		if curr_score > best_score and meets_val_score_req:
			#To save would need object without CV generator
			best_model, feat_mask = make_LR_model_from_fitted_CV_model(curr_lr, keep_zero_coef = False)
			best_model_feats = train_data.get_masked_feats(feat_mask)
			
			best_score = curr_score
			best_val_score = curr_val_score
			best_combo = cutoff_combo

			print('So far - Best score = %.2f, Best val score = %.2f with %d features and CV cutoff = %.2f, logFC cutoff = %.2f' % (best_score, best_val_score,
				len(best_model_feats), best_combo['cv'], best_combo['logFC']))
			
	print('Best score = %.2f, Best val score = %.2f with %d features and CV cutoff = %.2f, logFC cutoff = %.2f' % (best_score, best_val_score, 
		len(best_model_feats), best_combo['cv'], best_combo['logFC']))

	train_data, val_data = make_train_val_ML_data_obj(train_rnaseq_meta, val_rnaseq_meta, y_col, group_col, best_model_feats, **kwargs)
	if val_data is not None:
		train_data.join(val_data)

	calib_clf = CalibratedClassifierCV(base_estimator = best_model, method = 'sigmoid', cv = 'prefit')
	calib_clf.fit(train_data.X, train_data.y)

	return {'model' : best_model, 'calib_model' : calib_clf, 'combo' : best_combo, 'score' : best_score, 'features' : best_model_feats}


def get_classification_results(dataset_label, model, data_to_use, threshold = 0.5):
	'''
	Utility to print classification model results
	Input:
		dataset_label - str, labeling dataset during printing
		model - fitted model object
		data_to_use - ML_data obj instance
	Prints AUC, confusion matrix, and classification report
	'''
	print('%s results:' % dataset_label)
	
	model_prob = model.predict_proba(data_to_use.X)[:, 1]
	model_pred = get_class_label(data_to_use, model, threshold)

	if data_to_use.y.unique().shape[0] > 1: #Can only calc AUC, precision, recall with 2 data classes
		print('ROC AUC = %0.2f' % roc_auc_score(y_true = data_to_use.y, y_score = model_prob))
	

	print('Report:')
	print(classification_report(data_to_use.y, model_pred))

	print('Confusion matrix:')
	print(confusion_matrix(data_to_use.y, model_pred))
	print()
	return

def make_probPE_and_gene_matrix(ML_data_obj_dict, model, threshold):
	'''
	Utility to create df with probability(PE) per sample for multiple datasets 
	Input: 
		ML_data_obj_dict - Dict where keys are str denoting the dataset name (e.g. 'Discovery') and values are ML_data instances {dataset_name : ML_data_obj}
		model - Fitted model
	Return: 
		df - pd df that contains prob_PE, predicted value, and whether a sample was correctly predicted
		gene_vals - pd df that contains gene vals for genes used in model for all datasets
	'''
	sample_indices = [obj.y.index.to_series() for obj in ML_data_obj_dict.values()]
	sample_indices = pd.concat(sample_indices, axis = 0)
	df = pd.DataFrame(columns = ['dataset', 'case', 'prob_PE', 
								'prediction', 'correctly_predicted'], index = sample_indices)

	gene_vals = None

	for key, ML_data_obj in ML_data_obj_dict.items():
		#Get df vals
		idx_vals = ML_data_obj.y.index
		df.loc[idx_vals, 'dataset'] = key
		df.loc[idx_vals, ['case']] = ML_data_obj.y
		df.loc[idx_vals, 'prob_PE'] = model.predict_proba(ML_data_obj.X)[:, 1] #0 = p(Not PE), 1 = p(PE)
		df.loc[idx_vals, 'prediction'] = get_class_label(ML_data_obj, model, threshold)			

		#Get gene_vals vals
		gene_vals = ML_data_obj.X if gene_vals is None else pd.concat((gene_vals, ML_data_obj.X), axis = 0)

	df.prob_PE = df.prob_PE.astype(float)
	df.correctly_predicted = (df.case == df.prediction)

	gene_vals = gene_vals.join(df.loc[:, ['case', 'dataset']])
	return df, gene_vals

def get_auc_roc_CI(fitted_model, ML_data_obj, seed = 37, to_bootstrap = False, ci_interval = 0.05):
	'''
	Utility to calculate values corresponding to ROC curve and its area under curve (AUC)/corresponding confidence interval
	Input: 
		fitted_model - fitted model to use
		ML_data_obj - instance of ML_data to use for calculating AUC
		seed - optional, random seed for bootstrapping CI
		ci_interval - optional, float denoting confidence interval desired. Default = 0.05 [Corresponds to 90% confidence interval]
	Return: dictionary
		'fpr' - Array of FPR for ROC curve
		'tpr' - Array of TPR for ROC curve
		'val' - AUC value
		'ci_lb' - Lower bound on AUC
		'ci_ub' - Upper bound on AUC
	'''
	y_true = ML_data_obj.y
	y_prob = fitted_model.predict_proba(ML_data_obj.X)[:, 1]
	fpr, tpr, tshlds = roc_curve(y_true = y_true, y_score = y_prob)

	pr, rcall, pr_tshlds = precision_recall_curve(y_true = y_true, probas_pred = y_prob, pos_label = 1)
	roc_auc = roc_auc_score(y_true, y_prob)
	
	if to_bootstrap:
		n_iters = 1000
		np.random.seed(seed)
		delta_auc = []

		for i in range(n_iters):
			rand_idx = np.random.choice(np.arange(0, len(y_true), 1), size = (len(y_true)), replace = True)
			#Check if y_true is all 0's or 1's. Cannot calculate AUC in that case so redo random sampling
			while (np.sum(y_true[rand_idx]) == len(y_true)) or (np.sum(y_true[rand_idx]) == 0): 
				rand_idx = np.random.choice(np.arange(0, len(y_true), 1), size = (len(y_true)), replace = True)
			delta_auc.append((roc_auc - roc_auc_score(y_true[rand_idx], y_prob[rand_idx])))

		delta_auc = np.array(delta_auc)
		delta_auc.sort()

		delta_upper = delta_auc[int((ci_interval * n_iters))]
		delta_lower = delta_auc[int(((1 - ci_interval) * n_iters))]
	else:
		n1 = ML_data_obj.y.sum()
		n2 = ML_data_obj.y.shape[0] - n1
		q1 = roc_auc / (2-roc_auc)
		q2 = (2*roc_auc**2)/(1+roc_auc)
		se_auc_num = roc_auc*(1-roc_auc) + (n1-1)*(q1-roc_auc**2) + (n2-1)*(q2-roc_auc**2)
		se_auc = (se_auc_num / (n1*n2))**0.5
		delta_lower = ci_interval*se_auc
		delta_upper = -1*delta_lower
	
	out = {'fpr' : fpr,
			'tpr' : tpr, 
			'roc_curve_tshlds' : tshlds,
			'val' : roc_auc.round(2),
			'ci_lb' : (roc_auc - delta_lower).round(3), 
			'ci_ub' : (roc_auc - delta_upper).round(3) ,
			'precision' : pr,
			'recall' : rcall,
			'pr_curve_tshlds' : pr_tshlds,
			'pr_auc' : auc(rcall, pr).round(2)
			}

	return out

def add_pred_to_meta(meta_df, model, ml_data_obj, threshold):
	'''
	Utlity to add pred data to meta
	'''
	pred = get_class_label(ml_data_obj, model, threshold)
	score = pd.Series(model.predict_proba(ml_data_obj.X)[:, 1], index = ml_data_obj.y.index, name = 'score')
	pred_score = pred.to_frame().join(score)
	meta_df_aug = meta_df.join(pred_score)
	meta_df_aug['is_correct'] = (meta_df_aug.prediction == meta_df_aug.case)
	return meta_df_aug

def get_agreement_btwn_mult_samples_per_subj(meta_df_w_pred, group_col = 'subject'):
	'''
	Utility to check agreement between multiple samples from same person
	'''
	mult_samples_per_x = meta_df_w_pred.groupby(group_col).case.count() > 1
	mult_samples_per_x = mult_samples_per_x.loc[mult_samples_per_x].index

	same_grp_stats = pd.DataFrame(columns = ['delta_prob', 'frac_same_pred', 'std_if_geq_3_samples'], index = mult_samples_per_x)
	for group_id, ds in meta_df_w_pred.loc[meta_df_w_pred[group_col].isin(mult_samples_per_x)].groupby(group_col):
		ds.sort_values('ga_at_collection', inplace = True)

		same_grp_stats.loc[group_id, 'delta_prob'] = ds.score.iloc[1] - ds.score.iloc[0]
		same_grp_stats.loc[group_id, 'frac_same_pred'] = ds.prediction.value_counts().iloc[0] / ds.shape[0]

		if ds.shape[0] > 2:
			same_grp_stats.loc[group_id, 'std_if_geq_3_samples'] = ds.score.std()

	return same_grp_stats
		