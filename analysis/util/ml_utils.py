## ML specific utilities

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import re

from util.gen_utils import read_sample_meta_table
from util.class_def.ml_obj_classes import *
from util.class_def.de_obj_classes import de_data

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score, classification_report, roc_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

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
		feat_mask - np boolean array indicating which coef were kept. Only importance if keep_zero_coef = True
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
								Cs = inv_reg_strength_arr, penalty = penalty, l1_ratios = l1_ratio_arr, scoring = scoring, 
								solver = 'saga', max_iter = 10e6, random_state = seed
								)
	lr.fit(train_data.X, train_data.y)

	return lr

def training_pipeline(train_meta, train_rnaseq, train_logFC_per_group, relevant_group_names_in_logFC, 
						cv_cutoffs_to_try, logFC_cutoffs_to_try, expr_cutoffs_to_try, n_max_coef = 25):
	'''
	Full training pipeline with feature selection for logistic regression model with sklearn default CV 
	Input: 
		train_meta - metadata associated with training data
		train_rnaseq - rnaseq obj instance associated with training data, see rnaseq_data in obj_classes for more info
		train_logFC_per_group - logFC_data_by_group obj instance asso with feature selection
		relevant_group_names_in_logFC - list with relevant column names from train_logFC_per_group to be used during feature selection
		cv_cutoffs_to_try - array of possible cutoff values for CV in train_logFC_per_group
		logFC_cutoffs_to_try - array of possible cutoff values for logFC in train_logFC_per_group
		expr_cutoffs_to_try - array of possible cutoff values for CPM expression in train_rnaseq.CPM.
		n_max_coef - max coefficient number a model can have to be accepted [Helpful to avoid overfitting given small data]
	Return: 
		dict with {'model' : CalibratedClassifierCV fitted model, 'score' : best score, 'combo' : best combo of CV, logFC, expr cutoffs}
	'''

	cv_logFC_combos = list(product(cv_cutoffs_to_try, logFC_cutoffs_to_try))
	cv_logFC_high_expr_combos = list(product(cv_logFC_combos, expr_cutoffs_to_try))
	
	n_combos = len(cv_logFC_high_expr_combos)
	cutoff_combos = {'cutoff_combo_' + str(i + 1) : {'cv' : cv_logFC_high_expr_combos[i][0][0], 
													'logFC' : cv_logFC_high_expr_combos[i][0][1], 
													'expr' : cv_logFC_high_expr_combos[i][1]}
					for i in np.arange(n_combos)
					}
					
	#Holder vars for best model
	best_score = 0
	best_combo = None
	best_model = None
	best_model_feats = None
	
	#Init train_data
	train_data = ML_data(meta = train_meta, rnaseq_inst = train_rnaseq, y_col = 'case', to_batch_correct = True, group_col = 'subject')

	#Set up other LR hyperparams
	n_cv = len(train_data.groups.unique()) #LOOCV - 1 fold per subject
	Cs = np.logspace(start = -3, stop = 1, num = 30) #Inv reg strengths to test
	l1_ratios = np.linspace(0.25, 1, num = 5) #L1 ratios to test, Want some sparsity. 0 = L2, 1 = L1
	
	i = 0
	for cutoff_combo_name, cutoff_combo in cutoff_combos.items():
		train_logFC_per_group.mod_CV_mask(cutoff_combo['cv'])
		train_logFC_per_group.mod_logFC_mask(cutoff_combo['logFC'])
		
		init_feat_mask = np.logical_and(train_logFC_per_group.CV_mask.loc[:, relevant_group_names_in_logFC], 
										train_logFC_per_group.logFC_mask.loc[:, relevant_group_names_in_logFC]).sum(axis = 1) > 0
		
		high_expr_mask = train_rnaseq.CPM.loc[train_logFC_per_group.logFC.index, train_data.y.index].median(axis = 1) > cutoff_combo['expr']		
		init_feats = train_logFC_per_group.logFC.loc[np.logical_and(high_expr_mask, init_feat_mask)].index

		if init_feats.shape[0] == 0: #Sometimes no genes come up
			continue
		
		train_data = ML_data(meta = train_meta, rnaseq_inst = train_rnaseq, y_col = 'case', to_batch_correct = True, group_col = 'subject', features = init_feats)

		curr_lr = LR_train_w_sklearnCV(train_data, n_cv_folds = n_cv, inv_reg_strength_arr = Cs, scoring = 'accuracy',
										 penalty = 'elasticnet', l1_ratio_arr = l1_ratios)
	
		curr_score = roc_auc_score(y_true = train_data.y, y_score = curr_lr.predict(train_data.X))
		#fbeta_score(y_true = train_data.y, y_pred = curr_lr.predict(train_data.X), beta = 1.0)
		curr_n_coef = curr_lr.coef_.shape[1]
		
		if curr_score > best_score and  curr_n_coef <= n_max_coef:
			#To save would need object without CV generator
			lr, feat_mask = make_LR_model_from_fitted_CV_model(curr_lr, keep_zero_coef = False)
			best_model_feats = train_data.get_masked_feats(feat_mask)
			train_data.shrink_X_filter_genes(feat_mask)
			
			best_score = curr_score
			best_model = CalibratedClassifierCV(lr, method = 'sigmoid', cv = 'prefit').fit(train_data.X, train_data.y)
			best_combo = cutoff_combo

		i += 1
		if i % 50 == 0:
			print('Now completed %d iterations' % i)
			
	print('Best score = %.2f with %d features and CV cutoff = %.2f, logFC cutoff = %.2f, CPM cutoff = %d' % (best_score, len(best_model_feats), 
		best_combo['cv'], best_combo['logFC'], best_combo['expr']))
	return {'model' : best_model, 'combo' : best_combo, 'score' : best_score, 'features' : best_model_feats}

def get_classification_results(dataset_label, model, data_to_use):
	'''
	Utility to print classification model results
	Input:
		dataset_label - str, labeling dataset during printing
		model - fitted model object
		data_to_use - ML_data obj instance
	Prints AUC, confusion matrix, and classification report
	'''
	print('%s: Calibrated LR Classification report:' % dataset_label)
	print('ROC AUC = %0.2f' % roc_auc_score(y_true = data_to_use.y, y_score = model.predict(data_to_use.X)))
	print(confusion_matrix(data_to_use.y, model.predict(data_to_use.X)))
	print(classification_report(data_to_use.y, model.predict(data_to_use.X)))
	return

def read_delvecchio_meta(biosample_results_path, sra_results_path):
	'''
	Utility to read biosample_result.txt file and pull out relevant information. 
	Specific to the DelVecchio et al BioSample file

	Input:
		biosample_results_path - filepath to file that contains Biosample results
		sra_results_path - filepath to file that contains SRA Run Table 
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
	return meta

def make_fig3B_matrix(ML_data_obj_dict, model, meta):
	'''
	Utility to create pd df for figure 3B 
	Input: 
		ML_data_obj_dict - Dict where keys are str denoting the dataset name (e.g. 'Training') and values are ML_data instances {dataset_name : ML_data_obj}
		model - Fitted model
		meta - pd df containing metadata for all samples included in ML_data_obj_dict
	Return: pd df that contains relevant info for fig 3B including prob_PE, predicted value, and whether a sample was correclty predicted
	'''
	sample_indices = [obj.y.index.to_series() for obj in ML_data_obj_dict.values()]
	sample_indices = pd.concat(sample_indices, axis = 0)
	df = pd.DataFrame(columns = ['dataset', 'case', 'prob_PE', 'ga_at_collection', 
								'pe_onset_ga_wk', 'pe_feature', 'delta_collection_onset',
								'prediction', 'correctly_predicted'], index = sample_indices)

	for key, ML_data_obj in ML_data_obj_dict.items():
		idx_vals = ML_data_obj.y.index
		df.loc[idx_vals, 'dataset'] = key
		df.loc[idx_vals, ['case']] = ML_data_obj.y
		df.loc[idx_vals, 'prob_PE'] = model.predict_proba(ML_data_obj.X)[:, 1] #0 = p(Not PE), 1 = p(PE)
		df.loc[idx_vals, 'prediction'] = model.predict(ML_data_obj.X)
		df.loc[idx_vals, ['ga_at_collection', 'pe_onset_ga_wk', 'pe_feature']] = meta.loc[idx_vals, ['ga_at_collection', 'pe_onset_ga_wk', 'pe_feature']]

	df.ga_at_collection = df.ga_at_collection.astype(int)
	df.prob_PE = df.prob_PE.astype(float)
	df.pe_onset_ga_wk = df.pe_onset_ga_wk.astype(float)
	df.delta_collection_onset = df.ga_at_collection - df.pe_onset_ga_wk
	df.correctly_predicted = df.case == df.prediction
	return df

def get_auc_roc_CI(fitted_model, ML_data_obj, seed = 37, ci_interval = 0.025):
	'''
	Utility to calculate values corresponding to ROC curve and its area under curve (AUC)/corresponding confidence interval
	Input: 
		fitted_model - fitted model to use
		ML_data_obj - instance of ML_data to use for calculating AUC
		seed - optional, random seed for bootstrapping CI
		ci_interval - optional, float denoting confidence interval desired. Default = 0.025 [Corresponds to 95% confidence interval]
	Return: dictionary
		'fpr' - Array of FPR for ROC curve
		'tpr' - Array of TPR for ROC curve
		'auc' - AUC value
		'ci_auc_lb' - Lower bound on AUC
		'ci_auc_ub' - Upper bound on AUC
	'''
	y_true = ML_data_obj.y
	y_prob = fitted_model.predict(ML_data_obj.X)
	fpr, tpr, _ = roc_curve(y_true = y_true, y_score = y_prob)
	auc = roc_auc_score(y_true, y_prob)
	
	print(confusion_matrix(y_true, y_prob))
	
	n_iters = 1000
	np.random.seed(seed)
	delta_auc = []

	for i in range(n_iters):
		rand_idx = np.random.choice(np.arange(0, len(y_true), 1), size = (len(y_true)), replace = True)
		#Check if y_true is all 0's or 1's. Cannot calculate AUC in that case so redo random sampling
		while (np.sum(y_true[rand_idx]) == len(y_true)) or (np.sum(y_true[rand_idx]) == 0): 
			rand_idx = np.random.choice(np.arange(0, len(y_true), 1), size = (len(y_true)), replace = True)
		delta_auc.append((auc - roc_auc_score(y_true[rand_idx], y_prob[rand_idx])))

	delta_auc = np.array(delta_auc)
	delta_auc.sort()

	delta_upper = delta_auc[int((ci_interval * n_iters))]
	delta_lower = delta_auc[int(((1 - ci_interval) * n_iters))]
	
	out = {'fpr' : fpr,
			'tpr' : tpr, 
			'auc' : auc,
			'ci_auc_lb' : (auc - delta_lower), 
			'ci_auc_ub' : (auc - delta_upper)
			}

	return out


		