import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from util.class_def.ml_obj_classes import *

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score, classification_report, roc_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def preprocess_data(fitted_preprocessor, data):
	return pd.DataFrame(fitted_preprocessor.transform(data), index = data.index, columns = data.columns)

def get_best_CV_score(cv_true_labels, cv_predictions, l1_ratios, beta = 1.5, to_plot = True):
	_, n_reg_strengths, n_l1_ratios = cv_predictions.shape
	cv_scores = np.zeros((n_reg_strengths, n_l1_ratios))

	for i in range(n_reg_strengths):
		for j in range(n_l1_ratios):
			#Beta in fbeta tunes f1 score biasing for recall or precision. Biasing for better recall scores here (Want to make sure to capture all PE)
			cv_scores[i, j] = fbeta_score(y_true = cv_true_labels, y_pred = cv_predictions[:, i, j], beta = beta)

	best_score_inv_reg_strength_idx, best_score_l1_ratio_idx = np.unravel_index(np.argmax(cv_scores), cv_scores.shape)
	print('Best fbeta score after CV = %0.3f' % cv_scores[best_score_inv_reg_strength_idx, best_score_l1_ratio_idx])

	if to_plot:
		plt.figure()
		for j in range(n_l1_ratios):
			plt.plot(np.arange(n_reg_strengths), cv_scores[:, j], label = 'L1 ratio = %.2f' % l1_ratios[j])
		plt.legend()
		plt.xlabel('Inverse regularization strength index')
		plt.ylabel('Fbeta score (Beta = %0.1f)' % beta)

	return (best_score_inv_reg_strength_idx, best_score_l1_ratio_idx)

def get_avg_coef_importance(cv_coef):
	importance = np.abs(cv_coef)
	importance /= np.sum(importance, axis = 1)[:, np.newaxis]
	return np.median(importance, axis = 0)

def make_LR_model_from_fitted_CV_model(og_model, seed = 37, keep_zero_coef = False): 
	#Cannot save LogisticRegressionCV obj as pkl so once fitted use params from CV obj to fit LR
	
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

def LR_train_w_sklearnCV(train_data, n_cv_folds, inv_reg_strength_arr, penalty = 'elasticnet', scoring = 'roc_auc', seed = 37, l1_ratio_arr = None, batch_corrector = None):
	# if batch_corrector is not None:
	# 	train_data.X = preprocess_data(batch_corrector, train_data.X)
		
	lr = LogisticRegressionCV(cv = GroupKFold(n_splits = n_cv_folds).split(X = train_data.X, y = train_data.y, groups = train_data.groups),
								Cs = inv_reg_strength_arr, penalty = penalty, l1_ratios = l1_ratio_arr, scoring = scoring, 
								solver = 'saga', max_iter = 10e6, random_state = seed
								)
	lr.fit(train_data.X, train_data.y)

	print('Fitted LR model has %d non-zero coefficients' % np.sum(lr.coef_[0, :] != 0))
	print('Classification report:')
	print(classification_report(train_data.y, lr.predict(train_data.X)))

	return lr

def LR_train_w_CV_controlled(train_data, n_cv_folds, inv_reg_strength_arr, penalty = 'elasticnet', seed = 37, l1_ratio_arr = None, batch_corrector = None):
	n_samples, n_feats = train_data.X.shape
	n_reg_strengths = len(inv_reg_strength_arr)
	n_l1_ratios = len(l1_ratio_arr) if l1_ratio_arr is not None else 1

	#Init CV stats arrays
	cv_coef = np.zeros((n_cv_folds, n_reg_strengths, n_l1_ratios, n_feats))
	cv_predictions = np.zeros((n_samples, n_reg_strengths, n_l1_ratios))
	cv_true_labels = np.ones((n_samples)) * -1 

	#Loop counter
	reg_strength_i = 0

	#Make sure loop works with all penalties
	if penalty == 'l1':
		l1_ratio_arr = np.array([1.0]) #L1 is equivalent to elasticnet with l1_ratio = 1

	if penalty == 'l2':
		l1_ratio_arr = np.array([0.0]) #L2 is equivalent to elasticnet with l1_ratio = 1
	
	# og_train_data = deepcopy(train_data)
	# if batch_corrector is not None:
	# 	train_data.X = preprocess_data(batch_corrector, train_data.X)

	#Loop and CV
	for inv_reg_strength in inv_reg_strength_arr:
		l1_ratio_i = 0
		print('Now fitting model %d of %d with inverse regularization strength = %0.3f' % (reg_strength_i + 1, n_reg_strengths, inv_reg_strength))

		for l1_ratio in l1_ratio_arr:
			cv_fold_i = 0
			sample_i = 0 #Sometimes multiple samples per cv fold
			if penalty == 'elasticnet':
				print('Now fitting model %d of %d with l1_ratio = %0.2f' % (l1_ratio_i + 1, n_l1_ratios, l1_ratio))

			for cv_train_idx, cv_test_idx in GroupKFold(n_splits = n_cv_folds).split(X = train_data.X, y = train_data.y, groups = train_data.groups):
				#Pull training samples + features (no zero skew) for this round
				cv_X, cv_y = train_data.filter_samples(cv_train_idx, is_iloc = True)
				feat_sel_mask = np.ones(cv_X.shape[1], dtype = bool) #og_train_data.get_no_zero_skew_mask(cv_train_idx, is_iloc = True, no_zero_skew_cutoff = 0.01) 

				#Fit model
				lr_cv = LogisticRegression(C = inv_reg_strength, penalty = 'elasticnet', l1_ratio = l1_ratio, 
											solver = 'saga', max_iter = 10e6, random_state = seed)
				lr_cv.fit(cv_X.loc[:, feat_sel_mask], cv_y)

				#Save stats on withheld fold
				n_test_samples = len(cv_test_idx)
				cv_test_X, cv_test_y = train_data.filter_samples(cv_test_idx, is_iloc = True)
				cv_coef[cv_fold_i, reg_strength_i, l1_ratio_i, feat_sel_mask] = lr_cv.coef_[0, :]
				cv_predictions[sample_i : (sample_i + n_test_samples), reg_strength_i, l1_ratio_i] = lr_cv.predict(cv_test_X.loc[:, feat_sel_mask])

				if reg_strength_i == 0:
					cv_true_labels[sample_i : (sample_i + n_test_samples)] = cv_test_y

				cv_fold_i += 1
				sample_i += n_test_samples

			l1_ratio_i += 1

		reg_strength_i += 1

	best_score_inv_reg_strength_idx, best_score_l1_ratio_idx = get_best_CV_score(cv_true_labels, cv_predictions, l1_ratio_arr, beta = 1.5, to_plot = True)
	return {'best_inv_strength' : inv_reg_strength_arr[best_score_inv_reg_strength_idx], 
			'best_l1_ratio' : l1_ratio_arr[best_score_l1_ratio_idx], 
			'coef' : cv_coef[:, best_score_inv_reg_strength_idx, best_score_l1_ratio_idx, :]
			}

def make_fig3B_matrix(ML_data_obj_dict, model, meta):
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


		