from util.gen_utils import *
from util.ml_utils import *
import argparse
import os

def argparser():
	parser = argparse.ArgumentParser(description = "Train two LR models to predict PE using RNAseq data")
	parser.add_argument("counts_table_path", type = str, help = "Path to training csv with counts table")
	parser.add_argument("tmm_path", type = str, help = "Path to training csv with TMM generated using EdgeR")
	parser.add_argument("metadata_path", type = str, help = "Path to training metadata csv ")
	parser.add_argument("out_prefix", type = str, help = "Prefix used including absolute path used to save output")
	parser.add_argument("--holdout_set_path", type = str, default = "", help = "Path to separate holdout set csv with counts table")
	parser.add_argument("--holdout_tmm_path", type = str, default = "", help = "Path to separate holdout set csv with TMM generated using EdgeR")
	parser.add_argument("--holdout_meta_path", type = str, default = "", help = "Path to separate holdout set csv with metadata")
	parser.add_argument("--to_batch_correct", "-bc", type = bool, default = False, help="Include zero-centering per batch")
	parser.add_argument("--gene_types_included", "-gt", type = str, default = "", 
		help = "Comma separated string containing ENSG defined gene types to include in analysis (i.e. 'protein_coding,lncRNA') Note no error checking done right now")
	return parser.parse_args()

def training_pipeline(train_data):

	#Define search space for CV 
	n_cv = len(train_data.groups.unique()) #LOOCV - 1 fold per subject
	Cs = np.logspace(start = -2, stop = 1, num = 15) #Inv reg strengths to test
	l1_ratios = np.linspace(0.25, 1, num = 4) #L1 ratios to test, Want some sparsity. 0 = L2, 1 = L1

	#Round 1 - Coarse grained search
	coarse_grain_results = LR_train_w_CV_controlled(train_data, n_cv_folds = n_cv, inv_reg_strength_arr = Cs, 
		penalty = 'elasticnet', seed = 37, l1_ratio_arr = l1_ratios)

	os.system('echo Coarse grain search finished')

	## Fine grained CV
	#Round 2 - Fine grained search 
	#Use only features previously identified as important (based on normalized median coef values across folds) where CV < 1.0
	coef_importance = get_avg_coef_importance(coarse_grain_results['coef'])
	feat_mask = np.round(coef_importance, 3) > 0 
	lr_feats = train_data.get_masked_feats(feat_mask)

	train_data.shrink_X_filter_genes(feat_mask)

	#Focus CV around previously identfied best inv_reg_strength 
	fine_grain_inv_reg_strength =  np.linspace(coarse_grain_results['best_inv_strength'] - coarse_grain_results['best_inv_strength']/2, 
								 coarse_grain_results['best_inv_strength'] + coarse_grain_results['best_inv_strength']/2, 25)

	#6-fold CV with roc_auc as scoring metric to optimize performance across whole dataset
	lr_cv_obj = LR_train_w_sklearnCV(train_data, n_cv_folds = 6, inv_reg_strength_arr = fine_grain_inv_reg_strength, scoring = 'roc_auc',
									 penalty = 'elasticnet', l1_ratio_arr = np.array([coarse_grain_results['best_l1_ratio']]))

	os.system('echo Fine grain search finished')

	#Utility for later
	#Cannot save LogisticRegressionCV obj as pkl (cv = generator obj) so once fittued use param to fit LR
	#Remove features with 0 as coefficient since do not contribute to score
	lr, feat_mask = make_LR_model_from_fitted_CV_model(lr_cv_obj, keep_zero_coef = False)
	lr_feats = train_data.get_masked_feats(feat_mask)
	
	train_data.shrink_X_filter_genes(feat_mask)

	## Calibration
	calibrated_lr = CalibratedClassifierCV(lr, method = 'sigmoid', cv = 'prefit').fit(train_data.X, train_data.y)
	return {'fitted_model' : calibrated_lr, 'features_included' : lr_feats}

def get_classification_results(dataset_label, model, data_to_use):
	print('%s: Calibrated LR Classification report:' % dataset_label)
	print('ROC AUC = %0.2f' % roc_auc_score(y_true = data_to_use.y, y_score = model.predict(data_to_use.X)))
	print(confusion_matrix(data_to_use.y, model.predict(data_to_use.X)))
	print(classification_report(data_to_use.y, model.predict(data_to_use.X)))
	return

def train_and_test_model(train_data, to_batch_correct, holdout_meta, holdout_rnaseq, model_name):
	model = training_pipeline(train_data)
	os.system('echo Done training model')
	
	#Get holdout data with matching features
	holdout_data = ML_data(meta = holdout_meta, rnaseq_inst = holdout_rnaseq, y_col = 'case', to_batch_correct = to_batch_correct,
		group_col = 'subject', features = model['features_included']) 
	
	#Report results
	get_classification_results(model_name + ' (training)', model['fitted_model'], train_data)
	get_classification_results(model_name + ' (holdout)', model['fitted_model'], holdout_data)
	
	return model

def main():
	args = argparser()

	os.system('echo Starting')

	all_meta = read_sample_meta_table(args.metadata_path)
	rnaseq = rnaseq_data(args.counts_table_path, args.tmm_path, mygene_db = mygene.MyGeneInfo())
	if args.gene_types_included is not "":
		rnaseq.filter_to_gene_types(args.gene_types_included.split(","))

	#Either holdout set is included or made out of training DS by splitting training dataset
	has_sep_holdout = (args.holdout_set_path != "")
	training_frac = 0.8
	rnaseq_holdout = rnaseq

	if has_sep_holdout: 
		training_frac = 1.0
		meta_holdout = read_sample_meta_table(args.holdout_meta_path)
		rnaseq_holdout = rnaseq_data(args.holdout_set_path, args.holdout_tmm_path, mygene_db = mygene.MyGeneInfo())

	#Init masks
	masks = data_masks(train_frac = training_frac, seed = 1041, label_col = 'case') 
	masks.add_mask('is_collected_pre17wks', (all_meta.ga_at_collection <= 16))
	masks.add_mask('is_training', masks.get_sampled_mask(all_meta, addtnl_mask_label = 'is_collected_pre17wks', blocking_col = 'subject'))
	masks.add_mask('is_pp', (all_meta.is_pp == 1)) #Want to filter post-partum samples
	
	#Logical combinations
	masks.add_mask_logical_and_combinations('is_training', 'is_collected_pre17wks')
	masks.add_mask_logical_and_combinations('is_training', 'is_pp')

	#Load training data
	pre_17wks_training_data = ML_data(meta = all_meta.loc[masks.masks['is_training_and_is_collected_pre17wks']], rnaseq_inst = rnaseq, 
		y_col = 'case', to_batch_correct = args.to_batch_correct, group_col = 'subject')
	all_ga_training_data = ML_data(meta = all_meta.loc[masks.masks['is_training_and_not_is_pp']], rnaseq_inst = rnaseq, 
		y_col = 'case', to_batch_correct = args.to_batch_correct, group_col = 'subject')

	#Train and test pre 17 wk model
	pre_17wks_holdout_meta = meta_holdout.loc[meta_holdout.ga_at_collection <= 16] if has_sep_holdout else all_meta.loc[masks.masks['not_is_training_and_is_collected_pre17wks']]
	pre_17wks = train_and_test_model(pre_17wks_training_data, args.to_batch_correct, pre_17wks_holdout_meta, rnaseq_holdout, 'Pre 17 weeks')
	
	#Train and test all ga model
	all_ga_holdout_meta = meta_holdout.loc[meta_holdout.is_pp == 0] if has_sep_holdout else all_meta.loc[masks.masks['not_is_training_and_not_is_pp']]
	all_ga = train_and_test_model(all_ga_training_data, args.to_batch_correct, all_ga_holdout_meta, rnaseq_holdout, 'All GA')

	## Save models, masks, and features used
	write_pkl(masks, args.out_prefix + "masks.pkl")
	write_pkl(pre_17wks['fitted_model'], args.out_prefix + "_pre17wk_model.pkl")
	write_pkl(all_ga['fitted_model'], args.out_prefix + "_allGA_model.pkl")

	pre_17wks['features_included'].to_frame().reset_index(drop = True).to_csv(args.out_prefix + "_pre17wk_features.csv", index = False)
	all_ga['features_included'].to_frame().reset_index(drop = True).to_csv(args.out_prefix + "_allGA_features.csv", index = False)

	return

if __name__ == '__main__':
	main()