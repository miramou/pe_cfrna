## DE Analysis specific utilities

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from util.class_def.de_obj_classes import *
from util.gen_utils import *
from util.plot_utils import DE_plot_order

from gprofiler import GProfiler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

def get_hc_stats(dataset_name, cm, **kwargs):
	'''
	Utility fxn to get Spec, Sens from 2x2 confusion matrix
	Input:
		dataset_name - str, dataset label
		cm - confusion_matrix
		**kwargs - key word args to pass to get_stats
	Return dictionary with following keys
		Each key has dict value with 3 keys 'val', 'ci_lb', 'ci_ub' corresponding to the value, CI lower bound, and CI upper bound

		Sensitivity = TP / (TP + FN)
		Specificity = TN / (TN + FP)
	'''
	specs_to_get = ['Sensitivity', 'Specificity']
	specs = get_stats(cm, specs_to_get, **kwargs)

	all_spec_vals = [(prop, specs[prop]['val'], specs[prop]['ci_lb'], specs[prop]['ci_ub']) for prop in specs_to_get]

	print('%s: %s = %d%% [%d-%d%%], %s = %d%% [%d-%d%%]' % (dataset_name, *all_spec_vals[0], *all_spec_vals[1]))
	return specs

def get_go_table(queries, **kwargs):
	'''
	Utility fxn to get GO table for given queries using GProfiler
	For IEA and other definitions - see https://biit.cs.ut.ee/gprofiler/page/docs#electronic_annotations_iea

	Inputs:
		queries - dict where key = query name and val = ensg IDs for genes
		**kwargs - arguments to pass to GProfiler().profile(**kwargs)
	Returns:
		go_table - df with GO terms identified
	'''
	go_table = GProfiler(return_dataframe = True).profile(**kwargs, query = queries).sort_values(by = ['query','source'])
	print('GO table has %d terms' % (go_table.shape[0]))
	return go_table

def prune_go_table(go_table_og, choose_parent = True):
	'''
	Utility fxn to prune GO table
	Sometimes GO table can have too many terms to plot, if so can prune s.t. only plot parent terms

	Inputs:
		go_table_og - df, initial GO table from GProfiler
		choose_parent - bool, if True, prune to only parents, if False, prune to child
	Returns:
		go_table_pruned - pruned df with only parent terms
	'''

	ordered_idx = go_table_og.index.to_numpy() 
	ordered_idx_only_keep = go_table_og.index.to_numpy() if choose_parent else None #Init

	#Get idx for parent
	for group, go_grouped in go_table_og.groupby('query'):
		#Cannot use sort values as is since lists are not hashable
		is_parent = [go_grouped.native.isin(parent_list) for parent_list in go_grouped.parents] #True = parent
		
		if not choose_parent:
			no_parents_idx = go_grouped.index[[is_parent_i.sum() == 0 for is_parent_i in is_parent]]
			parent_idx = [go_grouped[is_parent_i].index[0] for is_parent_i in is_parent if np.sum(is_parent_i) > 0]
			head_parent_idx = np.array(list(set(parent_idx).intersection(set(no_parents_idx))))

			leaf_idx = go_grouped.index[[is_parent_i.sum() > 0 for is_parent_i in is_parent]].difference(parent_idx)
			is_parent_of_leaf = [go_grouped.native.isin(parent_list) for parent_list in go_grouped.loc[leaf_idx].parents]
			parent_of_leaf_idx = set([go_grouped[is_parent_i].index[0] for is_parent_i in is_parent_of_leaf if np.sum(is_parent_i) > 0])
			
			to_keep = leaf_idx.union(parent_of_leaf_idx).difference(head_parent_idx)
			
			ordered_idx_only_keep = to_keep if ordered_idx_only_keep is None else ordered_idx_only_keep.union(to_keep)

			continue

		child_i = -1
		for elem in is_parent:
			child_i += 1
			if np.sum(elem) == 0: #All false
				continue
			child_idx = go_grouped.iloc[child_i].name
			parent_idx = go_grouped[elem].index.to_numpy()
			
			ordered_idx = ordered_idx[~np.isin(ordered_idx, parent_idx)] #Choose all elem except parents			
			ordered_idx = np.concatenate((ordered_idx[: (remove_iloc + 1)], parent_idx, ordered_idx[(remove_iloc + 1):]), axis = 0)

			ordered_idx_only_keep = ordered_idx_only_keep[~np.isin(ordered_idx_only_keep, child_idx)] #Choose all elem except children
	
	go_table_pruned = go_table_og.loc[ordered_idx_only_keep].sort_values(by = ['query', 'p_value'])
	print('GO table has %d terms after pruning' % (go_table_pruned.shape[0]))
	return go_table_pruned

def get_and_prune_go_table(queries, query_type = 'kmeans', choose_parent = True, **kwargs):
	'''
	Wrapper that combines get_go_table and prune_go_table

	Inputs:
		Same as get_go_table

	Returns: Tuple
		original_go_table
		pruned_go_table
	'''
	
	go_table_og = get_go_table(queries, **kwargs)
	go_table_pruned = prune_go_table(go_table_og, choose_parent)

	#Make kmeans_clusters a category var
	go_table_og.loc[:, 'query'] = DE_plot_order(go_table_og.loc[:, 'query'], query_type)
	go_table_pruned.loc[:, 'query'] = DE_plot_order(go_table_pruned.loc[:, 'query'], query_type)

	return go_table_og, go_table_pruned	

def permute(init_df, n_total_permutations = 1000, axis = 1, seed = 37):
	'''
	Utility fxn to randomly permute df along axis yielding n_total_permutations

	Inputs:
		init_df - df that contains data you wish to permute
		n_total_permutations - total number of permutations
		axis - axis along which to permute, default = 1
		seed - random state

	Return: list with n_total_permutations of df.
	'''
	permutations = []

	np.random.seed(seed)
	for i in np.arange(n_total_permutations):
		other_axis = 0 if axis == 1 else 1
		init_idx_shuffled = np.array([np.random.permutation(np.arange(init_df.shape[axis])) for i in np.arange(init_df.shape[other_axis])])
		init_idx_shuffled = init_idx_shuffled.T if axis == 0 else init_idx_shuffled

		other_axis_idx = np.repeat(np.arange(init_df.shape[other_axis]), init_df.shape[axis])
	
		idx_both_axis = (other_axis_idx, init_idx_shuffled.flatten()) if axis == 1 else (init_idx_shuffled.flatten(), other_axis_idx)
		permuted = pd.DataFrame(init_df.to_numpy()[idx_both_axis].reshape(init_df.shape), index = init_df.index, columns = init_df.columns)
		
		permutations.append(permuted)

	return permutations

def kmeans_cluster(df, n_max_clusters, manually_choose_k = False, selected_k = 2, seed = 37, to_print_plot = True):
	'''
	Utility fxn to identify best k and then appropriately cluster data

	Inputs:
		df - df that contains data you wish to cluster with shape [n_features, n_groups]
		n_max_clusters - n_groups raised to n_possibilities where n_groups is jth dim of df and n_possibililties is usually binary
		seed - seed to ensure repeatability
		to_print - bool, whether to print stats

	Return: Dict with following keys / values
		'elbow_plot' - visualization of best k, y-axis = sum sq distance
		'clusters' - series of shape [n_features,] where each row corresponds to the cluster for that feature
		'n_clusters' - num of clusters
	'''
	sum_sq_dist = []
	clusters = np.arange(1, (n_max_clusters + 1))

	models = {}
	for k in clusters:
		models[k] = KMeans(n_clusters = k, random_state = seed).fit(df)
		sum_sq_dist.append(models[k].inertia_)

	kneedle = KneeLocator(clusters, sum_sq_dist, S = 1.0, curve = 'convex', direction = 'decreasing')
	n_clusters = kneedle.elbow if not manually_choose_k else selected_k

	if to_print_plot:

		fig, ax = plt.subplots()
		plt.plot(clusters, sum_sq_dist, marker = 'o', color = 'navy', label = 'Data')
		plt.vlines(x = n_clusters, ymin = min(sum_sq_dist) - 10, ymax = max(sum_sq_dist) + 10, linestyle = 'dashed', label = 'Chosen K')
		plt.xlabel('K')
		plt.ylabel('Sum of squared distances')
		plt.legend()

		print('Identified %d clusters' % n_clusters)

	#Add 1 - Bcz starts at 0, want to start at 1
	clusters = pd.Series((models[n_clusters].predict(df) + 1), index = df.index, name = 'kmeans_cluster') 
	
	#Sort cluster ID by n_feats in it in ascending order
	n_feats_per_cluster = clusters.value_counts(ascending = True)
	clusters_sorted = clusters.copy()
	i = 1
	for cluster_i in n_feats_per_cluster.keys():
		clusters_sorted.loc[clusters == cluster_i] = i
		i += 1

	out = {'clusters' : clusters_sorted, 'n_clusters' : n_clusters}

	if to_print_plot:
		print('N genes per cluster')
		print(clusters_sorted.value_counts(ascending = True))

		print('Percent features per cluster')
		print((clusters_sorted.value_counts(ascending = True) / clusters_sorted.shape[0]).round(2))

		out['elbow_plot'] = fig

	return out

def fit_line_get_coefs(x_vals, y_vals):
	'''
	Utility fxn to get slope [Used here with cluster data]

	Inputs:
		x_vals - array of x_vals
		y_vals - array of y_vals
		
	Return: Slope of fitted line
	'''
	return np.polyfit(x_vals, y_vals, deg = 1) #highest order 1st

def permute_and_kmeans(df, n_total_permutations, n_max_clusters, **kwargs):
	'''
	Wrapper to run permutation test related to kmeans clustering of logFC over time

	Inputs:
		df - df that contains data that has been clustered with shape [n_features, n_groups]
		n_total_permutations - n permutations to run
		n_max_clusters - n max possible kmeans clusters
		kwargs for kmeans_cluster

	Returns: dict with following vals
		permutation_ex - example permutation to visualize
		kmeans_permutation - kmeans_clustering for permutation_ex
	'''
	permutations = permute(df, n_total_permutations = n_total_permutations)
	kmeans_permuted = [kmeans_cluster(permutation, n_max_clusters = n_max_clusters, to_print_plot = False, **kwargs) for permutation in permutations]

	return {'permutation_ex' : permutations[0], 'kmeans' : kmeans_permuted[0]}

def read_ref_ds(gini_df_path, mean_exp_df_path, gini_cutoff, is_tsp):
	gini = pd.read_csv(gini_df_path, index_col = 0)
	mean_exp = pd.read_csv(mean_exp_df_path, index_col = 0)

	idx_geq_cutoff = gini.loc[gini.Gini >= gini_cutoff].index
	gini_geq_cutoff_w_max_exp = gini.loc[idx_geq_cutoff].join(mean_exp.loc[idx_geq_cutoff].idxmax(axis = 1).to_frame().rename(columns = {0 : 'max_exp_in'}), how = 'inner')

	if is_tsp:
		#Fix ENSG to match DEG list
		ensg_only = pd.Series([ensg[0] for ensg in gini_geq_cutoff_w_max_exp.index.str.split('.', expand = True)])
		ensg_to_keep_mask = (~ensg_only.duplicated(keep = 'first')).to_list()
		gini_geq_cutoff_w_max_exp = gini_geq_cutoff_w_max_exp.loc[ensg_to_keep_mask]
		gini_geq_cutoff_w_max_exp['gene_num'] = ensg_only[ensg_to_keep_mask].to_list()
		gini_geq_cutoff_w_max_exp.reset_index(drop = True, inplace = True)
		gini_geq_cutoff_w_max_exp.set_index('gene_num', inplace = True)

		#For purposes here, lump together neutrophils. 
		#The subtypes are based on clustering on manifold and it's unclear if actually dif. Usually a gene with high expr in one type of neutrophil has high in all
		gini_geq_cutoff_w_max_exp.loc[gini_geq_cutoff_w_max_exp.max_exp_in.str.contains('neutrophil'), 'max_exp_in'] = 'neutrophil' 
		#The subtypes are based on clustering on manifold and it's unclear if actually dif. 
		gini_geq_cutoff_w_max_exp.loc[gini_geq_cutoff_w_max_exp.max_exp_in.str.contains('endothelial'), 'max_exp_in'] = 'endothelial cell' 


	return gini_geq_cutoff_w_max_exp

def get_de_ref_kmeans_intersection_and_fracs(de_sig_df, ref_df, kmeans_df = None):
	'''
	Utility to get intersection of DEGs, kmeans_clusters, and reference (HPA, TSP+)
	Input:
		de_sig_df - dataframe with DEGs
		ref_df - reference df
		kmeans_df - kmeans df
	Return:
		Dict with following keys
		deg_ref_isec - the intersected df
		pct_isec - % of genes that intersected
		pct_non_isec - 100 - pct_isec
	'''
	print(de_sig_df.index.get_level_values('gene_num').intersection(ref_df))
	intersection_df = de_sig_df.join(ref_df, on = 'gene_num', how = 'inner')
	if kmeans_df is not None:
		intersection_df = intersection_df.join(kmeans_df)
	else:
		intersection_df['kmeans_cluster'] = 0
	n_DEGs_specific = intersection_df.shape[0]
	
	intersection_pct = round((n_DEGs_specific / de_sig_df.shape[0]) * 100, 0)

	return {'deg_ref_isec' : intersection_df, 'pct_isec' : intersection_pct, 'pct_non_isec' : 100 - intersection_pct}

def test_DEGs_for_ref_enrichment(deg_kmeans_isec_df, ref, ref_specific_to, mult_hyp_corr_method = 'fdr_bh', 
	alpha = 0.05, more_than_2_only = True):
	'''
	Utility to use hypergeometric test _ mult hypo correction to test for gene enrichment
	Input:
		deg_kmeans_isec_df - df from get_de_ref_kmeans_intersection_and_fracs
		ref - reference df
		ref_specific_to - either cell_type or tissue
		mult_hyp_corr_method - choose one of options eg fdr_bh, bonferroni
		alpha - sig level
		more_than_two_only - only test DEG intersections with size 2+
	Return:
		df with adj pvals and sig
	'''
	n_genes_specific_to_any = ref.shape[0]

	groupby_cols = ['kmeans_cluster', 'max_exp_in'] 

	n_DEGs_high_gini_per_cluster = deg_kmeans_isec_df.groupby('kmeans_cluster').kmeans_cluster.count()

	ref_DEG_enrichment = pd.DataFrame(columns = ['n_DEG_specific_to_this', 'n_genes_specific_to_this', 'n_DEGs_specific_to_any_name_for_query', 'p_val', 'DEGs_included', 'ref'], 
									  index = pd.MultiIndex.from_tuples(deg_kmeans_isec_df.groupby(groupby_cols).count().index, 
									  									names = ['kmeans_cluster', 'specific_to']))
	
	for kmeans_cluster_specific_to, cluster_DEG_specific in deg_kmeans_isec_df.groupby(groupby_cols):
		kmeans_cluster, specific_to = kmeans_cluster_specific_to

		n_genes_specific_to_this = ref.loc[ref['max_exp_in'] == specific_to].shape[0]
		n_DEGs_specific_to_this = cluster_DEG_specific.shape[0]
		
		#Only look at 2+ gene intersection
		if more_than_2_only and n_DEGs_specific_to_this < 2: 
			continue
		
		pval = scipy.stats.hypergeom(M = n_genes_specific_to_any, n = n_genes_specific_to_this, 
									N = n_DEGs_high_gini_per_cluster[kmeans_cluster]).sf((n_DEGs_specific_to_this - 1))
		
		ref_DEG_enrichment.loc[(kmeans_cluster, specific_to)] = [n_DEGs_specific_to_this, n_genes_specific_to_this, n_DEGs_high_gini_per_cluster[kmeans_cluster],
																 pval, ','.join(cluster_DEG_specific.index.get_level_values('gene_name').to_list()), ref_specific_to]

	ref_DEG_enrichment.dropna(inplace = True)
	ref_DEG_enrichment['adj_pval'] = multitest_corr(ref_DEG_enrichment.p_val, method = mult_hyp_corr_method)[1].astype(float)
	ref_DEG_enrichment['is_sig'] = ref_DEG_enrichment['adj_pval'].round(3) <= alpha
	return ref_DEG_enrichment

def compare_logFC(combo_logFC, col1_name, col2_name):
	'''
	Compare logFC between 2 groups (e.g., Val vs Discovery, Severe vs Mild)
	Input:
		combo_logFC - df with both logFC as cols
		col1_name - name of first group col
		col2_name - name of second group col

	Returns:
		dict with comparisons
			pct_same_sign - percent of logFC with same sign
			spearman_corr - spearman correlation coefficient and p-value
	'''
	comparisons = {}
	
	sign_logFC = np.sign(combo_logFC)
	comparisons['pct_same_sign'] = ((sign_logFC[col1_name] == sign_logFC[col2_name]).sum() / sign_logFC.shape[0]).round(2) * 100
	comparisons['spearman_corr']  = scipy.stats.spearmanr(a = combo_logFC[col1_name], b = combo_logFC[col2_name])
	comparisons['best_fit_line_coefs'] = fit_line_get_coefs(combo_logFC[col1_name], combo_logFC[col2_name])
	return comparisons

def plot_and_compare_logFC(disc_logFC, val_logFC, disc_CV, disc_ds_name, val_ds_name, labels, 
									   CV_max_cutoff = None, logFC_min_cutoff = None, figsize = (15, 5)):
	'''
	Plot scatterplot comparing 2 groups logFC and calc comparison stats using compare_logFC
	Input:
		disc_logFC - logFC for first cohort to compare
		val_logFC - logFC for second cohort to compare
		disc_CV - CV for first cohort
		disc_ds_name, val_ds_name - str, names of the two groups to use
		labels - dict with labels [subgroups] to compare - eg dif time points
		CV_max_cutoff, logFC_min_cutoff - optional, floats to filter prior to comparison
		figsize - optional, figsize for plot
	Return:
		fig - fig handle
		comparisons - dict of comparisons across all sub-groupings (eg time-points) 
	'''
	n_subplots = min(disc_logFC.shape[1], val_logFC.shape[1])
	
	ratios = np.repeat(1, n_subplots)
	fig, ax = plt.subplots(1, n_subplots, figsize = figsize, sharex = True, sharey = True, gridspec_kw={'width_ratios': ratios})
	
	hline_min_max = {'min' : disc_logFC.melt().value.min(), 'max' : disc_logFC.melt().value.max()}
	vline_min_max = {'min' : val_logFC.melt().value.min(), 'max' : val_logFC.melt().value.max()}

	comparisons = {}
	ax_i = 0
	for key, val in labels.items():
		if val not in val_logFC.columns:
			continue
			
		mask = np.repeat(True, disc_logFC.shape[0]) if CV_max_cutoff is None else disc_CV[val] < CV_max_cutoff
		mask2 = np.repeat(True, disc_logFC.shape[0]) if logFC_min_cutoff is None else disc_logFC[val].abs() > logFC_min_cutoff
		mask = np.logical_and(mask, mask2)
		
		to_plot = disc_logFC.loc[:, val].to_frame().rename(columns = {val : disc_ds_name}).join(val_logFC.loc[:, val].to_frame().rename(columns = {val : val_ds_name}), how = 'inner')
		
		to_plot_permuted = permute(to_plot, n_total_permutations = 1, axis = 0)[0]
				
		to_plot = to_plot.join(disc_CV.loc[:, val].to_frame().rename(columns = {val : 'CV'}), how = 'inner')
		
		comparisons[val] = compare_logFC(to_plot, col1_name = disc_ds_name, col2_name = val_ds_name)
		comparisons['PERMUTED ' + val] = compare_logFC(to_plot_permuted, col1_name = disc_ds_name, col2_name = val_ds_name)

		ax[ax_i].hlines(y = 0, xmin = hline_min_max['min'], xmax = hline_min_max['max'], color = 'gray', linestyle = 'dashed')
		ax[ax_i].vlines(x = 0, ymin = vline_min_max['min'], ymax = vline_min_max['max'], color = 'gray', linestyle = 'dashed')
		sns.scatterplot(x = disc_ds_name, y = val_ds_name, data = to_plot, color = '#08306b', edgecolor = 'None', ax = ax[ax_i])
		
		#Set x and y labels
		ax[ax_i].set_xlabel(disc_ds_name + ' logFC')
		ax[ax_i].set_ylabel(val_ds_name + ' logFC')
		ax[ax_i].set_title(val)
		ax_i += 1
	
	return fig, comparisons
