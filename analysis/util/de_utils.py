## DE Analysis specific utilities

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util.class_def.de_obj_classes import *
from util.gen_utils import *
from util.plot_utils import DE_plot_order

from gprofiler import GProfiler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

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

def prune_go_table(go_table_og):
	'''
	Utility fxn to prune GO table
	Sometimes GO table can have too many terms to plot, if so can prune s.t. only plot parent terms

	Inputs:
		go_table_og - df, initial GO table from GProfiler
	Returns:
		go_table_pruned - pruned df with only parent terms
	'''

	ordered_idx = go_table_og.index.to_numpy() 
	ordered_idx_only_parents = go_table_og.index.to_numpy() #Init

	#Get idx for parent
	for group, go_grouped in go_table_og.groupby('query'):
		#Cannot use sort values as is since lists are not hashable
		is_parent = [go_grouped.native.isin(parent_list) for parent_list in go_grouped.parents] #True = parent
		
		child_i = -1
		for elem in is_parent:
			child_i += 1
			if np.sum(elem) == 0: #All false
				continue
			child_idx = go_grouped.iloc[child_i].name
			parent_idx = go_grouped[elem].index.to_numpy()

			ordered_idx = ordered_idx[~np.isin(ordered_idx, parent_idx)] #Choose all elem except parents			
			child_iloc = np.where((child_idx == ordered_idx))[0][0]
			ordered_idx = np.concatenate((ordered_idx[: (child_iloc + 1)], parent_idx, ordered_idx[(child_iloc + 1):]), axis = 0)

			ordered_idx_only_parents = ordered_idx_only_parents[~np.isin(ordered_idx_only_parents, child_idx)] #Choose all elem except children
	
	go_table_pruned = go_table_og.loc[ordered_idx_only_parents]
	print('GO table has %d terms after pruning' % (go_table_pruned.shape[0]))
	return go_table_pruned

def get_and_prune_go_table(queries, query_type = 'kmeans', **kwargs):
	'''
	Wrapper that combines get_go_table and prune_go_table

	Inputs:
		Same as get_go_table

	Returns: Tuple
		original_go_table
		pruned_go_table
	'''
	
	go_table_og = get_go_table(queries, **kwargs)
	go_table_pruned = prune_go_table(go_table_og)

	#Make kmeans_clusters a category var
	go_table_og.loc[:, 'query'] = DE_plot_order(go_table_og.loc[:, 'query'], query_type)
	go_table_pruned.loc[:, 'query'] = DE_plot_order(go_table_pruned.loc[:, 'query'], query_type)

	return go_table_og, go_table_pruned	
	
def kmeans_cluster(df, n_max_clusters, seed = 37):
	'''
	Utility fxn to identify best k and then appropriately cluster data

	Inputs:
		df - df that contains data you which to cluster with shape [n_features, n_groups]
		n_max_clusters - n_groups raised to n_possibilities where n_groups is jth dim of df and n_possibililties is usually binary
		seed - seed to ensure repeatability

	Return: Dict with following keys / values
		'elbow_plot' - visualization of best k, y-axis = sum sq distance
		'clusters' - series of shape [n_features,] where each row corresponds to the cluster for that feature
	'''
	sum_sq_dist = []
	clusters = np.arange(1, (n_max_clusters + 1))

	models = {}
	for k in clusters:
		models[k] = KMeans(n_clusters = k, random_state = seed).fit(df)
		sum_sq_dist.append(models[k].inertia_)

	kneedle = KneeLocator(clusters, sum_sq_dist, S = 1.0, curve = 'convex', direction = 'decreasing')
	
	fig, ax = plt.subplots()
	plt.plot(clusters, sum_sq_dist, marker = 'o', color = 'navy', label = 'Data')
	plt.vlines(x = kneedle.elbow, ymin = min(sum_sq_dist) - 10, ymax = max(sum_sq_dist) + 10, linestyle = 'dashed', label = 'Optimal K (Elbow)')
	plt.xlabel('K')
	plt.ylabel('Sum of squared distances')
	plt.legend()

	print('Identified %d clusters' % kneedle.elbow)

	#Add 1 - Bcz starts at 0, want to start at 1
	clusters = pd.Series((models[kneedle.elbow].predict(df) + 1), index = df.index, name = 'kmeans_cluster') 
	
	#Sort cluster ID by n_feats in it in ascending order
	n_feats_per_cluster = clusters.value_counts(ascending = True)
	clusters_sorted = clusters.copy()
	i = 1
	for cluster_i in n_feats_per_cluster.keys():
		clusters_sorted.loc[clusters == cluster_i] = i
		i += 1

	print('Percent features per cluster')
	print((clusters_sorted.value_counts(ascending = True) / clusters_sorted.shape[0]).round(2))

	return {'elbow_plot' : fig, 'clusters' : clusters_sorted, 'n_clusters' : kneedle.elbow}
