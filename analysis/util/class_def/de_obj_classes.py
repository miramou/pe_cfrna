### Differential expression specific class def

import pandas as pd
import numpy as np

class de_data():
	"""
	Class to easily load DE table from R
	"""
	def __init__(self, de_df_path, alpha = 0.05, de_type = 'case', to_round = False):
		"""
		Init fxn for de_data
		Input:
			de_df_path - path to DE table csv
			alpha - cutoff value
			de_type - what was contrasted
			to_round - whether to round adj_pval prior to getting mask
		Attributes:
			de_path - see above
			de_type - see above
			alpha - see above
			de - differential expression table [n_genes, n_features]
			is_sig_mask - array [n_genes] indicating whether each gene passed sig threshold - adj_pval <= alpha
		"""
		self.de_path = de_df_path
		
		self.de_type = de_type		
		self.alpha = alpha

		de_init = self._read_de_file()
		self.de = self._parse_de_file(de_init)
		self.is_sig_mask = self.get_sig_mask(to_round)
		self.sig_genes = self.de.loc[self.is_sig_mask].index
		
	def _read_de_file(self):
		'''
		Private class method to read DE table
		'''
		#Again DE file generated with R so rename columns
		return pd.read_csv(self.de_path, index_col = 0).rename(columns = {"P.Value" : "p_val", "adj.P.Val" : "adj_pval"})

	def _parse_de_file(self, de_init):
		'''
		Private class method to parse DE file and read gene ids where gene name and gene num are combined by an _
		'''
		#Change index to match counts df
		gene_1idx = de_init.index.get_level_values(0).str.split("_ENSG", expand=True)

		de_init = de_init.assign(gene_name = gene_1idx.get_level_values(0), gene_num = "ENSG" + gene_1idx.get_level_values(1))
		return de_init.reset_index(drop = True).set_index(["gene_name", "gene_num"]).sort_values("adj_pval")

	def get_sig_mask(self, to_round = False):
		'''
		Identifies which genes are DEGs
		to_round - whether to round adj_pval prior to getting mask
		'''
		if to_round:
			rounded_padj = self.de.adj_pval.round(2)
			self.de.adj_pval = rounded_padj

		return (self.de.adj_pval <= self.alpha)

	def intersect_with_gene_set(self, gene_idx):
		'''
		Method to intersects DEGs with a set of genes, gene_idx, passed in

		Input:
			gene_idx - pd index of gene names

		Modified attributes:
			de - changed to only include gene_idx [n_genes_in_intersection, n_features]
			is_sig_mask - changed to match updated de [n_genes_in_intersection]
		'''
		self.de = self.de.reindex(gene_idx).dropna()
		self.is_sig_mask = self.get_sig_mask()
