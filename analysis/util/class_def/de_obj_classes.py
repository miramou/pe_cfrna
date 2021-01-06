import pandas as pd
import numpy as np

class de_data():
	def __init__(self, de_df_path, alpha = 0.05, de_type = 'case', de_col = 'case'):
		self.de_path = de_df_path
		
		self.de_type = de_type
		self.de_col = de_col
		
		self.alpha = alpha

		de_init = self._read_de_file()
		self.de = self._parse_de_file(de_init)
		self.is_sig_mask = self.get_sig_mask()

	def _read_de_file(self):
		#Again DE file generated with R so rename columns
		return pd.read_csv(self.de_path, index_col = 0).rename(columns = {"P.Value" : "p_val", "adj.P.Val" : "adj_pval"})

	def _parse_de_file(self, de_init):
		#Change index to match counts df
		gene_1idx = de_init.index.get_level_values(0).str.split("_ENSG", expand=True)

		de_init = de_init.assign(gene_name = gene_1idx.get_level_values(0), gene_num = "ENSG" + gene_1idx.get_level_values(1))
		return de_init.reset_index(drop = True).set_index(["gene_name", "gene_num"]).sort_values("adj_pval")

	def get_sig_mask(self):
		return (self.de.adj_pval <= self.alpha)

	def intersect_with_gene_set(self, gene_idx):
		self.de = self.de.reindex(gene_idx).dropna()
		self.is_sig_mask = self.get_sig_mask()
