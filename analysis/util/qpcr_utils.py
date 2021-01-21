## RT-qPCR Utils

import pandas as pd

def read_excel_sheet(excel_path, sheet_name, index_col = None):
	return pd.read_excel(excel_path, sheet_name = sheet_name, index_col = index_col)

def read_plate_map(excel_path, sheet_name, to_melt = False, val_name = None, index_col = None):
	out = read_excel_sheet(excel_path, sheet_name, index_col)
	if to_melt:
		out = out.melt(id_vars = 'row', var_name = 'col', value_name = val_name).set_index(['row', 'col'])
	return out


