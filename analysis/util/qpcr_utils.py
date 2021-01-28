## RT-qPCR Utils

import pandas as pd
from util.class_def.qpcr_obj_class import *


def read_excel_sheet(excel_path, sheet_name, index_col = None):
	'''
	Utility fxn to read an excel sheet
	Inputs:
		excel_path - filepath to xls/x doc
		sheet_name - the name of the sheet in the xlsx doc to be read
		index_col - column to index on
	'''
	return pd.read_excel(excel_path, sheet_name = sheet_name, index_col = index_col)

def read_plate_map(excel_path, sheet_name, to_melt = False, val_name = None, index_col = None):
	'''
	Utility fxn to read a plate map
	Inputs:
		excel_path - filepath to xls/x doc
		sheet_name - the name of the sheet in the xlsx doc to be read
		to_melt - bool, if True melt the plate into long form [row, col, data]
		val_name - only relevant if to_melt = True, name of data values
		index_col - column to index on
	'''
	out = read_excel_sheet(excel_path, sheet_name, index_col)
	if to_melt:
		out = out.melt(id_vars = 'row', var_name = 'col', value_name = val_name).set_index(['row', 'col'])
	return out


