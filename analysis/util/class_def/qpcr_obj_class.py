### RT-qPCR data class def

import numpy as np
import pandas as pd
from itertools import product

class rt_qpcr_data():
    """
    Class to read and process raw qPCR data
    """
    def __init__(self, plate_paths_dict, plate_map, target_map, target_info, has_RT_control, rt_control_name = 'RT', NTC_gen_threshold = None):
        """
        Init fxn for rt_qpcr_data
        Input:
            plate_paths_dict - dictionary where key = plate name, val = file path to plate data from Biorad-384 machine
            plate_map - map that indicates which sample is in which well
            target_map - map that indicates which gene primers are in which well
            target_info - additional info about target like fluorophore [e.g. FAM], location, etc
            has_RT_control - bool, indicates whether reverse transcription (RT) control was run per sample
            rt_control_name - only useful if has_RT_control = True, name of RT control
            NTC_gen_threshold - assuming NTC = not detected, what general threshold should be imposed as NTC
        Attributes:
            has_RT_control - see above
            rt_control_name - see above
            calculated_dCq - bool, indicates whether delta Cq has been calculated
            dCq - array that will contain delta Cq values
            dCq_cntrl_target - the name of the target gene used as control in dCq calc
            NTC_gen_threshold - see above
            qPCR - pandas df with qPCR info
            ntc_per_target - NTC threshold per gene target
        """
        self.has_RT_control = has_RT_control
        self.rt_control_name = rt_control_name

        self.calculated_dCq = False
        self.dCq = None
        self.dCq_cntrl_target = None
        self.NTC_gen_threshold = NTC_gen_threshold

        self.qPCR = pd.DataFrame(columns = ['to_include', 'below_ntc', 'mean_Cq', 'std_Cq', 'sample_Cqs', 'has_outliers'], 
                                dtype = float,
                                index = pd.MultiIndex.from_tuples(product(target_info.index.to_list(), plate_map.loc[:, 'sample'].unique())))
        
        self.ntc_per_target = pd.DataFrame(columns = ['ntc'], 
                                dtype = float,
                                index = target_info.index.to_list())

        self._process_qPCR_data(plate_paths_dict, plate_map, target_map, target_info)

        return 
    
    def _check_if_calculated_dCq(self):
        '''
        Error check to ensure dCq is calc prior to any manipulation that requires it
        '''
        if not self.calculated_dCq:
            raise(NameError('dCq has not been calculated yet'))

    def _read_qpcr_data(self, plate_xlsx, sheet_name = "0"):
        '''
        Private method to read a single qPCR plate data and pull out relevant info [fluor values for gene target]
        Inputs:
            plate_xlsx - path to plate excel doc
            sheet_name - name of sheet, default is 0 based on Biorad 384 machine naming conventions

        Returns:
            pd df with Cq values from relevant fluor for plate
        '''
        data = pd.read_excel(plate_xlsx, sheet_name = sheet_name)
        rows_cols_fluor = pd.MultiIndex.from_tuples([(data.Well[i][0], int(data.Well[i][1:]), data.Fluor[i]) for i in range(data.shape[0])], 
            names = ["row", "col", "fluor"])
        return pd.DataFrame(data = {'Cq' : data.Cq.round(2).to_list()}, index = rows_cols_fluor) 

    def _process_qPCR_data(self, plate_paths_dict, plate_map, target_map, target_info):
        '''
        Private method to process qPCR plate data, wrapper around _read_qpcr_data
        Inputs:
            See init for descriptions

        Modified attributes:
            qPCR - mean_Cq, std_Cq, and tech rep Cq vals modified, outlier measurements [inconsistent Cqs axs tech reps] identified and flagged
            ntc_per_target - modified to reflect NTC for each gene target, if none detected, default to gen_threshold set
        '''
        grouping = ['gene_target', 'sample']
        plate_data = {} #Init dict to store raw values

        for label, plate_path in plate_paths_dict.items():
            plate_data[label] = self._read_qpcr_data(plate_path)
            targets_in_plate = target_info.loc[target_info.plate.str.lower() == label.lower()]

            plate_target_measurements = None
            for gene_target, row in targets_in_plate.iterrows():
                wells_fluor_w_target = target_map.loc[target_map.which_target == row.loc_idx]
                wells_fluor_w_target.insert(0, 'fluor', row.fluorophore)
                wells_fluor_w_target = wells_fluor_w_target.reset_index().set_index(['row', 'col', 'fluor']).index

                target_measurements = plate_data[label].loc[wells_fluor_w_target].reset_index(level = 'fluor', drop = True)
                target_measurements = target_measurements.join(plate_map, on = ['row', 'col'], how = 'left')
                target_measurements.insert(0, 'gene_target', gene_target)

                plate_target_measurements = target_measurements if plate_target_measurements is None else pd.concat((plate_target_measurements, target_measurements), axis = 0)

            target_mean = plate_target_measurements.groupby(grouping).mean().squeeze()
            target_std = plate_target_measurements.groupby(grouping).std().squeeze()
            target_Cqs = plate_target_measurements.groupby(grouping)['Cq'].apply(list)
            target_ntc_Cq_min = target_mean.loc[target_mean.index.get_level_values('sample').astype(str).str.contains('NTC')].groupby('gene_target').min()
            self.ntc_per_target.loc[target_ntc_Cq_min.index, 'ntc'] = target_ntc_Cq_min.squeeze()

            self.qPCR.loc[target_mean.index, 'mean_Cq'] = target_mean
            self.qPCR.loc[target_std.index, 'std_Cq'] = target_std
            self.qPCR.loc[target_Cqs.index, 'sample_Cqs'] = target_Cqs
        
        self.qPCR.loc[:, 'to_include'] = np.round(2*self.qPCR.std_Cq, 2) <= 1.5 #95% of tech rep distribution (2*sigma) should be within X Cq of mean
        self.qPCR.loc[:, 'has_outliers'] = np.logical_and(~self.qPCR.to_include, ~self.qPCR.mean_Cq.isna()) #If 2*sigma varies by more than X Cq, there is/are outlier technical replicate(s)
        self.qPCR.index.names = grouping

        if self.NTC_gen_threshold is not None:
            self.ntc_per_target.loc[self.ntc_per_target.ntc.isna(), 'ntc'] = self.NTC_gen_threshold

        merged_df = self.qPCR.merge(self.ntc_per_target, how = 'left', left_on = 'gene_target', right_index = True)
        self.qPCR.loc[:, 'below_ntc'] = np.logical_or(merged_df.mean_Cq.round(1) < merged_df.ntc.round(0), merged_df.ntc.isna()).to_numpy()
        self.qPCR = self.qPCR.loc[~self.qPCR.index.get_level_values('sample').astype(str).str.contains("NTC"), :]
        return

    def _get_isec_Cq_df_w_gene(self, gene_for_isec, col_name = 'mean_Cq_cntrl'):
        '''
        Private method used when calculating dCq to get intersection of values
        Inputs:
            gene_for_isec - gene to intersect with mean_Cq_cntrl

        Returns:
            intersected qPCR table with gene_target and control values
        '''
        Cq_vals_cntrl = self.qPCR.loc[gene_for_isec].mean_Cq.squeeze()
        Cq_vals_cntrl.name = col_name
        return self.qPCR.merge(Cq_vals_cntrl, left_on = 'sample', right_index = True, how = 'inner')

    def filter_qPCR(self):
        '''
        Method to filter qPCR plate data to only those that pass QC [below_NTC, no outlier tech reps, pass RT_control]
        

        Modified attributes:
            qPCR - filtered to only samples that passed
        '''
        init_pass_mask = np.logical_and(self.qPCR.to_include, self.qPCR.below_ntc)
        n_before = self.qPCR.shape[0]
        self.qPCR = self.qPCR.loc[init_pass_mask]
        if self.has_RT_control:
            pass_rt_cntrl = self._get_isec_Cq_df_w_gene(self.rt_control_name)
            self.qPCR = self.qPCR.reindex(pass_rt_cntrl.index).dropna()
        n_after = self.qPCR.shape[0]
        print('%d of %d (%.2f) sample gene measurements were measured consistently at a Cq below the NTC threshold' % (n_after, n_before, (n_after/n_before)))
        return

    def get_dCq(self, cntrl_gene_target):
        '''
        Method to calculate delta Cq between a gene of interest and control_gene_target
        Inputs:
            cntrl_gene_target - the gene that is used as a control s.t. dCq = Cq - Cq_cntrl

        Modified attributes:
            dCq, dCq_cntrl_target, calculated_dCq 
        '''
        self.dCq_cntrl_target = cntrl_gene_target
        merged_df = self._get_isec_Cq_df_w_gene(cntrl_gene_target)
        self.dCq = (merged_df.mean_Cq - merged_df.mean_Cq_cntrl)
        self.calculated_dCq = True
        return
        
    def get_ddCq(self, metadata, gene, group_col = 'case', num_group = 1, denom_group = 0):
        '''
        Method to calc delta delta Cq
        Inputs:
            metadata - metadata df
            gene - gene of interest
            group_col - column_name in metadata for which ddCq will be calculated
            num_group - numerator or left hand side of ddCq
            denom_group - denominator or right hand side of ddCq

        Returns:
            ddCq for given gene
        '''
        self._check_if_calculated_dCq()

        dCq_avg = {}
        for group, df in metadata.groupby(group_col):
            idx = pd.MultiIndex.from_tuples(list(product([gene], df.index.to_list())))
            dCq_avg[group] = self.dCq.reindex(idx).dropna().median()

        return (dCq_avg[num_group] - dCq_avg[denom_group]).round(2)

    def get_copy_num_using_std_curve(self, slope, intercept):
        '''
        Method to get copy num using standard curve
        Inputs:
            Slope - best fit slope according to standard curve that can be used to back out copy num
            Intercept - best fit intercept according to standard curve

        Return:
            copy_num - Based on std curve
        '''
        return 2**((self.qPCR.mean_Cq - intercept) / slope)

    def get_fold_change(self, metadata, which_method, gene, group_col = 'case', num_group = 1, denom_group = 0, **kwargs):
        '''
        Method to calculate fold change using qPCR data
        Inputs:
            Metadata - metadata df
            which_method - str, either ddCq or copy_num
            gene - gene for which to calc fold change
            group_col, num_group_denom_group - see get_ddCq
            **kwargs - additional arguments to pass for when using copy_num, slope and intercept

        Returns:
            Calculated fold change
        '''
        assert(which_method in ['ddCq', 'copy_num']), 'Method must be ddCq or copy_num'

        if which_method == 'ddCq':
            ddCq = self.get_ddCq(metadata, gene, group_col, num_group, denom_group)
            fc = 2**(-ddCq)

        if which_method == 'copy_num':
            copy_num = self.get_copy_num_using_std_curve(**kwargs)
            idx = {group_i : pd.MultiIndex.from_tuples(list(product([gene], metadata.loc[(metadata.loc[:, group_col] == group_i)].index.to_list()))) for group_i in [num_group, denom_group]}

            fc = (copy_num.reindex(idx[num_group]).dropna().median() / copy_num.reindex(idx[denom_group]).dropna().median())

        return fc.round(2)
   