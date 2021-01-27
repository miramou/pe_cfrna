library(edgeR)

#Run within RStudio
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

get_save_tmm = function(counts_path, to_save_path, idx_cols_to_remove = c(-1, -2)) {
  counts = read.csv(counts_path, header=T, strip.white = T, check.names = F)
  counts_matrix = as.matrix(counts[,idx_cols_to_remove])
  rownames(counts_matrix) = counts[,1]
  
  all = DGEList(counts_matrix)
  
  #Calc norm factors
  all_filt = calcNormFactors(all, method="TMM")
  write.csv(all_filt$samples, file=to_save_path)
}

#Discovery dataset TMM calculated in 2.0_DE.R

#Validation datasets TMM 
get_save_tmm('../../../data/rnaseq_validation_data/htseq_postQC.csv', '../../../data/rnaseq_validation_data/TMM_postQC.csv')
get_save_tmm('../../../data/delvecchio_data/htseq_merged.csv', '../../../data/delvecchio_data/TMM.csv')

get_save_tmm('../../../data/munchel_data/S1_counts_only.csv', '../../../data/munchel_data/S1_TMM.csv', c(-1))
get_save_tmm('../../../data/munchel_data/S2_counts_only.csv', '../../../data/munchel_data/S2_TMM.csv', c(-1))
get_save_tmm('../../../data/munchel_data/S3_counts_only.csv', '../../../data/munchel_data/S3_TMM.csv', c(-1))