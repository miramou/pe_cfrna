library(edgeR)

#Run within RStudio
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

get_save_tmm = function(counts_path, to_save_path) {
  counts = read.csv(counts_path, header=T, strip.white = T, check.names = F)
  counts_matrix = as.matrix(counts[,c(-1, -2)])
  rownames(counts_matrix) = counts[,1]
  
  all = DGEList(counts_matrix)
  
  #Calc norm factors
  all_filt = calcNormFactors(all, method="TMM")
  write.csv(all_filt$samples, file=to_save_path)
}

#Discovery dataset TMM calculated in 2.0_DE.R

#Validation datasets TMM 
get_save_tmm('../../../data/rnaseq_validation_data/htseq_postQC.csv', '../../../data/rnaseq_validation_data/TMM_postQC.csv')