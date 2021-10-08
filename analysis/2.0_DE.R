library(reshape2)
library(plyr)
library(caret)
library(splines)
library(ggplot2)
library(dplyr)
library(limma)
library(edgeR)

#Run within RStudio
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Data paths
path_prefix = "../../data/rnaseq_stanford_all/discovery/"
counts_file = paste0(path_prefix, "htseq_postQC.csv")
meta_file = paste0(path_prefix, "sample_meta_postQC.csv")

#Load data
counts = read.csv(counts_file, header=T, strip.white=T)
meta = read.csv(meta_file, header=T, strip.white=T)

#Make counts into DGE list
counts_matrix = as.matrix(counts[,c(-1, -2)])
rownames(counts_matrix) = paste(counts[,1], counts[,2], sep = "_") 
all = DGEList(counts_matrix)

#Calc TMM norm
all_filt = calcNormFactors(all, method="TMM")
write.csv(all_filt$samples, file=paste0(path_prefix, "TMM_postQC.csv"))

#Make sure samples are in same order
all_filt$counts = all_filt$counts[, match(as.character(meta$sample), as.character(colnames(all_filt$counts)))]
all_filt$samples = all_filt$samples[match(as.character(meta$sample), as.character(rownames(all_filt$samples))),]

n_samples = dim(meta)[1]
if (sum(as.character(meta$sample) == as.character(rownames(all_filt$samples))) != n_samples) {
  print("Metadata and all_filt$counts dimensions do not match")
}
if (sum(as.character(meta$sample) == as.character(colnames(all_filt$counts))) != n_samples) {
  print("Metadata and all_filt$samples do not match")
}
  
#Factorize
meta$subject = factor(meta$subject)
meta$case = factor(meta$case)
meta$is_obese = factor(meta$is_obese)
meta$batch = factor(meta$batch)
meta$race = factor(meta$race)
meta$ethnicity = factor(meta$ethnicity)
meta$is_pp = factor(meta$is_pp)
meta$pe_feature = factor(meta$pe_feature)
str(meta)
  
#Make time cubic spline [Samples collected across a range]
time_spline = ns(meta$time_to_pe_onset, df=4)
  
#Include sample quality to reduce weight of outlier samples
#Based on manuscript - using this fxn alone performs comparably to:
fits = c()
designs = c()
  
for (i in c(1,2)) {
  
  #Design matrix
  design = model.matrix(~case*time_spline+case*is_pp+race+ethnicity+fetal_sex+bmi_grp+batch, data = meta)
  #Run again to remove batch as a var
  if (i == 2) {design = model.matrix(~case*time_spline+case*is_pp+race+ethnicity+fetal_sex+bmi_grp, data = meta)}
  
  str(design)
  colnames(design) = make.names(colnames(design))
  
  designs[[i]] = design
  
  dim(design)
  dim(all_filt)
  
  all_voom_1 = voomWithQualityWeights(
    all_filt,
    design,
    normalize.method = "none",
    method = "genebygene",
    maxiter = 100,
    tol = 1e-6,
    trace = TRUE,
    plot = TRUE
  )
  
  corfit = duplicateCorrelation(all_voom_1, design, block = meta$subject)
  corfit$consensus #Just to check
  
  all_voom_2 = voomWithQualityWeights(
    all_filt,
    design,
    normalize.method = "none",
    method = "genebygene",
    maxiter = 100,
    tol = 1e-6,
    block = meta$subject,
    correlation = corfit$consensus,
    plot = TRUE
  )
  
  if (i == 1) {
    fit = lmFit(all_voom_2,
                design,
                block = meta$subject,
                correlation = corfit$consensus
                )
    
    #With regression - look at main effect + interaction params from spline
    fits[[i]] = eBayes(fit, trend = TRUE, robust = TRUE)
  }
  if (i == 2) {
    no_batch = removeBatchEffect(all_voom_2, batch = meta$batch, design = design, 
                                 block = meta$subject, correlation = corfit$consensus)
  }
}
  
#ID changes
cutoff = Inf #Will filter in 2.1 Script for alpha but would like to export all
cutoff_to_check = 0.05

#PE changes over gestation
dTime_PE_onlyGA_w_covar = topTable(fits[[1]], coef=c(2,18:21), sort.by="B", resort.by = "logFC", p.value = cutoff, number = Inf, confint = TRUE)
n_changes_overGA = dim(topTable(fits[[1]], coef=c(2,18:21), sort.by="B", resort.by = "logFC", p.value = cutoff_to_check, number = Inf, confint = TRUE))[1]

#PE changes PP
dTime_PE_PP_w_covar = topTable(fits[[1]], coef=c(2,22), sort.by="B", resort.by = "logFC", p.value = cutoff, number = Inf, confint = TRUE)
n_changes_PP = dim(topTable(fits[[1]], coef=c(2,22), sort.by="B", resort.by = "logFC", p.value = cutoff_to_check, number = Inf, confint = TRUE))[1]

out_prefix = 'out/de/'
write.csv(no_batch, paste0(path_prefix, "logCPM_postQC_RemovedBatch.csv"))

if (n_changes_overGA > 0) {
  write.csv(dTime_PE_onlyGA_w_covar, paste0(out_prefix, "DE_PEspecific_onlyGA_changes_timeToPE_w_covar_bmi_fsex_w_batch.csv"))
}

if (n_changes_PP > 0) {
  write.csv(dTime_PE_PP_w_covar, paste0(out_prefix, "DE_PEspecific_PP_changes_timeToPE_w_covar_bmi_fsex_w_batch.csv"))
}