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
meta$is_pp = factor(meta$is_pp)
str(meta)
  
#Make time cubic spline [Samples collected across a range]
time_spline = ns(meta$time_to_pe_onset, df=4)
  
#Include sample quality to reduce weight of outlier samples
#Based on manuscript - using this fxn alone performs comparably to:
fits = c()
fits_no_bayes = c()
designs = c()
batch_removal = c()

for (i in c(1,2,3,4)) {
  
  #Design matrix w PE with or without severe features
  design = model.matrix(~pe_feature*time_spline+pe_feature*is_pp+race+ethnicity+fetal_sex+bmi_grp+mom_age+batch, data = meta)
  #Run again to remove batch as a var
  if (i == 2) {design = model.matrix(~pe_feature*time_spline+pe_feature*is_pp+race+ethnicity+fetal_sex+bmi_grp+mom_age, data = meta)}
  
  #Design matrix with PE as syndrome
  if (i == 3) {design = model.matrix(~case*time_spline+case*is_pp+race+ethnicity+fetal_sex+bmi_grp+batch, data = meta)}
  #Run again to remove batch as a var
  if (i == 4) {design = model.matrix(~case*time_spline+case*is_pp+race+ethnicity+fetal_sex+bmi_grp, data = meta)}
  
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
  
  if (i %in% c(1,3)) {
    fit = lmFit(all_voom_2,
                design,
                block = meta$subject,
                correlation = corfit$consensus
                )
    
    fits_no_bayes[[i]] = fit
    #With regression - look at main effect + interaction params from spline
    fits[[i]] = eBayes(fit, trend = TRUE, robust = TRUE)
  }
  if (i %in% c(2,4)) {
    no_batch = removeBatchEffect(all_voom_2, batch = meta$batch, design = design, 
                                 block = meta$subject, correlation = corfit$consensus)
    batch_removal[[i]] = no_batch
  }
}
  
#ID changes
cutoff = Inf #Will filter in 2.1 Script for alpha but would like to export all

#Test for general changes related to PE (mild or severe)
dTime_PE_mild_GA_PP_w_covar = topTable(fits[[1]], coef=c(2,20,22,24,26), sort.by="B", resort.by = "logFC", p.value = cutoff, number = Inf, confint = TRUE)
dTime_PE_severe_GA_PP_w_covar = topTable(fits[[1]], coef=c(3,21,23,25,27), sort.by="B", resort.by = "logFC", p.value = cutoff, number = Inf, confint = TRUE)

#Test for gen dif between severe and mild
sev_v_mild_cont = makeContrasts(
  pe_featuresevere - pe_featuremild,
  pe_featuresevere.time_spline1 - pe_featuremild.time_spline1,
  pe_featuresevere.time_spline2 - pe_featuremild.time_spline2,
  pe_featuresevere.time_spline3 - pe_featuremild.time_spline3,
  pe_featuresevere.time_spline4 - pe_featuremild.time_spline4,
  levels = designs[[1]]
)

fit_mild_v_sev = contrasts.fit(fits_no_bayes[[1]], sev_v_mild_cont)
fit_mild_v_sev = eBayes(fit_mild_v_sev)
dTime_sev_v_mild = topTable(fit_mild_v_sev, sort.by="B", resort.by = "logFC", p.value = cutoff, number = Inf, confint = TRUE)

#PE changes over gestation
dTime_PE_onlyGA_w_covar = topTable(fits[[3]], coef=c(2,18:21), sort.by="B", resort.by = "logFC", p.value = cutoff, number = Inf, confint = TRUE)

out_prefix = 'out/de/w_mild_severe/'
out_prefix1 = 'out/de/'

write.csv(batch_removal[[2]], paste0(path_prefix, "w_mild_severe/", "logCPM_postQC_RemovedBatch.csv"))
write.csv(batch_removal[[4]], paste0(path_prefix, "logCPM_postQC_RemovedBatch.csv"))

write.csv(dTime_PE_mild_GA_PP_w_covar, paste0(out_prefix, "DE_PEspecific_GA_PP_mild_changes_timeToPE_w_covar_bmi_fsex_w_batch.csv"))
write.csv(dTime_PE_severe_GA_PP_w_covar, paste0(out_prefix, "DE_PEspecific_GA_PP_severe_changes_timeToPE_w_covar_bmi_fsex_w_batch.csv"))
write.csv(dTime_sev_v_mild, paste0(out_prefix, 'DE_PEspecific_GA_PP_sev_v_mild_changes_timeToPE_w_covar_bmi_fsex_w_batch.csv'))

write.csv(dTime_PE_onlyGA_w_covar, paste0(out_prefix1, "DE_PEspecific_onlyGA_changes_timeToPE_w_covar_bmi_fsex_w_batch.csv"))