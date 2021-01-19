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
counts_file = "../../data/rnaseq_discovery_data/htseq_postQC.csv" 
meta_file = "../../data/rnaseq_discovery_data/sample_meta_postQC.csv"

#Load data
counts = read.csv(counts_file, header=T, strip.white=T)
meta = read.csv(meta_file, header=T, strip.white=T)

#Make counts into DGE list
counts_matrix = as.matrix(counts[,c(-1, -2)])
rownames(counts_matrix) = paste(counts[,1], counts[,2], sep = "_") 
all = DGEList(counts_matrix)

#Calc TMM norm
all_filt = calcNormFactors(all, method="TMM") #Want to do TMM with all samples for post-hoc analysis
write.csv(all_filt$samples, file="../../data/rnaseq_discovery_data/TMM_postQC.csv")

#Filter meta and counts to only contain samples during gestation [No PP]
meta = meta %>% filter(is_pp == 0)
preg_mask = as.character(rownames(all_filt$samples)) %in% as.character(meta$sample)
all_filt$samples = all_filt$samples[preg_mask, ]
all_filt$counts = all_filt$counts[ ,preg_mask]

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

str(meta)

#Make time cubic spline [Samples collected across a range]
time_spline = ns(meta$ga_at_collection, df=4)

#Design matrix
design = model.matrix(~case*time_spline, data=meta)
str(design)
colnames(design) = make.names(colnames(design))

dim(design)
head(design)
colSums(design)

#Include sample quality to reduce weight of outlier samples
#Based on manuscript - using this fxn alone performs comparably to:

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

fit = lmFit(all_voom_2,
            design,
            block = meta$subject,
            correlation = corfit$consensus
            )

#With regression - look at main effect + interaction params from spline
fit_case = eBayes(fit, trend = TRUE, robust = TRUE)

#ID changes
cutoff = Inf #Will filter in 2.1 Script for alpha = 0.05 but would like to export all

#All changes with time
dTime = topTable(fit_case, coef=c(3:6), sort.by="B", resort.by = "logFC", p.value = cutoff, number = Inf, confint = TRUE)

#dPE changes with time
dTime_PE = topTable(fit_case, coef=c(2, 7:10), sort.by="B", resort.by = "logFC", p.value = cutoff, number = Inf, confint = TRUE)

write.csv(dTime, "out/de/DE_all_changes_over_gestation.csv")
write.csv(dTime_PE, "out/de/DE_PEspecific_changes_over_gestation.csv")