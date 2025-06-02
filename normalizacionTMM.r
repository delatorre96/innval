if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("edgeR")

library(edgeR)

counts <- read.csv("C:/Users/delatorrei/Desktop/Proyectos/INNVAL/innval/df_raw_Proteines_Normalizar.csv")
dge <- DGEList(counts = counts)
dge <- calcNormFactors(dge, method = "TMM")
dge$samples$norm.factors

eff.lib.size <- dge$samples$lib.size * dge$samples$norm.factors

cpm_tmm <- cpm(dge, normalized.lib.sizes = TRUE)

write.csv(cpm_tmm, file = "C:/Users/delatorrei/Desktop/Proyectos/INNVAL/innval/cpm_tmmProteines.csv")