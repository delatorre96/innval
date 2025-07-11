if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("edgeR")

library(edgeR)

normalize <- function(csv_input_path,csv_output_path) { 
  counts <- read.csv(csv_input_path)
  dge <- DGEList(counts = counts)
    dge <- calcNormFactors(dge, method = "TMM")
    dge$samples$norm.factors

    eff.lib.size <- dge$samples$lib.size * dge$samples$norm.factors

    cpm_tmm <- cpm(dge, normalized.lib.sizes = TRUE)
    write.csv(cpm_tmm, file = csv_output_path)
}

 #"C:/Users/delatorrei/Desktop/Proyectos/INNVAL/innval/cpm_tmmProteines.csv"
normalize('C:/Users/delatorrei/Desktop/Proyectos/INNVAL/innval/lncRNA_validacion.csv','C:/Users/delatorrei/Desktop/Proyectos/INNVAL/innval/lncRNA_validacion_norm.csv')

