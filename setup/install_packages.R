#!/usr/bin/env Rscript

# vector of CRAN packages to be installed
cran_packages <- c("ggplot2",
"GGally",
"ggdendro",
"RColorBrewer",
"dendextend",
"gplots",
"doMC",
"doParallel",
"dbscan",
"cluster",
"methods",
"devtools",
"caret",
"mlbench",
"plot3D",
"pROC",
"ROCR",
"Rtsne",
"scatterplot3d",
"UsingR",
"rafalib",
"downloader",
"lattice",
"e1071",
"corrplot",
"class",
"randomForest",
"rpart",
"rpart.plot",
"party",
"stepPlr",
"caTools",
"arm",
"kernlab",
"nnet",
"neuralnet",
"MASS",
"NeuralNetTools",
"ISLR",
"caTools",
"boot",
"faraway",
"CVST",
"readr"
)



# vector of bioconductor packages to be installed
bioc_packages <- c("EBImage")

# install CRAN packages
install.packages(cran_packages, repos='http://mirrors.ebi.ac.uk/CRAN/')

# install bioconductor packages
source("https://bioconductor.org/biocLite.R")
biocLite(bioc_packages)

# install packages from other repositories
devtools::install_github("SheffieldML/vargplvm/vargplvmR")
devtools::install_github("ManchesterBioinference/DEtime")




