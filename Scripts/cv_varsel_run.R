# ATTENTION ! 
# Procédure cv varsel appliquée sur le modèle complet durait ~ 3h sur ma machine

library(rstanarm)
library(projpred)
library(doParallel)

load("post_ref.RData") #charger le modèle de référence

ncores <- 8L #nombre de cores
doParallel::registerDoParallel(ncores)
options(projpred.parallel_proj_trigger = 0)

cat("Backend:", foreach::getDoParName(), "\n")
cat("Workers:", foreach::getDoParWorkers(), "\n")

cvvs <- cv_varsel(
  post_ref,
  method = "forward",
  cv_method = "LOO",
  validate_search = TRUE,
  nterms_max = 15,
  nclusters = 20, #20 ou 50/100
  parallel = TRUE,
  nclusters_pred = 400 #400 ou 1000
)

doParallel::stopImplicitCluster()
foreach::registerDoSEQ()
saveRDS(cvvs, "cv_varsel_p5_4_2_1.rds") #save le résultat
cat("Terminé :", format(Sys.time()), "\n")
