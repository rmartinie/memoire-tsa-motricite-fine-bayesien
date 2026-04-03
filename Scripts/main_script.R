# ============================================================
# LIBRAIRIES
# ============================================================

library(tidyverse)
library(caret)
library(rstanarm)
library(bayesplot); theme_set(bayesplot::theme_default(base_family = "sans"))
library(loo)
library(projpred)
library(dplyr)
library(readxl)
library(patchwork)

# ============================================================
# 1) PARAMÈTRES 
# ============================================================

SEED <- 123
set.seed(SEED)
#setwd("") #éventuellemet paramétrer le chemin

# ============================================================
# 2) CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================

data <- read_excel("~/data/data.xlsx")

names(data)[names(data) == "Group"] <- "TSA"
data$TSA  <- factor(ifelse(data$TSA  == "ASD", 1, 0), levels = c(0, 1))
data$Sexe <- factor(ifelse(data$Sexe == 2,     1, 0), levels = c(0, 1))

# Centrage-réduction de toutes les variables continues
data_scaled <- data %>%
  mutate(across(
    .cols = -c(TSA, Sexe),
    .fns  = ~ scale(.)[, 1]
  ))

# Nettoyage des noms (tirets -> underscores)
names(data_scaled) <- gsub("-", "_", names(data_scaled))

# Formule complète
pred_vars   <- setdiff(names(data_scaled), "TSA")
pred_vars   <- paste0("`", pred_vars, "`")
reg_formula <- as.formula(paste("TSA ~", paste(pred_vars, collapse = " + ")))

# TSA en numérique pour les fonctions ppc
data_scaled$TSA <- as.numeric(as.character(data_scaled$TSA))

# Suppression des NA et de l'outlier identifié (Z score > 5)
data_scaled <- na.omit(data_scaled)
data_scaled <- data_scaled[-67, ]

y <- data_scaled$TSA

# ============================================================
# 3) PRIOR HORSESHOE 
# ============================================================

D           <- length(pred_vars)
p0          <- 10
sigma_logit <- 2
tau0        <- (p0 / (D - p0)) * sigma_logit / sqrt(nrow(data_scaled))

hs_prior <- hs(global_scale = tau0, slab_df = 4, slab_scale = 2, df = 1, global_df = 1)
t_prior  <- student_t(df = 7, location = 0, scale = 2.5)

# ============================================================
# 4) PRIOR PREDICTIVE CHECK
# ============================================================

model_prior <- stan_glm(
  formula         = reg_formula,
  data            = data_scaled,
  family          = binomial("logit"),
  prior           = hs_prior,
  prior_intercept = t_prior,
  seed            = SEED,
  chains          = 2,
  iter            = 1000,
  prior_PD        = TRUE,
  refresh         = 0
)

yrep_prior <- posterior_predict(model_prior, draws = 50)
ppc_dens_overlay(y = y, yrep = yrep_prior)
ppc_bars(y, yrep_prior)

# ============================================================
# 5) MODÈLE DE RÉFÉRENCE
# ============================================================

post_ref <- stan_glm(
  formula         = reg_formula,
  data            = data_scaled,
  family          = binomial("logit"),
  prior           = hs_prior,
  prior_intercept = t_prior,
  seed            = SEED,
  chains          = 4,
  iter            = 4000,
  adapt_delta     = 0.95,
  refresh         = 0
)

# Diagnostics
print(summary(post_ref), digits = 3)

yrep_ref <- posterior_predict(post_ref, draws = 100)
ppc_dens_overlay(y = y, yrep = yrep_ref)

ppc_bars(y, yrep_ref) 

ppc_stat(y, yrep_ref, stat = "mean")
ppc_stat(y, yrep_ref, stat = "sd")

vars_to_plot <- c("(Intercept)", names(coef(post_ref))[6:min(12, length(coef(post_ref)))])


p1 <- plot(post_ref, plotfun = "trace", pars = "L2_large") 
p2 <- plot(post_ref, plotfun = "acf", pars = "L2_large")
p3 <- plot(post_ref, plotfun = "hist", pars = "L2_large", bins = 50)
p4 <- plot(post_ref, plotfun = "dens", pars = "L2_large")
p_f <- (p1 | p2) / (p3 | p4)

p_f <- p_f & theme(
  strip.text = element_blank(),       
  strip.background = element_blank()  
)
p_f <- p_f + plot_annotation(subtitle = "Diagnostics MCMC pour le paramètre L2_large")
p_f

mcmc_hist(post_ref,         pars = vars_to_plot)
mcmc_dens(post_ref,         pars = vars_to_plot)
mcmc_dens_overlay(post_ref, pars = vars_to_plot)
mcmc_areas(post_ref, prob = 0.95, prob_outer = 1)

# ============================================================
# 6) PSIS-LOO ET DIAGNOSTIC
# ============================================================

loo_ref <- loo(post_ref, save_psis = TRUE)
print(loo_ref)

kvals <- pareto_k_values(loo_ref)
cat("Proportion pareto_k > 0.7 :", mean(kvals > 0.7), "\n")

cv_method_used <- "LOO"

#save(post_ref, file = "post_ref.RData")

# ============================================================
# 7) SÉLECTION PAR PROJECTION PREDICTIVE
# ============================================================

# cv_varsel généré dans un script séparé, chargé ici
#cvvs <- readRDS("cv_varsel_t1.rds")
    
plot(cvvs, stats = c("elpd", "pctcorr"), deltas = TRUE)
plot(cv_proportions(cvvs)) 

# Taille optimale retenue après inspection de la courbe ELPD
k_opt    <- 2
rank_obj <- ranking(cvvs)
sel_vars <- rank_obj$fulldata[1:k_opt]
cat("Variables sélectionnées (k_opt =", k_opt, ") :\n"); print(sel_vars)

# ============================================================
# 8) MODÈLE PROJETÉ
# ============================================================

proj2      <- project(cvvs, nterms = k_opt, ns = 4000)
proj2draws <- as.matrix(proj2)
colnames(proj2draws) <- c("Intercept", sel_vars)

# Coefficients et intervalles crédibles à 95 %
coefs_mean <- colMeans(proj2draws)
ci95       <- apply(proj2draws, 2, quantile, probs = c(0.025, 0.5, 0.975))
round(coefs_mean, 3)
round(t(ci95), 3)

mcmc_areas(proj2draws, prob = 0.95, prob_outer = 1,
           pars = c(sel_vars))

# Matrice de confusion pour Germain
# Probabilités prédites
formula_sub <- as.formula(paste("~", paste(sel_vars, collapse = " + ")))
X_sub       <- model.matrix(formula_sub, data = data_scaled)

log_odds <- X_sub %*% apply(proj2draws, 2, median)
p_pred   <- plogis(log_odds)

# Classe prédite + matrice de confusion
y_pred <- ifelse(p_pred > 0.5, 1, 0)
table(Prédit = y_pred, Observé = data_scaled$TSA)


# Performance (ELPD et accuracy LOO)
perf_sum    <- summary(cvvs, stats = c("elpd", "pctcorr"))
pctcorr_ref <- perf_sum$perf_ref[["pctcorr"]][[1]]
pctcorr_sub <- perf_sum$perf_sub[["pctcorr"]][[k_opt + 1]]
elpd_ref    <- perf_sum$perf_ref[[1]]
elpd_sub    <- perf_sum$perf_sub$elpd[[k_opt + 1]]



cat(sprintf(
  "\n=== RÉSUMÉ ===\nD=%d | p0=%d | tau0=%.3f | CV=%s | k=%d\n\nELPD  — réf : %.2f  | sous-modèle : %.2f\nAccuracy — réf : %.3f | sous-modèle : %.3f\n\nVariables : %s\n",
  D, p0, tau0, toupper(cv_method_used), k_opt,
  elpd_ref, elpd_sub,
  pctcorr_ref, pctcorr_sub,
  paste(sel_vars, collapse = ", ")
))

# ============================================================
# 9) MODÈLES PAR TÂCHE — horseshoe/tâche
# ============================================================

# Définition des variables par tâche
vars_t1 <- names(data_scaled)[4:27]
vars_t2 <- names(data_scaled)[28:47]
vars_t3 <- names(data_scaled)[48:75]
vars_t4 <- names(data_scaled)[76:145]

cat("T1:", length(vars_t1), "vars\n")
cat("T2:", length(vars_t2), "vars\n")
cat("T3:", length(vars_t3), "vars\n")
cat("T4:", length(vars_t4), "vars\n")

# Fonction générique d'ajustement par tâche
fit_task_model <- function(vars, data, p0 = 5, seed = 1234) {
  
  D          <- length(vars)
  sigma_logit <- 2
  tau0       <- (p0 / (D - p0)) * sigma_logit / sqrt(nrow(data))
  
  hs_prior <- hs(global_scale = tau0, slab_df = 4, slab_scale = 2, df = 1, global_df = 1)
  t_prior  <- student_t(df = 7, location = 0, scale = 2.5)
  
  formula_obj <- as.formula(paste("TSA ~", paste(vars, collapse = " + ")))
  
  fit <- stan_glm(
    formula         = formula_obj,
    data            = data,
    family          = binomial("logit"),
    prior           = hs_prior,
    prior_intercept = t_prior,
    seed            = seed,
    chains          = 4,
    iter            = 4000,
    adapt_delta     = 0.95,
    refresh         = 0
  )
  
  return(fit)
}

# Ajustement des 4 modèles par tâche
ref_t1 <- fit_task_model(vars_t1, data_scaled, p0 = 5)
ref_t2 <- fit_task_model(vars_t2, data_scaled, p0 = 5)
ref_t3 <- fit_task_model(vars_t3, data_scaled, p0 = 5)
ref_t4 <- fit_task_model(vars_t4, data_scaled, p0 = 5)

# saveRDS(ref_t1, "ref_t1.rds")
# saveRDS(ref_t2, "ref_t2.rds")
# saveRDS(ref_t3, "ref_t3.rds")
# saveRDS(ref_t4, "ref_t4.rds")

# ============================================================
# 10) LOO PAR MODÈLE ET COMPARAISON
# ============================================================

loo_t1 <- loo(ref_t1, save_psis = TRUE)
loo_t2 <- loo(ref_t2, save_psis = TRUE)
loo_t3 <- loo(ref_t3, save_psis = TRUE)
loo_t4 <- loo(ref_t4, save_psis = TRUE)

# Diagnostic k-hat par modèle
for (nm in c("loo_ref", "loo_t1", "loo_t2", "loo_t3", "loo_t4")) {
  k <- pareto_k_values(get(nm))
  cat(nm, "— prop k > 0.7 :", round(mean(k > 0.7), 3), "\n")
}

# Comparaison ELPD entre modèles (non nichés — indicatif)
comp <- loo_compare(loo_ref, loo_t1, loo_t2, loo_t3, loo_t4)
print(comp, simplify = FALSE)

# ============================================================
# 11) VISUALISATION LOO_COMPARE
# ============================================================

comp_df        <- as.data.frame(comp)
comp_df$model  <- rownames(comp_df)
comp_df$model  <- factor(comp_df$model, levels = rev(comp_df$model))

levels(comp_df$model) <- recode(levels(comp_df$model),
                                "ref_t1"   = "PA (D=24 vars)",
                                "post_ref" = "Full (D=148 vars)",
                                "ref_t3"   = "Epow (D=28 vars)",
                                "ref_t2"   = "ll_ln (D=20 vars)",
                                "ref_t4"   = "PS_lin (D=70 vars)"
)

ggplot(comp_df, aes(x = elpd_diff, y = model)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.4) +
  geom_errorbar(
    aes(xmin = elpd_diff - se_diff, xmax = elpd_diff + se_diff),
    width = 0.2, color = "grey40", linewidth = 0.5
  ) +
  geom_point(size = 3, color = "#2E86AB") +
  labs(
    x        = expression(Delta * "ELPD vs meilleur modèle"),
    y        = NULL,
    #title    = "Comparaison des modèles par tâche",
    #subtitle = "Points = elpd_diff ± se_diff | Ligne 0 = meilleur modèle"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.text.y        = element_text(size = 11)
  )

# ============================================================
# 12) CV_VARSEL PAR TÂCHE (T1)
# ============================================================

# cvvs_t1 <- cv_varsel(
#   ref_t1,
#   method          = "forward",
#   cv_method       = "LOO",
#   validate_search = TRUE,
#   nterms_max      = 15,
#   nclusters       = 20,
#   parallel        = TRUE,
#   nclusters_pred  = 400
# )

#saveRDS(cvvs_t1, "cv_varsel_t1_p10.rds") 

#cvvs <- readRDS("cv_varsel_t1.rds")
# refaire tourner section cvvs


# ============================================================
# 13) VISUALISATIONS RESULTATS
# ============================================================
data_scaled$TSA <- factor(data_scaled$TSA, levels = c(0, 1),
                          labels = c("TD", "TSA"))
ggplot(data = data_scaled, aes(x = L2_large, y = Sparc_small, color = TSA, shape = TSA)) +
  geom_point(size = 2.5) +
  scale_color_manual(values = c("TD" = "#2E86AB", "TSA" = "#E84855")) +
  scale_shape_manual(values = c("TD" = 16, "TSA" = 17)) +
  theme_minimal()
  
  
ggplot(data = data_scaled, aes(y = Sparc_small, fill=TSA))+ geom_density()+ facet_wrap(~TSA)
ggplot(data = data_scaled, aes(y = L2_large, fill=TSA))+ geom_density()+ facet_wrap(~TSA)

boxplot(data_scaled$L2_large~data_scaled$TSA)
boxplot(data_scaled$Sparc_small~data_scaled$TSA)

# Probabilité de direction
pd_L2    <- mean(proj2draws[, "L2_large"]    < 0)
pd_Sparc <- mean(proj2draws[, "Sparc_small"] < 0)
print(paste("P(L2_large < 0)    : ", pd_L2))
print(paste("P(Sparc_small < 0) : ", pd_Sparc))

