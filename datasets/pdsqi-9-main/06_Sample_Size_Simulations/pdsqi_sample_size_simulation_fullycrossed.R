# ============================================================
# PDSQI-9 Multi-Site Reliability Sample Size Simulation
# Fully Crossed Design
# ============================================================

suppressPackageStartupMessages({
  library(ordinal)
  library(dplyr)
  library(parallel)
})

cat("========================================\n")
cat("  PDSQI-9 Sample Size — Fully Crossed\n")
cat("========================================\n")
cat("Started:", format(Sys.time()), "\n")
cat(sprintf("Interactive: %s\n\n", interactive()))

# ============================================================
# FUNCTIONS
# ============================================================

make_design <- function(K, N_cases, n_sites = 3) {
  do.call(rbind, lapply(seq_len(n_sites), function(s) {
    site <- paste0("S", s)
    expand.grid(
      site     = site,
      reviewer = paste0(site, "_R", seq_len(K)),
      case     = paste0(site, "_C", seq_len(N_cases)),
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )
  }))
}

simulate_items <- function(K, N_cases, item_params, n_sites = 3, seed = 1) {
  set.seed(seed)
  var_resid <- (pi^2) / 3
  logistic  <- function(x) 1 / (1 + exp(-x))
  
  design <- make_design(K, N_cases, n_sites)
  design$site          <- factor(design$site)
  design$case_site     <- factor(paste(design$site, design$case, sep = "."))
  design$reviewer_site <- factor(paste(design$site, design$reviewer, sep = "."))
  
  for (nm in names(item_params)) {
    p <- item_params[[nm]]
    var_case <- (p$icc * (p$var_rev + var_resid)) / (1 - p$icc)
    
    u <- rnorm(nlevels(design$site), 0, sqrt(p$var_site)); names(u) <- levels(design$site)
    v <- rnorm(nlevels(design$case_site), 0, sqrt(var_case)); names(v) <- levels(design$case_site)
    w <- rnorm(nlevels(design$reviewer_site), 0, sqrt(p$var_rev)); names(w) <- levels(design$reviewer_site)
    
    eta <- u[as.character(design$site)] +
      v[as.character(design$case_site)] +
      w[as.character(design$reviewer_site)]
    
    cum <- sapply(p$thresholds, function(tk) logistic(tk - eta))
    cum <- cbind(0, cum, 1)
    pr  <- cum[, 2:6] - cum[, 1:5]
    design[[nm]] <- ordered(
      apply(pr, 1, function(row) sample.int(5, 1, prob = row)),
      levels = 1:5
    )
  }
  design
}

fit_icc <- function(dat, item_col) {
  form <- as.formula(paste0(item_col, " ~ 1 + (1|site) + (1|case_site) + (1|reviewer_site)"))
  m <- tryCatch(
    clmm(form, data = dat, link = "logit", Hess = FALSE, nAGQ = 1),
    error = function(e) NULL
  )
  if (is.null(m)) return(list(model = NULL, icc = NA_real_, formula = form))
  
  vc <- VarCorr(m)
  var_case  <- as.numeric(vc$case_site)
  var_rev   <- as.numeric(vc$reviewer_site)
  var_resid <- (pi^2) / 3
  
  list(model = m, icc = var_case / (var_case + var_rev + var_resid), formula = form)
}

boot_icc_ci <- function(fitted_model, orig_data, item_col, form,
                        nsim = 20, seed = 1) {
  if (is.null(fitted_model)) return(c(lower = NA, upper = NA))
  set.seed(seed)
  
  vc <- VarCorr(fitted_model)
  sd_case <- sqrt(as.numeric(vc$case_site))
  sd_rev  <- sqrt(as.numeric(vc$reviewer_site))
  sd_site <- if ("site" %in% names(vc)) sqrt(as.numeric(vc$site)) else 0
  th <- fitted_model$alpha
  logistic  <- function(x) 1 / (1 + exp(-x))
  var_resid <- (pi^2) / 3
  
  site_levels <- levels(orig_data$site)
  case_levels <- levels(orig_data$case_site)
  rev_levels  <- levels(orig_data$reviewer_site)
  
  n_ok <- 0
  icc_boot <- numeric(nsim)
  
  for (b in seq_len(nsim)) {
    u <- rnorm(length(site_levels), 0, sd_site); names(u) <- site_levels
    v <- rnorm(length(case_levels), 0, sd_case); names(v) <- case_levels
    w <- rnorm(length(rev_levels),  0, sd_rev);  names(w) <- rev_levels
    
    eta <- u[as.character(orig_data$site)] +
      v[as.character(orig_data$case_site)] +
      w[as.character(orig_data$reviewer_site)]
    
    cum <- sapply(th, function(tk) logistic(tk - eta))
    cum <- cbind(0, cum, 1)
    pr  <- cum[, 2:6] - cum[, 1:5]
    y_new <- apply(pr, 1, function(row) sample.int(5, 1, prob = row))
    
    dat_b <- orig_data
    dat_b[[item_col]] <- ordered(y_new, levels = 1:5)
    
    m2 <- tryCatch(
      clmm(form, data = dat_b, link = "logit", Hess = FALSE, nAGQ = 1),
      error = function(e) NULL
    )
    
    if (is.null(m2)) {
      icc_boot[b] <- NA_real_
    } else {
      vc2 <- VarCorr(m2)
      icc_boot[b] <- as.numeric(vc2$case_site) /
        (as.numeric(vc2$case_site) + as.numeric(vc2$reviewer_site) + var_resid)
      n_ok <- n_ok + 1
    }
  }
  
  icc_valid <- icc_boot[is.finite(icc_boot)]
  if (length(icc_valid) < 3) return(c(lower = NA, upper = NA))
  c(lower = unname(quantile(icc_valid, 0.025)),
    upper = unname(quantile(icc_valid, 0.975)))
}

# ============================================================
# ITEM PARAMS — all 9 PDSQI-9 items
# ============================================================
item_params <- list(
  item1 = list(icc = 0.80, var_rev = 0.30, var_site = 0.10,
               thresholds = c(-3.0, -2.0, -1.0, 0.0)),
  item2 = list(icc = 0.80, var_rev = 0.30, var_site = 0.10,
               thresholds = c(-3.0, -2.0, -1.0, 0.0)),
  item3 = list(icc = 0.80, var_rev = 0.30, var_site = 0.10,
               thresholds = c(-3.0, -2.0, -1.0, 0.0)),
  item4 = list(icc = 0.80, var_rev = 0.30, var_site = 0.10,
               thresholds = c(-3.0, -2.0, -1.0, 0.0)),
  item5 = list(icc = 0.72, var_rev = 0.55, var_site = 0.10,
               thresholds = c(-2.2, -1.4, -0.6, 0.4)),
  item6 = list(icc = 0.80, var_rev = 0.30, var_site = 0.10,
               thresholds = c(-3.0, -2.0, -1.0, 0.0)),
  item7 = list(icc = 0.80, var_rev = 0.30, var_site = 0.10,
               thresholds = c(-3.0, -2.0, -1.0, 0.0)),
  item8 = list(icc = 0.88, var_rev = 0.20, var_site = 0.10,
               thresholds = c(-3.4, -2.5, -1.4, -0.3)),
  item9 = list(icc = 0.88, var_rev = 0.20, var_site = 0.10,
               thresholds = c(-3.4, -2.5, -1.4, -0.3))
)

# ============================================================
# SETTINGS
# ============================================================
N_REP        <- 50
BOOT_NSIM    <- 300
N_CORES      <- 50
WIDTH_TARGET <- 0.15
HARD_ITEM    <- "item5"

grid <- expand.grid(
  K       = 3:10,
  N_cases = c(50, 100, 150),
  stringsAsFactors = FALSE
)

cat(sprintf("Grid: %d scenarios | Reps: %d | Boot: %d | Cores: %d\n",
            nrow(grid), N_REP, BOOT_NSIM, N_CORES))
cat(sprintf("Total jobs: %d\n\n", nrow(grid) * N_REP))

# ============================================================
# RUN ALL SCENARIOS
# ============================================================
t_total <- Sys.time()
all_summaries <- list()
all_raw <- list()

for (sc in seq_len(nrow(grid))) {
  K_VAL       <- grid$K[sc]
  N_CASES_VAL <- grid$N_cases[sc]
  
  cat(sprintf("\n========== Scenario %d/%d: K=%d, N=%d ==========\n",
              sc, nrow(grid), K_VAL, N_CASES_VAL))
  cat(sprintf("[%s] Launching %d reps on %d cores...\n", format(Sys.time()), N_REP, N_CORES))
  t0 <- Sys.time()
  
  BATCH_SIZE <- N_CORES
  n_batches <- ceiling(N_REP / BATCH_SIZE)
  all_results <- vector("list", N_REP)
  
  for (batch in seq_len(n_batches)) {
    idx_start <- (batch - 1) * BATCH_SIZE + 1
    idx_end <- min(batch * BATCH_SIZE, N_REP)
    batch_reps <- idx_start:idx_end
    
    cat(sprintf("[%s]   Batch %d/%d (reps %d-%d)...\n",
                format(Sys.time()), batch, n_batches, idx_start, idx_end))
    
    batch_results <- mclapply(batch_reps, function(rep_id) {
      seed_sim <- 1000 * K_VAL + 10 * N_CASES_VAL + rep_id
      dat <- simulate_items(K = K_VAL, N_cases = N_CASES_VAL,
                            item_params = item_params, n_sites = 3, seed = seed_sim)
      
      widths <- numeric(length(item_params))
      iccs   <- numeric(length(item_params))
      names(widths) <- names(iccs) <- names(item_params)
      
      for (nm in names(item_params)) {
        fit <- fit_icc(dat, nm)
        iccs[nm] <- fit$icc
        if (is.null(fit$model)) {
          widths[nm] <- NA_real_
        } else {
          ci <- boot_icc_ci(fit$model, dat, nm, fit$formula,
                            nsim = BOOT_NSIM,
                            seed = seed_sim * 10 + which(names(item_params) == nm))
          widths[nm] <- as.numeric(ci["upper"] - ci["lower"])
        }
      }
      
      converged   <- is.finite(widths)
      n_converged <- sum(converged)
      
      data.frame(
        K = K_VAL, N_cases = N_CASES_VAL, rep_id = rep_id,
        hard_width  = widths["item5"],
        worst_width = if (any(converged)) max(widths[converged]) else NA_real_,
        hard_icc    = iccs["item5"],
        pass_hard   = is.finite(widths["item5"]) && widths["item5"] <= WIDTH_TARGET,
        pass_all    = if (n_converged == 0) NA else all(widths[converged] <= WIDTH_TARGET),
        n_converged = n_converged,
        stringsAsFactors = FALSE
      )
    }, mc.cores = N_CORES, mc.set.seed = TRUE)
    
    all_results[batch_reps] <- batch_results
    
    # Batch progress
    batch_df <- do.call(rbind, Filter(is.data.frame, batch_results))
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    batch_fail <- sum(is.na(batch_df$hard_width))
    
    cat(sprintf("[%s]   Batch %d done — %d/%d reps — %.1f min — failures: %d\n",
                format(Sys.time()), batch, idx_end, N_REP, elapsed, batch_fail))
    if (nrow(batch_df) > 0) {
      cat(sprintf("    hard_width: median=%.3f  range=[%.3f, %.3f]\n",
                  median(batch_df$hard_width, na.rm = TRUE),
                  min(batch_df$hard_width, na.rm = TRUE),
                  max(batch_df$hard_width, na.rm = TRUE)))
    }
    flush.console()
  }
  
  # Summarize this scenario
  df <- do.call(rbind, Filter(is.data.frame, all_results))
  all_raw[[sc]] <- df
  
  summary_row <- data.frame(
    K                    = K_VAL,
    N_cases              = N_CASES_VAL,
    median_hard_width    = median(df$hard_width, na.rm = TRUE),
    p90_hard_width       = as.numeric(quantile(df$hard_width, 0.90, na.rm = TRUE)),
    median_worst_width   = median(df$worst_width, na.rm = TRUE),
    p90_worst_width      = as.numeric(quantile(df$worst_width, 0.90, na.rm = TRUE)),
    pr_hard_meets_target = mean(df$pass_hard, na.rm = TRUE),
    pr_all_meet_target   = mean(df$pass_all, na.rm = TRUE),
    row.names = NULL
  )
  all_summaries[[sc]] <- summary_row
  
  sc_elapsed <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  cat(sprintf("[%s] Scenario %d/%d done (%.1f min): median_hard_w=%.3f  pr_hard=%.2f\n",
              format(Sys.time()), sc, nrow(grid), sc_elapsed,
              summary_row$median_hard_width, summary_row$pr_hard_meets_target))
  
  # Incremental save after each scenario
  results_so_far <- do.call(rbind, all_summaries)
  write.csv(results_so_far, "pdsqi9_ordinal_icc_precision_fullcrossed_PARTIAL.csv", row.names = FALSE)
}

elapsed_total <- as.numeric(difftime(Sys.time(), t_total, units = "hours"))

# ============================================================
# FINAL OUTPUT
# ============================================================
results_sorted <- do.call(rbind, all_summaries)
results_sorted <- results_sorted[order(results_sorted$median_hard_width, results_sorted$p90_hard_width), ]

raw_df <- do.call(rbind, all_raw)

write.csv(results_sorted, "pdsqi9_ordinal_icc_precision_fullcrossed.csv", row.names = FALSE)
write.csv(raw_df, "pdsqi9_ordinal_icc_raw_replicates.csv", row.names = FALSE)

cat("\n========================================\n")
cat("  FINAL RESULTS\n")
cat("========================================\n")
print(results_sorted)
cat(sprintf("\nTotal runtime: %.1f hours on %d cores\n", elapsed_total, N_CORES))
cat(sprintf("Failed reps: %d/%d\n", sum(is.na(raw_df$hard_width)), nrow(raw_df)))
cat("Saved: pdsqi9_ordinal_icc_precision_fullcrossed.csv\n")
cat("Saved: pdsqi9_ordinal_icc_raw_replicates.csv\n")