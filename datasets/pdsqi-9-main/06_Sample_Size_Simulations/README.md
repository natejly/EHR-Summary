# PDSQI-9 External Validation Simulation

**Precision-Based Sample Size Planning for Multi-Site Ordinal Reliability Studies**

Monte Carlo simulation framework for precision-based sample size planning of the **Provider Documentation Summarization Quality Instrument (PDSQI-9)** in multi-site reliability studies.

---

## 🎯 Objective

This repository provides simulation code to determine the number of:

- **Reviewers per site (K)**
- **Case notes per site (N_cases)**
- (For partially crossed designs) **Reviewers per case (r)**

required to achieve a target **95% confidence interval width ≤ 0.15** for ordinal intraclass correlation coefficients (ICCs).

Precision planning is driven by the most difficult-to-estimate item (ICC = 0.72).

---

## 🧠 Statistical Framework

Each of the nine PDSQI-9 items is modeled using a:

> **Cumulative link mixed-effects model (CLMM)**
> with ordered logit link

### Random Effects Structure

- Site-level random intercept
- Case-level random intercept
- Reviewer-level random intercept

### Reliability Tiers

Items are grouped into three ICC tiers:

| Tier | PDSQI-9 Item | ICC |
|---| ----- | ---|
| High | Comprehensible, Accurate | 0.88 |
| Moderate | Cited, Thorough, Organized, Succinct, Useful| 0.80 |
| Challenging (Hardest) | Synthesized | 0.72 |

Sample size decisions are based on achieving target precision for the **hardest item (ICC = 0.72)**.

---

## 🧪 Simulation Design

For each design scenario:

- **50 Monte Carlo replicates**
- **300 parametric bootstrap refits per replicate**
- ICC precision evaluated via 95% CI width

Two rating designs are supported:

---

### 🔁 Fully Crossed Design

Every reviewer rates every case within a site.

Script:

    pdsqi_sample_size_simulation_fullycrossed.R

---

### 🔄 Partially Crossed Design

Each case is rated by *r* of *K* reviewers
(assigned round-robin within site).

Script:

    pdsqi_sample_size_simulation_partiallycrossed.R

---

## ▶️ Running the Simulations

Each script is self-contained and executable from the command line:

```bash
Rscript pdsqi_sample_size_simulation_fullycrossed.R
Rscript pdsqi_sample_size_simulation_partiallycrossed.R
```

### Parallelization

- Uses `parallel::mclapply`
- Number of cores controlled by `N_CORES`
- Incremental checkpoint files (`*_PARTIAL.csv`) written after each grid point

---

## ⚙️ Configuration Parameters

Defined at the top of each script:

| Parameter | Default | Description |
|---|---|---|
| `N_REP` | 50 | Monte Carlo replicates per design scenario |
| `BOOT_NSIM` | 300 | Parametric bootstrap refits per replicate |
| `N_CORES` | 100 | Parallel cores used |
| `WIDTH_TARGET` | 0.15 | Target 95% CI width |

---

## 📊 Output Files

### Fully Crossed

| File | Description |
|---|---|
| `pdsqi9_ordinal_icc_precision_fullcrossed.csv` | Summary metrics per scenario |
| `pdsqi9_ordinal_icc_raw_replicates.csv` | Per-replicate results |

---

### Partially Crossed

| File | Description |
|---|---|
| `pdsqi9_ordinal_icc_precision_partiallycrossed.csv` | Summary metrics per scenario |
| `pdsqi9_ordinal_icc_raw_replicates_partiallycrossed.csv` | Per-replicate results |

---

## 📈 Summary Output Structure

(`*_precision_*.csv`)

One row per design scenario:

| Column | Description |
|---|---|
| `K` | Reviewers per site |
| `N_cases` | Case notes per site |
| `r` | Reviewers per case (partially crossed only) |
| `median_hard_width` | Median CI width for hardest item |
| `p90_hard_width` | 90th percentile CI width (hardest item) |
| `median_worst_width` | Median of widest CI across all items |
| `p90_worst_width` | 90th percentile of widest CI |
| `pr_hard_meets_target` | Proportion meeting width target (hardest item) |
| `pr_all_meet_target` | Proportion meeting width target (all items) |

---

## 📉 Raw Replicate Output Structure

(`*_raw_replicates*.csv`)

One row per replicate per scenario:

| Column | Description |
|---|---|
| `K`, `N_cases`, (`r`) | Design parameters |
| `rep_id` | Replicate index (1–50) |
| `hard_width` | CI width for hardest item |
| `worst_width` | Widest CI width across items |
| `hard_icc` | ICC estimate for hardest item |
| `pass_hard` | Indicator: hardest item met target |
| `pass_all` | Indicator: all items met target |
| `n_converged` | Number of items (out of 9) with successful CLMM convergence |

---

## 💻 Computational Requirements

| Resource | Specification |
|---|---|
| **R version** | 4.1.2 |
| **Packages** | `ordinal` (2025.12-29), `dplyr` (1.2.0), `parallel` (base) |
| **CPU** | 50–100 cores recommended |
| **RAM** | ~36 GiB |
| **Runtime** | ~4–6 hours per grid point (K × N combination) |

---

## 📌 Notes

- Precision is driven by CI width rather than hypothesis testing.
- The framework is designed for multi-site generalizability.
- Checkpoint files allow recovery from interrupted runs.