# 🧬 Breast Cancer ML Classification Using Gene Expression Profiles
### *With Emphasis on Indian Population Generalizability*

> **Can a model trained on Western genomic data correctly classify Indian breast cancer patients?**  
> This project builds, evaluates, and stress-tests machine learning pipelines for breast cancer classification — deliberately probing the cross-population generalization gap that most studies ignore.

---

## 📌 Table of Contents

- [Overview](#overview)
- [The Core Problem](#the-core-problem)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Models](#models)
- [Results Summary](#results-summary)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Biomarker Discovery](#biomarker-discovery)
- [Limitations & Future Work](#limitations--future-work)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project develops a complete machine learning pipeline for **binary breast cancer classification** (Tumour vs. Normal) using gene expression profiles. Unlike most existing work that trains and tests on the same Western cohort, this study explicitly evaluates **cross-population generalizability** — training on a large amalgamated Western dataset and testing on an independent Indian cohort.

Two model families are evaluated across two experimental settings:

| Experiment | Train | Test |
|---|---|---|
| **WesternSVM / WesternXGB** | Western (80%) | Western (20%) |
| **WestIndiSVM / WestIndiXGB** | Western (100%) | Indian (GSE89116) |

Each experiment includes baseline models, SHAP-driven refined models, cross-validation, threshold tuning, and full biomarker analysis.

---

## The Core Problem

Most breast cancer ML models are built on predominantly Western (European/American ancestry) datasets such as TCGA-BRCA and GEO microarray series. These models achieve near-perfect internal metrics — but their real-world clinical utility for **Indian patients** remains largely untested.

This matters because:

- 🧬 **Gene expression landscapes differ across populations** — driven by differences in genetic ancestry, allele frequencies, hormonal profiles, lifestyle, and environmental exposures
- 🏥 **Indian patients present differently** — younger age of onset, more advanced stage at diagnosis, higher prevalence of aggressive subtypes (Triple Negative Breast Cancer, TNBC)
- 📊 **Indian genomic data is severely underrepresented** — Indian patients constitute only **3.5%** of the combined dataset used in this study
- ⚠️ **Deployment on unvalidated populations risks clinical harm** — a model that silently fails on Indian samples could delay diagnosis

This project directly quantifies that failure, and explores what mitigates it.

---

## Datasets

### Western Dataset (Training) — 6,322 samples

An amalgamation of four publicly available Western breast cancer gene expression cohorts, preprocessed and batch-corrected into a unified feature matrix (`shapfeatures.csv`):

| Dataset | Samples | Platform | Source | Cohort |
|---|---|---|---|---|
| **GSE96058** | 3,273 | Illumina RNA-seq | NCBI GEO | Sweden |
| **TCGA-BRCA** | 1,218 | Illumina HiSeq | UCSC Xena | Multi-centre US |
| **ICGC-BRCA** | 1,172 | Illumina RNA-seq | ICGC Xena | European |
| **GTEx (Mammary)** | 514 | Illumina RNA-seq | GTEx Portal | Normal baseline (GTEx v8) |

> GTEx mammary samples serve as the normal tissue reference baseline, providing biologically grounded negative class examples.

### Indian Dataset (External Test) — 232 samples

| Dataset | Samples | Platform | Source | Notes |
|---|---|---|---|---|
| **GSE89116** | 232 (116 tumour · 116 normal) | Affymetrix HG-U133 Plus 2.0 | NCBI GEO | Indian breast cancer cohort — perfectly balanced |

> ⚠️ The Indian dataset was **never used during training** at any stage. It serves exclusively as a held-out external test set.

### Feature Alignment

After cross-platform gene symbol harmonisation, **566 common genes** were identified across Western and Indian datasets. All models in the cross-population experiments operate on this shared feature space.

| Stat | Value |
|---|---|
| Total samples | 6,554 |
| Western patients | 6,322 |
| Indian patients | 232 |
| Common features | **566 genes** |
| Indian share of total | **3.5%** |

---
---

Raw Gene Expression Data
        │
        ▼
┌─────────────────────────┐
│   Data Preprocessing    │  ← dropna, deduplication, LabelEncoder
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Feature Alignment     │  ← common gene identification (Western ∩ Indian)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   StandardScaler        │  ← fit on Western train, transform Indian test
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   SMOTE                 │  ← synthetic oversampling on Western training set only
└─────────────────────────┘
        │
        ▼
┌──────────────┬──────────────────┐
│  Baseline    │  Biomarker       │
│  Model       │  Discovery       │
│  (all genes) │  RF + SHAP       │
└──────┬───────┴────────┬─────────┘
       │                │
       │         Top 20–30 SHAP genes
       │                │
       ▼                ▼
┌─────────────────────────┐
│   Refined Model         │  ← SVM (RBF) or XGBoost on top SHAP features
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   5-Fold Stratified CV  │  ← on Western data (internal validation)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Threshold Tuning      │  ← sweep 0.10–0.90, optimise tumour recall ≥ 0.85
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   External Evaluation   │  ← Indian test set (GSE89116)
└─────────────────────────┘
```
---
---

## Models

### Support Vector Machine (SVM)

| Variant | Kernel | Features | Class Handling |
|---|---|---|---|
| Baseline | Linear | All common genes | `class_weight='balanced'` |
| Refined | RBF (C=10, γ=scale) | Top 20–30 SHAP genes | `class_weight='balanced'` |
| Tuned | RBF | Top SHAP genes | Custom threshold (≠ 0.50) |

- Cross-population experiment uses **LinearSVC** wrapped in `CalibratedClassifierCV` for speed (10–100× faster than `SVC(kernel='linear')`), with no sacrifice in accuracy
- SMOTE removed in cross-population SVM — `class_weight='balanced'` handles imbalance without the noise introduced by synthetic oversampling on high-dimensional genomic data

### XGBoost

| Variant | Estimators | Features | Regularization |
|---|---|---|---|
| Baseline | 300 | All common genes | `scale_pos_weight` |
| Refined | 400 | Top 30 SHAP genes | `reg_alpha=0.1`, `reg_lambda=1.0` |
| Tuned | 400 | Top SHAP genes | Custom threshold (≠ 0.50) |

- Three native importance types computed: `weight`, `gain`, `cover`
- SHAP computed via `TreeExplainer` on test set samples

---

## Results Summary

### Experiment 1: Train + Test on Western Data

| Model | Accuracy | ROC-AUC | Tumour Recall | Normal Recall | Macro F1 |
|---|---|---|---|---|---|
| Baseline SVM (linear, all features) | 0.707 | 0.840 | 0.712 | 0.672 | 0.599 |
| Refined SVM (RBF, top-30, t=0.50) | 0.791 | 0.910 | 0.767 | 0.938 | 0.710 |
| **Refined SVM (RBF, top-30, t=0.23 tuned)** | **0.927** | **0.910** | **0.996** | **0.503** | **0.809** |
| Baseline XGB (all features) | 0.998 | 0.9998 | 0.997 | 1.000 | 0.995 |
| Refined XGB (top-30, t=0.50) | 0.998 | 0.9998 | 0.998 | 1.000 | 0.997 |

**Western CV (5-fold):**

| Model | CV ROC-AUC | CV Accuracy | CV Macro F1 |
|---|---|---|---|
| SVM (RBF, top-30) | 0.9226 ± 0.0111 | 0.8020 ± 0.0101 | 0.7228 ± 0.0119 |
| XGBoost (top-30) | 0.9999 ± 0.0000 | 0.9979 ± 0.0011 | 0.9957 ± 0.0022 |

---

### Experiment 2: Train on Western → Test on Indian (GSE89116)

| Model | Accuracy | ROC-AUC | Tumour Recall | Normal Recall | Macro F1 |
|---|---|---|---|---|---|
| Baseline LinearSVC (all genes) | 0.500 | 0.997 | 1.000 | 0.000 | 0.333 |
| Refined SVM (RBF, top-20, t=0.50) | 0.703 | 0.747 | 0.405 | 1.000 | 0.674 |
| **Refined SVM (RBF, top-20, t=0.10 tuned)** | **0.741** | **0.747** | **0.483** | **1.000** | **0.723** |
| Baseline XGB (all common features) | 0.500 | 0.990 | 1.000 | 0.000 | 0.333 |
| Refined XGB (top-30, t=0.50) | 0.500 | 0.978 | 1.000 | 0.000 | 0.333 |
| Refined XGB (top-30, t=0.90 tuned) | 0.556 | 0.978 | 1.000 | 0.112 | 0.447 |

> **Western CV vs Indian Test (XGBoost):** ROC-AUC drops from 0.9999 → 0.978; Macro F1 collapses from 0.9957 → 0.333 at default threshold — the most stark generalization failure in this study.

---

## Key Findings

### 1. XGBoost fails catastrophically on Indian data at default threshold
XGBoost, despite near-perfect Western performance, classified virtually every Indian sample as tumour when using the default 0.50 threshold — achieving 0% normal recall. This is not a subtle degradation; it is a complete failure of the decision boundary to transfer across populations. Even after aggressive threshold tuning (t=0.90), normal recall only recovered to 11.2%.

### 2. SVM transfers better than XGBoost across populations
The refined SVM (RBF kernel, top-20 SHAP genes) achieved 70.3% accuracy and 74.7% ROC-AUC on the Indian test set — far from perfect, but meaningfully better than XGBoost's effective coin-flip performance. Margin-based classifiers appear less prone to memorizing population-specific expression patterns during training.

### 3. SHAP-based feature selection helps — but doesn't solve the gap
Reducing from all 566 common genes to the top 20 SHAP-ranked genes improved SVM's cross-population macro F1 from 0.333 (baseline) to 0.674. SHAP selection forces the model to focus on genes with interpretable, consistent contributions rather than population-specific noise features.

### 4. Threshold tuning is non-negotiable in cross-population deployment
The default 0.50 threshold, optimised for Western training data, is inappropriate for Indian samples. Tuning the SVM threshold to 0.10 improved macro F1 from 0.674 to 0.723. Any clinical deployment across population boundaries must treat the decision threshold as a population-specific hyperparameter.

### 5. Biomarker consistency across methods
- **SVM experiment:** 14 genes overlapped between RF top-30 and SHAP top-30
- **XGBoost experiment:** 10 genes overlapped between Gain top-30 and SHAP top-30

These overlapping genes represent the most robust candidates for population-agnostic breast cancer biomarkers — consistently identified regardless of the importance method or model used.

### 6. The 3.5% problem
Indian patients represent only 3.5% of the combined dataset. This severe imbalance at the population level — not just the class level — is a root cause of the generalization gap. No amount of algorithmic tuning fully compensates for training data that does not represent the target population.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Breast-Cancer-ML-Classification-Using-Gene-Expression-Profiles-with-Emphasis-on-Indian-Population.git
cd Breast-Cancer-ML-Classification-Using-Gene-Expression-Profiles-with-Emphasis-on-Indian-Population

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
xgboost>=1.7.0
shap>=0.42.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Usage

Update the `DATA_PATH` / `WESTERN_PATH` / `INDIAN_PATH` / `RESULTS_DIR` variables at the top of each script to match your local paths, then run:

```bash
# Experiment 1: Train & test SVM on Western data
python scripts/WesternSVM.py

# Experiment 1: Train & test XGBoost on Western data
python scripts/WesternXGB.py

# Experiment 2: Train on Western, test on Indian — SVM
python scripts/WestIndiSVM.py

# Experiment 2: Train on Western, test on Indian — XGBoost
python scripts/WestIndiXGB.py
```

Each script runs the full pipeline end-to-end and saves all outputs (CSVs + plots) to its designated results directory.

### Key Configuration Parameters (WestIndiSVM.py)

```python
RF_N_ESTIMATORS  = 200    # Trees for biomarker discovery
SHAP_N_SAMPLES   = 200    # Samples used for SHAP computation
TOP_N_GENES      = 20     # Top SHAP genes for refined SVM
SVM_C_BASE       = 1.0    # Baseline SVM regularization
SVM_C_REFINED    = 10.0   # Refined SVM regularization
CV_FOLDS         = 5      # Stratified K-fold splits
USE_OVERSAMPLING = False   # Toggle BorderlineSMOTE
```

---

## Biomarker Discovery

Each experiment produces ranked gene lists through two independent methods:

### Random Forest (Gini Importance)
- 200–400 estimators, `max_features='sqrt'`, `class_weight='balanced'`
- Top 50 genes saved to `top50_RF_biomarkers.csv`

### SHAP (SHapley Additive exPlanations)
- `TreeExplainer` applied to RF (SVM experiments) or XGBoost (XGB experiments)
- Mean absolute SHAP value computed across test samples
- Top 50 genes saved to `top50_SHAP_biomarkers.csv`

### Overlap Analysis
Genes appearing in **both** RF top-30 and SHAP top-30 are considered the most robust biomarker candidates:

| Experiment | Overlap Count |
|---|---|
| WestIndiSVM (RF ∩ SHAP) | **14 genes** |
| WestIndiXGB (Gain ∩ SHAP) | **10 genes** |

These overlapping genes are the strongest candidates for further biological validation, as they are consistently prioritised by fundamentally different importance methods.

---

## Limitations & Future Work

### Current Limitations
- **Platform heterogeneity:** Western datasets use Illumina RNA-seq; the Indian dataset uses Affymetrix microarray. Despite gene-level alignment, platform-specific expression distributions may contribute to the generalization gap beyond true biological differences.
- **Small Indian cohort:** 232 samples (116 tumour, 116 normal) limits statistical power and may not represent the full diversity of Indian breast cancer patients.
- **No domain adaptation:** Models are trained purely on Western data with no explicit domain shift correction.
- **Threshold tuning leakage:** Optimal thresholds are selected using the Indian test set, which may yield optimistically biased estimates in practice.
- **Binary classification only:** Tumour vs. Normal; does not address molecular subtype classification (Luminal A/B, HER2+, TNBC).

### Future Directions
- **Domain adaptation:** Apply techniques such as CORAL, MMD minimization, or adversarial domain adaptation to explicitly align Western and Indian feature distributions.
- **Transfer learning:** Fine-tune Western-trained models on a small labelled Indian subset.
- **Multi-omics integration:** Incorporate CNV, methylation, or proteomic data for richer population-inclusive representation.
- **Larger Indian cohorts:** Partner with Indian cancer genomics initiatives (IndiGen, NCDIR) to expand training representation.
- **Molecular subtype classification:** Extend beyond binary to multi-class subtype prediction relevant to treatment decisions.
- **Federated learning:** Enable collaborative model training across Indian hospital systems without centralizing patient data.

---


## Acknowledgements

This project was conducted as part of the **Interdisciplinary Research Internship** organized by the **International Institute of Medical Science and Technology Council (IIMSTC)** in collaboration with **Visvesvaraya Technological University (VTU)**, February 2025 onwards.

Datasets accessed from:
- [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/) — GSE96058, GSE89116
- [UCSC Xena](https://xena.ucsc.edu/) — TCGA-BRCA
- [ICGC Data Portal](https://dcc.icgc.org/) — ICGC-BRCA
- [GTEx Portal](https://gtexportal.org/) — GTEx v8 Mammary

---

<div align="center">

**Built with purpose — to make breast cancer ML more equitable, one population at a time.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

</div>
