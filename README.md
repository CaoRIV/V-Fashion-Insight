<div align="center">

<!-- Logo -->
<!-- <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 120' width='80' height='80'%3E%3Crect width='120' height='120' fill='%23f0f0f0' rx='16'/%3E%3Cpath d='M30 50 Q60 20 90 50' stroke='%23d4436e' stroke-width='3' fill='none' stroke-linecap='round'/%3E%3Ccircle cx='50' cy='60' r='6' fill='%2307c' opacity='0.8'/%3E%3Ccircle cx='60' cy='75' r='6' fill='%2307c' opacity='0.6'/%3E%3Ccircle cx='70' cy='65' r='6' fill='%2307c' opacity='0.7'/%3E%3Cpath d='M30 85 L90 85 L85 100 L35 100 Z' fill='%2307c' opacity='0.15'/%3E%3C/svg%3E" alt="V-Fashion Insight Logo"> -->

# V-Fashion Insight

*Aspect-Based Sentiment Analysis for Vietnamese Fashion Reviews*

[![Python](https://img.shields.io/badge/python-3.11+-3776ab.svg?logo=python&logoColor=white)](https://python.org)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/CaoRIV/V-Fashion-Insight/releases)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-f7931e.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/streamlit-ready-ff69b4.svg?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/CaoRIV/V-Fashion-Insight.svg)](https://github.com/CaoRIV/V-Fashion-Insight)

[Overview](#overview) • [Features](#features) • [Setup](#environment-setup) • [Usage](#usage) • [Project Structure](#project-structure)

</div>

---

## 📋 Overview

V-Fashion Insight is an aspect-based sentiment analysis project for Vietnamese fashion reviews. A review is classified independently across **five aspects**: material, design, size, price, and service.

The project provides:
- **Data Pipeline** – Comprehensive data download, validation, profiling, and preprocessing
- **Baseline Models** – TF-IDF with Linear SVM and Logistic Regression classifiers
- **Error Analysis** – Detailed validation prediction analysis and aspect-specific error breakdown
- **Reproducibility** – Versioned artifacts with checksums and comprehensive audit trails

## ✨ Features

- **Multi-Aspect Classification** – Independent sentiment labels for 5 fashion review aspects
- **Leakage-Safe Splits** – Proper train/validation/test isolation with duplicate handling
- **Robust Data Processing** – Exact and near-duplicate detection with SimHash LSH
- **TF-IDF Baseline** – Five tuned text-feature configurations (word, char, combined)
- **Comprehensive Metrics** – Confusion matrices, precision/recall analysis per aspect
- **PhoBERT Ready** – Architecture prepared for transformer-based models
- **Streamlit Integration** – Production-ready application framework

## 🏷️ Label Schema

Labels are consistent across all aspects:

| Label | Meaning |
|:---:|---|
| **0** | Not mentioned |
| **1** | Negative |
| **2** | Neutral |
| **3** | Positive |

**Aspect Order** (stable): `material`, `design`, `size`, `price`, `service`

**Default Random Seed**: `42`

## 📁 Project Structure

```text
configs/                     Experiment configuration files
data/                        Raw, interim, and processed datasets
├── raw/                    Downloaded datasets
├── interim/                Preprocessed data
└── processed/              Final training data
models/                      Model artifacts and baselines
├── baseline/               TF-IDF frozen baseline
└── selected.json           Selected model metadata
notebooks/                   Exploratory and analysis notebooks
reports/                     Metrics, analysis, and visualizations
├── metrics/                Quantitative results (JSON, CSV)
├── analysis/               Error analysis reports (Markdown)
└── figures/                Confusion matrices and plots
src/v_fashion_insight/      Application source package
tests/                       Automated test suite
```

**Note**: Downloaded datasets, generated models, local Python environments (`.python/`, `.venv/`), and workflow files are excluded from Git.

## 🛠️ Requirements

- **Windows PowerShell** (or PowerShell Core)
- **Python 3.11+** (native Windows or via `uv`)
- **Git**
- Optional: `uv` for managing a project-local Python runtime

## 🚀 Environment Setup

### Option 1: Existing Native Windows Python 3.11

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

**Note**: Confirm that `Get-Command python` does not resolve to an MSYS2 path.

### Option 2: Project-Local Python with uv

Use this when native Windows Python 3.11 is unavailable:

```powershell
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv python install 3.11
uv venv --python 3.11 --seed .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

### Verification

Run commands through the virtual environment:

```powershell
.\.venv\Scripts\python.exe --version          # Expected: Python 3.11.x
.\.venv\Scripts\python.exe -m pip check       # Dependency check
.\.venv\Scripts\python.exe -m pytest -q       # Test suite
```

## 📊 Usage

### 1. Download the Dataset

```powershell
.\.venv\Scripts\vfi-download-data.exe
```

Or via module:

```powershell
.\.venv\Scripts\python.exe -m v_fashion_insight.data.download
```

Use `--help` to override dataset, revision, cache directory, or metadata path. Add `--force-redownload` to refresh cached content.

### 2. Validate the Raw Dataset

```powershell
.\.venv\Scripts\vfi-validate-data.exe
```

Checks for:
- Required columns and data types
- Null or empty review text
- Invalid aspect labels
- Duplicated rows

Exit codes: `0` = passed, `1` = errors found, `2` = validation failed

### 3. Profile and Analyze Data

```powershell
# Profile label distribution
.\.venv\Scripts\vfi-profile-labels.exe

# Profile review text characteristics
.\.venv\Scripts\vfi-profile-text.exe

# Analyze exact duplicates
.\.venv\Scripts\vfi-analyze-duplicates.exe

# Detect near-duplicates (with optional sampling)
.\.venv\Scripts\vfi-analyze-near-duplicates.exe --sample-size 5000
```

### 4. Build EDA Report

```powershell
.\.venv\Scripts\vfi-build-eda.exe
```

Generates:
- `notebooks/01_eda.ipynb`
- `reports/eda_report.md`
- `reports/figures/` (distribution plots, duplicate summaries)

### 5. Preprocess Data

```powershell
.\.venv\Scripts\vfi-preprocess-data.exe
```

Applies:
- NFKC and whitespace normalization
- Duplicate grouping (exact and high-confidence near-duplicates)
- Stable review ID generation
- Validation while preserving known missing labels

### 6. Train Baseline Models

```powershell
.\.venv\Scripts\python.exe -m v_fashion_insight.models.train_baseline `
  --config configs\baseline.yaml
```

Trains all experiments in `baseline.yaml` and selects the best via validation Macro F1.

### 7. Predict with Frozen Baseline

```powershell
.\.venv\Scripts\python.exe -m v_fashion_insight.models.predict `
  --text "Áo đẹp, vải mềm nhưng giao hàng chậm." --pretty
```

Returns labels in stable aspect order.

### 8. Evaluate on Validation

```powershell
.\.venv\Scripts\python.exe -m v_fashion_insight.models.evaluate `
  --split validation
```

### 9. Analyze Validation Predictions

```powershell
.\.venv\Scripts\python.exe -m v_fashion_insight.analysis.validation_predictions
```

Generates:
- Missed mentions, false mentions, and sentiment confusions
- Error analysis by aspect and label
- Detailed error examples (CSV)

---

## 📈 Baseline Performance

Selected model: **`combined_svc`** (Linear SVM with combined TF-IDF features)

| Run | Features | Classifier | Mean Macro F1 | Exact Match | Latency (ms/review) | Size (MB) |
|---|---|---|---:|---:|---:|---:|
| **combined_svc** | combined | linear_svc | **0.8662** | 0.5640 | 0.016 | 13.07 |
| combined_svc_balanced | combined | linear_svc | 0.8659 | 0.5599 | 0.011 | 13.09 |
| word_logreg | word | logistic_regression | 0.8658 | 0.5611 | 0.002 | 6.56 |
| combined_logreg | combined | logistic_regression | 0.8656 | 0.5612 | 0.010 | 7.34 |
| char_svc | char_wb | linear_svc | 0.8272 | 0.4760 | 0.009 | 4.67 |

**Selection Criteria**: Highest validation mean Macro F1 across five aspects; latency and artifact size are deterministic tie-breakers.

---

## 🔬 Advanced Workflows

### Conflict Handling in Preprocessing

Use different conflict policies for duplicate resolution:

```powershell
# Default: retain conflicting labels
.\.venv\Scripts\python.exe -m v_fashion_insight.data.preprocess `
  --conflict-policy retain

# Exclude conflicting duplicates
.\.venv\Scripts\python.exe -m v_fashion_insight.data.preprocess `
  --conflict-policy exclude

# Manual review mode
.\.venv\Scripts\python.exe -m v_fashion_insight.data.preprocess `
  --conflict-policy manual_review
```

### Aspect-Specific Error Analysis

```powershell
.\.venv\Scripts\python.exe -m v_fashion_insight.analysis.aspect_errors
```

Generates per-aspect error reports with false positive/negative breakdown by label.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 👤 Author

**Cao Van Ha** – [@CaoRIV](https://github.com/CaoRIV)

---

<div align="center">

**[↑ Back to Top](#v-fashion-insight)**

</div>
