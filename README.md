# V-Fashion Insight

V-Fashion Insight is an aspect-based sentiment analysis project for Vietnamese
fashion reviews. A review is classified independently across five aspects:
material, design, size, price, and service.

## Requirements

- Windows PowerShell
- Native Windows CPython 3.11
- Git
- Optional: `uv` for installing a project-local Python runtime

The initial machine inspection found:

- `python --version`: Python 3.11.6 from MSYS2
- `py -0p`: no Python installations detected by the Windows Python Launcher

The MSYS2 interpreter does not use standard Windows scientific-package wheels.
This project therefore uses native Windows CPython 3.11.14 installed locally
under `.python/`. Both `.python/` and `.venv/` are ignored by Git.

## Environment Setup

### Existing native Windows Python 3.11

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Before creating the environment, confirm that `Get-Command python` does not
resolve to an MSYS2 path.

### Project-local Python with uv

Use this option when a native Windows Python 3.11 installation is unavailable:

```powershell
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv python install 3.11
uv venv --python 3.11 --seed .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Verification

Run commands through the virtual environment explicitly:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

Expected Python version: `3.11.x`.

## Project Structure

```text
configs/                     Experiment configuration
data/                        Local raw, interim, and processed data
models/                      Local model artifacts
notebooks/                   Exploratory notebooks
reports/                     Metrics, figures, and analysis
src/v_fashion_insight/       Application source package
tests/                       Automated tests
```

Downloaded datasets, generated models, local Python environments, `plan.md`,
and `workflow.md` are intentionally excluded from Git.

## Shared Contracts

The stable aspect order is:

```text
material, design, size, price, service
```

Labels are:

| Label | Meaning |
|---:|---|
| 0 | Not mentioned |
| 1 | Negative |
| 2 | Neutral |
| 3 | Positive |

The default random seed is `42`.

## Current Status

The data foundation and leakage-safe split are complete. The baseline modeling
phase now provides:

- Five declared TF-IDF/linear-classifier experiments.
- Independent classifiers for material, design, size, price, and service.
- Missing-target masking for the 70 known null label cells.
- Validation-only model selection with the test split locked.
- Versioned artifacts, checksums, comparison metrics, confusion matrices, and
  prediction/evaluation commands.

The frozen benchmark is recorded in `models/baseline/selected.json`. PhoBERT
and product application work follow only after baseline error analysis.

## Download the Dataset

Download the default FashionReviews dataset and write reproducibility metadata:

```powershell
.\.venv\Scripts\vfi-download-data.exe
```

Equivalent module command:

```powershell
.\.venv\Scripts\python.exe -m v_fashion_insight.data.download
```

The command resolves `main` to an immutable Hugging Face commit SHA before
loading the dataset. Cached data is stored under `data/raw/huggingface/`, and
`data/raw/metadata.json` records the resolved revision, split sizes, schema,
and split fingerprints.

Use `--help` to override the dataset, revision, cache directory, or metadata
path. Add `--force-redownload` only when cached content must be refreshed.

## Validate the Raw Dataset

Validate the exact revision recorded by the download metadata:

```powershell
.\.venv\Scripts\vfi-validate-data.exe
```

The validator reads from the local Hugging Face cache by default and checks:

- Required source columns and data types.
- Null or empty review text.
- Null, non-integer, and out-of-range aspect labels.
- Fully duplicated rows and repeated review text.

The machine-readable report is written to
`reports/metrics/data_validation.json`. Exit code `0` means the dataset passed,
`1` means data-quality errors were found, and `2` means validation could not
run. Duplicate review text is reported as a warning because it must be grouped
before the train/validation/test split.

## Profile Label Distribution

Generate deterministic label and aspect-mention summaries:

```powershell
.\.venv\Scripts\vfi-profile-labels.exe
```

The profiler writes:

- `reports/metrics/label_distribution.json`
- `reports/metrics/label_distribution.csv`
- `reports/metrics/mentioned_aspect_distribution.csv`

Reports include counts and proportions for labels `0-3`, missing-label counts,
the comparison between mentioned and not-mentioned aspects, and the number of
mentioned aspects per review. Missing labels remain separate and are never
silently converted to label `0`.

## Profile Review Text

Measure review length and text patterns without changing the source text:

```powershell
.\.venv\Scripts\vfi-profile-text.exe
```

The profiler writes:

- `reports/metrics/text_profile.json`
- `reports/metrics/text_length_percentiles.csv`
- `reports/metrics/text_pattern_counts.csv`

The report contains character, whitespace-token, and line-length percentiles;
IQR outlier counts; short and long review counts; and observed URLs,
emoji-like symbols, unusual whitespace, punctuation, and digits. Only review
IDs and lengths are stored for the longest samples, not duplicated review
text.

## Analyze Exact Duplicates

Group exact duplicates after conservative analysis-only normalization:

```powershell
.\.venv\Scripts\vfi-analyze-duplicates.exe
```

Normalization applies Unicode NFKC, case folding, zero-width character
removal, and whitespace collapsing. Vietnamese diacritics, punctuation, and
digits are preserved. The source dataset is not changed.

The analyzer writes:

- `reports/metrics/exact_duplicate_analysis.json`
- `reports/metrics/exact_duplicate_groups.csv`

Reports distinguish raw duplicates from normalized duplicates and identify
groups with conflicting aspect labels or mixed missing labels. Group IDs are
stable SHA-256 hashes of normalized text.

## Analyze Near-Duplicates and Augmentation

Find likely augmented review variants with scalable lexical similarity:

```powershell
.\.venv\Scripts\vfi-analyze-near-duplicates.exe
```

For a faster deterministic calibration run:

```powershell
.\.venv\Scripts\vfi-analyze-near-duplicates.exe --sample-size 5000
```

The analyzer uses 64-bit SimHash blocking over word unigrams and bigrams,
followed by character trigram Jaccard and sequence similarity verification.
Exact normalized duplicates are excluded because they are handled by the exact
duplicate task. Only high-confidence links form cluster candidates; lower
confidence links are marked `needs_review`.

Outputs:

- `reports/metrics/near_duplicate_analysis.json`
- `reports/metrics/near_duplicate_pairs.csv`
- `reports/metrics/near_duplicate_clusters.csv`

LSH is a scalable candidate generator and does not guarantee that every
semantic paraphrase will be found. Candidate groups are analysis artifacts and
do not modify the dataset.

## Build the Phase 1 EDA Report

Generate the consolidated report, summary, and charts from the pinned local
dataset and the preceding Phase 1 analysis artifacts:

```powershell
.\.venv\Scripts\vfi-build-eda.exe
```

Outputs:

- `notebooks/01_eda.ipynb`
- `reports/eda_report.md`
- `reports/metrics/eda_summary.json`
- `reports/figures/label_distribution.png`
- `reports/figures/review_length_distribution.png`
- `reports/figures/duplicate_summary.png`

The notebook calls the same reusable source function as the CLI and stores no
dataset rows in its outputs. Open it in a Jupyter-compatible editor and run all
cells from a fresh kernel after the CLI and test suite pass.

## Build the Interim Dataset

After the exact and near-duplicate analysis artifacts exist, build the
deterministic interim dataset:

```powershell
.\.venv\Scripts\vfi-preprocess-data.exe
```

Equivalent module command:

```powershell
.\.venv\Scripts\python.exe -m v_fashion_insight.data.preprocess
```

The pipeline:

- Validates the pinned raw dataset while preserving known missing labels.
- Applies conservative NFKC and whitespace normalization.
- Generates stable review IDs from the dataset revision, source split, and
  source identifier.
- Unions exact duplicates with high-confidence near-duplicate clusters.
- Preserves source text and all original labels without deduplication or label
  overwrite.

Outputs:

- `data/interim/reviews.csv`
- `reports/metrics/preprocessing_audit.json`

The default conflict policy is `retain`. Use `--conflict-policy exclude` or
`--conflict-policy manual_review` only when that decision is intentional.
Existing outputs are not replaced unless `--force` is supplied.

## Train and Select the TF-IDF Baseline

Run every experiment declared in `configs/baseline.yaml` and freeze the best
model using validation mean Macro F1:

```powershell
.\.venv\Scripts\python.exe -m v_fashion_insight.models.train_baseline `
  --config configs\baseline.yaml
```

Use `--force` to intentionally replace an existing set of baseline artifacts.
Training fits TF-IDF only on training text, masks missing targets separately
for each aspect, evaluates on validation, and never includes test rows in the
modeling frame. Outputs include:

- `models/baseline/<run>/artifact.joblib`
- `models/baseline/<run>/metadata.json`
- `models/baseline/<run>/validation_metrics.json`
- `models/baseline/<run>/validation_predictions.csv`
- `models/baseline/selected.json`
- `reports/metrics/baseline_comparison.csv`
- Validation confusion matrices under `reports/figures/`

## Predict with the Frozen Baseline

```powershell
.\.venv\Scripts\python.exe -m v_fashion_insight.models.predict `
  --text "Áo đẹp, vải mềm nhưng giao hàng chậm." --pretty
```

The command verifies the selected artifact checksum and returns labels in the
stable aspect order. Logistic regression artifacts expose probabilities;
Linear SVM artifacts expose decision scores, which are not probabilities.

## Evaluate a Saved Baseline

Re-evaluate the frozen model on validation without retraining:

```powershell
.\.venv\Scripts\python.exe -m v_fashion_insight.models.evaluate `
  --split validation
```

Train and validation evaluation are available explicitly. Test evaluation is
intentionally locked until final model assessment and requires the literal
token shown by `--help`; do not unlock it during model selection or error
analysis.

## Analyze Frozen Validation Predictions

Analyze the saved `combined_svc` validation labels without loading a model or
running inference:

```powershell
.\.venv\Scripts\python.exe -m `
  v_fashion_insight.analysis.validation_predictions
```

The command verifies exact validation-ID coverage, joins review text by stable
ID, reproduces the recorded metrics, and separates missed mentions, false
mentions, and sentiment confusions. It writes:

- `reports/analysis/combined_svc_validation_prediction_analysis.md`
- `reports/analysis/combined_svc_validation_error_examples.csv`
- `reports/metrics/combined_svc_validation_prediction_analysis.json`
- `reports/metrics/combined_svc_validation_aspect_errors.csv`
- `reports/metrics/combined_svc_validation_confusions.csv`

The saved label export has no Linear SVM decision scores, so confidence-based
error sampling remains intentionally out of scope until a separate inference
export is run.

### Inspect Errors by Aspect and Label

Create one report for each aspect and quantify false positives, false
negatives, dominant confusion directions, and error families for labels `0-3`:

```powershell
.\.venv\Scripts\python.exe -m `
  v_fashion_insight.analysis.aspect_errors
```

Compact reports are written to
`reports/analysis/combined_svc_aspects/` and `reports/metrics/`. The complete
row-level error export is generated locally as
`combined_svc_validation_aspect_errors_detailed.csv`; it is excluded from Git
because it is reproducible, large, and contains review text.
