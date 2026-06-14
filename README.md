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

Phase 0 establishes:

- A `src`-layout Python package.
- Core data-science dependencies.
- Shared constants, logging, and reproducibility helpers.
- A pytest test suite.

Dataset download and validation begin in Phase 1.

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
