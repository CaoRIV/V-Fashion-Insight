# combined_svc Validation Prediction Analysis

This report analyzes saved validation predictions only. No model was loaded, no inference was run, and the test split was not accessed.

The saved CSV contains predicted labels but no Linear SVM decision scores, so this stage does not make confidence-based claims.

## Executive Summary

- Validation rows: 13,368.
- Mean Macro F1: 0.866187.
- Exact-match ratio on complete targets: 0.564049.
- Rows with at least one aspect error: 5,831 (43.62%).
- Total aspect-level errors: 7,788.
- Weakest aspect: `size` (Macro F1 0.810502).

## Aspect Results

| Aspect | Evaluated | Errors | Error rate | Macro F1 | Missed mention | False mention | Sentiment confusion | Weakest label |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| material | 13,367 | 1,670 | 12.49% | 0.863851 | 513 | 319 | 838 | negative (0.8302) |
| design | 13,365 | 1,630 | 12.20% | 0.876142 | 483 | 395 | 752 | negative (0.7909) |
| size | 13,366 | 2,063 | 15.43% | 0.810502 | 479 | 219 | 1,365 | neutral (0.6232) |
| price | 13,366 | 1,087 | 8.13% | 0.898666 | 261 | 79 | 747 | neutral (0.8438) |
| service | 13,365 | 1,338 | 10.01% | 0.881772 | 505 | 269 | 564 | neutral (0.8239) |

## Largest Confusions

| Aspect | True | Predicted | Count | Share of true label | Share of aspect errors |
|---|---|---|---:|---:|---:|
| size | neutral | positive | 460 | 25.68% | 22.30% |
| design | negative | positive | 303 | 11.43% | 18.59% |
| price | neutral | positive | 297 | 12.25% | 27.32% |
| service | negative | not_mentioned | 283 | 10.92% | 21.15% |
| size | neutral | negative | 276 | 15.41% | 13.38% |
| price | positive | neutral | 256 | 8.66% | 23.55% |
| design | positive | negative | 249 | 6.50% | 15.28% |
| size | positive | neutral | 239 | 5.97% | 11.59% |
| material | positive | not_mentioned | 236 | 6.63% | 14.13% |
| material | negative | not_mentioned | 218 | 8.27% | 13.05% |
| design | negative | not_mentioned | 216 | 8.14% | 13.25% |
| material | neutral | positive | 207 | 9.76% | 12.40% |
| design | positive | not_mentioned | 206 | 5.37% | 12.64% |
| size | positive | not_mentioned | 204 | 5.10% | 9.89% |
| size | negative | not_mentioned | 200 | 6.02% | 9.69% |

## Errors per Review

| Incorrect aspects | Reviews |
|---:|---:|
| 0 | 7,537 |
| 1 | 4,203 |
| 2 | 1,331 |
| 3 | 265 |
| 4 | 32 |
| 5 | 0 |

## Interpretation

- `size` is the first priority for manual review, especially neutral size targets.
- Missed/false mention errors separate aspect detection problems from sentiment-polarity problems.
- The deterministic examples CSV contains review text for every populated off-diagonal confusion cell and is the input to the next manual taxonomy step.
- These results reproduce the frozen validation metrics exactly.
