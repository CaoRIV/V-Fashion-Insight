# Design Validation Errors

Analysis source: saved `combined_svc` validation predictions. No model inference and no test-set access.

## Summary

- Evaluated targets: 13,365.
- Errors: 1,630 (12.20%).
- Macro F1: 0.876142.
- Weakest label by F1: `negative` (0.7909).
- Highest target error rate: `negative` (21.72%).

## Error Families

| Family | Count | Share of aspect errors |
|---|---:|---:|
| missed_mention | 483 | 29.63% |
| false_mention | 395 | 24.23% |
| sentiment_confusion | 752 | 46.13% |

## Label Diagnostics

| Label | Support | Predicted | Correct | FN | FP | Error rate | Precision | Recall | F1 | Most often predicted as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| not_mentioned | 4,501 | 4,589 | 4,106 | 395 | 483 | 8.78% | 0.8947 | 0.9122 | 0.9034 | negative (194) |
| negative | 2,652 | 2,598 | 2,076 | 576 | 522 | 21.72% | 0.7991 | 0.7828 | 0.7909 | positive (303) |
| neutral | 2,379 | 2,278 | 2,190 | 189 | 88 | 7.94% | 0.9614 | 0.9206 | 0.9405 | negative (79) |
| positive | 3,833 | 3,900 | 3,363 | 470 | 537 | 12.26% | 0.8623 | 0.8774 | 0.8698 | negative (249) |

## Confusion Directions

| True | Predicted | Count | Share of aspect errors |
|---|---|---:|---:|
| negative | positive | 303 | 18.59% |
| positive | negative | 249 | 15.28% |
| negative | not_mentioned | 216 | 13.25% |
| positive | not_mentioned | 206 | 12.64% |
| not_mentioned | negative | 194 | 11.90% |
| not_mentioned | positive | 185 | 11.35% |
| neutral | negative | 79 | 4.85% |
| neutral | not_mentioned | 61 | 3.74% |
| negative | neutral | 57 | 3.50% |
| neutral | positive | 49 | 3.01% |
| not_mentioned | neutral | 16 | 0.98% |
| positive | neutral | 15 | 0.92% |

## Evidence-based Priority

Start manual review with `negative` targets, then inspect the largest confusion direction above. The detailed error CSV retains stable IDs and review text for every row.
