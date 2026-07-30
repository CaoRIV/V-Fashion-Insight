# Service Validation Errors

Analysis source: saved `combined_svc` validation predictions. No model inference and no test-set access.

## Summary

- Evaluated targets: 13,365.
- Errors: 1,338 (10.01%).
- Macro F1: 0.881772.
- Weakest label by F1: `neutral` (0.8239).
- Highest target error rate: `neutral` (20.45%).

## Error Families

| Family | Count | Share of aspect errors |
|---|---:|---:|
| missed_mention | 505 | 37.74% |
| false_mention | 269 | 20.10% |
| sentiment_confusion | 564 | 42.15% |

## Label Diagnostics

| Label | Support | Predicted | Correct | FN | FP | Error rate | Precision | Recall | F1 | Most often predicted as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| not_mentioned | 5,022 | 5,258 | 4,753 | 269 | 505 | 5.36% | 0.9040 | 0.9464 | 0.9247 | negative (164) |
| negative | 2,591 | 2,482 | 2,138 | 453 | 344 | 17.48% | 0.8614 | 0.8252 | 0.8429 | not_mentioned (283) |
| neutral | 1,726 | 1,607 | 1,373 | 353 | 234 | 20.45% | 0.8544 | 0.7955 | 0.8239 | negative (152) |
| positive | 4,026 | 4,018 | 3,763 | 263 | 255 | 6.53% | 0.9365 | 0.9347 | 0.9356 | not_mentioned (139) |

## Confusion Directions

| True | Predicted | Count | Share of aspect errors |
|---|---|---:|---:|
| negative | not_mentioned | 283 | 21.15% |
| not_mentioned | negative | 164 | 12.26% |
| neutral | negative | 152 | 11.36% |
| positive | not_mentioned | 139 | 10.39% |
| negative | neutral | 123 | 9.19% |
| neutral | positive | 118 | 8.82% |
| positive | neutral | 96 | 7.17% |
| not_mentioned | positive | 90 | 6.73% |
| neutral | not_mentioned | 83 | 6.20% |
| negative | positive | 47 | 3.51% |
| positive | negative | 28 | 2.09% |
| not_mentioned | neutral | 15 | 1.12% |

## Evidence-based Priority

Start manual review with `neutral` targets, then inspect the largest confusion direction above. The detailed error CSV retains stable IDs and review text for every row.
