# Size Validation Errors

Analysis source: saved `combined_svc` validation predictions. No model inference and no test-set access.

## Summary

- Evaluated targets: 13,366.
- Errors: 2,063 (15.43%).
- Macro F1: 0.810502.
- Weakest label by F1: `neutral` (0.6232).
- Highest target error rate: `neutral` (45.28%).

## Error Families

| Family | Count | Share of aspect errors |
|---|---:|---:|
| missed_mention | 479 | 23.22% |
| false_mention | 219 | 10.62% |
| sentiment_confusion | 1,365 | 66.17% |

## Label Diagnostics

| Label | Support | Predicted | Correct | FN | FP | Error rate | Precision | Recall | F1 | Most often predicted as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| not_mentioned | 4,252 | 4,512 | 4,033 | 219 | 479 | 5.15% | 0.8938 | 0.9485 | 0.9204 | negative (108) |
| negative | 3,321 | 3,395 | 2,871 | 450 | 524 | 13.55% | 0.8457 | 0.8645 | 0.8550 | not_mentioned (200) |
| neutral | 1,791 | 1,354 | 980 | 811 | 374 | 45.28% | 0.7238 | 0.5472 | 0.6232 | positive (460) |
| positive | 4,002 | 4,105 | 3,419 | 583 | 686 | 14.57% | 0.8329 | 0.8543 | 0.8435 | neutral (239) |

## Confusion Directions

| True | Predicted | Count | Share of aspect errors |
|---|---|---:|---:|
| neutral | positive | 460 | 22.30% |
| neutral | negative | 276 | 13.38% |
| positive | neutral | 239 | 11.59% |
| positive | not_mentioned | 204 | 9.89% |
| negative | not_mentioned | 200 | 9.69% |
| positive | negative | 140 | 6.79% |
| negative | neutral | 126 | 6.11% |
| negative | positive | 124 | 6.01% |
| not_mentioned | negative | 108 | 5.24% |
| not_mentioned | positive | 102 | 4.94% |
| neutral | not_mentioned | 75 | 3.64% |
| not_mentioned | neutral | 9 | 0.44% |

## Evidence-based Priority

Start manual review with `neutral` targets, then inspect the largest confusion direction above. The detailed error CSV retains stable IDs and review text for every row.
