# Price Validation Errors

Analysis source: saved `combined_svc` validation predictions. No model inference and no test-set access.

## Summary

- Evaluated targets: 13,366.
- Errors: 1,087 (8.13%).
- Macro F1: 0.898666.
- Weakest label by F1: `neutral` (0.8438).
- Highest target error rate: `neutral` (17.94%).

## Error Families

| Family | Count | Share of aspect errors |
|---|---:|---:|
| missed_mention | 261 | 24.01% |
| false_mention | 79 | 7.27% |
| sentiment_confusion | 747 | 68.72% |

## Label Diagnostics

| Label | Support | Predicted | Correct | FN | FP | Error rate | Precision | Recall | F1 | Most often predicted as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| not_mentioned | 6,122 | 6,304 | 6,043 | 79 | 261 | 1.29% | 0.9586 | 0.9871 | 0.9726 | negative (36) |
| negative | 1,863 | 1,796 | 1,654 | 209 | 142 | 11.22% | 0.9209 | 0.8878 | 0.9041 | not_mentioned (121) |
| neutral | 2,425 | 2,292 | 1,990 | 435 | 302 | 17.94% | 0.8682 | 0.8206 | 0.8438 | positive (297) |
| positive | 2,956 | 2,974 | 2,592 | 364 | 382 | 12.31% | 0.8716 | 0.8769 | 0.8742 | neutral (256) |

## Confusion Directions

| True | Predicted | Count | Share of aspect errors |
|---|---|---:|---:|
| neutral | positive | 297 | 27.32% |
| positive | neutral | 256 | 23.55% |
| negative | not_mentioned | 121 | 11.13% |
| positive | not_mentioned | 84 | 7.73% |
| neutral | negative | 82 | 7.54% |
| neutral | not_mentioned | 56 | 5.15% |
| negative | positive | 49 | 4.51% |
| negative | neutral | 39 | 3.59% |
| not_mentioned | negative | 36 | 3.31% |
| not_mentioned | positive | 36 | 3.31% |
| positive | negative | 24 | 2.21% |
| not_mentioned | neutral | 7 | 0.64% |

## Evidence-based Priority

Start manual review with `neutral` targets, then inspect the largest confusion direction above. The detailed error CSV retains stable IDs and review text for every row.
