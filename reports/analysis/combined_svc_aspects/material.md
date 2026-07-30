# Material Validation Errors

Analysis source: saved `combined_svc` validation predictions. No model inference and no test-set access.

## Summary

- Evaluated targets: 13,367.
- Errors: 1,670 (12.49%).
- Macro F1: 0.863851.
- Weakest label by F1: `negative` (0.8302).
- Highest target error rate: `neutral` (19.67%).

## Error Families

| Family | Count | Share of aspect errors |
|---|---:|---:|
| missed_mention | 513 | 30.72% |
| false_mention | 319 | 19.10% |
| sentiment_confusion | 838 | 50.18% |

## Label Diagnostics

| Label | Support | Predicted | Correct | FN | FP | Error rate | Precision | Recall | F1 | Most often predicted as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| not_mentioned | 5,050 | 5,244 | 4,731 | 319 | 513 | 6.32% | 0.9022 | 0.9368 | 0.9192 | negative (159) |
| negative | 2,636 | 2,618 | 2,181 | 455 | 437 | 17.26% | 0.8331 | 0.8274 | 0.8302 | not_mentioned (218) |
| neutral | 2,120 | 1,916 | 1,703 | 417 | 213 | 19.67% | 0.8888 | 0.8033 | 0.8439 | positive (207) |
| positive | 3,561 | 3,589 | 3,082 | 479 | 507 | 13.45% | 0.8587 | 0.8655 | 0.8621 | not_mentioned (236) |

## Confusion Directions

| True | Predicted | Count | Share of aspect errors |
|---|---|---:|---:|
| positive | not_mentioned | 236 | 14.13% |
| negative | not_mentioned | 218 | 13.05% |
| neutral | positive | 207 | 12.40% |
| not_mentioned | negative | 159 | 9.52% |
| negative | positive | 158 | 9.46% |
| neutral | negative | 151 | 9.04% |
| not_mentioned | positive | 142 | 8.50% |
| positive | negative | 127 | 7.60% |
| positive | neutral | 116 | 6.95% |
| negative | neutral | 79 | 4.73% |
| neutral | not_mentioned | 59 | 3.53% |
| not_mentioned | neutral | 18 | 1.08% |

## Evidence-based Priority

Start manual review with `neutral` targets, then inspect the largest confusion direction above. The detailed error CSV retains stable IDs and review text for every row.
