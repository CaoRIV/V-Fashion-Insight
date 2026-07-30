# Frozen TF-IDF Baseline

Selected run: `combined_svc`.

Selection used validation mean Macro F1 only. The test split was not accessed.

| Run | Features | Classifier | Class weight | Mean Macro F1 | Exact match | Latency ms/review | Size MB |
|---|---|---|---|---:|---:|---:|---:|
| combined_svc | combined | linear_svc | none | 0.866187 | 0.564049 | 0.0160 | 13.07 |
| combined_svc_balanced | combined | linear_svc | balanced | 0.865870 | 0.559931 | 0.0113 | 13.09 |
| word_logreg | word | logistic_regression | none | 0.865844 | 0.561054 | 0.0024 | 6.56 |
| combined_logreg | combined | logistic_regression | none | 0.865626 | 0.561204 | 0.0103 | 7.34 |
| char_svc | char_wb | linear_svc | none | 0.827175 | 0.476005 | 0.0094 | 4.67 |

## Selection rationale

Selected the highest validation mean Macro F1 across the five aspects; latency, artifact size, and run name are deterministic tie-breakers.

The registry stores the artifact checksum so later loading can verify immutability.
