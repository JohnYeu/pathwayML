# Manuscript Status

`main.tex` is a legacy draft file from Claude's previous run. The previous PDF
has been renamed to `main_legacy_unsynced.pdf` so it is not mistaken for a
fresh build. The manuscript text should not be treated as synchronized with the
corrected code until it is updated from:

```text
../tables/results_no_embedding.json
../tables/table1_performance.csv
../tables/table3_ablation.csv
../tables/candidate_results.csv
```

Critical corrected values:

```text
D = 80
Selected GO terms = 74
XGBoost CV AUROC = 0.952 +/- 0.002 SE
XGBoost Test AUROC = 0.951
C1 novel = false
```
