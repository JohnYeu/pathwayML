# Manuscript Status

`main.tex` has been regenerated from the V7 DOCX manuscript in
`../paper/v7_final/PathwayML_Ath_Paper_V7_recomputed_final.docx`.
It is a LaTeX working copy for editing and audit, while the DOCX/PDF files in
`../paper/v7_final/` remain the original V7 paper files from ChatGPT.

The previous LaTeX draft was preserved as `main_legacy_pre_v7.tex.bak`, and the
older PDF remains `main_legacy_unsynced.pdf`.

The V7 LaTeX file should be checked against these current recomputed tables:

```text
../tables/results_no_embedding.json
../tables/table1_performance.csv
../tables/table3_ablation.csv
../tables/candidate_results.csv
../tables/paper_reproducibility_manifest.csv
../tables/paper_key_results_recomputed.json
../tables/paper_table_lofo_recomputed_rounded.csv
```

Current canonical values:

```text
D = 80
Selected GO terms = 74
XGBoost CV AUROC = 0.952 +/- 0.002 SE
XGBoost Test AUROC = 0.948
XGBoost Test AUPRC = 0.892
20-seed XGBoost Test AUROC = 0.955 +/- 0.010 SD
Recomputed LOFO AUROC range = 0.861-0.957
C1 novel = false
```

Note: no local LaTeX compiler was available in this environment, so the source
was updated and checked textually but not compiled to a new PDF here.
