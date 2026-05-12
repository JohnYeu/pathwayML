# PathwayML-Ath: Reproducible No-Embedding Pipeline

If you are continuing this project in Codex on another machine, read
`START_HERE_MAC_CODEX.md` first.

This repository has been corrected so the main analysis matches the
no-embedding manuscript direction. The canonical entry point is:

```bash
python run_no_embedding_reproducible.py
```

The old notebook outputs mixed an embedding model with no-embedding manuscript
claims. The corrected pipeline removes SVD/UMAP features from the final model
and regenerates tables from one source of truth.

## Reproducible Averaged Run

For the manuscript robustness result, run a fixed list of seeds. This is still
reproducible because the seed list is part of the protocol:

```bash
python run_no_embedding_reproducible.py --seeds 1-20 --no-figures
```

Each seed regenerates hard negatives, the train/test split, GO feature
selection, and model random states. Per-seed samples, splits, selected GO
terms, candidate sets, and feature names are saved under
`tables/multiseed/reproducibility/`.

Regenerated on 2026-05-11 with seeds 1--20:

| Model | Test AUROC (mean +/- SD) | Test AUROC SE | Test AUPRC mean | Test F1 mean |
|---|---:|---:|---:|---:|
| XGBoost | 0.955 +/- 0.010 | 0.002 | 0.909 | 0.839 |
| Random Forest | 0.945 +/- 0.012 | 0.003 | 0.886 | 0.824 |
| Logistic Regression | 0.918 +/- 0.021 | 0.005 | 0.856 | 0.781 |

The exact run-level and summary outputs are:

```text
tables/multiseed/results_no_embedding_multiseed.json
tables/multiseed/multiseed_runs.csv
tables/multiseed/multiseed_summary.csv
tables/multiseed/multiseed_candidate_results.csv
```

## Reference Seed Run

The single-seed reference run is retained for exact inspection and the original
table/figure workflow:

```bash
python run_no_embedding_reproducible.py --seed 42
```

Regenerated on this Mac with Python 3.13 / XGBoost 3.2.0 on 2026-05-11.

| Model | CV AUROC | Test AUROC | Test AUPRC | Test F1 |
|---|---:|---:|---:|---:|
| XGBoost | 0.952 +/- 0.002 SE | 0.948 | 0.892 | 0.831 |
| Random Forest | 0.946 +/- 0.002 SE | 0.940 | 0.879 | 0.821 |
| Logistic Regression | 0.911 +/- 0.003 SE | 0.924 | 0.860 | 0.802 |

Feature vector:

```text
D = 80 = 74 GO frequency features + 4 GO Jaccard features + 2 size features
```

Important candidate result:

```text
C1 score = 0.883, but it is NOT novel in the corrected run:
max overlap fraction = 0.412, ORA significant = true.
```

So the manuscript should not claim that C1 is a novel ORA-invisible module
unless the candidate construction is redesigned and revalidated.

## Reproducibility Fixes

- No embedding features are used in the final model.
- GO feature selection is performed on the training split only.
- Random negatives are generated with a fixed RNG and saved.
- Train/test split IDs are saved.
- Selected GO terms are saved.
- Candidate gene sets are saved.
- Tables are regenerated from `tables/results_no_embedding.json`.

Saved reproducibility artifacts:

```text
tables/reproducibility/samples.json
tables/reproducibility/splits.json
tables/reproducibility/selected_go_terms.json
tables/reproducibility/candidate_gene_sets.json
tables/reproducibility/feature_names.json
```

## Key Outputs

```text
tables/results_no_embedding.json      # complete machine-readable results
tables/final_no_emb.json              # compact result summary
tables/table1_performance.csv         # model performance
tables/table2_shap_importance.csv     # no-embedding SHAP features
tables/table3_ablation.csv            # no-embedding ablation
tables/feature_selection_cv.csv       # k selection on training split
tables/selected_go_terms.csv          # MI-ranked GO terms
tables/candidate_results.csv          # deterministic candidate scoring
tables/multiseed/                     # fixed-seed-list averaged robustness outputs
tables/paper_reproducibility_manifest.csv  # map paper components to source files
tables/paper_key_results_recomputed.json   # compact recomputed values for paper writing
```

For the latest paper-writing handoff, read `RECOMPUTED_RESULTS_HANDOFF.md`.

## Optional Embedding Comparison

`embedding_comparison.py` is retained only as a separate optional comparison
script. It is not part of the final no-embedding model and should not be used
to populate the main manuscript tables.

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.12 and XGBoost 3.2.0.
