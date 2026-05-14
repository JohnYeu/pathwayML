# PathwayML-Ath: Reproducible No-Embedding Pipeline

If you are continuing this project in Codex on another machine, read
`START_HERE_MAC_CODEX.md` first.

The canonical entry point is:

```bash
python run_no_embedding_reproducible.py
```

The current code removes dense embedding features from the final model and uses
the revised four-class synthetic decoy scheme:

```text
empirical-size-matched random / full-replacement shuffled /
corrupted pathway / cross-pathway mixture
```

The old notebook outputs and older manuscript drafts used different negative
definitions. Do not mix old paper numbers with the current method.

Pure 50-80% pathway subsets are not primary training negatives in the current
pipeline. They are scored only as boundary probes after training because such
subsets may retain real pathway-like GO coherence.

## Reproducible Averaged Run

For the manuscript robustness result, run a fixed list of seeds. This is still
reproducible because the seed list is part of the protocol:

```bash
python run_no_embedding_reproducible.py --seeds 1-20
```

Each seed regenerates negatives, the train/test split, GO feature selection,
and model random states. Per-seed samples, splits, selected GO terms, candidate
sets, and feature names are saved under:

```text
tables/multiseed/reproducibility/
```

Numeric model-input arrays are also saved for audit/replay:

```text
outputs/intermediate/seed_XXXX/model_input_arrays.npz
outputs/intermediate/seed_XXXX/model_input_manifest.json
```

Regenerated on 2026-05-13 with seeds 1-20:

| Model | Test AUROC (mean +/- SD) | Test AUPRC mean | Test F1 mean |
|---|---:|---:|---:|
| XGBoost | 0.983 +/- 0.004 | 0.966 | 0.913 |
| Random Forest | 0.977 +/- 0.006 | 0.949 | 0.889 |
| Logistic Regression | 0.961 +/- 0.009 | 0.931 | 0.859 |

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

Regenerated on this Mac with Python 3.13 / XGBoost 3.2.0 on 2026-05-13.

| Model | CV AUROC | Test AUROC | Test AUPRC | Test F1 |
|---|---:|---:|---:|---:|
| XGBoost | 0.983 +/- 0.002 SE | 0.982 | 0.971 | 0.901 |
| Random Forest | 0.974 +/- 0.005 SE | 0.978 | 0.954 | 0.886 |
| Logistic Regression | 0.959 +/- 0.002 SE | 0.973 | 0.957 | 0.869 |

Feature vector:

```text
D = 76 = 70 GO frequency features + 4 GO Jaccard features + 2 size features
```

## Supplementary Analyses

Two supplementary analyses were added for thesis writing:

```bash
python supplementary_analysis.py --cv-splits 5 --cv-repeats 1 --ratios 1-5 --ratio-seeds 1-5
```

Outputs:

```text
tables/supplementary_model_comparison.csv
tables/supplementary_negative_ratio_sensitivity.csv
tables/paper_supplementary_13_model_comparison_rounded.csv
tables/paper_supplementary_negative_ratio_summary_rounded.csv
tables/supplementary_analysis_summary.json
figures/fig_supp_model_comparison.png
figures/fig_supp_negative_ratio_sensitivity.png
figures/supplementary_13_model_comparison.png
figures/supplementary_negative_ratio_sensitivity.png
```

In the 13-model comparison, advanced boosting models are the strongest
held-out models in the current seed-42 run:

```text
CatBoost test AUROC = 0.985, test AUPRC = 0.973
XGBoost test AUROC = 0.982, test AUPRC = 0.971
```

The negative-ratio sensitivity analysis runs ratios 1:1 to 1:5 over fixed seeds
1-5. It shows AUROC is fairly stable, while AUPRC decreases as the positive
class becomes rarer.

A size-only sanity check was also added:

```bash
python size_only_sanity_check.py --seeds 1-20 --reference-seed 42
```

Outputs:

```text
tables/paper_size_only_baseline_by_negative_type_compact.csv
tables/size_only_baseline_by_negative_type.csv
tables/size_only_baseline_by_negative_type_per_seed.csv
tables/size_only_baseline_summary.json
figures/size_only_baseline_by_negative_type.png
```

Seed-42 all-mixed size-only performance is AUROC 0.516 and AUPRC 0.347.
Across seeds 1-20, all-mixed size-only AUROC is 0.528 +/- 0.016 SD. This means
size alone does not explain the mixed benchmark, although it partially explains
the cross-pathway contrast.

## Reproducibility Fixes

- No embedding features are used in the final model.
- GO feature selection is performed on the training split only.
- Negative samples are generated with fixed RNG seeds and saved.
- Train/test split IDs are saved.
- Selected GO terms are saved.
- Candidate gene sets are saved.
- Tables are regenerated from the current scripts instead of mixed from old runs.

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
tables/results_no_embedding.json
tables/final_no_emb.json
tables/table1_seed42_performance.csv
tables/table2_shap_importance.csv
tables/table3_ablation.csv
tables/table5_negative_type_performance.csv
tables/table7_lofo_generalization.csv
tables/feature_selection_cv.csv
tables/selected_go_terms.csv
tables/candidate_results.csv
tables/multiseed/
tables/paper_reproducibility_manifest.csv
tables/paper_key_results_recomputed.json
tables/paper_size_only_baseline_by_negative_type_compact.csv
```

For the latest paper-writing handoff, read `RECOMPUTED_RESULTS_HANDOFF.md`.

`generalization_fast.py` is the canonical source for the paper-facing
per-negative-type table and LOFO table.

## Optional Embedding Comparison

`embedding_comparison.py` is retained only as a separate optional comparison
script. It is not part of the final no-embedding model and should not be used
to populate the main manuscript tables.

## Requirements

```bash
pip install -r requirements.txt
```

Tested locally with Python 3.13, XGBoost 3.2.0, LightGBM 4.6.0, and CatBoost
1.2.10.
