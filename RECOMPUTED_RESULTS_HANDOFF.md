# Recomputed Results Handoff

Generated locally: 2026-05-11 17:01:57

This handoff treats the current `data/` directory as frozen input and regenerates code-derived outputs in `tables/` and `figures/`. The LaTeX draft was intentionally not updated; use the CSV/JSON files listed below as the source of truth for a new paper draft.

## Commands Run
- `python -m py_compile *.py`
- `python run_no_embedding_reproducible.py --seed 42 --no-figures`
- `python run_no_embedding_reproducible.py --seeds 1-20 --no-figures`
- `python analyze_stability.py --repo .`
- `python generalization_fast.py`
- `python run_no_embedding_reproducible.py --seed 42`

## Core Recomputed Values

- Dataset: 539 pathways (156 KEGG, 383 AraCyc), 27435 genes with GO, 790 filtered GO terms.
- Seed-42 features: D=80 = 74 GO + 4 Jaccard + 2 size.
- Seed-42 XGBoost: CV AUROC 0.952 +/- 0.002 SE; test AUROC 0.948; test AUPRC 0.892; F1 0.831.
- 20-seed XGBoost: test AUROC 0.955 +/- 0.010 SD; test AUPRC 0.909 +/- 0.024 SD.
- GO selection stability: 184 unique selected GO terms, 57 core terms selected in >=75% seeds, 127 recurrent terms selected in >=50% seeds, selected GO range 54-157.
- Negative-type all mixed negatives: AUROC 0.948, AUPRC 0.892. Chimera: AUROC 0.876, AUPRC 0.911.
- Recomputed LOFO: AUROC range 0.861-0.957; AUPRC range 0.728-0.935.

## Paper-Ready Tables
- `tables/paper_table_model_performance_rounded.csv`
- `tables/paper_table_ablation_rounded.csv`
- `tables/paper_table_negative_type_performance_rounded.csv`
- `tables/paper_table_lofo_recomputed_rounded.csv`
- `tables/paper_table_multiseed_summary_rounded.csv`
- `tables/paper_table_candidate_uncertainty_rounded.csv`
- `tables/paper_lofo_previous_v6_vs_recomputed.csv`
- `tables/paper_reproducibility_manifest.csv`
- `tables/paper_key_results_recomputed.json`

## Important LOFO Change

The previous V6 paper Table 6 values came from a cached CSV. `generalization_fast.py` now recomputes LOFO by default and writes the recomputed result to both `tables/table8_lofo_generalization.csv` and `tables/table8_lofo_generalization_training_only.csv`. Use `--use-cached-lofo` only for explicit cache inspection, not for producing new manuscript numbers.

For the new paper draft, replace the old LOFO Table 6 with `tables/paper_table_lofo_recomputed_rounded.csv`. The old-vs-new comparison is in `tables/paper_lofo_previous_v6_vs_recomputed.csv`.

## Do Not Use As Current Paper Source

- `latex/main.tex` remains a legacy draft and was not updated in this pass.
- Exploratory LOFO scripts and old partial outputs are retained for audit only; use `generalization_fast.py` and the manifest above for the manuscript workflow.
