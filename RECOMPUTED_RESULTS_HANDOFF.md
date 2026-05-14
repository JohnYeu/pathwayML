# Recomputed Results Handoff

Generated locally: 2026-05-13 18:46

This handoff treats the current `data/` directory as frozen input and regenerates
code-derived outputs in `tables/`, `figures/`, and `outputs/`. The current
primary negative-sampling design is:

```text
empirical_size_matched_random / full_replacement_shuffled /
corrupted_pathway / cross_pathway_mixture
```

Pure 50-80% pathway subsets are **not** primary training negatives. They are
saved only as post-training boundary probes.

## Commands

```bash
python3 -m py_compile *.py
python3 run_no_embedding_reproducible.py --seeds 1-20 --cv-repeats 1
python3 analyze_stability.py --repo .
python3 generalization_fast.py
python3 size_only_sanity_check.py --seeds 1-20 --reference-seed 42
python3 supplementary_analysis.py --cv-splits 5 --cv-repeats 1 --ratios 1-5 --ratio-seeds 1-5
python3 finalize_recomputed_outputs.py
python3 make_paper_ready_tables.py
```

## Core Values

- Dataset: 539 pathways (156 KEGG, 383 AraCyc), 27435 genes with GO, 790 filtered GO terms.
- Canonical negatives: 1078 total at a 1:2 positive:negative ratio.
- Seed-42 features: D=76 = 70 GO + 4 Jaccard + 2 size.
- Seed-42 XGBoost: CV AUROC 0.983 +/- 0.002 SE; test AUROC 0.982; test AUPRC 0.971; F1 0.901.
- 20-seed XGBoost: test AUROC 0.983 +/- 0.004 SD; test AUPRC 0.966 +/- 0.010 SD.
- GO selection: selected GO count range 37-155; unique selected terms 179; core >=75% terms 111; recurrent >=50% terms 145.
- LOFO AUROC range 0.969-0.994; AUPRC range 0.926-0.988.
- Size-only all-mixed seed-42 AUROC 0.516; 20-seed all-mixed AUROC 0.528 +/- 0.016 SD.

## Source Tables For Paper Writing

- `tables/paper_table_model_performance_rounded.csv`
- `tables/paper_table_multiseed_summary_rounded.csv`
- `tables/paper_table_ablation_rounded.csv`
- `tables/paper_table_go_selection_stability_rounded.csv`
- `tables/paper_table_negative_type_performance_rounded.csv`
- `tables/paper_table_negative_design_summary_rounded.csv`
- `tables/paper_table_lofo_recomputed_rounded.csv`
- `tables/paper_table_candidate_uncertainty_rounded.csv`
- `tables/paper_supplementary_13_model_comparison_rounded.csv`
- `tables/paper_supplementary_negative_ratio_summary_rounded.csv`
- `tables/paper_size_only_baseline_by_negative_type_compact.csv`
- `tables/negative_metadata.csv`
- `tables/boundary_partial_probe_scores.csv`
- `outputs/intermediate/seed_XXXX/model_input_arrays.npz`
- `outputs/manifest.json`
- `outputs/negative_sampling_audit.json`
- `tables/paper_key_results_recomputed.json`
- `tables/paper_reproducibility_manifest.csv`

## Notes

- `latex/main.tex` remains a legacy draft and is not the current source of truth.
- Do not mix older paper numbers with this handoff; changing the negative
  design changes performance, GO selection, LOFO, candidate scores, and
  supplementary analyses.
