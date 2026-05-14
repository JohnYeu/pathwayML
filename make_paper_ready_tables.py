#!/usr/bin/env python3
"""Create rounded paper-facing tables from regenerated analysis outputs.

The analysis scripts save detailed machine-readable CSV/JSON files. This helper
adds compact rounded versions for paper drafting and rewrites
`RECOMPUTED_RESULTS_HANDOFF.md` so ChatGPT/Claude can use a single current
handoff without mixing numbers from older negative-sampling schemes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────
TABLE_DIR = Path("tables")
OUTPUT_DIR = Path("outputs")
FIG_DIR = Path("figures")
HANDOFF = Path("RECOMPUTED_RESULTS_HANDOFF.md")  # human/AI readable summary of all key results


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, failing early with a clear path if the prerequisite script was not run."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_json(path: Path, payload: Any) -> None:
    """Write JSON with deterministic formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def round_numeric(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Round all numeric columns to `digits` decimals for paper presentation."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(digits)
    return out


def mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    """Format mean +/- SD for paper-facing compact tables."""
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def make_tables() -> Dict[str, Any]:
    """Round raw analysis CSVs into paper-ready copies and build a key-results JSON.

    Each source table (table1-table8) comes from an upstream analysis script.
    This function never recomputes results -- it only reformats.
    """
    outputs: Dict[str, Any] = {}

    # ── Per-table rounding and aliasing ─────────────────────────────────
    perf = read_csv(TABLE_DIR / "table1_seed42_performance.csv")
    perf.to_csv(TABLE_DIR / "paper_table_model_performance_rounded.csv", index=False)

    ablation = round_numeric(read_csv(TABLE_DIR / "table3_ablation.csv"))
    ablation.to_csv(TABLE_DIR / "paper_table_ablation_rounded.csv", index=False)

    neg_type = round_numeric(read_csv(TABLE_DIR / "table5_negative_type_performance.csv"))
    neg_type.to_csv(TABLE_DIR / "paper_table_negative_type_performance_rounded.csv", index=False)

    lofo = round_numeric(read_csv(TABLE_DIR / "table7_lofo_generalization.csv"))
    lofo.to_csv(TABLE_DIR / "paper_table_lofo_recomputed_rounded.csv", index=False)

    multiseed = read_csv(TABLE_DIR / "table2_multiseed_summary.csv")
    multiseed_round = round_numeric(multiseed)
    multiseed_round.to_csv(TABLE_DIR / "paper_table_multiseed_summary_rounded.csv", index=False)

    candidate = round_numeric(read_csv(TABLE_DIR / "candidate_uncertainty_summary.csv"))
    candidate.to_csv(TABLE_DIR / "paper_table_candidate_uncertainty_rounded.csv", index=False)

    go_stability = round_numeric(read_csv(TABLE_DIR / "table4_go_selection_stability.csv"))
    go_stability.to_csv(TABLE_DIR / "paper_table_go_selection_stability_rounded.csv", index=False)

    design = round_numeric(read_csv(TABLE_DIR / "negative_design_summary.csv"))
    design.to_csv(TABLE_DIR / "paper_table_negative_design_summary_rounded.csv", index=False)

    # ── Reproducibility manifest: one-stop lookup for all paper tables ──
    paper_manifest = pd.DataFrame(
        [
            {"artifact": "model_performance", "path": "tables/paper_table_model_performance_rounded.csv"},
            {"artifact": "multiseed_summary", "path": "tables/paper_table_multiseed_summary_rounded.csv"},
            {"artifact": "ablation", "path": "tables/paper_table_ablation_rounded.csv"},
            {"artifact": "go_selection_stability", "path": "tables/paper_table_go_selection_stability_rounded.csv"},
            {"artifact": "negative_type_performance", "path": "tables/paper_table_negative_type_performance_rounded.csv"},
            {"artifact": "size_only_by_negative_type", "path": "tables/paper_size_only_baseline_by_negative_type_compact.csv"},
            {"artifact": "lofo_generalization", "path": "tables/paper_table_lofo_recomputed_rounded.csv"},
            {"artifact": "candidate_uncertainty", "path": "tables/paper_table_candidate_uncertainty_rounded.csv"},
            {"artifact": "supplementary_model_comparison", "path": "tables/paper_supplementary_13_model_comparison_rounded.csv"},
            {"artifact": "negative_ratio_sensitivity", "path": "tables/paper_supplementary_negative_ratio_summary_rounded.csv"},
            {"artifact": "negative_metadata", "path": "tables/negative_metadata.csv"},
            {"artifact": "intermediate_arrays", "path": "outputs/intermediate/seed_XXXX/model_input_arrays.npz"},
        ]
    )
    paper_manifest.to_csv(TABLE_DIR / "paper_reproducibility_manifest.csv", index=False)

    # ── Assemble key numbers for the handoff document ───────────────────
    results = json.loads((TABLE_DIR / "results_no_embedding.json").read_text(encoding="utf-8"))
    xgb = results["performance"]["XGBoost"]
    xgb_ms = multiseed.loc[multiseed["model"] == "XGBoost"].iloc[0]
    go = go_stability.iloc[0]
    # "All mixed negatives" is the primary 4-decoy evaluation condition
    neg_mixed = neg_type.loc[neg_type["comparison"] == "All mixed negatives"].iloc[0]
    size = read_csv(TABLE_DIR / "table6_size_only_by_negative_type.csv")
    size_mixed = size.loc[size["comparison"] == "All mixed negatives"].iloc[0]

    outputs = {
        "generated_by": "make_paper_ready_tables.py",
        "negative_scheme": [
            "empirical_size_matched_random",
            "full_replacement_shuffled",
            "corrupted_pathway",
            "cross_pathway_mixture",
        ],
        "dataset": results["dataset"],
        "seed42_xgboost": {
            "n_go_selected": results["dataset"]["n_go_selected"],
            "D": results["dataset"]["D"],
            "cv_auroc_mean": xgb["cv_auroc_mean"],
            "cv_auroc_se": xgb["cv_auroc_se"],
            "test_auroc": xgb["test_auroc"],
            "test_auprc": xgb["test_auprc"],
            "test_f1": xgb["test_f1"],
        },
        "multiseed_xgboost": {
            "test_auroc_mean": float(xgb_ms["test_auroc_mean"]),
            "test_auroc_sd": float(xgb_ms["test_auroc_sd"]),
            "test_auprc_mean": float(xgb_ms["test_auprc_mean"]),
            "test_auprc_sd": float(xgb_ms["test_auprc_sd"]),
            "n_go_selected_min": float(xgb_ms["n_go_selected_min"]),
            "n_go_selected_max": float(xgb_ms["n_go_selected_max"]),
        },
        "go_selection_stability": go.to_dict(),
        "negative_type_mixed_seed42": neg_mixed.to_dict(),
        "lofo": {
            "auroc_min": float(lofo["test_auroc"].min()),
            "auroc_max": float(lofo["test_auroc"].max()),
            "auprc_min": float(lofo["test_auprc"].min()),
            "auprc_max": float(lofo["test_auprc"].max()),
        },
        "size_only_mixed": size_mixed.to_dict(),
    }
    save_json(TABLE_DIR / "paper_key_results_recomputed.json", outputs)
    return outputs


def write_handoff(key: Dict[str, Any]) -> None:
    """Rewrite the AI/human handoff doc so it always reflects the latest run."""
    dataset = key["dataset"]
    xgb = key["seed42_xgboost"]
    ms = key["multiseed_xgboost"]
    go = key["go_selection_stability"]
    lofo = key["lofo"]
    size = key["size_only_mixed"]

    text = f"""# Recomputed Results Handoff

Generated locally: {datetime.now().strftime('%Y-%m-%d %H:%M')}

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

- Dataset: {dataset['n_pathways']} pathways ({dataset['n_kegg']} KEGG, {dataset['n_aracyc']} AraCyc), {dataset['n_genes']} genes with GO, {dataset['n_go_filtered']} filtered GO terms.
- Canonical negatives: {dataset['negatives']['n_neg']} total at a 1:2 positive:negative ratio.
- Seed-42 features: D={xgb['D']} = {xgb['n_go_selected']} GO + 4 Jaccard + 2 size.
- Seed-42 XGBoost: CV AUROC {xgb['cv_auroc_mean']:.3f} +/- {xgb['cv_auroc_se']:.3f} SE; test AUROC {xgb['test_auroc']:.3f}; test AUPRC {xgb['test_auprc']:.3f}; F1 {xgb['test_f1']:.3f}.
- 20-seed XGBoost: test AUROC {mean_sd(ms['test_auroc_mean'], ms['test_auroc_sd'])} SD; test AUPRC {mean_sd(ms['test_auprc_mean'], ms['test_auprc_sd'])} SD.
- GO selection: selected GO count range {int(go['selected_go_count_min'])}-{int(go['selected_go_count_max'])}; unique selected terms {int(go['n_unique_go_terms_selected_at_least_once'])}; core >=75% terms {int(go['n_core_terms_ge_75pct'])}; recurrent >=50% terms {int(go['n_recurrent_terms_ge_50pct'])}.
- LOFO AUROC range {lofo['auroc_min']:.3f}-{lofo['auroc_max']:.3f}; AUPRC range {lofo['auprc_min']:.3f}-{lofo['auprc_max']:.3f}.
- Size-only all-mixed seed-42 AUROC {float(size['seed42_auroc']):.3f}; 20-seed all-mixed AUROC {mean_sd(float(size['test_auroc_mean']), float(size['test_auroc_sd']))} SD.

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
"""
    HANDOFF.write_text(text, encoding="utf-8")


def main() -> Dict[str, Any]:
    """Entry point: round all tables, write key-results JSON, refresh handoff."""
    key = make_tables()
    write_handoff(key)
    print(f"Wrote {TABLE_DIR / 'paper_key_results_recomputed.json'}")
    print(f"Wrote {HANDOFF}")
    return key


if __name__ == "__main__":
    main()
