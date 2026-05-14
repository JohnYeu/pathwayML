#!/usr/bin/env python3
"""Size-only sanity check by negative type.

This supplementary diagnostic tests whether the current benchmark can be
solved mainly by gene-set size. It keeps the current negative-sampling scheme
and train/test split logic, but trains XGBoost using only:

* |G|
* log(1 + |G|)

Outputs are written under `tables/` and `figures/`. This script does not change
the main model, the main pipeline tables, or any manuscript text.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

# --- Third-party imports ---------------------------------------------------
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server/CI rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --- Canonical pipeline (shared data loading, model factory, utils) --------
import run_no_embedding_reproducible as core

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

TABLE_DIR = Path("tables")
FIG_DIR = Path("figures")
NEGATIVE_TYPES = core.PRIMARY_NEGATIVE_TYPES
# Display order for the summary table and figure x-axis.
COMPARISON_ORDER = [
    "All mixed negatives",
    "empirical size matched random",
    "full replacement shuffled",
    "corrupted pathway",
    "cross pathway mixture",
]


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction (size-only)
# ═══════════════════════════════════════════════════════════════════════════


def size_matrix(records: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Return the two size features used in the main pipeline: |G| and log(1+|G|)."""
    return np.vstack([core.size_features(record["genes"]) for record in records]).astype(float)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════


def metric_row(
    seed: int,
    comparison: str,
    labels: np.ndarray,
    scores: np.ndarray,
    sizes: np.ndarray,
) -> Dict[str, Any]:
    """Compute classification and score/size summaries for one comparison.

    Records both classification metrics and score/size distributions so the
    thesis can argue that size alone does not separate positives from negatives.
    """
    pred = (scores >= 0.5).astype(int)
    neg_scores = scores[labels == 0]
    pos_scores = scores[labels == 1]
    neg_sizes = sizes[labels == 0]
    pos_sizes = sizes[labels == 1]
    return {
        "seed": int(seed),
        "comparison": comparison,
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "test_auroc": float(roc_auc_score(labels, scores)),
        "test_auprc": float(average_precision_score(labels, scores)),
        "test_f1": float(f1_score(labels, pred)),
        "test_precision": float(precision_score(labels, pred, zero_division=0)),
        "test_recall": float(recall_score(labels, pred, zero_division=0)),
        # Score distributions help diagnose if the model is guessing.
        "positive_score_median": float(np.median(pos_scores)),
        "negative_score_median": float(np.median(neg_scores)),
        "negative_score_iqr": f"{np.percentile(neg_scores, 25):.3f}-{np.percentile(neg_scores, 75):.3f}",
        # Size distributions verify that size overlap is realistic.
        "positive_size_median": float(np.median(pos_sizes)),
        "negative_size_median": float(np.median(neg_sizes)),
        "negative_size_iqr": f"{np.percentile(neg_sizes, 25):.1f}-{np.percentile(neg_sizes, 75):.1f}",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-seed evaluation
# ═══════════════════════════════════════════════════════════════════════════


def run_seed(data: core.DataBundle, seed: int) -> pd.DataFrame:
    """Train/evaluate a size-only XGBoost model for one full-pipeline seed.

    Reuses the exact same split and negative samples as the full model so
    any difference in AUROC is attributable to feature content, not data.
    """
    records, sample_meta, _split_info, train_idx, test_idx = core.build_split_samples(data, seed=seed)
    if sample_meta.get("negative_multiplier") != 2:
        raise RuntimeError("This sanity check expects the canonical 1:2 negative ratio.")

    y = np.asarray([int(record["label"]) for record in records], dtype=int)

    # Only the two size features: |G| and log(1+|G|).
    x = size_matrix(records)
    # Borrow the same XGBoost hyperparameters as the full model.
    model = clone(core.make_models(seed)["XGBoost"])
    model.fit(x[train_idx], y[train_idx])

    scores = model.predict_proba(x[test_idx])[:, 1]
    labels = y[test_idx]
    sizes = x[test_idx, 0]  # raw |G| for distribution reporting
    types = np.asarray([str(records[int(i)].get("type", "")) for i in test_idx])

    rows: List[Dict[str, Any]] = []
    # Overall mixed-negative evaluation first.
    rows.append(metric_row(seed, "All mixed negatives", labels, scores, sizes))
    # Then one row per negative type (positives are always included).
    pos_mask = labels == 1
    for negative_type in NEGATIVE_TYPES:
        mask = pos_mask | ((labels == 0) & (types == negative_type))
        # Skip if either class is empty in this subset.
        if np.any(labels[mask] == 0) and np.any(labels[mask] == 1):
            rows.append(metric_row(seed, negative_type.replace("_", " "), labels[mask], scores[mask], sizes[mask]))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def summarize(per_seed_df: pd.DataFrame, seed42_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize seeds 1-20 and join the canonical seed-42 diagnostic values.

    Seed-42 values are included as a reference column so the thesis can show
    both the canonical run and the multi-seed mean in the same table.
    """
    rows: List[Dict[str, Any]] = []
    metric_cols = [
        "test_auroc",
        "test_auprc",
        "test_f1",
        "test_precision",
        "test_recall",
        "positive_score_median",
        "negative_score_median",
        "positive_size_median",
        "negative_size_median",
    ]
    seed42 = seed42_df.set_index("comparison")
    for comparison, group in per_seed_df.groupby("comparison", sort=False):
        row: Dict[str, Any] = {
            "comparison": comparison,
            "n_runs": int(len(group)),
            "n_positive_mean": float(group["n_positive"].mean()),
            "n_negative_mean": float(group["n_negative"].mean()),
        }
        # Attach seed-42 reference values for side-by-side comparison.
        if comparison in seed42.index:
            row["seed42_auroc"] = float(seed42.loc[comparison, "test_auroc"])
            row["seed42_auprc"] = float(seed42.loc[comparison, "test_auprc"])
            row["seed42_f1"] = float(seed42.loc[comparison, "test_f1"])
            row["seed42_negative_size_median"] = float(seed42.loc[comparison, "negative_size_median"])
        for metric in metric_cols:
            values = group[metric].astype(float)
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = sd
            row[f"{metric}_se"] = float(sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)

    summary = pd.DataFrame(rows)
    # Sort rows to match the thesis table layout.
    order = {name: i for i, name in enumerate(COMPARISON_ORDER)}
    summary["order"] = summary["comparison"].map(order).fillna(99).astype(int)
    return summary.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


def write_plot(summary_df: pd.DataFrame) -> None:
    """Plot mean AUROC/AUPRC for the size-only diagnostic.

    Low AUROC here supports the thesis claim that GO/Jaccard features, not
    gene-set size, drive the full model's discriminative power.
    """
    plot_df = summary_df.copy()
    x = np.arange(len(plot_df))
    labels = plot_df["comparison"].tolist()

    plt.figure(figsize=(8.5, 4.8))
    # Slight horizontal offset (-0.06 / +0.06) prevents error-bar overlap.
    plt.errorbar(
        x - 0.06,
        plot_df["test_auroc_mean"],
        yerr=plot_df["test_auroc_sd"],
        marker="o",
        capsize=4,
        color="#2B6CB0",
        label="AUROC",
    )
    plt.errorbar(
        x + 0.06,
        plot_df["test_auprc_mean"],
        yerr=plot_df["test_auprc_sd"],
        marker="s",
        capsize=4,
        color="#C53030",
        label="AUPRC",
    )
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Held-out test metric, mean +/- SD")
    plt.title("Size-only sanity check by negative type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "size_only_baseline_by_negative_type.png", dpi=300)
    plt.savefig(FIG_DIR / "size_only_baseline_by_negative_type.pdf")
    # Stable alias for thesis data handoff.
    plt.savefig(FIG_DIR / "fig_size_only_by_negative_type.png", dpi=300)
    plt.close()


def write_compact_paper_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Write a compact paper-facing table with the key sanity-check values.

    Combines mean +/- SD into single formatted strings for direct LaTeX use.
    """
    compact = pd.DataFrame(
        {
            "comparison": summary_df["comparison"],
            "seed42_auroc": summary_df["seed42_auroc"].round(3),
            "seed42_auprc": summary_df["seed42_auprc"].round(3),
            "auroc_mean_sd": [
                f"{mean:.3f} +/- {sd:.3f}"
                for mean, sd in zip(summary_df["test_auroc_mean"], summary_df["test_auroc_sd"])
            ],
            "auprc_mean_sd": [
                f"{mean:.3f} +/- {sd:.3f}"
                for mean, sd in zip(summary_df["test_auprc_mean"], summary_df["test_auprc_sd"])
            ],
            "f1_mean_sd": [
                f"{mean:.3f} +/- {sd:.3f}"
                for mean, sd in zip(summary_df["test_f1_mean"], summary_df["test_f1_sd"])
            ],
            "positive_size_median_mean": summary_df["positive_size_median_mean"].round(1),
            "negative_size_median_mean": summary_df["negative_size_median_mean"].round(1),
            "n_negative_mean": summary_df["n_negative_mean"].round(1),
        }
    )
    compact.to_csv(TABLE_DIR / "paper_size_only_baseline_by_negative_type_compact.csv", index=False)
    return compact


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    """Run the size-only sanity check and write all output tables/figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1-20", help="Fixed seed list/range for mean +/- SD.")
    parser.add_argument("--reference-seed", type=int, default=42)
    args = parser.parse_args(argv)

    core.ensure_dirs()
    data = core.load_data()
    seeds = core.parse_seed_list(args.seeds)

    # Canonical seed-42 run provides the single-seed reference for the thesis.
    print(f"Running size-only seed {args.reference_seed} reference...")
    seed42_df = run_seed(data, seed=args.reference_seed)
    seed42_df.to_csv(TABLE_DIR / "size_only_baseline_by_negative_type_seed42.csv", index=False)

    # Multi-seed runs provide mean +/- SD for robustness reporting.
    rows: List[pd.DataFrame] = []
    for i, seed in enumerate(seeds, start=1):
        print(f"Running size-only seed {seed} ({i}/{len(seeds)})...")
        rows.append(run_seed(data, seed=seed))
    per_seed_df = pd.concat(rows, ignore_index=True)
    per_seed_df.to_csv(TABLE_DIR / "size_only_baseline_by_negative_type_per_seed.csv", index=False)

    summary_df = summarize(per_seed_df, seed42_df)
    summary_df.to_csv(TABLE_DIR / "size_only_baseline_by_negative_type.csv", index=False)
    summary_df.to_csv(TABLE_DIR / "table6_size_only_by_negative_type.csv", index=False)

    # Rounded copy for paper tables; full precision kept in the main CSV.
    rounded = summary_df.copy()
    numeric_cols = rounded.select_dtypes(include=[np.number]).columns
    rounded[numeric_cols] = rounded[numeric_cols].round(3)
    rounded.to_csv(TABLE_DIR / "paper_size_only_baseline_by_negative_type_rounded.csv", index=False)
    compact = write_compact_paper_table(summary_df)

    payload = {
        "generated_by": "size_only_sanity_check.py",
        "feature_set": ["pathway_size", "log_size"],
        "reference_seed": args.reference_seed,
        "seeds": seeds,
        "negative_scheme": "empirical_size_matched_random__full_replacement_shuffled__corrupted_pathway__cross_pathway_mixture",
        "outputs": {
            "seed42": "tables/size_only_baseline_by_negative_type_seed42.csv",
            "per_seed": "tables/size_only_baseline_by_negative_type_per_seed.csv",
            "summary": "tables/size_only_baseline_by_negative_type.csv",
            "paper_numbered_summary": "tables/table6_size_only_by_negative_type.csv",
            "paper_ready": "tables/paper_size_only_baseline_by_negative_type_rounded.csv",
            "paper_ready_compact": "tables/paper_size_only_baseline_by_negative_type_compact.csv",
            "figure_png": "figures/size_only_baseline_by_negative_type.png",
            "paper_numbered_figure_png": "figures/fig_size_only_by_negative_type.png",
            "figure_pdf": "figures/size_only_baseline_by_negative_type.pdf",
        },
        "summary": summary_df.to_dict(orient="records"),
        "note": (
            "This is a supplementary diagnostic only. It trains XGBoost on size "
            "features alone and evaluates whether size can explain the mixed and "
            "per-negative-type contrasts."
        ),
    }
    core.save_json(TABLE_DIR / "size_only_baseline_summary.json", payload)
    write_plot(summary_df)

    print("\nSeed-42 size-only:")
    print(seed42_df.to_string(index=False))
    print("\nMean +/- SD summary:")
    print(rounded.to_string(index=False))
    print("\nCompact paper table:")
    print(compact.to_string(index=False))
    return payload


if __name__ == "__main__":
    main()
