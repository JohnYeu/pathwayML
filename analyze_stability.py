#!/usr/bin/env python3
"""Post-run stability analysis for PathwayML-Ath.

This script does not retrain the model. It analyzes the reproducibility
artifacts produced by

    python run_no_embedding_reproducible.py --seeds 1-20 --no-figures

and generates thesis-ready tables/figures for two questions:

1. Why does the number of selected GO terms vary across seeds?
2. Are candidate scores and test performance stable despite feature-set changes?

Outputs are written to tables/ and figures/.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json(path: Path):
    """Read and parse a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_go_names(repo: Path) -> Dict[str, str]:
    """Load human-readable GO term names for annotation in outputs."""
    candidates = [repo / "original_data" / "go_term_names.json"]
    for path in candidates:
        if path.exists():
            return load_json(path)
    return {}


def selected_terms_by_seed(repro_root: Path) -> Dict[int, List[str]]:
    """Load selected GO terms for each seed from multiseed reproducibility artifacts."""
    data: Dict[int, List[str]] = {}
    for path in sorted(repro_root.glob("seed_*/selected_go_terms.json")):
        seed_text = path.parent.name.replace("seed_", "")
        try:
            seed = int(seed_text)
        except ValueError:
            continue
        terms = load_json(path)
        data[seed] = list(terms)
    if not data:
        raise FileNotFoundError(f"No per-seed selected_go_terms.json found under {repro_root}")
    return data


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets; returns 1.0 for two empty sets."""
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def stability_tables(repo: Path, seed_terms: Dict[int, List[str]]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compute GO-term selection stability across seeds.

    Returns:
        freq_df: per-GO-term selection frequency and stability class
            (core >= 75%, recurrent >= 50%, variable < 50%)
        pair_df: pairwise Jaccard similarity of selected GO sets between seeds
        summary: aggregate statistics dict
    """
    go_names = load_go_names(repo)
    seeds = sorted(seed_terms)
    n = len(seeds)
    all_terms = sorted(set().union(*(set(v) for v in seed_terms.values())))
    rows = []
    for term in all_terms:
        selected_in = [seed for seed in seeds if term in set(seed_terms[seed])]
        rows.append(
            {
                "go_term": term,
                "name": go_names.get(term, ""),
                "n_selected": len(selected_in),
                "selection_frequency": len(selected_in) / n,
                "selected_seeds": ";".join(str(seed) for seed in selected_in),
                "stability_class": (
                    "core" if len(selected_in) >= 0.75 * n else
                    "recurrent" if len(selected_in) >= 0.50 * n else
                    "variable"
                ),
            }
        )
    freq_df = pd.DataFrame(rows).sort_values(
        ["selection_frequency", "n_selected", "go_term"], ascending=[False, False, True]
    )

    pair_rows = []
    for s1, s2 in combinations(seeds, 2):
        a, b = set(seed_terms[s1]), set(seed_terms[s2])
        pair_rows.append({"seed_a": s1, "seed_b": s2, "selected_go_jaccard": jaccard(a, b)})
    pair_df = pd.DataFrame(pair_rows)

    summary = {
        "n_seeds": n,
        "seed_list": seeds,
        "n_unique_go_terms_selected_at_least_once": int(len(all_terms)),
        "selected_go_count_mean": float(np.mean([len(seed_terms[s]) for s in seeds])),
        "selected_go_count_sd": float(np.std([len(seed_terms[s]) for s in seeds], ddof=1)),
        "selected_go_count_min": int(min(len(seed_terms[s]) for s in seeds)),
        "selected_go_count_max": int(max(len(seed_terms[s]) for s in seeds)),
        "n_core_terms_ge_75pct": int((freq_df["selection_frequency"] >= 0.75).sum()),
        "n_recurrent_terms_ge_50pct": int((freq_df["selection_frequency"] >= 0.50).sum()),
        "pairwise_go_set_jaccard_mean": float(pair_df["selected_go_jaccard"].mean()),
        "pairwise_go_set_jaccard_sd": float(pair_df["selected_go_jaccard"].std(ddof=1)),
        "pairwise_go_set_jaccard_min": float(pair_df["selected_go_jaccard"].min()),
        "pairwise_go_set_jaccard_max": float(pair_df["selected_go_jaccard"].max()),
    }
    return freq_df, pair_df, summary


def performance_stability(repo: Path) -> pd.DataFrame:
    """Compute Pearson correlation of GO count / dimensionality with AUROC across seeds."""
    path = repo / "tables" / "multiseed" / "multiseed_runs.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    runs = pd.read_csv(path)
    xgb = runs[runs["model"] == "XGBoost"].copy()
    corr_cols = ["n_go_selected", "D"]
    rows = []
    for col in corr_cols:
        rows.append(
            {
                "metric": col,
                "pearson_corr_with_test_auroc": float(xgb[col].corr(xgb["test_auroc"])),
                "pearson_corr_with_cv_auroc": float(xgb[col].corr(xgb["cv_auroc_mean"])),
            }
        )
    return pd.DataFrame(rows)


def candidate_uncertainty(repo: Path) -> pd.DataFrame:
    """Summarize candidate pathway scores across seeds (mean, SD, min, max)."""
    path = repo / "tables" / "multiseed" / "multiseed_candidate_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    cand = pd.read_csv(path)
    summary = (
        cand.groupby("candidate")
        .agg(
            n_runs=("seed", "count"),
            score_mean=("score", "mean"),
            score_sd=("score", "std"),
            score_min=("score", "min"),
            score_max=("score", "max"),
            n_above_0_5=("score", lambda x: int((x >= 0.5).sum())),
            ora_significant_all=("ora_significant", lambda x: bool(x.all())),
            novel_any=("novel", lambda x: bool(x.any())),
            max_overlap_fraction=("max_overlap_fraction", "first"),
            closest_by_overlap=("closest_by_overlap", "first"),
            best_ora_p_adj=("best_ora_p_adj", "first"),
        )
        .reset_index()
    )
    return summary


def plot_go_stability(repo: Path, freq_df: pd.DataFrame, summary: dict) -> None:
    """Bar chart of the top-30 most frequently selected GO features across seeds."""
    fig_dir = repo / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    top = freq_df.head(30).iloc[::-1]
    labels = [f"{row.go_term} {row['name'][:36]}".strip() for _, row in top.iterrows()]
    plt.figure(figsize=(10, 8))
    plt.barh(labels, top["selection_frequency"])
    plt.xlabel("Selection frequency across 20 seeds")
    plt.ylabel("GO term")
    plt.title("Most stable GO frequency features")
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig8_go_selection_stability.png", dpi=300)
    plt.savefig(fig_dir / "fig8_go_selection_stability.pdf")
    plt.close()


def plot_score_uncertainty(repo: Path) -> None:
    """Box plot of candidate pathway scores across 20 seeds."""
    fig_dir = repo / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(repo / "tables" / "multiseed" / "multiseed_candidate_results.csv")
    order = sorted(cand["candidate"].unique())
    data = [cand.loc[cand["candidate"] == c, "score"].values for c in order]
    plt.figure(figsize=(7, 4.5))
    plt.boxplot(data, tick_labels=order, showmeans=True)
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.ylabel("PathwayML-Ath score")
    plt.xlabel("Candidate gene set")
    plt.title("Candidate-score uncertainty across 20 seeds")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig9_candidate_score_uncertainty.png", dpi=300)
    plt.savefig(fig_dir / "fig9_candidate_score_uncertainty.pdf")
    plt.close()


def plot_performance_vs_dimension(repo: Path) -> None:
    """Scatter plot of selected GO count vs test AUROC with linear fit."""
    fig_dir = repo / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(repo / "tables" / "multiseed" / "multiseed_runs.csv")
    xgb = runs[runs["model"] == "XGBoost"].copy()
    plt.figure(figsize=(7, 4.5))
    plt.scatter(xgb["n_go_selected"], xgb["test_auroc"])
    m, b = np.polyfit(xgb["n_go_selected"], xgb["test_auroc"], deg=1)
    xs = np.linspace(xgb["n_go_selected"].min(), xgb["n_go_selected"].max(), 100)
    plt.plot(xs, m * xs + b, linewidth=1)
    plt.xlabel("Selected GO terms")
    plt.ylabel("Held-out test AUROC")
    plt.title("Performance is stable despite variable GO feature counts")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig10_performance_vs_go_count.png", dpi=300)
    plt.savefig(fig_dir / "fig10_performance_vs_go_count.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."), help="PathwayML-Ath repository root")
    args = parser.parse_args()
    repo = args.repo.resolve()
    tables = repo / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    seed_terms = selected_terms_by_seed(repo / "tables" / "multiseed" / "reproducibility")
    freq_df, pair_df, summary = stability_tables(repo, seed_terms)
    perf_corr = performance_stability(repo)
    cand_summary = candidate_uncertainty(repo)

    freq_df.to_csv(tables / "go_selection_stability.csv", index=False)
    pair_df.to_csv(tables / "go_selection_pairwise_jaccard.csv", index=False)
    perf_corr.to_csv(tables / "go_count_performance_correlation.csv", index=False)
    cand_summary.to_csv(tables / "candidate_uncertainty_summary.csv", index=False)
    (tables / "go_selection_stability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plot_go_stability(repo, freq_df, summary)
    plot_score_uncertainty(repo)
    plot_performance_vs_dimension(repo)

    print("Wrote stability outputs:")
    for rel in [
        "tables/go_selection_stability.csv",
        "tables/go_selection_pairwise_jaccard.csv",
        "tables/go_count_performance_correlation.csv",
        "tables/candidate_uncertainty_summary.csv",
        "tables/go_selection_stability_summary.json",
        "figures/fig8_go_selection_stability.png",
        "figures/fig9_candidate_score_uncertainty.png",
        "figures/fig10_performance_vs_go_count.png",
    ]:
        print(f"  {rel}")


if __name__ == "__main__":
    main()
