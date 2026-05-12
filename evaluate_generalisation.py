#!/usr/bin/env python3
"""Additional validation analyses for PathwayML-Ath.

This script adds two thesis-strength checks that were not part of the
original Claude manuscript:

1. Per-negative-type performance: after fitting the no-embedding XGBoost model,
   evaluate positives against each negative class separately. This tests whether
   the reported AUROC is driven by easy random negatives.
2. Leave-one-family-out (LOFO) validation: hold out one broad pathway family at
   a time, train on all other curated pathways plus mixed hard negatives, and
   test whether held-out pathways still receive pathway-like scores.

The script is deliberately conservative. GO feature selection is always fitted
on the training records only. No dense embedding features are used.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import run_no_embedding_reproducible as core

TABLE_DIR = Path("tables")
FIG_DIR = Path("figures")
MULTI_REPRO = TABLE_DIR / "multiseed" / "reproducibility"


def read_json(path: Path) -> Any:
    """Read and parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def fit_xgb_from_records(
    data: core.DataBundle,
    records: Sequence[Dict[str, Any]],
    train_ids: Sequence[str],
    selected_go: Sequence[str],
    seed: int,
):
    """Fit the canonical XGBoost model using saved samples/split/GO list."""
    x_all, feature_names, groups = core.build_feature_matrix(records, selected_go, data, seed=seed)
    y_all = np.asarray([int(r["label"]) for r in records])
    id_to_idx = {str(r["id"]): i for i, r in enumerate(records)}
    train_idx = np.asarray([id_to_idx[str(rid)] for rid in train_ids if str(rid) in id_to_idx], dtype=int)
    model = core.make_models(seed)["XGBoost"]
    model.fit(x_all[train_idx], y_all[train_idx])
    pred = model.predict_proba(x_all)[:, 1]
    return pred, y_all, feature_names


def evaluate_per_negative_type(data: core.DataBundle, seeds: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the model against each negative type using saved multiseed artifacts.

    For each seed, loads pre-saved samples/splits/GO terms, re-fits XGBoost,
    and evaluates positives vs each negative type separately.
    """
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        repro = MULTI_REPRO / f"seed_{seed:04d}"
        if not repro.exists():
            raise FileNotFoundError(f"Missing multiseed reproducibility directory: {repro}")
        records = read_json(repro / "samples.json")
        splits = read_json(repro / "splits.json")
        selected_go = read_json(repro / "selected_go_terms.json")
        pred, y_all, _names = fit_xgb_from_records(data, records, splits["train_ids"], selected_go, seed=seed)
        test_ids = [str(x) for x in splits["test_ids"]]
        id_to_idx = {str(r["id"]): i for i, r in enumerate(records)}
        test_idx = [id_to_idx[x] for x in test_ids if x in id_to_idx]
        test_pos_idx = [i for i in test_idx if records[i]["label"] == 1]
        neg_types = sorted({str(records[i].get("type", "unknown")) for i in test_idx if records[i]["label"] == 0})
        for neg_type in neg_types:
            neg_idx = [i for i in test_idx if records[i]["label"] == 0 and records[i].get("type") == neg_type]
            subset = test_pos_idx + neg_idx
            if len(neg_idx) == 0 or len(test_pos_idx) == 0:
                continue
            y = y_all[subset]
            p = pred[subset]
            labels = (p >= 0.5).astype(int)
            rows.append({
                "seed": seed,
                "negative_type": neg_type,
                "n_positive": int(len(test_pos_idx)),
                "n_negative": int(len(neg_idx)),
                "test_auroc": float(roc_auc_score(y, p)),
                "test_auprc": float(average_precision_score(y, p)),
                "test_f1": float(f1_score(y, labels)),
                "positive_score_mean": float(np.mean(pred[test_pos_idx])),
                "negative_score_mean": float(np.mean(pred[neg_idx])),
            })
    raw = pd.DataFrame(rows)
    summary_rows: List[Dict[str, Any]] = []
    for neg_type, g in raw.groupby("negative_type"):
        row = {
            "negative_type": neg_type,
            "n_runs": int(len(g)),
            "n_positive_mean": float(g["n_positive"].mean()),
            "n_negative_mean": float(g["n_negative"].mean()),
        }
        for metric in ["test_auroc", "test_auprc", "test_f1", "positive_score_mean", "negative_score_mean"]:
            vals = g[metric].astype(float)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_min"] = float(vals.min())
            row[f"{metric}_max"] = float(vals.max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("test_auroc_mean")
    return raw, summary


# Regex-based family classification for AraCyc pathways (order matters: first match wins)
ARACYC_PATTERNS = [
    ("Biosynthesis (specialized)", [r"biosynth", r"synthesis", r"formation", r"glucosinolate", r"flavonoid", r"phenylpropanoid", r"alkaloid", r"terpenoid", r"anthocyanin", r"lignin"]),
    ("Degradation", [r"degrad", r"catabol", r"breakdown", r"salvage"]),
    ("Amino acid metabolism", [r"amino", r"alanine", r"arginine", r"aspart", r"cysteine", r"glutamate", r"glutamine", r"glycine", r"histidine", r"isoleucine", r"leucine", r"lysine", r"methionine", r"phenylalanine", r"proline", r"serine", r"threonine", r"tryptophan", r"tyrosine", r"valine"]),
    ("Lipid metabolism", [r"lipid", r"fatty", r"acyl", r"glycerol", r"phospholipid", r"sterol", r"wax", r"cutin", r"suberin"]),
    ("Carbohydrate metabolism", [r"carbohydrate", r"starch", r"sucrose", r"sugar", r"glucose", r"fructose", r"cellulose", r"pectin", r"hemicellulose", r"glycan"]),
    ("Nucleotide metabolism", [r"nucleotide", r"purine", r"pyrimidine"]),
    ("Hormone/signaling", [r"hormone", r"auxin", r"ethylene", r"jasmon", r"cytokinin", r"abscisic", r"gibberell", r"signaling", r"signal"]),
    ("Detoxification", [r"detox", r"glutathione", r"oxidative", r"stress", r"xenobiotic"]),
    ("Energy metabolism", [r"photosynthesis", r"respiration", r"electron transport"]),
]

# Keyword-based family classification for KEGG pathways
KEGG_KEYWORDS = [
    ("Carbohydrate metabolism", ["carbohydrate", "glycolysis", "citrate", "pentose", "fructose", "galactose", "starch", "sucrose", "pyruvate", "glyoxylate", "propanoate", "butanoate"]),
    ("Energy metabolism", ["energy", "photosynthesis", "carbon fixation", "oxidative phosphorylation", "methane", "nitrogen", "sulfur"]),
    ("Lipid metabolism", ["lipid", "fatty acid", "glycerolipid", "glycerophospholipid", "sphingolipid", "steroid", "sterol", "cutin", "suberine", "wax"]),
    ("Nucleotide metabolism", ["nucleotide", "purine", "pyrimidine"]),
    ("Amino acid metabolism", ["amino", "alanine", "arginine", "aspartate", "cysteine", "methionine", "glycine", "serine", "threonine", "lysine", "histidine", "phenylalanine", "tyrosine", "tryptophan", "valine", "leucine", "isoleucine", "glutathione"]),
    ("Glycan metabolism", ["glycan", "glycosyl", "glycosphingolipid"]),
    ("Specialized metabolism", ["secondary", "terpenoid", "polyketide", "phenylpropanoid", "flavonoid", "alkaloid", "glucosinolate", "isoquinoline", "indole", "quinone"]),
    ("Signaling and transport", ["signal", "hormone", "transport", "circadian", "environmental adaptation", "plant-pathogen"]),
    ("Genetic information processing", ["ribosome", "spliceosome", "rna", "dna", "replication", "repair", "translation", "transcription", "proteasome", "ubiquitin"]),
]


def assign_family(pid: str, name: str) -> str:
    text = (name or pid).lower()
    if pid.startswith("AC_"):
        for fam, patterns in ARACYC_PATTERNS:
            if any(re.search(p, text) for p in patterns):
                return fam
        return "Other (AraCyc)"
    for fam, patterns in KEGG_KEYWORDS:
        if any(p in text for p in patterns):
            return fam
    return "Other (KEGG)"


def family_table(data: core.DataBundle) -> pd.DataFrame:
    rows = []
    for pid, genes in data.pathways.items():
        src = "AraCyc" if pid.startswith("AC_") else "KEGG"
        name = data.pathway_names.get(pid, pid)
        fam = assign_family(pid, name)
        rows.append({"pathway_id": pid, "source": src, "name": name, "family": fam, "n_genes": len(genes)})
    return pd.DataFrame(rows)


def balanced_test_negatives(
    rng: np.random.Generator,
    records: Sequence[Dict[str, Any]],
    train_negative_ids: set,
    n_needed: int,
) -> List[str]:
    """Select type-balanced negative IDs for the test set, avoiding training negatives."""
    candidates_by_type: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        if r["label"] == 0 and r["id"] not in train_negative_ids:
            candidates_by_type[str(r.get("type", "unknown"))].append(r["id"])
    selected: List[str] = []
    types = sorted(candidates_by_type)
    if not types:
        return selected
    per_type = max(1, int(np.ceil(n_needed / len(types))))
    for t in types:
        vals = candidates_by_type[t]
        if not vals:
            continue
        choose = min(per_type, len(vals), n_needed - len(selected))
        if choose > 0:
            selected.extend([vals[int(i)] for i in rng.choice(len(vals), size=choose, replace=False)])
        if len(selected) >= n_needed:
            break
    if len(selected) < n_needed:
        remaining = [r["id"] for r in records if r["label"] == 0 and r["id"] not in set(selected) and r["id"] not in train_negative_ids]
        if remaining:
            choose = min(n_needed - len(selected), len(remaining))
            selected.extend([remaining[int(i)] for i in rng.choice(len(remaining), size=choose, replace=False)])
    return selected


def evaluate_lofo(
    data: core.DataBundle,
    seed: int,
    min_family_size: int,
    max_families: int | None,
    cv_splits: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """LOFO validation using pre-generated negative pools with training-only GO selection."""
    fam_df = family_table(data)
    fam_summary = fam_df.groupby("family").agg(
        n_pathways=("pathway_id", "count"),
        n_kegg=("source", lambda s: int((s == "KEGG").sum())),
        n_aracyc=("source", lambda s: int((s == "AraCyc").sum())),
        median_size=("n_genes", "median"),
    ).reset_index().sort_values("n_pathways", ascending=False)
    fam_summary.to_csv(TABLE_DIR / "pathway_family_distribution.csv", index=False)

    candidate_families = fam_summary[fam_summary["n_pathways"] >= min_family_size]["family"].tolist()
    if max_families:
        candidate_families = candidate_families[:max_families]

    records, sample_meta = core.build_samples(data, seed=seed + 7000)
    id_to_record = {r["id"]: r for r in records}
    neg_ids_all = [r["id"] for r in records if r["label"] == 0]
    rng = np.random.default_rng(seed + 9000)
    rows: List[Dict[str, Any]] = []
    for fam in candidate_families:
        heldout_pos_ids = fam_df.loc[fam_df["family"] == fam, "pathway_id"].tolist()
        heldout_pos_ids = [pid for pid in heldout_pos_ids if pid in id_to_record]
        if len(heldout_pos_ids) < min_family_size:
            continue
        train_pos_ids = [pid for pid in data.pathways if pid not in set(heldout_pos_ids)]
        n_test_neg = min(len(heldout_pos_ids) * 2, len(neg_ids_all) // 4)
        test_neg_ids = balanced_test_negatives(rng, records, train_negative_ids=set(), n_needed=n_test_neg)
        train_neg_pool = [nid for nid in neg_ids_all if nid not in set(test_neg_ids)]
        n_train_neg = min(len(train_pos_ids) * 2, len(train_neg_pool))
        train_neg_ids = [train_neg_pool[int(i)] for i in rng.choice(len(train_neg_pool), size=n_train_neg, replace=False)]
        train_ids = train_pos_ids + train_neg_ids
        test_ids = heldout_pos_ids + test_neg_ids
        train_records = [id_to_record[i] for i in train_ids]
        selected_go, _fs, _mi = core.select_go_terms(train_records, data, cv_splits=cv_splits, seed=seed)
        x_all, feature_names, _groups = core.build_feature_matrix(records, selected_go, data, seed=seed)
        y_all = np.asarray([int(r["label"]) for r in records])
        ridx = {r["id"]: i for i, r in enumerate(records)}
        train_idx = np.asarray([ridx[i] for i in train_ids], dtype=int)
        test_idx = np.asarray([ridx[i] for i in test_ids], dtype=int)
        model = core.make_models(seed)["XGBoost"]
        model.fit(x_all[train_idx], y_all[train_idx])
        pred = model.predict_proba(x_all[test_idx])[:, 1]
        y = y_all[test_idx]
        labels = (pred >= 0.5).astype(int)
        pos_scores = pred[:len(heldout_pos_ids)]
        neg_scores = pred[len(heldout_pos_ids):]
        rows.append({
            "family": fam,
            "n_heldout_pathways": int(len(heldout_pos_ids)),
            "median_pathway_size": float(fam_df.loc[fam_df["family"] == fam, "n_genes"].median()),
            "n_train_positive": int(len(train_pos_ids)),
            "n_train_negative": int(len(train_neg_ids)),
            "n_test_negative": int(len(test_neg_ids)),
            "n_selected_go": int(len(selected_go)),
            "D": int(len(feature_names)),
            "test_auroc": float(roc_auc_score(y, pred)),
            "test_auprc": float(average_precision_score(y, pred)),
            "test_f1": float(f1_score(y, labels)),
            "heldout_positive_score_mean": float(np.mean(pos_scores)),
            "test_negative_score_mean": float(np.mean(neg_scores)),
        })
    lofo = pd.DataFrame(rows).sort_values("test_auroc")
    return fam_summary, lofo


def make_plots(neg_summary: pd.DataFrame, family_distribution: pd.DataFrame, lofo: pd.DataFrame) -> None:
    """Generate Fig 11 (negative-type performance), Fig 12 (family distribution), Fig 13 (LOFO)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Per-negative-type bar plot
    plt.figure(figsize=(7.5, 4.5))
    df = neg_summary.sort_values("test_auroc_mean")
    plt.bar(df["negative_type"], df["test_auroc_mean"], yerr=df["test_auroc_sd"], capsize=4)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Test AUROC (mean across seeds)")
    plt.title("Performance by negative type")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_negative_type_performance.png", dpi=300)
    plt.savefig(FIG_DIR / "fig11_negative_type_performance.pdf")
    plt.close()

    # Family distribution
    top = family_distribution.sort_values("n_pathways", ascending=False).head(12).iloc[::-1]
    plt.figure(figsize=(8.2, 5.0))
    plt.barh(top["family"], top["n_pathways"])
    plt.xlabel("Number of curated pathways")
    plt.title("Broad pathway-family distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig12_family_distribution.png", dpi=300)
    plt.savefig(FIG_DIR / "fig12_family_distribution.pdf")
    plt.close()

    # LOFO AUROC
    if not lofo.empty:
        plot_df = lofo.sort_values("test_auroc").copy()
        plt.figure(figsize=(8.5, 5.0))
        plt.barh(plot_df["family"], plot_df["test_auroc"])
        plt.xlim(0.0, 1.0)
        plt.xlabel("Held-out family AUROC")
        plt.title("Leave-one-family-out validation")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig13_lofo_performance.png", dpi=300)
        plt.savefig(FIG_DIR / "fig13_lofo_performance.pdf")
        plt.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1-20", help="Seed list for per-negative-type evaluation, e.g. 1-20")
    parser.add_argument("--lofo-seed", type=int, default=42)
    parser.add_argument("--lofo-min-family-size", type=int, default=10)
    parser.add_argument("--lofo-max-families", type=int, default=8)
    parser.add_argument("--lofo-cv-splits", type=int, default=3)
    args = parser.parse_args(argv)

    TABLE_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    data = core.load_data()
    seeds = core.parse_seed_list(args.seeds)

    neg_raw, neg_summary = evaluate_per_negative_type(data, seeds)
    neg_raw.to_csv(TABLE_DIR / "negative_type_performance_by_seed.csv", index=False)
    neg_summary.to_csv(TABLE_DIR / "negative_type_performance_summary.csv", index=False)

    family_dist, lofo = evaluate_lofo(
        data=data,
        seed=args.lofo_seed,
        min_family_size=args.lofo_min_family_size,
        max_families=args.lofo_max_families,
        cv_splits=args.lofo_cv_splits,
    )
    lofo.to_csv(TABLE_DIR / "lofo_performance.csv", index=False)

    make_plots(neg_summary, family_dist, lofo)
    print("Wrote:")
    print("  tables/negative_type_performance_summary.csv")
    print("  tables/lofo_performance.csv")
    print("  figures/fig11_negative_type_performance.png")
    print("  figures/fig12_family_distribution.png")
    print("  figures/fig13_lofo_performance.png")


if __name__ == "__main__":
    main()
