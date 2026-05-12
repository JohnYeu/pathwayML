#!/usr/bin/env python3
"""Additional validation analyses for the PathwayML-Ath thesis paper.

This script extends the canonical no-embedding analysis with two checks requested
for the final thesis version:

1. Per-negative-type evaluation. The model is trained on the same mixed training
   set as the main pipeline and evaluated separately against each negative type
   in the held-out test split.
2. Leave-one-pathway-family-out (LOFO) validation. A coarse pathway family is
   excluded from the positive training set, a model is trained on the remaining
   positive pathways plus newly generated hard negatives, and the excluded family
   is evaluated against size-matched negatives generated specifically for that
   family.

Run from the repository root:

    python additional_validation.py --seeds 1-20

The script writes CSV tables and PNG/PDF figures under tables/ and figures/.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from run_no_embedding_reproducible import (
    DataBundle,
    build_feature_matrix,
    build_samples,
    co_annotation_cluster,
    ensure_dirs,
    jaccard_stats,
    load_data,
    make_models,
    pathway_jaccard_mean,
    rng_choice_list,
    save_json,
    select_go_terms,
)

TABLE_DIR = Path("tables")
FIG_DIR = Path("figures")

# ---------------------------------------------------------------------------
# Keyword lists for heuristic pathway family assignment.
# These define broad, thesis-level categories for LOFO validation.
# ---------------------------------------------------------------------------
AMINO_WORDS = [
    "alanine", "arginine", "asparagine", "aspartate", "cysteine", "glutamine",
    "glutamate", "glutathione", "glycine", "histidine", "isoleucine", "leucine",
    "lysine", "methionine", "ornithine", "phenylalanine", "proline", "serine",
    "threonine", "tryptophan", "tyrosine", "valine", "branched chain", "branched-chain",
    "chorismate", "citrulline", "homoserine",
]
LIPID_WORDS = [
    "lipid", "fatty", "acyl", "phospholipid", "triacylglycerol", "glycerolipid",
    "glycerophospholipid", "sphingolipid", "steroid", "sterol", "wax", "cutin",
    "suberin", "linoleic", "linolenic", "arachidonic",
]
CARBO_WORDS = [
    "carbohydrate", "carbon", "glycolysis", "gluconeogenesis", "glucose", "sucrose",
    "starch", "fructose", "mannose", "galactose", "pentose", "cellulose", "pectin",
    "xylan", "glycan", "glycosyl", "sugar", "calvin", "photosynthesis", "pyruvate",
    "citrate", "tca", "glyoxylate",
]
DEGRAD_WORDS = ["degradation", "catabolism", "degrad", "breakdown", "salvage"]
SIGNAL_WORDS = [
    "signaling", "signalling", "hormone", "auxin", "cytokinin", "gibberellin",
    "ethylene", "jasmonate", "abscisic", "brassinosteroid", "signal transduction",
]
DETOX_WORDS = ["detox", "superoxide", "reactive oxygen", "glutathione", "xenobiotic"]
COFACTOR_WORDS = ["cofactor", "vitamin", "coenzyme", "heme", "chlorophyll", "porphyrin"]
NUCLEOTIDE_WORDS = ["nucleotide", "purine", "pyrimidine", "nucleoside"]
SECONDARY_WORDS = [
    "secondary", "phenylpropanoid", "flavonoid", "terpenoid", "terpene", "alkaloid",
    "glucosinolate", "camalexin", "anthocyanin", "stilbenoid", "betalain", "isoprenoid",
]
CELL_WORDS = [
    "cell cycle", "replication", "repair", "transcription", "translation", "ribosome",
    "proteasome", "ubiquitin", "autophagy", "endocytosis", "transport", "sorting",
    "peroxisome", "spliceosome", "protein processing", "folding",
]


def has_any(text: str, words: Sequence[str]) -> bool:
    """Check if any keyword appears in text (case-sensitive)."""
    return any(word in text for word in words)


def assign_family(pid: str, name: str) -> str:
    """Assign a defensible coarse family using pathway identifiers and names.

    The labels are intentionally coarse because the two pathway databases use
    different classification systems. The goal is to test broad-family
    generalisation, not to reproduce KEGG BRITE exactly.
    """
    text = f"{pid} {name}".lower()
    is_aracyc = pid.startswith("AC_")

    if is_aracyc:
        if has_any(text, AMINO_WORDS):
            return "AraCyc amino acid / glutathione metabolism"
        if has_any(text, LIPID_WORDS):
            return "AraCyc lipid metabolism"
        if has_any(text, CARBO_WORDS):
            return "AraCyc carbohydrate / energy metabolism"
        if has_any(text, DEGRAD_WORDS):
            return "AraCyc degradation"
        if has_any(text, DETOX_WORDS):
            return "AraCyc detoxification / redox"
        if has_any(text, SIGNAL_WORDS):
            return "AraCyc hormone / signalling"
        if "biosynthesis" in text or "biosynthetic" in text or "synthesis" in text:
            return "AraCyc specialized biosynthesis"
        return "AraCyc other"

    # KEGG pathways. Broad maps and disease-like inherited KEGG maps are grouped
    # as KEGG other to avoid over-interpreting small BRITE categories.
    if pid in {"ath01100", "ath01110", "ath01200", "ath01210", "ath01212", "ath01230", "ath01232", "ath01240", "ath01250"}:
        return "KEGG global / overview maps"
    if has_any(text, AMINO_WORDS):
        return "KEGG amino acid / related metabolism"
    if has_any(text, LIPID_WORDS):
        return "KEGG lipid metabolism"
    if has_any(text, CARBO_WORDS):
        return "KEGG carbohydrate / energy metabolism"
    if has_any(text, COFACTOR_WORDS) or has_any(text, NUCLEOTIDE_WORDS):
        return "KEGG cofactor / nucleotide metabolism"
    if has_any(text, SECONDARY_WORDS):
        return "KEGG secondary metabolism"
    if has_any(text, SIGNAL_WORDS) or has_any(text, CELL_WORDS):
        return "KEGG cellular / signalling processes"
    return "KEGG other"


def family_table(data: DataBundle) -> pd.DataFrame:
    """Build a table mapping each pathway to its coarse family, source, size, and coherence."""
    rows = []
    for pid, genes in data.pathways.items():
        name = data.pathway_names.get(pid, pid)
        family = assign_family(pid, name)
        rows.append({
            "pathway_id": pid,
            "name": name,
            "source": "AraCyc" if pid.startswith("AC_") else "KEGG",
            "family": family,
            "n_genes": len(genes),
            "jaccard_mean": pathway_jaccard_mean(genes, data.gene_go, salt=f"fam:{pid}", seed=42),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["family", "pathway_id"])


def generate_negatives_for_sizes(
    data: DataBundle,
    sizes: Sequence[int],
    seed: int,
    prefix: str,
    n_multiplier: int = 2,
) -> List[Dict[str, Any]]:
    """Generate mixed hard negatives using a supplied held-out size distribution."""
    rng = np.random.default_rng(seed)
    n_neg = len(sizes) * n_multiplier
    if n_neg == 0:
        return []
    counts = {
        "jaccard_matched": n_neg // 4,
        "co_annotation": n_neg // 4,
        "chimera": n_neg // 4,
        "shuffled": n_neg - 3 * (n_neg // 4),
    }
    pathway_ids = sorted(data.pathways)
    pathway_sets = [data.pathways[pid] for pid in pathway_ids]
    real_jaccards = [
        pathway_jaccard_mean(genes, data.gene_go, salt=f"real:{pid}", seed=seed)
        for pid, genes in zip(pathway_ids, pathway_sets)
    ]
    jac_p25 = float(np.percentile(real_jaccards, 25))
    bg_annotated = sorted(g for g in data.background_genes if len(data.gene_go.get(g, set())) >= 3)
    negatives: List[Dict[str, Any]] = []

    def sample_size() -> int:
        return int(rng.choice(list(sizes)))

    attempts = 0
    while len([x for x in negatives if x["type"] == "jaccard_matched"]) < counts["jaccard_matched"]:
        attempts += 1
        size = sample_size()
        genes = set(rng_choice_list(rng, data.background_genes, size, replace=False))
        jm = pathway_jaccard_mean(genes, data.gene_go, salt=f"{prefix}:A:{attempts}", seed=seed)
        if jm >= jac_p25 or attempts >= max(100, counts["jaccard_matched"] * 25):
            negatives.append({
                "id": f"{prefix}_NEG_A_{len(negatives):04d}",
                "label": 0,
                "type": "jaccard_matched",
                "genes": sorted(genes),
            })

    for i in range(counts["co_annotation"]):
        seed_gene = rng_choice_list(rng, bg_annotated, 1, replace=False)[0]
        size = min(sample_size(), 50)
        genes = co_annotation_cluster(seed_gene, size, data.gene_go, bg_annotated, rng)
        negatives.append({
            "id": f"{prefix}_NEG_B_{i:04d}",
            "label": 0,
            "type": "co_annotation",
            "genes": sorted(genes),
        })

    for i in range(counts["chimera"]):
        n_pathways = int(rng.choice([2, 3]))
        chosen_idx = rng.choice(len(pathway_sets), size=n_pathways, replace=False)
        target_size = sample_size()
        per_pathway = max(2, target_size // n_pathways)
        genes = set()
        source_ids: List[str] = []
        for idx in chosen_idx:
            source_ids.append(pathway_ids[int(idx)])
            genes.update(rng_choice_list(rng, sorted(pathway_sets[int(idx)]), per_pathway, replace=False))
        negatives.append({
            "id": f"{prefix}_NEG_C_{i:04d}",
            "label": 0,
            "type": "chimera",
            "source_pathways": sorted(source_ids),
            "genes": sorted(genes),
        })

    for i in range(counts["shuffled"]):
        idx = int(rng.integers(0, len(pathway_sets)))
        source_genes = sorted(pathway_sets[idx])
        frac = float(rng.uniform(0.4, 0.6))
        n_replace = max(1, int(len(source_genes) * frac))
        keep = rng_choice_list(rng, source_genes, len(source_genes) - n_replace, replace=False)
        add = rng_choice_list(rng, data.background_genes, n_replace, replace=False)
        negatives.append({
            "id": f"{prefix}_NEG_D_{i:04d}",
            "label": 0,
            "type": "shuffled",
            "source_pathway": pathway_ids[idx],
            "replace_fraction": frac,
            "genes": sorted(set(keep + add)),
        })
    return negatives


def train_xgb_no_cv(train_records: Sequence[Dict[str, Any]], data: DataBundle, seed: int) -> Tuple[Any, List[str]]:
    """Train XGBoost on the given records with training-only GO selection (no CV scoring)."""
    selected_go, _fs_df, _mi_df = select_go_terms(train_records, data, cv_splits=5, seed=seed)
    x_train, _feature_names, _groups = build_feature_matrix(train_records, selected_go, data, seed=seed)
    y_train = np.asarray([r["label"] for r in train_records], dtype=int)
    model = make_models(seed)["XGBoost"]
    model.fit(x_train, y_train)
    return model, selected_go


def predict_records(model: Any, selected_go: Sequence[str], records: Sequence[Dict[str, Any]], data: DataBundle, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build features for records and return (true_labels, predicted_probabilities)."""
    x, _feature_names, _groups = build_feature_matrix(records, selected_go, data, seed=seed)
    y = np.asarray([r["label"] for r in records], dtype=int)
    pred = model.predict_proba(x)[:, 1]
    return y, pred


def metric_row(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute AUROC, AUPRC, F1, and mean scores for positive/negative classes."""
    labels = (pred >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, pred)),
        "auprc": float(average_precision_score(y, pred)),
        "f1": float(f1_score(y, labels)),
        "mean_positive_score": float(pred[y == 1].mean()) if np.any(y == 1) else np.nan,
        "mean_negative_score": float(pred[y == 0].mean()) if np.any(y == 0) else np.nan,
    }


def per_negative_type_analysis(data: DataBundle, seeds: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate model performance against each negative type separately across seeds.

    This tests whether the overall AUROC is driven by easy negative types (e.g.,
    random sets) or if the model also discriminates against harder negatives
    (co-annotation clusters, chimeras).
    """
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        records, _meta = build_samples(data, seed=seed)
        y_all = np.asarray([r["label"] for r in records], dtype=int)
        idx = np.arange(len(records))
        train_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=seed, stratify=y_all)
        train_records = [records[int(i)] for i in train_idx]
        test_records = [records[int(i)] for i in test_idx]
        model, selected_go = train_xgb_no_cv(train_records, data, seed=seed)
        test_pos = [r for r in test_records if r["label"] == 1]
        neg_types = sorted({r.get("type") for r in test_records if r["label"] == 0})
        for neg_type in neg_types:
            subset = test_pos + [r for r in test_records if r["label"] == 0 and r.get("type") == neg_type]
            y, pred = predict_records(model, selected_go, subset, data, seed=seed)
            row = metric_row(y, pred)
            rows.append({
                "seed": seed,
                "negative_type": neg_type,
                "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
                **row,
            })
        y_all_test, pred_all_test = predict_records(model, selected_go, test_records, data, seed=seed)
        row = metric_row(y_all_test, pred_all_test)
        rows.append({
            "seed": seed,
            "negative_type": "mixed_all",
            "n_positive": int((y_all_test == 1).sum()),
            "n_negative": int((y_all_test == 0).sum()),
            **row,
        })

    df = pd.DataFrame(rows)
    summary_rows = []
    for neg_type, g in df.groupby("negative_type", sort=False):
        row: Dict[str, Any] = {"negative_type": neg_type, "n_runs": len(g)}
        for col in ["auroc", "auprc", "f1", "mean_positive_score", "mean_negative_score", "n_positive", "n_negative"]:
            vals = g[col].astype(float)
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{col}_min"] = float(vals.min())
            row[f"{col}_max"] = float(vals.max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("auroc_mean")
    return df, summary


def lofo_analysis(data: DataBundle, seed: int, min_family_size: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-family-out validation: hold out each broad pathway family and evaluate.

    For each family with >= min_family_size pathways, a model is trained on all
    other curated pathways plus newly generated hard negatives, then the held-out
    family is evaluated against its own size-matched negatives.
    """
    fam_df = family_table(data)
    fam_df.to_csv(TABLE_DIR / "family_assignments.csv", index=False)
    dist = fam_df.groupby("family").agg(
        n_pathways=("pathway_id", "count"),
        n_kegg=("source", lambda s: int((s == "KEGG").sum())),
        n_aracyc=("source", lambda s: int((s == "AraCyc").sum())),
        median_size=("n_genes", "median"),
        median_jaccard=("jaccard_mean", "median"),
    ).reset_index().sort_values("n_pathways", ascending=False)
    dist.to_csv(TABLE_DIR / "family_distribution_v6.csv", index=False)

    families = dist.loc[dist["n_pathways"] >= min_family_size, "family"].tolist()
    rows: List[Dict[str, Any]] = []
    all_pos_records = [
        {"id": pid, "label": 1, "type": "curated_pathway", "name": data.pathway_names.get(pid, pid), "genes": sorted(genes)}
        for pid, genes in data.pathways.items()
    ]
    family_map = dict(zip(fam_df["pathway_id"], fam_df["family"]))

    for i, family in enumerate(families, start=1):
        test_pos = [r for r in all_pos_records if family_map[r["id"]] == family]
        train_pos = [r for r in all_pos_records if family_map[r["id"]] != family]
        train_sizes = [len(r["genes"]) for r in train_pos]
        test_sizes = [len(r["genes"]) for r in test_pos]
        train_neg = generate_negatives_for_sizes(data, train_sizes, seed=seed + i * 101, prefix=f"LOFO_TRAIN_{i}", n_multiplier=2)
        test_neg = generate_negatives_for_sizes(data, test_sizes, seed=seed + i * 101 + 1, prefix=f"LOFO_TEST_{i}", n_multiplier=2)
        train_records = train_pos + train_neg
        test_records = test_pos + test_neg
        model, selected_go = train_xgb_no_cv(train_records, data, seed=seed + i)
        y, pred = predict_records(model, selected_go, test_records, data, seed=seed + i)
        metrics = metric_row(y, pred)
        rows.append({
            "family": family,
            "n_test_pathways": len(test_pos),
            "n_train_pathways": len(train_pos),
            "n_test_negatives": len(test_neg),
            "median_test_pathway_size": float(np.median(test_sizes)),
            "median_test_jaccard": float(fam_df.loc[fam_df["family"] == family, "jaccard_mean"].median()),
            "n_go_selected": len(selected_go),
            **metrics,
        })
        print(f"LOFO {i}/{len(families)} {family}: AUROC={metrics['auroc']:.3f}, AUPRC={metrics['auprc']:.3f}, n={len(test_pos)}")

    lofo_df = pd.DataFrame(rows).sort_values("auroc")
    lofo_df.to_csv(TABLE_DIR / "lofo_family_results.csv", index=False)
    return lofo_df, dist


def make_figures(per_neg_summary: pd.DataFrame, lofo_df: pd.DataFrame, dist: pd.DataFrame) -> None:
    """Generate Fig 8 (negative-type AUROC), Fig 9 (LOFO AUROC), Fig 10 (family distribution)."""
    # Per-negative type performance.
    plot_df = per_neg_summary.copy()
    order_map = {"co_annotation": 0, "jaccard_matched": 1, "chimera": 2, "shuffled": 3, "mixed_all": 4}
    plot_df["order"] = plot_df["negative_type"].map(order_map).fillna(10)
    plot_df = plot_df.sort_values("order")
    plt.figure(figsize=(8.5, 4.8))
    x = np.arange(len(plot_df))
    plt.bar(x, plot_df["auroc_mean"], yerr=plot_df["auroc_sd"], capsize=4)
    plt.xticks(x, plot_df["negative_type"], rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUROC (mean +/- SD across seeds)")
    plt.title("Performance by negative type")
    for xi, val in zip(x, plot_df["auroc_mean"]):
        plt.text(xi, min(val + 0.035, 0.98), f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig8_negative_type_performance.png", dpi=300)
    plt.savefig(FIG_DIR / "fig8_negative_type_performance.pdf")
    plt.close()

    # LOFO family generalization.
    plot_df = lofo_df.sort_values("auroc")
    plt.figure(figsize=(9.5, 5.2))
    y = np.arange(len(plot_df))
    labels = [f"{fam} (n={n})" for fam, n in zip(plot_df["family"], plot_df["n_test_pathways"])]
    plt.barh(y, plot_df["auroc"])
    plt.yticks(y, labels, fontsize=8)
    plt.xlim(0.0, 1.0)
    plt.xlabel("LOFO AUROC")
    plt.title("Leave-one-pathway-family-out validation")
    for yi, val in zip(y, plot_df["auroc"]):
        plt.text(min(val + 0.015, 0.96), yi, f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig9_lofo_family_validation.png", dpi=300)
    plt.savefig(FIG_DIR / "fig9_lofo_family_validation.pdf")
    plt.close()

    # Family distribution.
    top = dist.sort_values("n_pathways", ascending=True)
    plt.figure(figsize=(9.5, 5.5))
    y = np.arange(len(top))
    plt.barh(y, top["n_pathways"])
    plt.yticks(y, top["family"], fontsize=8)
    plt.xlabel("Curated pathways")
    plt.title("Coarse pathway-family distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_family_distribution.png", dpi=300)
    plt.savefig(FIG_DIR / "fig10_family_distribution.pdf")
    plt.close()


def parse_seeds(text: str) -> List[int]:
    """Parse seed ranges like '1-20' or '1,5,10' into a sorted deduplicated list."""
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1-20", help="Seed list/range for per-negative-type analysis")
    parser.add_argument("--lofo-seed", type=int, default=42, help="Seed for LOFO validation")
    parser.add_argument("--min-family-size", type=int, default=10)
    args = parser.parse_args(argv)

    ensure_dirs()
    data = load_data()
    seeds = parse_seeds(args.seeds)
    print(f"Loaded {len(data.pathways)} pathways. Per-negative-type seeds: {seeds}")

    per_neg_df, per_neg_summary = per_negative_type_analysis(data, seeds)
    per_neg_df.to_csv(TABLE_DIR / "per_negative_type_per_seed.csv", index=False)
    per_neg_summary.to_csv(TABLE_DIR / "per_negative_type_summary.csv", index=False)

    lofo_df, dist = lofo_analysis(data, seed=args.lofo_seed, min_family_size=args.min_family_size)
    make_figures(per_neg_summary, lofo_df, dist)

    payload = {
        "generated_by": "additional_validation.py",
        "per_negative_type_summary": per_neg_summary.to_dict(orient="records"),
        "lofo_family_results": lofo_df.to_dict(orient="records"),
        "family_distribution": dist.to_dict(orient="records"),
        "notes": [
            "Per-negative-type evaluation trains on the normal mixed training split and evaluates test positives against one negative type at a time.",
            "LOFO excludes a coarse pathway family from positive training samples and evaluates the held-out family against size-matched mixed hard negatives.",
        ],
    }
    save_json(TABLE_DIR / "additional_validation_results.json", payload)
    print("Wrote additional validation tables and figures.")


if __name__ == "__main__":
    main()
