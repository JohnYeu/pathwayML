#!/usr/bin/env python3
"""Additional validation analyses for the thesis manuscript.

This script adds two checks requested after manuscript review:

1. Per-negative-type evaluation on the seed-42 held-out test split.
2. Leave-one-family-out (LOFO) pathway generalisation across broad pathway families.

The script intentionally keeps dense embeddings excluded. It uses the same GO-frequency,
Jaccard, and size feature definitions as run_no_embedding_reproducible.py. LOFO uses
training-only GO selection with a fixed 70% cumulative-MI rule for speed and to avoid
nested test leakage.
"""

from __future__ import annotations

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
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

import run_no_embedding_reproducible as core

TABLE_DIR = Path("tables")
FIG_DIR = Path("figures")

# ---------------------------------------------------------------------------
# Keyword lists for heuristic pathway family assignment (LOFO categories).
# Categories are deliberately broad to keep held-out families large enough
# for meaningful AUROC computation.
# ---------------------------------------------------------------------------
AMINO_ACID_KEYWORDS = [
    "alanine", "arginine", "asparagine", "aspartate", "aspartic", "cysteine", "glutamate", "glutamic",
    "glutamine", "glycine", "histidine", "isoleucine", "leucine", "lysine", "methionine", "phenylalanine",
    "proline", "serine", "threonine", "tryptophan", "tyrosine", "valine", "amino acid", "branched-chain",
]
LIPID_KEYWORDS = [
    "lipid", "fatty", "glycerol", "glycerolipid", "glycerophospholipid", "sphingolipid", "cutin", "suberin",
    "wax", "sterol", "steroid", "linolenic", "linoleic", "phospholipid", "triacylglycerol", "acyl-lipid",
]
CARB_KEYWORDS = [
    "carbohydrate", "starch", "sucrose", "cellulose", "glycolysis", "gluconeogenesis", "glucose", "fructose",
    "galactose", "mannose", "xylose", "pentose", "glycan", "pectin", "hemicellulose", "cell wall", "trehalose",
]
ENERGY_KEYWORDS = [
    "photosynthesis", "oxidative phosphorylation", "carbon fixation", "nitrogen metabolism", "sulfur metabolism",
    "methane", "atp", "respiration", "electron transport", "calvin", "photorespiration",
]
SIGNAL_KEYWORDS = [
    "signaling", "signalling", "signal", "hormone", "auxin", "ethylene", "abscisic", "jasmonic", "salicylic",
    "brassinosteroid", "circadian", "mapk", "response", "transduction",
]
COFACTOR_KEYWORDS = [
    "cofactor", "vitamin", "folate", "riboflavin", "thiamine", "biotin", "porphyrin", "chlorophyll", "carotenoid",
    "heme", "tetrahydrofolate", "nicotinate", "pantothenate",
]
SPECIALIZED_KEYWORDS = [
    "flavonoid", "phenylpropanoid", "glucosinolate", "terpenoid", "alkaloid", "anthocyanin", "lignin",
    "isoprenoid", "phytoalexin", "secondary metabolite", "stilbenoid", "benzoxazinoid", "coumarin", "betalain",
]
DEGRADATION_KEYWORDS = ["degradation", "catabolism", "catabolic", "breakdown", "salvage", "detoxification", "detox"]


def contains_any(text: str, words: Sequence[str]) -> bool:
    """Check if any keyword appears in text."""
    return any(w in text for w in words)


def assign_family(pid: str, name: str) -> str:
    """Assign a broad, thesis-level pathway family from pathway ID/name.

    The categories are deliberately broad to keep held-out families large enough for AUROC.
    The assignment is heuristic but deterministic and is saved for audit.
    """
    s = f"{pid} {name}".lower()
    if contains_any(s, DEGRADATION_KEYWORDS):
        return "Degradation/catabolism"
    if contains_any(s, AMINO_ACID_KEYWORDS):
        return "Amino acid metabolism"
    if contains_any(s, LIPID_KEYWORDS):
        return "Lipid metabolism"
    if contains_any(s, CARB_KEYWORDS):
        return "Carbohydrate metabolism"
    if contains_any(s, ENERGY_KEYWORDS):
        return "Energy metabolism"
    if contains_any(s, COFACTOR_KEYWORDS):
        return "Cofactor/vitamin metabolism"
    if contains_any(s, SIGNAL_KEYWORDS):
        return "Signalling/regulatory"
    if contains_any(s, SPECIALIZED_KEYWORDS) or "biosynth" in s or "biosynthesis" in s or "synthesis" in s:
        return "Specialized/other biosynthesis"
    if pid.startswith("AC_"):
        return "Other AraCyc"
    return "Other KEGG/cellular"


def fast_select_go_terms(train_records: Sequence[Dict[str, Any]], data: core.DataBundle, seed: int, mi_fraction: float = 0.70) -> List[str]:
    """Select GO terms using a fixed MI cumulative fraction (faster than CV-based selection)."""
    y_train = np.asarray([r["label"] for r in train_records], dtype=int)
    freq_train = np.vstack([core.go_frequency(r["genes"], data.go_terms, data.gene_go) for r in train_records])
    vt = VarianceThreshold(threshold=0.001)
    freq_vt = vt.fit_transform(freq_train)
    kept_terms = [term for term, keep in zip(data.go_terms, vt.get_support()) if keep]
    mi = mutual_info_classif(freq_vt, y_train, random_state=seed, n_neighbors=5)
    order = np.argsort(mi)[::-1]
    mi_sorted = mi[order]
    total = float(mi_sorted.sum())
    if total <= 0:
        k = min(20, len(kept_terms))
    else:
        cummi = np.cumsum(mi_sorted) / total
        k = int(np.argmax(cummi >= mi_fraction) + 1)
    k = max(3, min(k, len(kept_terms)))
    return [kept_terms[int(i)] for i in order[:k]]


def train_reference_seed42() -> Dict[str, Any]:
    """Build the canonical seed-42 reference model with fast GO selection."""
    data = core.load_data()
    records, meta = core.build_samples(data, seed=42)
    y_all = np.asarray([r["label"] for r in records], dtype=int)
    idx = np.arange(len(records))
    train_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=42, stratify=y_all)
    train_records = [records[int(i)] for i in train_idx]
    selected_go = fast_select_go_terms(train_records, data, seed=42, mi_fraction=0.70)
    X_all, feature_names, groups = core.build_feature_matrix(records, selected_go, data, seed=42)
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    model = core.make_models(42)["XGBoost"]
    model.fit(X_train, y_train)
    pred_test = model.predict_proba(X_test)[:, 1]
    return {
        "data": data,
        "records": records,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "selected_go": selected_go,
        "feature_names": feature_names,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "pred_test": pred_test,
        "sample_meta": meta,
    }


def per_negative_type_analysis() -> pd.DataFrame:
    """Evaluate positives vs each negative type on the seed-42 held-out test set."""
    obj = train_reference_seed42()
    records = obj["records"]
    test_idx = obj["test_idx"]
    y_test = obj["y_test"]
    pred_test = obj["pred_test"]
    test_types = np.asarray([records[int(i)]["type"] for i in test_idx])
    positive_mask = y_test == 1
    rows: List[Dict[str, Any]] = []
    # overall first
    labels = y_test
    scores = pred_test
    rows.append({
        "comparison": "All mixed negatives",
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        "test_auroc": float(roc_auc_score(labels, scores)),
        "test_auprc": float(average_precision_score(labels, scores)),
        "negative_score_median": float(np.median(scores[labels == 0])),
        "negative_score_iqr": f"{np.percentile(scores[labels == 0],25):.3f}-{np.percentile(scores[labels == 0],75):.3f}",
    })
    for neg_type in ["jaccard_matched", "co_annotation", "chimera", "shuffled"]:
        mask = positive_mask | ((y_test == 0) & (test_types == neg_type))
        labels = y_test[mask]
        scores = pred_test[mask]
        if np.sum(labels == 0) == 0:
            continue
        rows.append({
            "comparison": neg_type.replace("_", " "),
            "n_positive": int(np.sum(labels == 1)),
            "n_negative": int(np.sum(labels == 0)),
            "test_auroc": float(roc_auc_score(labels, scores)),
            "test_auprc": float(average_precision_score(labels, scores)),
            "negative_score_median": float(np.median(scores[labels == 0])),
            "negative_score_iqr": f"{np.percentile(scores[labels == 0],25):.3f}-{np.percentile(scores[labels == 0],75):.3f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "table6_negative_type_performance.csv", index=False)
    # plot
    plot_df = df[df["comparison"] != "All mixed negatives"].copy()
    plt.figure(figsize=(8, 4.5))
    plt.bar(plot_df["comparison"], plot_df["test_auroc"])
    plt.ylim(0.5, 1.0)
    plt.ylabel("Held-out AUROC vs positives")
    plt.xticks(rotation=25, ha="right")
    plt.title("Performance by negative-set type (seed 42)")
    for i, v in enumerate(plot_df["test_auroc"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_negative_type_performance.png", dpi=300)
    plt.savefig(FIG_DIR / "fig11_negative_type_performance.pdf")
    plt.close()
    return df


def family_table(data: core.DataBundle) -> pd.DataFrame:
    """Map each pathway to its coarse family, with source, size, and GO-Jaccard coherence."""
    rows = []
    for pid, genes in data.pathways.items():
        name = data.pathway_names.get(pid, pid)
        fam = assign_family(pid, name)
        source = "AraCyc" if pid.startswith("AC_") else "KEGG"
        rows.append({
            "pathway_id": pid,
            "pathway_name": name,
            "family": fam,
            "source": source,
            "n_genes": len(genes),
            "jaccard_mean": core.pathway_jaccard_mean(genes, data.gene_go, salt=f"fam:{pid}", seed=42),
        })
    return pd.DataFrame(rows)


def lofo_analysis(seed: int = 42, min_family_size: int = 10) -> pd.DataFrame:
    """LOFO validation with training-only GO selection and independent negative pools.

    Negative sets are allowed to include held-out family genes because they are
    constructed (non-curated) sets, not pathway records from the held-out family.
    """
    data = core.load_data()
    fmap = family_table(data)
    fmap.to_csv(TABLE_DIR / "pathway_family_assignment.csv", index=False)
    family_summary = (fmap.groupby("family")
        .agg(n_pathways=("pathway_id", "count"),
             n_kegg=("source", lambda s: int((s == "KEGG").sum())),
             n_aracyc=("source", lambda s: int((s == "AraCyc").sum())),
             median_size=("n_genes", "median"),
             median_jaccard=("jaccard_mean", "median"))
        .reset_index()
        .sort_values("n_pathways", ascending=False))
    family_summary.to_csv(TABLE_DIR / "table7_family_distribution.csv", index=False)

    # Generate reproducible negative pools; do not exclude held-out family genes, because negatives are not curated pathways.
    train_records_all, _meta_train = core.build_samples(data, seed=seed + 3000)
    test_records_all, _meta_test = core.build_samples(data, seed=seed + 4000)
    neg_train_pool = [r for r in train_records_all if r["label"] == 0]
    neg_test_pool = [r for r in test_records_all if r["label"] == 0]
    rng = np.random.default_rng(seed + 5000)

    rows: List[Dict[str, Any]] = []
    pathway_to_family = dict(zip(fmap["pathway_id"], fmap["family"]))
    pathway_names = dict(zip(fmap["pathway_id"], fmap["pathway_name"]))
    for _, fam_row in family_summary.iterrows():
        fam = str(fam_row["family"])
        n_heldout = int(fam_row["n_pathways"])
        if n_heldout < min_family_size:
            continue
        heldout_ids = set(fmap.loc[fmap["family"] == fam, "pathway_id"])
        train_pos = []
        test_pos = []
        for pid, genes in data.pathways.items():
            rec = {"id": pid, "label": 1, "type": "curated_pathway", "name": pathway_names.get(pid, pid), "genes": sorted(genes)}
            if pid in heldout_ids:
                test_pos.append(rec)
            else:
                train_pos.append(rec)
        n_train_neg = min(len(neg_train_pool), 2 * len(train_pos))
        n_test_neg = min(len(neg_test_pool), 2 * len(test_pos))
        train_neg_idx = rng.choice(len(neg_train_pool), size=n_train_neg, replace=False)
        test_neg_idx = rng.choice(len(neg_test_pool), size=n_test_neg, replace=False)
        train_records = train_pos + [neg_train_pool[int(i)] for i in train_neg_idx]
        test_records = test_pos + [neg_test_pool[int(i)] for i in test_neg_idx]
        # Select features on training records only.
        selected_go = fast_select_go_terms(train_records, data, seed=seed + stable_family_seed(fam), mi_fraction=0.70)
        X_train, feature_names, _groups = core.build_feature_matrix(train_records, selected_go, data, seed=seed)
        X_test, _feature_names, _groups = core.build_feature_matrix(test_records, selected_go, data, seed=seed)
        y_train = np.asarray([r["label"] for r in train_records], dtype=int)
        y_test = np.asarray([r["label"] for r in test_records], dtype=int)
        model = core.xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=2,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        pred = (scores >= 0.5).astype(int)
        rows.append({
            "family": fam,
            "n_heldout_pathways": len(test_pos),
            "n_test_negatives": n_test_neg,
            "n_train_pathways": len(train_pos),
            "n_go_selected": len(selected_go),
            "D": len(feature_names),
            "median_size": float(fam_row["median_size"]),
            "median_jaccard": float(fam_row["median_jaccard"]),
            "test_auroc": float(roc_auc_score(y_test, scores)),
            "test_auprc": float(average_precision_score(y_test, scores)),
            "f1": float(f1_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "positive_score_median": float(np.median(scores[y_test == 1])),
            "negative_score_median": float(np.median(scores[y_test == 0])),
        })
    df = pd.DataFrame(rows).sort_values("test_auroc", ascending=False)
    df.to_csv(TABLE_DIR / "table8_lofo_generalization.csv", index=False)

    # plot
    plot_df = df.sort_values("test_auroc")
    plt.figure(figsize=(8.5, 5.2))
    plt.barh(plot_df["family"], plot_df["test_auroc"])
    plt.xlim(0.5, 1.0)
    plt.xlabel("LOFO AUROC")
    plt.title("Leave-one-family-out generalisation")
    for i, v in enumerate(plot_df["test_auroc"]):
        plt.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig12_lofo_generalization.png", dpi=300)
    plt.savefig(FIG_DIR / "fig12_lofo_generalization.pdf")
    plt.close()
    return df


def stable_family_seed(text: str) -> int:
    """Deterministic hash of family name for per-family seed offset."""
    return int(sum((i + 1) * ord(c) for i, c in enumerate(text)) % 100000)


def main() -> None:
    TABLE_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    print("Running per-negative-type analysis...")
    neg_df = per_negative_type_analysis()
    print(neg_df.to_string(index=False))
    print("\nRunning leave-one-family-out analysis...")
    lofo_df = lofo_analysis(seed=42, min_family_size=10)
    print(lofo_df.to_string(index=False))
    summary = {
        "negative_type_performance": neg_df.to_dict(orient="records"),
        "lofo_generalization": lofo_df.to_dict(orient="records"),
        "notes": [
            "Negative-type analysis uses seed-42 held-out positives compared with one negative type at a time.",
            "LOFO uses broad deterministic family labels, training-only GO selection, and held-out family positives with independently generated negatives.",
            "Negative sets are allowed to include held-out family genes because they are constructed non-pathway sets rather than curated held-out pathway records."
        ]
    }
    with open(TABLE_DIR / "generalization_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    main()
