#!/usr/bin/env python3
"""Reproducible no-embedding PathwayML-Ath pipeline.

This is the canonical analysis entry point for the manuscript version that
removes dense gene embeddings. It fixes the problems in the Claude handoff:

* no SVD/UMAP/Node2Vec features in the final model;
* feature selection is learned on the training split only;
* all random samples, train/test splits, candidates, and selected GO terms
  are saved under tables/reproducibility/;
* tables are regenerated from one run instead of being mixed from old runs.
* optional fixed-seed-list averaging reports reproducible mean +/- SD/SE
  across independent runs.

Run from the repository root:

    python run_no_embedding_reproducible.py

For reproducible robustness averaging over fixed seeds:

    python run_no_embedding_reproducible.py --seeds 1-20 --no-figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import hypergeom
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
DEFAULT_REFERENCE_SEED = 42
DEFAULT_CANDIDATE_SEED = 42
DATA_DIR = Path("data")
FIG_DIR = Path("figures")
TABLE_DIR = Path("tables")
REPRO_DIR = TABLE_DIR / "reproducibility"
# Cap pairwise Jaccard computation to at most this many genes for speed
MAX_JACCARD_GENES = 15


@dataclass
class DataBundle:
    """Container for all loaded biological data used throughout the pipeline."""
    gene_go: Dict[str, set]          # gene -> set of GO terms
    go_genes: Dict[str, set]         # GO term -> set of genes
    pathways: Dict[str, set]         # pathway ID -> set of member genes
    pathway_names: Dict[str, str]    # pathway ID -> human-readable name
    go_terms: List[str]              # frequency-filtered GO terms (20 <= coverage <= 30%)
    background_genes: List[str]      # all genes with at least one GO annotation
    pathway_sizes: List[int]         # gene counts for all curated pathways
    n_kegg: int                      # number of KEGG pathways
    n_aracyc: int                    # number of AraCyc pathways


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    for path in [FIG_DIR, TABLE_DIR, REPRO_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def stable_int(text: str, seed: int = DEFAULT_REFERENCE_SEED) -> int:
    """Derive a deterministic integer hash from text+seed for reproducible RNG seeding."""
    h = hashlib.blake2b(f"{seed}:{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False)


def stable_subsample(
    items: Sequence[str],
    n: int,
    salt: str,
    seed: int = DEFAULT_REFERENCE_SEED,
) -> List[str]:
    """Deterministically subsample up to n items using a content-derived RNG seed."""
    values = list(items)
    if len(values) <= n:
        return values
    rng = np.random.default_rng(stable_int("|".join(values) + "|" + salt, seed=seed))
    idx = rng.choice(len(values), size=n, replace=False)
    return [values[i] for i in sorted(idx)]


def rng_choice_list(
    rng: np.random.Generator, values: Sequence[str], size: int, replace: bool = False
) -> List[str]:
    """Wrapper around rng.choice that returns a list of strings (handles edge cases)."""
    values = list(values)
    if size <= 0 or not values:
        return []
    size = min(size, len(values)) if not replace else size
    idx = rng.choice(len(values), size=size, replace=replace)
    if np.isscalar(idx):
        return [values[int(idx)]]
    return [values[int(i)] for i in idx]


def load_data() -> DataBundle:
    """Load gene-GO annotations (TAIR GAF), KEGG, and AraCyc pathway databases.

    Pathways with fewer than 5 genes are excluded. GO terms are filtered to
    those annotating between 20 and 30% of all annotated genes.
    """
    gene_go: Dict[str, set] = defaultdict(set)
    go_genes: Dict[str, set] = defaultdict(set)
    with open(DATA_DIR / "tair.gaf", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue
            gene = parts[1].upper()
            go_term = parts[4]
            if re.match(r"AT[0-9]G[0-9]{5}", gene):
                gene_go[gene].add(go_term)
                go_genes[go_term].add(gene)

    kegg: Dict[str, set] = defaultdict(set)
    with open(DATA_DIR / "kegg_pathway_genes.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                pid = parts[0].replace("path:", "")
                gene = parts[1].replace("ath:", "").upper()
                kegg[pid].add(gene)

    pathway_names: Dict[str, str] = {}
    with open(DATA_DIR / "kegg_pathway_names.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                pid = parts[0].replace("path:", "")
                pathway_names[pid] = parts[1].split(" - ")[0].strip()

    aracyc: Dict[str, set] = defaultdict(set)
    aracyc_names: Dict[str, str] = {}
    with open(DATA_DIR / "aracyc_pathways.20251021", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 7:
                pid, name, gene = parts[0], parts[1], parts[6].upper()
                if gene != "NIL" and gene.startswith("AT"):
                    aracyc[pid].add(gene)
                    aracyc_names[pid] = name

    pathways: Dict[str, set] = {}
    for pid in sorted(kegg):
        if len(kegg[pid]) >= 5:
            pathways[pid] = set(kegg[pid])
    n_kegg = len(pathways)
    for pid in sorted(aracyc):
        if len(aracyc[pid]) >= 5:
            new_id = f"AC_{pid}"
            pathways[new_id] = set(aracyc[pid])
            pathway_names[new_id] = aracyc_names.get(pid, pid)
    n_aracyc = sum(1 for pid in pathways if pid.startswith("AC_"))

    total_genes = len(gene_go)
    upper_thresh = int(0.30 * total_genes)
    go_terms = sorted(
        go for go, genes in go_genes.items() if 20 <= len(genes) <= upper_thresh
    )
    background_genes = sorted(gene_go.keys())
    pathway_sizes = [len(pathways[pid]) for pid in sorted(pathways)]

    return DataBundle(
        gene_go=dict(gene_go),
        go_genes=dict(go_genes),
        pathways=dict(sorted(pathways.items())),
        pathway_names=pathway_names,
        go_terms=go_terms,
        background_genes=background_genes,
        pathway_sizes=pathway_sizes,
        n_kegg=n_kegg,
        n_aracyc=n_aracyc,
    )


def go_frequency(gene_set: Iterable[str], go_terms: Sequence[str], gene_go: Dict[str, set]) -> np.ndarray:
    """Compute the fraction of genes in gene_set annotated with each GO term."""
    genes = sorted(set(gene_set))
    if not genes:
        return np.zeros(len(go_terms), dtype=float)
    freqs = []
    for term in go_terms:
        freqs.append(sum(1 for gene in genes if term in gene_go.get(gene, set())) / len(genes))
    return np.asarray(freqs, dtype=float)


def jaccard_stats(
    gene_set: Iterable[str],
    gene_go: Dict[str, set],
    salt: str,
    seed: int = DEFAULT_REFERENCE_SEED,
) -> np.ndarray:
    """Compute pairwise GO-Jaccard statistics [mean, std, min, max] for a gene set.

    Subsamples to MAX_JACCARD_GENES for computational tractability.
    """
    genes = sorted(set(gene_set))
    genes = stable_subsample(genes, MAX_JACCARD_GENES, salt=salt, seed=seed)
    values: List[float] = []
    for i, g1 in enumerate(genes):
        go1 = gene_go.get(g1, set())
        for g2 in genes[i + 1 :]:
            go2 = gene_go.get(g2, set())
            union = len(go1 | go2)
            if union:
                values.append(len(go1 & go2) / union)
    if not values:
        return np.zeros(4, dtype=float)
    arr = np.asarray(values, dtype=float)
    return np.asarray([arr.mean(), arr.std(), arr.min(), arr.max()], dtype=float)


def size_features(gene_set: Iterable[str]) -> np.ndarray:
    """Return [gene_count, log1p(gene_count)] as size features."""
    n = len(set(gene_set))
    return np.asarray([float(n), float(np.log1p(n))], dtype=float)


def pathway_jaccard_mean(
    gene_set: Iterable[str],
    gene_go: Dict[str, set],
    salt: str,
    seed: int = DEFAULT_REFERENCE_SEED,
) -> float:
    """Return the mean pairwise GO-Jaccard for a gene set (coherence measure)."""
    return float(jaccard_stats(gene_set, gene_go, salt=salt, seed=seed)[0])


def co_annotation_cluster(
    seed_gene: str,
    target_size: int,
    gene_go: Dict[str, set],
    pool: Sequence[str],
    rng: np.random.Generator,
    noise_frac: float = 0.5,
) -> set:
    """Build a co-annotation cluster around a seed gene for hard negative generation.

    Selects genes sharing GO terms with seed_gene (coherent core), then pads
    with random noise genes to reach target_size. This produces negatives that
    resemble real pathways in GO coherence but are not curated.
    """
    seed_terms = gene_go.get(seed_gene, set())
    if not seed_terms:
        return set(rng_choice_list(rng, pool, target_size, replace=False))

    sampled_pool = rng_choice_list(rng, pool, min(500, len(pool)), replace=False)
    candidates: List[Tuple[str, float]] = []
    for gene in sampled_pool:
        if gene == seed_gene:
            continue
        terms = gene_go.get(gene, set())
        union = len(seed_terms | terms)
        if union:
            score = len(seed_terms & terms) / union
            if score > 0:
                candidates.append((gene, score))
    candidates.sort(key=lambda x: (-x[1], x[0]))

    n_coherent = max(2, int(target_size * (1.0 - noise_frac)))
    cluster = {seed_gene}
    for gene, _score in candidates:
        cluster.add(gene)
        if len(cluster) >= n_coherent + 1:
            break

    n_noise = max(0, target_size - len(cluster))
    noise_pool = [gene for gene in pool if gene not in cluster]
    cluster.update(rng_choice_list(rng, noise_pool, n_noise, replace=False))
    return cluster


def build_samples(data: DataBundle, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Construct the full dataset: curated pathways (positives) + 4 types of hard negatives.

    Negative types (each ~25% of negatives):
      A) jaccard_matched: random gene sets with GO-Jaccard >= P25 of real pathways
      B) co_annotation: coherent clusters built around a seed gene + noise
      C) chimera: gene subsets drawn from 2-3 real pathways (cross-pathway hybrids)
      D) shuffled: real pathway with 40-60% of genes replaced by random background
    """
    rng = np.random.default_rng(seed)
    pathway_ids = sorted(data.pathways)
    pathway_sets = [data.pathways[pid] for pid in pathway_ids]
    n_pos = len(pathway_sets)
    n_neg = n_pos * 2       # 2:1 negative-to-positive ratio
    n_per_type = n_neg // 4  # roughly equal split across 4 negative types
    n_remainder = n_neg - 4 * n_per_type

    # Compute Jaccard coherence of real pathways to set the acceptance threshold
    real_jaccards = [
        pathway_jaccard_mean(genes, data.gene_go, salt=f"real:{pid}", seed=seed)
        for pid, genes in zip(pathway_ids, pathway_sets)
    ]
    jac_p25 = float(np.percentile(real_jaccards, 25))
    # Pool of well-annotated genes for co-annotation negatives (>= 3 GO terms)
    bg_annotated = sorted(
        gene for gene in data.background_genes if len(data.gene_go.get(gene, set())) >= 3
    )

    negatives: List[Dict[str, Any]] = []

    # --- Type A: Jaccard-matched random negatives ---
    attempts = 0
    while len([n for n in negatives if n["type"] == "jaccard_matched"]) < n_per_type:
        attempts += 1
        size = int(rng.choice(data.pathway_sizes))
        genes = set(rng_choice_list(rng, data.background_genes, size, replace=False))
        jm = pathway_jaccard_mean(genes, data.gene_go, salt=f"negA:{attempts}", seed=seed)
        if jm >= jac_p25 or attempts >= n_per_type * 20:
            negatives.append(
                {
                    "id": f"NEG_A_{len(negatives):04d}",
                    "type": "jaccard_matched",
                    "genes": sorted(genes),
                    "jaccard_mean": jm,
                }
            )

    # --- Type B: Co-annotation cluster negatives ---
    for i in range(n_per_type):
        seed_gene = rng_choice_list(rng, bg_annotated, 1, replace=False)[0]
        size = min(int(rng.choice(data.pathway_sizes)), 50)
        genes = co_annotation_cluster(seed_gene, size, data.gene_go, bg_annotated, rng)
        negatives.append(
            {
                "id": f"NEG_B_{i:04d}",
                "type": "co_annotation",
                "seed_gene": seed_gene,
                "genes": sorted(genes),
            }
        )

    # --- Type C: Chimeric cross-pathway negatives ---
    for i in range(n_per_type):
        n_pathways = int(rng.choice([2, 3]))
        chosen_idx = rng.choice(len(pathway_sets), size=n_pathways, replace=False)
        target_size = int(rng.choice(data.pathway_sizes))
        per_pathway = max(2, target_size // n_pathways)
        genes: set = set()
        source_ids: List[str] = []
        for idx in chosen_idx:
            source_ids.append(pathway_ids[int(idx)])
            genes.update(rng_choice_list(rng, sorted(pathway_sets[int(idx)]), per_pathway, replace=False))
        negatives.append(
            {
                "id": f"NEG_C_{i:04d}",
                "type": "chimera",
                "source_pathways": sorted(source_ids),
                "genes": sorted(genes),
            }
        )

    # --- Type D: Shuffled pathway negatives (partial gene replacement) ---
    for i in range(n_per_type + n_remainder):
        idx = int(rng.integers(0, len(pathway_sets)))
        source_genes = sorted(pathway_sets[idx])
        frac = float(rng.uniform(0.4, 0.6))
        n_replace = max(1, int(len(source_genes) * frac))
        keep = rng_choice_list(rng, source_genes, len(source_genes) - n_replace, replace=False)
        add = rng_choice_list(rng, data.background_genes, n_replace, replace=False)
        negatives.append(
            {
                "id": f"NEG_D_{i:04d}",
                "type": "shuffled",
                "source_pathway": pathway_ids[idx],
                "replace_fraction": frac,
                "genes": sorted(set(keep + add)),
            }
        )

    records: List[Dict[str, Any]] = []
    for pid in pathway_ids:
        records.append(
            {
                "id": pid,
                "label": 1,
                "type": "curated_pathway",
                "name": data.pathway_names.get(pid, pid),
                "genes": sorted(data.pathways[pid]),
            }
        )
    for neg in negatives:
        rec = dict(neg)
        rec["label"] = 0
        records.append(rec)

    meta = {
        "n_pos": n_pos,
        "n_neg": len(negatives),
        "negative_counts": pd.Series([n["type"] for n in negatives]).value_counts().to_dict(),
        "real_pathway_jaccard_median": float(np.median(real_jaccards)),
        "real_pathway_jaccard_p25": jac_p25,
        "jaccard_matched_attempts": attempts,
    }
    return records, meta


def select_go_terms(
    train_records: Sequence[Dict[str, Any]],
    data: DataBundle,
    cv_splits: int,
    seed: int,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Select informative GO terms using mutual information on the training split.

    Procedure:
    1. Remove near-zero-variance GO features
    2. Rank remaining terms by MI with class label
    3. Try several cumulative-MI cutoffs (30%-95% + all)
    4. Pick k that maximizes CV AUROC (prefer smaller k on ties)
    """
    y_train = np.asarray([record["label"] for record in train_records], dtype=int)
    freq_train = np.vstack(
        [go_frequency(record["genes"], data.go_terms, data.gene_go) for record in train_records]
    )

    vt = VarianceThreshold(threshold=0.001)
    freq_vt = vt.fit_transform(freq_train)
    kept_terms = [term for term, keep in zip(data.go_terms, vt.get_support()) if keep]

    mi = mutual_info_classif(freq_vt, y_train, random_state=seed, n_neighbors=5)
    order = np.argsort(mi)[::-1]
    mi_sorted = mi[order]
    mi_total = float(mi_sorted.sum())
    if mi_total <= 0:
        cummi = np.linspace(0, 1, len(mi_sorted), endpoint=True)
    else:
        cummi = np.cumsum(mi_sorted) / mi_total

    k_values: List[int] = []
    for frac in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        k_values.append(int(np.argmax(cummi >= frac) + 1))
    k_values.append(len(kept_terms))
    k_values = sorted(set(k for k in k_values if k >= 3))

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    rows: List[Dict[str, Any]] = []
    for k in k_values:
        cols = order[:k]
        x_k = freq_vt[:, cols]
        aucs: List[float] = []
        for tr, va in skf.split(x_k, y_train):
            model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            )
            model.fit(x_k[tr], y_train[tr])
            aucs.append(roc_auc_score(y_train[va], model.predict_proba(x_k[va])[:, 1]))
        rows.append(
            {
                "k": k,
                "cumulative_mi": float(cummi[k - 1]),
                "cv_auroc_mean": float(np.mean(aucs)),
                "cv_auroc_std": float(np.std(aucs, ddof=1)),
            }
        )

    fs_df = pd.DataFrame(rows)
    best_row = fs_df.sort_values(["cv_auroc_mean", "k"], ascending=[False, True]).iloc[0]
    best_k = int(best_row["k"])
    selected_idx = order[:best_k]
    selected_terms = [kept_terms[int(i)] for i in selected_idx]

    mi_df = pd.DataFrame(
        {
            "go_term": [kept_terms[int(i)] for i in order],
            "mutual_info": mi_sorted,
            "rank": np.arange(1, len(mi_sorted) + 1),
            "cumulative_mi": cummi,
            "selected": [rank <= best_k for rank in range(1, len(mi_sorted) + 1)],
        }
    )
    return selected_terms, fs_df, mi_df


def build_feature_matrix(
    records: Sequence[Dict[str, Any]],
    selected_go: Sequence[str],
    data: DataBundle,
    seed: int,
) -> Tuple[np.ndarray, List[str], Dict[str, List[int]]]:
    """Assemble the no-embedding feature matrix: [GO freq | Jaccard stats | size].

    Returns the feature matrix, feature names, and column-index groups for ablation.
    """
    feature_names = (
        list(selected_go)
        + ["jaccard_mean", "jaccard_std", "jaccard_min", "jaccard_max"]
        + ["pathway_size", "log_size"]
    )
    rows = []
    for record in records:
        genes = record["genes"]
        rows.append(
            np.concatenate(
                [
                    go_frequency(genes, selected_go, data.gene_go),
                    jaccard_stats(genes, data.gene_go, salt=f"feature:{record['id']}", seed=seed),
                    size_features(genes),
                ]
            )
        )
    groups = {
        "go": list(range(0, len(selected_go))),
        "jaccard": list(range(len(selected_go), len(selected_go) + 4)),
        "size": list(range(len(selected_go) + 4, len(selected_go) + 6)),
    }
    return np.vstack(rows), feature_names, groups


def make_models(seed: int) -> Dict[str, Any]:
    """Return the three classifiers used in the main analysis."""
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
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
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=seed,
                solver="lbfgs",
            ),
        ),
    }


def evaluate_model(
    name: str,
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cv_splits: int,
    cv_repeats: int,
    seed: int,
) -> Tuple[Dict[str, Any], Any]:
    """Train a model with repeated stratified CV, then evaluate on held-out test set."""
    rskf = RepeatedStratifiedKFold(
        n_splits=cv_splits, n_repeats=cv_repeats, random_state=seed
    )
    aucs: List[float] = []
    auprcs: List[float] = []
    for tr, va in rskf.split(x_train, y_train):
        fold_model = clone(model)
        fold_model.fit(x_train[tr], y_train[tr])
        pred = fold_model.predict_proba(x_train[va])[:, 1]
        aucs.append(roc_auc_score(y_train[va], pred))
        auprcs.append(average_precision_score(y_train[va], pred))

    final_model = clone(model)
    final_model.fit(x_train, y_train)
    test_pred = final_model.predict_proba(x_test)[:, 1]
    test_label = (test_pred >= 0.5).astype(int)
    cv_std = float(np.std(aucs, ddof=1))
    result = {
        "model": name,
        "cv_auroc_mean": float(np.mean(aucs)),
        "cv_auroc_std": cv_std,
        "cv_auroc_se": float(cv_std / np.sqrt(len(aucs))),
        "cv_auprc_mean": float(np.mean(auprcs)),
        "cv_auprc_std": float(np.std(auprcs, ddof=1)),
        "n_cv_folds": len(aucs),
        "test_auroc": float(roc_auc_score(y_test, test_pred)),
        "test_auprc": float(average_precision_score(y_test, test_pred)),
        "test_f1": float(f1_score(y_test, test_label)),
        "test_precision": float(precision_score(y_test, test_label)),
        "test_recall": float(recall_score(y_test, test_label)),
    }
    return result, final_model


def evaluate_ablation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    groups: Dict[str, List[int]],
    cv_splits: int,
    cv_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Run leave-one-group-out and cumulative ablation to quantify feature group contributions."""
    idx_go, idx_jac, idx_size = groups["go"], groups["jaccard"], groups["size"]
    configs = {
        "Full model": idx_go + idx_jac + idx_size,
        "-GO freq": idx_jac + idx_size,
        "-Jaccard": idx_go + idx_size,
        "-Size": idx_go + idx_jac,
        "GO freq only": idx_go,
        "Jaccard only": idx_jac,
        "Size only": idx_size,
        "Cumul: Size": idx_size,
        "Cumul: +Jaccard": idx_size + idx_jac,
        "Cumul: +GO freq": idx_go + idx_jac + idx_size,
    }
    rows: List[Dict[str, Any]] = []
    base_model = make_models(seed)["XGBoost"]
    for cfg, idx in configs.items():
        result, _model = evaluate_model(
            cfg,
            base_model,
            x_train[:, idx],
            y_train,
            x_test[:, idx],
            y_test,
            cv_splits=cv_splits,
            cv_repeats=cv_repeats,
            seed=seed,
        )
        rows.append(
            {
                "configuration": cfg,
                "d": len(idx),
                "cv_auroc_mean": result["cv_auroc_mean"],
                "cv_auroc_se": result["cv_auroc_se"],
                "test_auroc": result["test_auroc"],
                "test_auprc": result["test_auprc"],
            }
        )
    df = pd.DataFrame(rows)
    full = float(df.loc[df["configuration"] == "Full model", "test_auroc"].iloc[0])
    df["delta_vs_full"] = df["test_auroc"] - full
    return df


def construct_candidates(data: DataBundle, candidate_seed: int) -> Dict[str, List[str]]:
    """Build 4 deterministic candidate gene sets for novelty scoring.

    C1: oxidative-stress genes from multiple pathway buckets + orphan stress genes
    C2: subset of photosynthesis pathway (ath00195)
    C3: subset of ubiquitin-mediated proteolysis (ath04120)
    C4: starch/sucrose subset + random background genes (designed to be ambiguous)
    """
    rng = np.random.default_rng(candidate_seed + 1000)
    go_target = "GO:0006979"
    stress_genes = sorted(gene for gene in data.gene_go if go_target in data.gene_go[gene])

    gene_to_pathways: Dict[str, set] = defaultdict(set)
    for pid, genes in data.pathways.items():
        for gene in genes:
            gene_to_pathways[gene].add(pid)

    pathway_buckets: Dict[str, List[str]] = defaultdict(list)
    no_pathway_stress: List[str] = []
    for gene in stress_genes:
        pids = sorted(gene_to_pathways.get(gene, set()))
        if not pids:
            no_pathway_stress.append(gene)
        for pid in pids:
            pathway_buckets[pid].append(gene)

    good_pathways = [
        (pid, sorted(set(genes)))
        for pid, genes in pathway_buckets.items()
        if 3 <= len(set(genes)) <= 20 and len(data.pathways[pid]) < 60
    ]
    good_pathways.sort(key=lambda item: (-len(item[1]), item[0]))

    c1: set = set()
    for pid, genes in good_pathways[:4]:
        c1.update(rng_choice_list(rng, genes, min(3, len(genes)), replace=False))
    need = 17 - len(c1)
    if need > 0:
        c1.update(rng_choice_list(rng, no_pathway_stress, need, replace=False))

    c2 = set(rng_choice_list(rng, sorted(data.pathways.get("ath00195", set())), 11, replace=False))
    c3 = set(rng_choice_list(rng, sorted(data.pathways.get("ath04120", set())), 14, replace=False))
    c4 = set(rng_choice_list(rng, sorted(data.pathways.get("ath00500", set())), 5, replace=False))
    c4.update(rng_choice_list(rng, data.background_genes, 4, replace=False))

    return {"C1": sorted(c1), "C2": sorted(c2), "C3": sorted(c3), "C4": sorted(c4)}


def run_ora(genes: set, data: DataBundle) -> pd.DataFrame:
    """Over-representation analysis using Fisher's exact / hypergeometric test.

    Tests whether the gene set overlaps significantly with each curated pathway.
    Bonferroni correction is applied (p_adj = p * n_pathways).
    """
    rows: List[Dict[str, Any]] = []
    bg_size = len(data.gene_go)
    for pid, pathway_genes in data.pathways.items():
        overlap = len(genes & pathway_genes)
        if overlap == 0:
            continue
        p_value = hypergeom.sf(overlap - 1, bg_size, len(pathway_genes), len(genes))
        rows.append(
            {
                "pathway": pid,
                "name": data.pathway_names.get(pid, pid),
                "overlap": overlap,
                "p_value": float(p_value),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["pathway", "name", "overlap", "p_value", "p_adj"])
    df = pd.DataFrame(rows).sort_values("p_value")
    df["p_adj"] = np.minimum(df["p_value"] * len(data.pathways), 1.0)
    return df


def score_candidates(
    candidates: Dict[str, List[str]],
    model: Any,
    selected_go: Sequence[str],
    data: DataBundle,
    seed: int,
) -> pd.DataFrame:
    """Score each candidate gene set with the trained model and assess novelty.

    A candidate is 'novel' if: score >= 0.5, max overlap with any pathway < 30%,
    and no ORA-significant pathway overlap (p_adj < 0.05).
    """
    rows: List[Dict[str, Any]] = []
    for cid, genes in candidates.items():
        fv, _names, _groups = build_feature_matrix(
            [{"id": cid, "genes": genes, "label": -1}], selected_go, data, seed=seed
        )
        score = float(model.predict_proba(fv)[0, 1])
        gene_set = set(genes)
        max_j = 0.0
        max_overlap_frac = 0.0
        closest_j = ""
        closest_overlap = ""
        for pid, pathway_genes in data.pathways.items():
            if len(pathway_genes) > 200:
                continue
            union = len(gene_set | pathway_genes)
            jaccard = len(gene_set & pathway_genes) / union if union else 0.0
            overlap_frac = len(gene_set & pathway_genes) / len(gene_set) if gene_set else 0.0
            if jaccard > max_j:
                max_j, closest_j = jaccard, pid
            if overlap_frac > max_overlap_frac:
                max_overlap_frac, closest_overlap = overlap_frac, pid
        ora = run_ora(gene_set, data)
        ora_significant = bool((ora["p_adj"] < 0.05).any()) if len(ora) else False
        best_ora = float(ora.iloc[0]["p_adj"]) if len(ora) else 1.0
        rows.append(
            {
                "candidate": cid,
                "n": len(genes),
                "score": score,
                "max_jaccard": float(max_j),
                "closest_by_jaccard": data.pathway_names.get(closest_j, closest_j),
                "max_overlap_fraction": float(max_overlap_frac),
                "closest_by_overlap": data.pathway_names.get(closest_overlap, closest_overlap),
                "ora_significant": ora_significant,
                "best_ora_p_adj": best_ora,
                "novel": bool(score >= 0.5 and max_overlap_frac < 0.30 and not ora_significant),
            }
        )
    return pd.DataFrame(rows)


def save_json(path: Path, obj: Any) -> None:
    """Write a Python object to JSON with numpy type coercion."""
    def default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=default), encoding="utf-8")


def write_tables(  # noqa: C901 (complexity unavoidable for single-pass output)
    seed: int,
    repro_dir: Path,
    data: DataBundle,
    records: Sequence[Dict[str, Any]],
    split_info: Dict[str, Any],
    sample_meta: Dict[str, Any],
    selected_go: Sequence[str],
    feature_names: Sequence[str],
    model_results: Dict[str, Dict[str, Any]],
    ablation_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    fs_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    candidates: Dict[str, List[str]],
    candidate_df: pd.DataFrame,
    cv_splits: int,
    cv_repeats: int,
) -> Dict[str, Any]:
    """Write all CSV tables, reproducibility artifacts, and JSON result summaries."""
    perf_df = pd.DataFrame(
        {
            name: {
                "CV AUROC": f"{res['cv_auroc_mean']:.3f} +/- {res['cv_auroc_se']:.3f} SE",
                "Test AUROC": f"{res['test_auroc']:.3f}",
                "Test AUPRC": f"{res['test_auprc']:.3f}",
                "Test F1": f"{res['test_f1']:.3f}",
            }
            for name, res in model_results.items()
        }
    ).T
    perf_df.to_csv(TABLE_DIR / "table1_performance.csv")

    shap_df.to_csv(TABLE_DIR / "table2_shap_importance.csv", index=False)
    ablation_df.to_csv(TABLE_DIR / "table3_ablation.csv", index=False)
    ablation_df.to_json(TABLE_DIR / "ablation_full.json", orient="records", indent=2)
    fs_df.to_csv(TABLE_DIR / "feature_selection_cv.csv", index=False)
    mi_df.to_csv(TABLE_DIR / "selected_go_terms.csv", index=False)
    candidate_df.to_csv(TABLE_DIR / "candidate_results.csv", index=False)

    repro_dir.mkdir(parents=True, exist_ok=True)
    save_json(repro_dir / "samples.json", records)
    save_json(repro_dir / "splits.json", split_info)
    save_json(repro_dir / "selected_go_terms.json", list(selected_go))
    save_json(repro_dir / "candidate_gene_sets.json", candidates)
    save_json(repro_dir / "feature_names.json", list(feature_names))

    final = {
        "generated_by": "run_no_embedding_reproducible.py",
        "seed": seed,
        "cv": {"splits": cv_splits, "repeats": cv_repeats, "folds": cv_splits * cv_repeats},
        "dataset": {
            "n_pathways": len(data.pathways),
            "n_kegg": data.n_kegg,
            "n_aracyc": data.n_aracyc,
            "n_genes": len(data.gene_go),
            "n_go_filtered": len(data.go_terms),
            "n_go_selected": len(selected_go),
            "D": len(feature_names),
            "features": f"GO freq ({len(selected_go)}) + Jaccard (4) + Size (2) = {len(feature_names)}",
            "negatives": sample_meta,
        },
        "split": {
            "n_train": len(split_info["train_ids"]),
            "n_test": len(split_info["test_ids"]),
            "feature_selection": "GO terms selected on training split only",
        },
        "performance": model_results,
        "ablation": ablation_df.to_dict(orient="records"),
        "candidates": candidate_df.to_dict(orient="records"),
        "note": "NO EMBEDDING in this version. SVD/UMAP features are not used by the final model.",
    }
    save_json(TABLE_DIR / "results_no_embedding.json", final)

    summary = {
        "dataset": final["dataset"],
        "performance": {
            name: {
                "cv": f"{res['cv_auroc_mean']:.3f} +/- {res['cv_auroc_se']:.3f} SE",
                "test_auroc": round(res["test_auroc"], 3),
            }
            for name, res in model_results.items()
        },
        "note": final["note"],
    }
    save_json(TABLE_DIR / "final_no_emb.json", summary)
    return final


def plot_outputs(  # noqa: C901
    model_results: Dict[str, Dict[str, Any]],
    ablation_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    fs_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    data: DataBundle,
) -> None:
    """Generate all manuscript figures (Fig 2-7): performance, SHAP, candidates, ablation, etc."""
    names = list(model_results)
    test_aurocs = [model_results[name]["test_auroc"] for name in names]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(names, test_aurocs, color=["#2B6CB0", "#38A169", "#D69E2E"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Held-out test AUROC")
    plt.title("Classifier performance (no embedding)")
    for bar, value in zip(bars, test_aurocs):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_classifier_performance.png", dpi=300)
    plt.savefig(FIG_DIR / "fig2_classifier_performance.pdf")
    plt.close()

    top = shap_df.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2B6CB0")
    plt.xlabel("Mean |SHAP|")
    plt.title("Top SHAP features (no embedding)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_shap_analysis.png", dpi=300)
    plt.savefig(FIG_DIR / "fig3_shap_analysis.pdf")
    plt.close()

    plt.figure(figsize=(6.5, 5))
    colors = candidate_df["novel"].map({True: "#38A169", False: "#718096"})
    plt.scatter(candidate_df["max_jaccard"], candidate_df["score"], s=120, c=colors, edgecolor="black")
    for _, row in candidate_df.iterrows():
        plt.annotate(row["candidate"], (row["max_jaccard"], row["score"]), xytext=(6, 4), textcoords="offset points")
    plt.axhline(0.5, color="#A0AEC0", linestyle="--", linewidth=1)
    plt.xlabel("Max Jaccard to curated pathway")
    plt.ylabel("ML score")
    plt.title("Candidate scoring")
    plt.xlim(-0.02, 1.0)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_candidates.png", dpi=300)
    plt.savefig(FIG_DIR / "fig4_candidates.pdf")
    plt.close()

    plt.figure(figsize=(9, 4.8))
    plot_df = ablation_df.copy()
    plt.bar(plot_df["configuration"], plot_df["test_auroc"], color="#2B6CB0")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Held-out test AUROC")
    plt.title("No-embedding ablation")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_ablation.png", dpi=300)
    plt.savefig(FIG_DIR / "fig6_ablation.pdf")
    plt.close()

    plt.figure(figsize=(6.5, 4.5))
    plt.errorbar(fs_df["k"], fs_df["cv_auroc_mean"], yerr=fs_df["cv_auroc_std"], marker="o", capsize=4)
    best = fs_df.sort_values(["cv_auroc_mean", "k"], ascending=[False, True]).iloc[0]
    plt.axvline(best["k"], color="#C53030", linestyle="--", label=f"k={int(best['k'])}")
    plt.xlabel("Number of selected GO terms")
    plt.ylabel("Training-CV AUROC")
    plt.title("GO feature selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_feature_selection.png", dpi=300)
    plt.savefig(FIG_DIR / "fig7_feature_selection.pdf")
    plt.close()

    sizes = [len(v) for v in data.pathways.values()]
    go_cov = sorted([len(v) for v in data.go_genes.values()], reverse=True)
    plt.figure(figsize=(11, 4.5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(sizes, bins=30, color="#2B6CB0", alpha=0.8)
    ax1.set_xlabel("Pathway gene count")
    ax1.set_ylabel("Count")
    ax1.set_title("Pathway sizes")
    ax2 = plt.subplot(1, 2, 2)
    ax2.semilogy(range(1, len(go_cov) + 1), go_cov, color="#2B6CB0")
    ax2.axhline(20, color="#D69E2E", linestyle="--", linewidth=1)
    ax2.axhline(int(0.30 * len(data.gene_go)), color="#C53030", linestyle="--", linewidth=1)
    ax2.set_xlabel("GO term rank")
    ax2.set_ylabel("Gene coverage")
    ax2.set_title("GO coverage")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_dataset_stats.png", dpi=300)
    plt.savefig(FIG_DIR / "fig5_dataset_stats.pdf")
    plt.close()


def parse_seed_list(value: str) -> List[int]:
    """Parse comma-separated seeds and inclusive ranges such as 1-5,10,42."""
    seeds: List[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(token))
    seen = set()
    unique: List[int] = []
    for seed in seeds:
        if seed not in seen:
            unique.append(seed)
            seen.add(seed)
    if not unique:
        raise ValueError("At least one seed is required")
    return unique


def write_reproducibility_artifacts(  # used by multi-seed runs
    repro_dir: Path,
    records: Sequence[Dict[str, Any]],
    split_info: Dict[str, Any],
    selected_go: Sequence[str],
    candidates: Dict[str, List[str]],
    feature_names: Sequence[str],
) -> None:
    """Save samples, splits, GO terms, candidates, and feature names for reproducibility."""
    repro_dir.mkdir(parents=True, exist_ok=True)
    save_json(repro_dir / "samples.json", records)
    save_json(repro_dir / "splits.json", split_info)
    save_json(repro_dir / "selected_go_terms.json", list(selected_go))
    save_json(repro_dir / "candidate_gene_sets.json", candidates)
    save_json(repro_dir / "feature_names.json", list(feature_names))


def compact_dataset_summary(  # lightweight version for multi-seed runs
    data: DataBundle,
    selected_go: Sequence[str],
    feature_names: Sequence[str],
    sample_meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "n_pathways": len(data.pathways),
        "n_kegg": data.n_kegg,
        "n_aracyc": data.n_aracyc,
        "n_genes": len(data.gene_go),
        "n_go_filtered": len(data.go_terms),
        "n_go_selected": len(selected_go),
        "D": len(feature_names),
        "features": f"GO freq ({len(selected_go)}) + Jaccard (4) + Size (2) = {len(feature_names)}",
        "negatives": sample_meta,
    }


def run_single_analysis(
    data: DataBundle,
    seed: int,
    candidate_seed: int,
    cv_splits: int,
    cv_repeats: int,
    write_outputs: bool,
    write_figures: bool,
    with_ablation: bool,
    with_shap: bool,
    repro_dir: Path,
) -> Dict[str, Any]:
    """Execute the full no-embedding pipeline for a single seed.

    Steps: generate negatives -> train/test split -> GO selection on train ->
    build features -> train & evaluate models -> ablation -> SHAP -> candidates.
    """
    np.random.seed(seed)

    print(f"Generating deterministic hard negatives (seed={seed})...")
    records, sample_meta = build_samples(data, seed=seed)
    y_all = np.asarray([record["label"] for record in records], dtype=int)
    all_idx = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, random_state=seed, stratify=y_all
    )
    train_records = [records[int(i)] for i in train_idx]
    split_info = {
        "random_state": seed,
        "train_ids": [records[int(i)]["id"] for i in train_idx],
        "test_ids": [records[int(i)]["id"] for i in test_idx],
    }

    print("Selecting GO features on training split only...")
    selected_go, fs_df, mi_df = select_go_terms(
        train_records, data, cv_splits=cv_splits, seed=seed
    )
    print(f"  Selected GO terms: {len(selected_go)}")

    print("Building no-embedding features...")
    x_all, feature_names, groups = build_feature_matrix(records, selected_go, data, seed=seed)
    x_train, x_test = x_all[train_idx], x_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    print(f"  D={x_all.shape[1]} = GO({len(selected_go)}) + Jaccard(4) + Size(2)")

    print("Training models...")
    models = make_models(seed)
    model_results: Dict[str, Dict[str, Any]] = {}
    fitted_models: Dict[str, Any] = {}
    for name, model in models.items():
        result, fitted = evaluate_model(
            name,
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            cv_splits=cv_splits,
            cv_repeats=cv_repeats,
            seed=seed,
        )
        model_results[name] = result
        fitted_models[name] = fitted
        print(
            f"  {name}: CV={result['cv_auroc_mean']:.3f} +/- "
            f"{result['cv_auroc_se']:.3f} SE, Test={result['test_auroc']:.3f}"
        )

    if with_ablation:
        print("Running no-embedding ablation...")
        ablation_df = evaluate_ablation(
            x_train,
            y_train,
            x_test,
            y_test,
            groups,
            cv_splits=cv_splits,
            cv_repeats=cv_repeats,
            seed=seed,
        )
    else:
        ablation_df = pd.DataFrame()

    if with_shap:
        print("Computing SHAP values...")
        xgb_model = fitted_models["XGBoost"]
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(x_test)
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        shap_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
                "pct_total": mean_abs / mean_abs.sum() * 100,
            }
        ).sort_values("mean_abs_shap", ascending=False)
    else:
        xgb_model = fitted_models["XGBoost"]
        shap_df = pd.DataFrame()

    print("Scoring deterministic candidates...")
    candidates = construct_candidates(data, candidate_seed=candidate_seed)
    candidate_df = score_candidates(candidates, xgb_model, selected_go, data, seed=seed)

    if write_outputs:
        if ablation_df.empty or shap_df.empty:
            raise ValueError("Main output writing requires ablation and SHAP results")
        print("Writing tables and reproducibility artifacts...")
        results = write_tables(
            seed=seed,
            repro_dir=repro_dir,
            data=data,
            records=records,
            split_info=split_info,
            sample_meta=sample_meta,
            selected_go=selected_go,
            feature_names=feature_names,
            model_results=model_results,
            ablation_df=ablation_df,
            shap_df=shap_df,
            fs_df=fs_df,
            mi_df=mi_df,
            candidates=candidates,
            candidate_df=candidate_df,
            cv_splits=cv_splits,
            cv_repeats=cv_repeats,
        )

        if write_figures:
            print("Writing figures...")
            plot_outputs(model_results, ablation_df, shap_df, fs_df, candidate_df, data)
    else:
        results = {
            "generated_by": "run_no_embedding_reproducible.py",
            "seed": seed,
            "candidate_seed": candidate_seed,
            "cv": {"splits": cv_splits, "repeats": cv_repeats, "folds": cv_splits * cv_repeats},
            "dataset": compact_dataset_summary(data, selected_go, feature_names, sample_meta),
            "split": {
                "n_train": len(split_info["train_ids"]),
                "n_test": len(split_info["test_ids"]),
                "feature_selection": "GO terms selected on training split only",
            },
            "performance": model_results,
            "candidates": candidate_df.to_dict(orient="records"),
            "note": "NO EMBEDDING in this version. SVD/UMAP features are not used by the final model.",
        }
        write_reproducibility_artifacts(
            repro_dir=repro_dir,
            records=records,
            split_info=split_info,
            selected_go=selected_go,
            candidates=candidates,
            feature_names=feature_names,
        )

    return results


def summarize_multiseed_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed metrics into mean/SD/SE/min/max summary per model."""
    metric_cols = [
        "cv_auroc_mean",
        "cv_auprc_mean",
        "test_auroc",
        "test_auprc",
        "test_f1",
        "test_precision",
        "test_recall",
        "n_go_selected",
        "D",
    ]
    rows: List[Dict[str, Any]] = []
    for model, group in run_df.groupby("model", sort=False):
        row: Dict[str, Any] = {"model": model, "n_runs": int(len(group))}
        for metric in metric_cols:
            values = group[metric].astype(float)
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = sd
            row[f"{metric}_se"] = float(sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def run_multiseed_analysis(
    data: DataBundle,
    seeds: Sequence[int],
    candidate_seed: int,
    cv_splits: int,
    cv_repeats: int,
    output_dir: Path,
    save_artifacts: bool,
) -> Dict[str, Any]:
    """Run the pipeline across multiple seeds and report reproducible mean +/- SD/SE.

    Each seed independently regenerates negatives, train/test split, GO selection,
    and model random states. Results are saved as CSV and JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    repro_root = output_dir / "reproducibility"
    rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    run_payloads: List[Dict[str, Any]] = []

    for i, seed in enumerate(seeds, start=1):
        print(f"\nMulti-seed run {i}/{len(seeds)} (seed={seed})")
        seed_repro_dir = repro_root / f"seed_{seed:04d}"
        if not save_artifacts:
            seed_repro_dir = output_dir / "_tmp_repro_not_saved"
        result = run_single_analysis(
            data=data,
            seed=seed,
            candidate_seed=candidate_seed,
            cv_splits=cv_splits,
            cv_repeats=cv_repeats,
            write_outputs=False,
            write_figures=False,
            with_ablation=False,
            with_shap=False,
            repro_dir=seed_repro_dir,
        )
        if not save_artifacts:
            for path in seed_repro_dir.glob("*"):
                path.unlink()
            seed_repro_dir.rmdir()

        run_payloads.append(result)
        for model_name, metrics in result["performance"].items():
            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "n_go_selected": result["dataset"]["n_go_selected"],
                    "D": result["dataset"]["D"],
                    "n_train": result["split"]["n_train"],
                    "n_test": result["split"]["n_test"],
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key
                        in {
                            "cv_auroc_mean",
                            "cv_auroc_std",
                            "cv_auroc_se",
                            "cv_auprc_mean",
                            "cv_auprc_std",
                            "test_auroc",
                            "test_auprc",
                            "test_f1",
                            "test_precision",
                            "test_recall",
                        }
                    },
                }
            )
        for candidate in result["candidates"]:
            candidate_rows.append({"seed": seed, **candidate})

    run_df = pd.DataFrame(rows)
    summary_df = summarize_multiseed_runs(run_df)
    candidate_df = pd.DataFrame(candidate_rows)
    run_df.to_csv(output_dir / "multiseed_runs.csv", index=False)
    summary_df.to_csv(output_dir / "multiseed_summary.csv", index=False)
    candidate_df.to_csv(output_dir / "multiseed_candidate_results.csv", index=False)

    payload = {
        "generated_by": "run_no_embedding_reproducible.py --seeds",
        "seeds": list(seeds),
        "candidate_seed": candidate_seed,
        "cv": {"splits": cv_splits, "repeats": cv_repeats, "folds": cv_splits * cv_repeats},
        "summary": summary_df.to_dict(orient="records"),
        "runs": run_payloads,
        "note": (
            "Multi-seed results are reproducible for this fixed seed list. "
            "Each seed regenerates negatives, train/test split, GO feature selection, "
            "and model random states."
        ),
    }
    save_json(output_dir / "results_no_embedding_multiseed.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    """CLI entry point: single-seed or multi-seed reproducible analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_REFERENCE_SEED)
    parser.add_argument("--candidate-seed", type=int, default=DEFAULT_CANDIDATE_SEED)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional fixed seed list for reproducible averaging, e.g. '1-20' or '1,2,3'.",
    )
    parser.add_argument(
        "--reference-seed",
        type=int,
        default=DEFAULT_REFERENCE_SEED,
        help="Reference seed used for standard tables when --seeds is provided.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="With --seeds, skip regenerating the standard single-seed tables/figures.",
    )
    parser.add_argument(
        "--multi-output-dir",
        type=Path,
        default=TABLE_DIR / "multiseed",
        help="Directory for multi-seed CSV/JSON outputs.",
    )
    parser.add_argument(
        "--no-multiseed-artifacts",
        action="store_true",
        help="With --seeds, skip saving per-seed samples/splits/selected terms.",
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()

    print("Loading data...")
    data = load_data()
    print(f"  Pathways: {len(data.pathways)} (KEGG={data.n_kegg}, AraCyc={data.n_aracyc})")
    print(f"  Genes with GO: {len(data.gene_go)}")
    print(f"  Filtered GO terms: {len(data.go_terms)}")

    if args.seeds:
        seeds = parse_seed_list(args.seeds)
        if not args.no_reference:
            print(f"\nReference run (seed={args.reference_seed})")
            run_single_analysis(
                data=data,
                seed=args.reference_seed,
                candidate_seed=args.candidate_seed,
                cv_splits=args.cv_splits,
                cv_repeats=args.cv_repeats,
                write_outputs=True,
                write_figures=not args.no_figures,
                with_ablation=True,
                with_shap=True,
                repro_dir=REPRO_DIR,
            )
        print(f"\nRunning reproducible multi-seed average: seeds={seeds}")
        results = run_multiseed_analysis(
            data=data,
            seeds=seeds,
            candidate_seed=args.candidate_seed,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            output_dir=args.multi_output_dir,
            save_artifacts=not args.no_multiseed_artifacts,
        )
        print("Done.")
        print(f"  Multi-seed results: {args.multi_output_dir / 'results_no_embedding_multiseed.json'}")
        print(f"  Multi-seed summary: {args.multi_output_dir / 'multiseed_summary.csv'}")
        return results

    results = run_single_analysis(
        data=data,
        seed=args.seed,
        candidate_seed=args.candidate_seed,
        cv_splits=args.cv_splits,
        cv_repeats=args.cv_repeats,
        write_outputs=True,
        write_figures=not args.no_figures,
        with_ablation=True,
        with_shap=True,
        repro_dir=REPRO_DIR,
    )

    print("Done.")
    print(f"  Main results: {TABLE_DIR / 'results_no_embedding.json'}")
    print(f"  Reproducibility artifacts: {REPRO_DIR}")
    return results


if __name__ == "__main__":
    main()
