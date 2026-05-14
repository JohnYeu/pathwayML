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
from scipy.stats import hypergeom, ks_2samp
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
OUTPUT_DIR = Path("outputs")
# Cap pairwise Jaccard computation to at most this many genes for speed
MAX_JACCARD_GENES = 15
PRIMARY_NEGATIVE_TYPES = [
    "empirical_size_matched_random",
    "full_replacement_shuffled",
    "corrupted_pathway",
    "cross_pathway_mixture",
]


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
    for path in [FIG_DIR, TABLE_DIR, REPRO_DIR, OUTPUT_DIR]:
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Feature engineering: GO frequency, Jaccard statistics, size
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pathway metadata and family assignment
# ---------------------------------------------------------------------------

def source_database(pathway_id: str) -> str:
    """Return the source database for a curated pathway ID."""
    return "AraCyc" if pathway_id.startswith("AC_") else "KEGG"


def contains_any(text: str, words: Sequence[str]) -> bool:
    """Check whether any keyword appears as a substring in text (case-sensitive)."""
    return any(word in text for word in words)


def assign_pathway_family(pathway_id: str, pathway_name: str) -> str:
    """Deterministically assign broad pathway families for metadata and LOFO."""
    s = f"{pathway_id} {pathway_name}".lower()
    if contains_any(s, ["degradation", "catabolism", "catabolic", "breakdown", "salvage", "detoxification", "detox"]):
        return "Degradation/catabolism"
    if contains_any(s, ["alanine", "arginine", "asparagine", "aspartate", "aspartic", "cysteine", "glutamate", "glutamic", "glutamine", "glycine", "histidine", "isoleucine", "leucine", "lysine", "methionine", "phenylalanine", "proline", "serine", "threonine", "tryptophan", "tyrosine", "valine", "amino acid", "branched-chain"]):
        return "Amino acid metabolism"
    if contains_any(s, ["lipid", "fatty", "glycerol", "glycerolipid", "glycerophospholipid", "sphingolipid", "cutin", "suberin", "wax", "sterol", "steroid", "linolenic", "linoleic", "phospholipid", "triacylglycerol", "acyl-lipid"]):
        return "Lipid metabolism"
    if contains_any(s, ["carbohydrate", "starch", "sucrose", "cellulose", "glycolysis", "gluconeogenesis", "glucose", "fructose", "galactose", "mannose", "xylose", "pentose", "glycan", "pectin", "hemicellulose", "cell wall", "trehalose"]):
        return "Carbohydrate metabolism"
    if contains_any(s, ["photosynthesis", "oxidative phosphorylation", "carbon fixation", "nitrogen metabolism", "sulfur metabolism", "methane", "atp", "respiration", "electron transport", "calvin", "photorespiration"]):
        return "Energy metabolism"
    if contains_any(s, ["cofactor", "vitamin", "folate", "riboflavin", "thiamine", "biotin", "porphyrin", "chlorophyll", "carotenoid", "heme", "tetrahydrofolate", "nicotinate", "pantothenate"]):
        return "Cofactor/vitamin metabolism"
    if contains_any(s, ["signaling", "signalling", "signal", "hormone", "auxin", "ethylene", "abscisic", "jasmonic", "salicylic", "brassinosteroid", "circadian", "mapk", "response", "transduction"]):
        return "Signalling/regulatory"
    if contains_any(s, ["flavonoid", "phenylpropanoid", "glucosinolate", "terpenoid", "alkaloid", "anthocyanin", "lignin", "isoprenoid", "phytoalexin", "secondary metabolite", "stilbenoid", "benzoxazinoid", "coumarin", "betalain"]) or "biosynth" in s or "biosynthesis" in s or "synthesis" in s:
        return "Specialized/other biosynthesis"
    if pathway_id.startswith("AC_"):
        return "Other AraCyc"
    return "Other KEGG/cellular"


def curated_pathway_records(data: DataBundle) -> List[Dict[str, Any]]:
    """Return positive pathway records with source/family metadata."""
    records = []
    for pid in sorted(data.pathways):
        name = data.pathway_names.get(pid, pid)
        records.append(
            {
                "id": pid,
                "label": 1,
                "type": "curated_pathway",
                "negative_type": "NA",
                "name": name,
                "genes": sorted(data.pathways[pid]),
                "source_database": source_database(pid),
                "source_family": assign_pathway_family(pid, name),
                "split": "unsplit",
            }
        )
    return records


def pathway_family_table(data: DataBundle) -> pd.DataFrame:
    """Canonical pathway-family assignment table used by all LOFO analyses.

    Keeping the family mapping in one place prevents different scripts from
    silently grouping the same pathway into different held-out families.
    """
    rows = []
    for pid, genes in data.pathways.items():
        name = data.pathway_names.get(pid, pid)
        rows.append(
            {
                "pathway_id": pid,
                "pathway_name": name,
                "family": assign_pathway_family(pid, name),
                "source": source_database(pid),
                "n_genes": len(genes),
                "jaccard_mean": pathway_jaccard_mean(
                    genes,
                    data.gene_go,
                    salt=f"fam:{pid}",
                    seed=DEFAULT_REFERENCE_SEED,
                ),
            }
        )
    return pd.DataFrame(rows)


def record_source_families(record: Dict[str, Any]) -> set[str]:
    """Return all pathway families recorded as sources for a sample/decoy."""
    families = set()
    for key in ("source_family", "source_family_1", "source_family_2", "source_family_3"):
        value = record.get(key)
        if value and str(value) != "NA":
            families.add(str(value))
    return families


def record_uses_source_family(record: Dict[str, Any], family: str) -> bool:
    """Whether a negative sample was generated from a pathway in `family`."""
    return family in record_source_families(record)


# ---------------------------------------------------------------------------
# Synthetic negative (decoy) generation
# ---------------------------------------------------------------------------

def overlap_coefficient(a: Iterable[str], b: Iterable[str]) -> float:
    """Overlap coefficient len(A & B) / min(len(A), len(B))."""
    set_a, set_b = set(a), set(b)
    denom = min(len(set_a), len(set_b))
    return float(len(set_a & set_b) / denom) if denom else 0.0


def closest_curated_overlap(genes: Iterable[str], curated_records: Sequence[Dict[str, Any]]) -> Tuple[float, str]:
    """Return max overlap coefficient and closest curated pathway ID."""
    gene_set = set(genes)
    best_overlap = 0.0
    best_id = ""
    for record in curated_records:
        ov = overlap_coefficient(gene_set, record["genes"])
        if ov > best_overlap:
            best_overlap = ov
            best_id = str(record["id"])
    return best_overlap, best_id


def split_counts(total: int, n_parts: int) -> List[int]:
    """Split total into n_parts approximately equal integer counts."""
    base, remainder = divmod(int(total), int(n_parts))
    return [base + (1 if i < remainder else 0) for i in range(n_parts)]


def empty_negative_record(sample_id: str, negative_type: str, seed: int, split: str) -> Dict[str, Any]:
    """Base negative metadata row with NA defaults for non-applicable fields."""
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "label": 0,
        "type": negative_type,
        "negative_type": negative_type,
        "name": sample_id,
        "genes": [],
        "gene_ids": "",
        "target_size": "NA",
        "actual_size": "NA",
        "seed": int(seed),
        "generation_seed": int(seed),
        "split": split,
        "source_pathway_id": "NA",
        "source_database": "NA",
        "source_family": "NA",
        "source_pathway_id_1": "NA",
        "source_pathway_id_2": "NA",
        "source_pathway_id_3": "NA",
        "source_family_1": "NA",
        "source_family_2": "NA",
        "source_family_3": "NA",
        "kept_fraction": "NA",
        "replaced_fraction": "NA",
        "replacement_fraction": "NA",
        "n_kept": "NA",
        "n_replaced": "NA",
        "random_fill_count": 0,
        "max_overlap_with_curated": "NA",
        "closest_curated_pathway": "NA",
    }


def finalize_negative_record(
    record: Dict[str, Any],
    genes: Iterable[str],
    curated_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Attach sorted genes, size, and closest-curated overlap metadata."""
    gene_list = sorted(set(genes))
    max_overlap, closest = closest_curated_overlap(gene_list, curated_records)
    record["genes"] = gene_list
    record["gene_ids"] = ",".join(gene_list)
    record["actual_size"] = len(gene_list)
    record["max_overlap_with_curated"] = float(max_overlap)
    record["closest_curated_pathway"] = closest
    return record


def valid_decoy(
    genes: Iterable[str],
    curated_records: Sequence[Dict[str, Any]],
    max_any_overlap: float = 0.80,
    min_size: int = 5,
) -> bool:
    """Return True if a generated decoy passes size and curated-overlap filters."""
    gene_set = set(genes)
    if len(gene_set) < min_size:
        return False
    max_overlap, _closest = closest_curated_overlap(gene_set, curated_records)
    return max_overlap <= max_any_overlap


def generate_empirical_size_matched_random(
    data: DataBundle,
    source_records: Sequence[Dict[str, Any]],
    curated_records: Sequence[Dict[str, Any]],
    rng: np.random.Generator,
    sample_id: str,
    seed: int,
    split: str,
    max_attempts: int = 2000,
) -> Dict[str, Any]:
    """Generate a size-matched random background decoy.

    Target sizes are drawn from the empirical positive pathway size
    distribution for the same split, ensuring no systematic size bias
    between positives and this negative class.
    """
    sizes = [len(record["genes"]) for record in source_records]
    for _attempt in range(max_attempts):
        target_size = int(rng.choice(sizes))
        genes = set(rng_choice_list(rng, data.background_genes, target_size, replace=False))
        if valid_decoy(genes, curated_records):
            rec = empty_negative_record(sample_id, "empirical_size_matched_random", seed, split)
            rec["target_size"] = target_size
            return finalize_negative_record(rec, genes, curated_records)
    raise RuntimeError(f"Could not generate empirical_size_matched_random after {max_attempts} attempts")


def generate_full_replacement_shuffled(
    data: DataBundle,
    source_records: Sequence[Dict[str, Any]],
    curated_records: Sequence[Dict[str, Any]],
    rng: np.random.Generator,
    sample_id: str,
    seed: int,
    split: str,
    max_attempts: int = 2000,
) -> Dict[str, Any]:
    """Generate a size-preserving, full-replacement pathway shuffle.

    Picks a source pathway's size but replaces ALL its genes with random
    background genes. This preserves the size distribution while removing
    all biological coherence — the strongest structural null hypothesis.
    """
    for _attempt in range(max_attempts):
        source = source_records[int(rng.integers(0, len(source_records)))]
        source_genes = set(source["genes"])
        target_size = len(source_genes)
        # Exclude source genes to ensure zero overlap with the original pathway
        replacement_pool = [gene for gene in data.background_genes if gene not in source_genes]
        exclusion_feasible = len(replacement_pool) >= target_size
        pool = replacement_pool if exclusion_feasible else data.background_genes
        genes = set(rng_choice_list(rng, pool, target_size, replace=False))
        if exclusion_feasible and genes & source_genes:
            continue
        if valid_decoy(genes, curated_records):
            rec = empty_negative_record(sample_id, "full_replacement_shuffled", seed, split)
            rec.update(
                {
                    "source_pathway_id": source["id"],
                    "source_database": source["source_database"],
                    "source_family": source["source_family"],
                    "target_size": target_size,
                    "replacement_fraction": 1.0,
                    "kept_fraction": 0.0,
                    "n_kept": 0,
                    "n_replaced": target_size,
                }
            )
            return finalize_negative_record(rec, genes, curated_records)
    raise RuntimeError(f"Could not generate full_replacement_shuffled after {max_attempts} attempts")


def generate_corrupted_pathway(
    data: DataBundle,
    source_records: Sequence[Dict[str, Any]],
    curated_records: Sequence[Dict[str, Any]],
    rng: np.random.Generator,
    sample_id: str,
    seed: int,
    split: str,
    max_attempts: int = 2000,
) -> Dict[str, Any]:
    """Generate a decoy with a small true pathway core plus random contamination.

    Keeps 20-50% of a real pathway's genes and replaces the rest with random
    background genes. The 0.60 self-overlap threshold ensures the result is
    sufficiently different from the source to be a meaningful negative.
    """
    for _attempt in range(max_attempts):
        source = source_records[int(rng.integers(0, len(source_records)))]
        source_genes = sorted(set(source["genes"]))
        target_size = len(source_genes)
        # Keep 20-50% of original genes — enough to retain partial signal
        # but too little to be a true pathway fragment
        kept_fraction = float(rng.uniform(0.20, 0.50))
        n_keep = int(round(kept_fraction * target_size))
        n_keep = max(1, min(n_keep, target_size - 1 if target_size > 1 else 1))
        kept = set(rng_choice_list(rng, source_genes, n_keep, replace=False))
        replacement_pool = [gene for gene in data.background_genes if gene not in source_genes and gene not in kept]
        n_replace = target_size - len(kept)
        if len(replacement_pool) < n_replace:
            continue
        genes = set(kept)
        genes.update(rng_choice_list(rng, replacement_pool, n_replace, replace=False))
        if len(genes) != target_size:
            continue
        # Reject if the result still looks too similar to the source
        if overlap_coefficient(genes, source_genes) > 0.60:
            continue
        if valid_decoy(genes, curated_records):
            rec = empty_negative_record(sample_id, "corrupted_pathway", seed, split)
            rec.update(
                {
                    "source_pathway_id": source["id"],
                    "source_database": source["source_database"],
                    "source_family": source["source_family"],
                    "target_size": target_size,
                    "kept_fraction": float(len(kept) / target_size),
                    "replaced_fraction": float(n_replace / target_size),
                    "replacement_fraction": float(n_replace / target_size),
                    "n_kept": len(kept),
                    "n_replaced": n_replace,
                }
            )
            return finalize_negative_record(rec, genes, curated_records)
    raise RuntimeError(f"Could not generate corrupted_pathway after {max_attempts} attempts")


def generate_cross_pathway_mixture(
    data: DataBundle,
    source_records: Sequence[Dict[str, Any]],
    curated_records: Sequence[Dict[str, Any]],
    rng: np.random.Generator,
    sample_id: str,
    seed: int,
    split: str,
    max_attempts: int = 4000,
) -> Dict[str, Any]:
    """Generate a heterogeneous cross-family pathway mixture.

    Draws genes from 2-3 pathways in different biological families, ensuring
    no single source contributes more than 60% of the final set. This creates
    plausible-looking decoys with pathway-like GO statistics but incoherent
    biological function.
    """
    sizes = [len(record["genes"]) for record in source_records]
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in source_records:
        by_family[str(record["source_family"])].append(record)
    families = sorted(by_family)

    for _attempt in range(max_attempts):
        target_size = int(rng.choice(sizes))
        if len(families) >= 2:
            fam1, fam2 = rng.choice(families, size=2, replace=False)
            p1 = by_family[str(fam1)][int(rng.integers(0, len(by_family[str(fam1)])))]
            p2 = by_family[str(fam2)][int(rng.integers(0, len(by_family[str(fam2)])))]
        else:
            p1, p2 = rng.choice(source_records, size=2, replace=False)

        n1 = max(1, target_size // 2)
        n2 = max(1, target_size - n1)
        genes: set = set()
        contrib: Dict[str, int] = {}
        pick1 = set(rng_choice_list(rng, p1["genes"], min(n1, len(p1["genes"])), replace=False))
        genes.update(pick1)
        contrib[str(p1["id"])] = len(pick1)
        pool2 = [gene for gene in p2["genes"] if gene not in genes]
        pick2 = set(rng_choice_list(rng, pool2, min(n2, len(pool2)), replace=False))
        genes.update(pick2)
        contrib[str(p2["id"])] = len(pick2)

        p3: Dict[str, Any] | None = None
        if len(genes) < target_size and len(families) >= 3:
            remaining_families = [fam for fam in families if fam not in {p1["source_family"], p2["source_family"]}]
            if remaining_families:
                fam3 = str(rng.choice(remaining_families))
                p3 = by_family[fam3][int(rng.integers(0, len(by_family[fam3])))]
                pool3 = [gene for gene in p3["genes"] if gene not in genes]
                pick3 = set(rng_choice_list(rng, pool3, min(target_size - len(genes), len(pool3)), replace=False))
                genes.update(pick3)
                contrib[str(p3["id"])] = len(pick3)

        random_fill_count = 0
        if len(genes) < target_size:
            fill_pool = [gene for gene in data.background_genes if gene not in genes]
            fill = set(rng_choice_list(rng, fill_pool, target_size - len(genes), replace=False))
            genes.update(fill)
            random_fill_count = len(fill)

        if len(genes) != target_size:
            continue
        # Reject if one source dominates — defeats the purpose of mixing
        if contrib and max(contrib.values()) / target_size > 0.60:
            continue
        if valid_decoy(genes, curated_records):
            rec = empty_negative_record(sample_id, "cross_pathway_mixture", seed, split)
            rec.update(
                {
                    "source_pathway_id_1": p1["id"],
                    "source_pathway_id_2": p2["id"],
                    "source_pathway_id_3": p3["id"] if p3 else "NA",
                    "source_family_1": p1["source_family"],
                    "source_family_2": p2["source_family"],
                    "source_family_3": p3["source_family"] if p3 else "NA",
                    "target_size": target_size,
                    "random_fill_count": random_fill_count,
                    "contribution_fraction_1": float(contrib.get(str(p1["id"]), 0) / target_size),
                    "contribution_fraction_2": float(contrib.get(str(p2["id"]), 0) / target_size),
                    "contribution_fraction_3": float(contrib.get(str(p3["id"]), 0) / target_size) if p3 else "NA",
                }
            )
            return finalize_negative_record(rec, genes, curated_records)
    raise RuntimeError(f"Could not generate cross_pathway_mixture after {max_attempts} attempts")


def generate_negative_records(
    data: DataBundle,
    source_records: Sequence[Dict[str, Any]],
    curated_records: Sequence[Dict[str, Any]],
    n_neg_total: int,
    seed: int,
    split: str,
    id_prefix: str,
) -> List[Dict[str, Any]]:
    """Generate all four primary synthetic negative classes for one split."""
    if not source_records:
        raise ValueError("source_records cannot be empty")
    counts = split_counts(n_neg_total, len(PRIMARY_NEGATIVE_TYPES))
    rng = np.random.default_rng(seed)
    generators = {
        "empirical_size_matched_random": generate_empirical_size_matched_random,
        "full_replacement_shuffled": generate_full_replacement_shuffled,
        "corrupted_pathway": generate_corrupted_pathway,
        "cross_pathway_mixture": generate_cross_pathway_mixture,
    }
    records: List[Dict[str, Any]] = []
    for negative_type, count in zip(PRIMARY_NEGATIVE_TYPES, counts):
        for i in range(count):
            sample_id = f"{id_prefix}_{negative_type}_{i:04d}"
            records.append(
                generators[negative_type](
                    data=data,
                    source_records=source_records,
                    curated_records=curated_records,
                    rng=rng,
                    sample_id=sample_id,
                    seed=seed,
                    split=split,
                )
            )
    return records


def build_split_samples(
    data: DataBundle,
    seed: int,
    negative_multiplier: int = 2,
    test_size: float = 0.20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    """Split positives first, then generate split-specific synthetic negatives."""
    if negative_multiplier < 1:
        raise ValueError("negative_multiplier must be >= 1")

    positives = curated_pathway_records(data)
    pos_idx = np.arange(len(positives))
    train_pos_idx, test_pos_idx = train_test_split(
        pos_idx,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    train_pos = [deepcopy(positives[int(i)]) for i in train_pos_idx]
    test_pos = [deepcopy(positives[int(i)]) for i in test_pos_idx]
    for record in train_pos:
        record["split"] = "train"
    for record in test_pos:
        record["split"] = "test"

    # Seed offsets (+10000 / +20000) ensure train and test negatives are
    # generated from independent RNG streams, preventing accidental overlap.
    train_neg = generate_negative_records(
        data=data,
        source_records=train_pos,
        curated_records=positives,
        n_neg_total=negative_multiplier * len(train_pos),
        seed=seed + 10_000,
        split="train",
        id_prefix=f"TRAIN_NEG_seed{seed}",
    )
    test_neg = generate_negative_records(
        data=data,
        source_records=test_pos,
        curated_records=positives,
        n_neg_total=negative_multiplier * len(test_pos),
        seed=seed + 20_000,
        split="test",
        id_prefix=f"TEST_NEG_seed{seed}",
    )

    records = train_pos + train_neg + test_pos + test_neg
    train_idx = np.arange(0, len(train_pos) + len(train_neg), dtype=int)
    test_idx = np.arange(len(train_idx), len(records), dtype=int)
    negative_records = train_neg + test_neg
    real_jaccards = [
        pathway_jaccard_mean(record["genes"], data.gene_go, salt=f"real:{record['id']}", seed=seed)
        for record in positives
    ]
    neg_type_counts = pd.Series([record["negative_type"] for record in negative_records]).value_counts().to_dict()
    neg_split_counts = (
        pd.DataFrame({"split": [record["split"] for record in negative_records], "negative_type": [record["negative_type"] for record in negative_records]})
        .groupby(["split", "negative_type"])
        .size()
        .to_dict()
    )
    split_info = {
        "random_state": seed,
        "test_size": test_size,
        "split_protocol": "positive_pathways_split_first_then_generate_split_specific_negatives",
        "train_ids": [records[int(i)]["id"] for i in train_idx],
        "test_ids": [records[int(i)]["id"] for i in test_idx],
        "train_positive_ids": [record["id"] for record in train_pos],
        "test_positive_ids": [record["id"] for record in test_pos],
        "train_negative_ids": [record["id"] for record in train_neg],
        "test_negative_ids": [record["id"] for record in test_neg],
    }
    meta = {
        "n_pos": len(positives),
        "n_train_pos": len(train_pos),
        "n_test_pos": len(test_pos),
        "n_neg": len(negative_records),
        "n_train_neg": len(train_neg),
        "n_test_neg": len(test_neg),
        "negative_multiplier": negative_multiplier,
        "negative_to_positive_ratio": f"{negative_multiplier}:1",
        "negative_counts": neg_type_counts,
        "negative_counts_by_split": {f"{k[0]}::{k[1]}": int(v) for k, v in neg_split_counts.items()},
        "real_pathway_jaccard_median": float(np.median(real_jaccards)),
        "negative_scheme": "empirical_size_matched_random__full_replacement_shuffled__corrupted_pathway__cross_pathway_mixture",
        "negative_type_definitions": {
            "empirical_size_matched_random": "Uniform background genes with target sizes sampled from the empirical positive pathway size distribution for the same split.",
            "full_replacement_shuffled": "A source pathway size is preserved but all original genes are replaced by background genes.",
            "corrupted_pathway": "A small true pathway core (20-50%) is kept and the remaining genes are replaced by random background genes.",
            "cross_pathway_mixture": "Genes are drawn from two or more different broad pathway families, with no single source contributing more than 60%.",
        },
        "pure_partial_training_negatives": False,
    }
    return records, meta, split_info, train_idx, test_idx


def build_samples(
    data: DataBundle,
    seed: int,
    negative_multiplier: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compatibility wrapper returning split-aware records and metadata."""
    records, meta, _split_info, _train_idx, _test_idx = build_split_samples(
        data=data,
        seed=seed,
        negative_multiplier=negative_multiplier,
    )
    return records, meta


# ---------------------------------------------------------------------------
# GO feature selection (training-only, MI-based)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

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
        "GO + Jaccard": idx_go + idx_jac,
        "GO + Size": idx_go + idx_size,
        "Jaccard + Size": idx_jac + idx_size,
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
                "test_f1": result["test_f1"],
                "test_precision": result["test_precision"],
                "test_recall": result["test_recall"],
            }
        )
    df = pd.DataFrame(rows)
    full = float(df.loc[df["configuration"] == "Full model", "test_auroc"].iloc[0])
    df["delta_vs_full"] = df["test_auroc"] - full
    return df


# ---------------------------------------------------------------------------
# Candidate scoring and boundary probes
# ---------------------------------------------------------------------------

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
    """Score each deterministic candidate gene set with the trained model.

    The `novel` flag is retained only as a conservative diagnostic column for
    downstream review. Manuscript text should not treat this model score as
    standalone evidence of novel pathway discovery.
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


def score_boundary_partial_probes(
    test_records: Sequence[Dict[str, Any]],
    model: Any,
    selected_go: Sequence[str],
    data: DataBundle,
    seed: int,
) -> pd.DataFrame:
    """Score pure partial pathway fragments as boundary probes, not negatives.

    These probes are generated only from held-out test positives and are never
    included in training or AUROC/AUPRC calculations. A high score is expected
    for some fragments because they may retain real pathway-like coherence.
    """
    # Seed offset +30000 keeps boundary probes independent from train/test
    # negative generation (+10000/+20000)
    rng = np.random.default_rng(seed + 30_000)
    rows: List[Dict[str, Any]] = []
    for record in test_records:
        if int(record.get("label", -1)) != 1:
            continue
        source_genes = sorted(set(record["genes"]))
        if len(source_genes) < 5:
            continue
        keep_fraction = float(rng.uniform(0.50, 0.80))
        probe_size = int(round(keep_fraction * len(source_genes)))
        probe_size = max(5, min(probe_size, len(source_genes)))
        genes = sorted(rng_choice_list(rng, source_genes, probe_size, replace=False))
        fv, _names, _groups = build_feature_matrix(
            [{"id": f"BOUNDARY_{record['id']}", "genes": genes, "label": -1}],
            selected_go,
            data,
            seed=seed,
        )
        score = float(model.predict_proba(fv)[0, 1])
        rows.append(
            {
                "seed": seed,
                "probe_type": "pure_partial_boundary_probe",
                "source_pathway_id": record["id"],
                "source_database": record.get("source_database", source_database(str(record["id"]))),
                "source_family": record.get("source_family", assign_pathway_family(str(record["id"]), str(record.get("name", record["id"])))),
                "source_size": len(source_genes),
                "probe_size": len(genes),
                "keep_fraction": float(len(genes) / len(source_genes)),
                "gene_ids": ",".join(genes),
                "model_score": score,
            }
        )
    return pd.DataFrame(rows)


def summarize_boundary_probes(boundary_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize boundary probe scores overall and by source family."""
    if boundary_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for family, group in [("ALL", boundary_df)] + list(boundary_df.groupby("source_family", sort=True)):
        scores = group["model_score"].astype(float)
        rows.append(
            {
                "source_family": family,
                "n_probes": int(len(group)),
                "mean_score": float(scores.mean()),
                "sd_score": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "median_score": float(scores.median()),
                "iqr_score": f"{scores.quantile(0.25):.3f}-{scores.quantile(0.75):.3f}",
                "pct_score_ge_0_5": float((scores >= 0.5).mean() * 100.0),
                "pct_score_ge_0_7": float((scores >= 0.7).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def plot_boundary_probe_scores(boundary_df: pd.DataFrame) -> None:
    """Plot boundary-probe score distribution by source family."""
    if boundary_df.empty:
        return
    order = (
        boundary_df.groupby("source_family")["model_score"]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    data_to_plot = [boundary_df.loc[boundary_df["source_family"] == fam, "model_score"].values for fam in order]
    plt.figure(figsize=(9, 4.8))
    plt.boxplot(data_to_plot, labels=order, showfliers=False)
    plt.axhline(0.5, color="#A0AEC0", linestyle="--", linewidth=1)
    plt.ylabel("Model score")
    plt.title("Pure partial pathway boundary probes")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_boundary_partial_probe_scores.png", dpi=300)
    plt.savefig(FIG_DIR / "fig_boundary_partial_probe_scores.pdf")
    plt.close()


# ---------------------------------------------------------------------------
# I/O helpers and table writing
# ---------------------------------------------------------------------------

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


def negative_metadata_frame(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Return one reproducibility row per generated negative sample.

    The table intentionally stores both human-readable metadata and the exact
    comma-separated gene set. This makes it possible to audit every synthetic
    decoy without re-running the sampler.
    """
    required_cols = [
        "sample_id",
        "negative_type",
        "gene_ids",
        "target_size",
        "actual_size",
        "seed",
        "split",
        "source_pathway_id",
        "source_database",
        "source_family",
        "source_pathway_id_1",
        "source_pathway_id_2",
        "source_pathway_id_3",
        "source_family_1",
        "source_family_2",
        "source_family_3",
        "kept_fraction",
        "replaced_fraction",
        "replacement_fraction",
        "n_kept",
        "n_replaced",
        "contribution_fraction_1",
        "contribution_fraction_2",
        "contribution_fraction_3",
        "random_fill_count",
        "max_overlap_with_curated",
        "closest_curated_pathway",
    ]
    rows = []
    for record in records:
        if int(record.get("label", -1)) != 0:
            continue
        row = {col: record.get(col, "NA") for col in required_cols}
        row["sample_id"] = record.get("sample_id", record.get("id", ""))
        row["negative_type"] = record.get("negative_type", record.get("type", ""))
        row["gene_ids"] = record.get("gene_ids", ",".join(record.get("genes", [])))
        row["actual_size"] = record.get("actual_size", len(record.get("genes", [])))
        rows.append(row)
    return pd.DataFrame(rows, columns=required_cols)


def negative_design_summary_frame(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Summarize negative balance, size matching, and size-distribution checks."""
    pos_sizes = np.asarray([len(record["genes"]) for record in records if int(record.get("label", -1)) == 1], dtype=float)
    neg_records = [record for record in records if int(record.get("label", -1)) == 0]
    neg_sizes = np.asarray([len(record["genes"]) for record in neg_records], dtype=float)
    rows: List[Dict[str, Any]] = []

    def add_row(name: str, subset: Sequence[Dict[str, Any]]) -> None:
        sizes = np.asarray([len(record["genes"]) for record in subset], dtype=float)
        if len(sizes) == 0:
            return
        ks = ks_2samp(pos_sizes, sizes) if len(pos_sizes) and len(sizes) else None
        rows.append(
            {
                "comparison": name,
                "n_positive": int(len(pos_sizes)),
                "n_negative": int(len(sizes)),
                "negative_to_positive_ratio": float(len(sizes) / len(pos_sizes)) if len(pos_sizes) else np.nan,
                "median_size_positive": float(np.median(pos_sizes)),
                "median_size_negative": float(np.median(sizes)),
                "mean_size_positive": float(np.mean(pos_sizes)),
                "mean_size_negative": float(np.mean(sizes)),
                "ks_statistic": float(ks.statistic) if ks else np.nan,
                "ks_p_value": float(ks.pvalue) if ks else np.nan,
            }
        )

    add_row("all_negatives", neg_records)
    for negative_type in PRIMARY_NEGATIVE_TYPES:
        add_row(negative_type, [record for record in neg_records if record.get("negative_type") == negative_type])
    return pd.DataFrame(rows)


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
    perf_df.to_csv(TABLE_DIR / "table1_seed42_performance.csv")

    shap_df.to_csv(TABLE_DIR / "table2_shap_importance.csv", index=False)
    ablation_df.to_csv(TABLE_DIR / "table3_ablation.csv", index=False)
    ablation_df.to_json(TABLE_DIR / "ablation_full.json", orient="records", indent=2)
    fs_df.to_csv(TABLE_DIR / "feature_selection_cv.csv", index=False)
    mi_df.to_csv(TABLE_DIR / "selected_go_terms.csv", index=False)
    candidate_df.to_csv(TABLE_DIR / "candidate_results.csv", index=False)
    candidate_df.to_csv(TABLE_DIR / "table8_candidate_scoring.csv", index=False)
    neg_meta_df = negative_metadata_frame(records)
    neg_meta_df.to_csv(TABLE_DIR / "negative_metadata.csv", index=False)
    negative_design_summary_frame(records).to_csv(TABLE_DIR / "negative_design_summary.csv", index=False)

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


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

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
    plt.savefig(FIG_DIR / "fig_ablation.png", dpi=300)
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


def write_intermediate_variables(
    seed: int,
    records: Sequence[Dict[str, Any]],
    split_info: Dict[str, Any],
    selected_go: Sequence[str],
    feature_names: Sequence[str],
    x_all: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Save model-input arrays needed to reproduce a run without resampling.

    JSON artifacts store the human-readable samples and selected GO terms.
    This NPZ stores the exact numeric feature matrix and train/test indices used
    by the classifiers. Together they are the smallest practical set of
    intermediate variables needed to audit or replay the fitted benchmark.
    """
    out_dir = OUTPUT_DIR / "intermediate" / f"seed_{seed:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = np.asarray([str(record["id"]) for record in records], dtype=str)
    sample_types = np.asarray([str(record.get("type", "")) for record in records], dtype=str)
    sample_splits = np.asarray([str(record.get("split", "")) for record in records], dtype=str)
    np.savez_compressed(
        out_dir / "model_input_arrays.npz",
        X_all=x_all,
        y_all=y_all,
        train_idx=train_idx,
        test_idx=test_idx,
        sample_ids=sample_ids,
        sample_types=sample_types,
        sample_splits=sample_splits,
        feature_names=np.asarray(list(feature_names), dtype=str),
        selected_go=np.asarray(list(selected_go), dtype=str),
    )
    save_json(
        out_dir / "model_input_manifest.json",
        {
            "seed": seed,
            "n_samples": int(len(records)),
            "n_features": int(x_all.shape[1]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "arrays": {
                "X_all": "Feature matrix in sample order.",
                "y_all": "Binary labels in sample order.",
                "train_idx": "Row indices used for training.",
                "test_idx": "Row indices used for held-out testing.",
                "sample_ids": "Sample IDs aligned to X_all rows.",
                "sample_types": "curated_pathway or negative type aligned to X_all rows.",
                "sample_splits": "train/test split label aligned to X_all rows.",
                "feature_names": "Column names aligned to X_all columns.",
                "selected_go": "GO terms selected using the training split only.",
            },
            "split_protocol": split_info.get("split_protocol"),
            "source_files": {
                "human_readable_samples": str(REPRO_DIR / "samples.json") if seed == DEFAULT_REFERENCE_SEED else "tables/multiseed/reproducibility/seed_XXXX/samples.json",
                "human_readable_splits": str(REPRO_DIR / "splits.json") if seed == DEFAULT_REFERENCE_SEED else "tables/multiseed/reproducibility/seed_XXXX/splits.json",
            },
        },
    )


def compact_dataset_summary(
    data: DataBundle,
    selected_go: Sequence[str],
    feature_names: Sequence[str],
    sample_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a lightweight dataset summary dict for multi-seed JSON output."""
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


# ---------------------------------------------------------------------------
# Pipeline orchestration (single-seed and multi-seed)
# ---------------------------------------------------------------------------

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

    print(f"Splitting positives and generating split-specific synthetic negatives (seed={seed})...")
    # The main benchmark now splits curated positives first. Negative decoys are
    # generated separately from the train and test positive pools so empirical
    # size matching cannot accidentally use test-positive sizes during training.
    records, sample_meta, split_info, train_idx, test_idx = build_split_samples(data, seed=seed)
    y_all = np.asarray([record["label"] for record in records], dtype=int)
    train_records = [records[int(i)] for i in train_idx]

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
    write_intermediate_variables(
        seed=seed,
        records=records,
        split_info=split_info,
        selected_go=selected_go,
        feature_names=feature_names,
        x_all=x_all,
        y_all=y_all,
        train_idx=train_idx,
        test_idx=test_idx,
    )

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
    boundary_df = score_boundary_partial_probes(
        test_records=[records[int(i)] for i in test_idx],
        model=xgb_model,
        selected_go=selected_go,
        data=data,
        seed=seed,
    )
    boundary_summary_df = summarize_boundary_probes(boundary_df)

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
        boundary_df.to_csv(TABLE_DIR / "boundary_partial_probe_scores.csv", index=False)
        boundary_summary_df.to_csv(TABLE_DIR / "boundary_partial_probe_summary.csv", index=False)
        plot_boundary_probe_scores(boundary_df)
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
            "boundary_partial_probes": boundary_summary_df.to_dict(orient="records"),
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
    run_df.to_csv(TABLE_DIR / "multiseed_per_seed_results.csv", index=False)
    summary_df.to_csv(TABLE_DIR / "table2_multiseed_summary.csv", index=False)

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
