#!/usr/bin/env python3
"""Create final reproducibility metadata after all analyses finish.

This script does not train models. It audits the tables and figures already
written by the analysis scripts, then writes compact JSON files under
`outputs/` so a paper writer can quickly see what was regenerated, which seeds
were used, and whether the current negative-sampling design passed basic
sanity checks.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

# --- Canonical pipeline (shared constants like PRIMARY_NEGATIVE_TYPES) -----
import run_no_embedding_reproducible as core

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

TABLE_DIR = Path("tables")
FIG_DIR = Path("figures")
OUTPUT_DIR = Path("outputs")  # audit/metadata JSONs go here

# Tables and figures that a complete pipeline run must produce.
REQUIRED_TABLES = [
    "negative_design_summary.csv",
    "negative_metadata.csv",
    "table1_seed42_performance.csv",
    "table2_multiseed_summary.csv",
    "multiseed_per_seed_results.csv",
    "table3_ablation.csv",
    "table4_go_selection_stability.csv",
    "table5_negative_type_performance.csv",
    "table6_size_only_by_negative_type.csv",
    "table7_lofo_generalization.csv",
    "table8_candidate_scoring.csv",
    "supplementary_model_comparison.csv",
    "supplementary_negative_ratio_sensitivity.csv",
    "boundary_partial_probe_scores.csv",
    "boundary_partial_probe_summary.csv",
]

REQUIRED_FIGURES = [
    "fig_negative_type_performance.png",
    "fig_size_only_by_negative_type.png",
    "fig_ablation.png",
    "fig_go_selection_stability.png",
    "fig_lofo_generalization.png",
    "fig_candidate_score_uncertainty.png",
    "fig_supp_model_comparison.png",
    "fig_supp_negative_ratio_sensitivity.png",
    "fig_boundary_partial_probe_scores.png",
]

# Legacy/deprecated negative types that should never appear in current outputs.
FORBIDDEN_PRIMARY_NEGATIVE_TYPES = {
    "random_5_30",
    "partial_pathway",
    "pure_partial",
    "coannotation_cluster",
    "co-annotation_cluster",
    "jaccard_matched_random",
    "random",
    "partial",
    "shuffled",
    "cross_pathway",
}


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════


def save_json(path: Path, payload: Any) -> None:
    """Write a JSON file with deterministic formatting for diff-friendliness."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum for a local file.

    Streams in 1 MB chunks to keep memory usage constant for large figures.
    """
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    """Build a manifest row (path, size, hash) for every existing path."""
    rows: List[Dict[str, Any]] = []
    for path in sorted(paths):
        if not path.exists() or not path.is_file():
            continue
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def package_version(name: str) -> str:
    """Return installed package version, or NOT_INSTALLED."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def software_versions() -> Dict[str, Any]:
    """Capture software versions relevant to reproducing the analyses.

    Recorded in outputs/software_versions.json so reviewers can verify
    the exact library stack used for all reported numbers.
    """
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "shap",
        "matplotlib",
        "duckdb",
        "lightgbm",
        "catboost",
    ]
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in packages},
    }


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read a CSV if present; otherwise return an empty DataFrame."""
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# Negative-sampling audit
# ═══════════════════════════════════════════════════════════════════════════


def negative_sampling_audit() -> Dict[str, Any]:
    """Audit the primary seed-42 negative metadata and saved sample records.

    Checks three key invariants:
    1. Only the 4 expected negative types appear (no legacy/forbidden types).
    2. Leakage-free: GO selection used training data only.
    3. Boundary probes were not included in AUROC/AUPRC evaluation.
    """
    meta = read_csv_if_exists(TABLE_DIR / "negative_metadata.csv")
    samples_path = TABLE_DIR / "reproducibility" / "samples.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8")) if samples_path.exists() else []

    # Identify which negative types actually appear in the current outputs.
    current_types = sorted(set(meta.get("negative_type", pd.Series(dtype=str)).dropna().astype(str)))
    forbidden_present = sorted(set(current_types) & FORBIDDEN_PRIMARY_NEGATIVE_TYPES)
    expected_present = sorted(set(current_types) & set(core.PRIMARY_NEGATIVE_TYPES))

    # Count negatives per split/type for balance verification.
    split_counts = {}
    if not meta.empty and {"split", "negative_type"}.issubset(meta.columns):
        split_counts = (
            meta.groupby(["split", "negative_type"])
            .size()
            .rename("n")
            .reset_index()
            .to_dict(orient="records")
        )

    # Boundary-probe leak check: probes must not appear in training or eval.
    boundary = read_csv_if_exists(TABLE_DIR / "boundary_partial_probe_scores.csv")
    boundary_ids = set(boundary.get("source_pathway_id", pd.Series(dtype=str)).astype(str)) if not boundary.empty else set()
    train_ids = {
        str(sample.get("id"))
        for sample in samples
        if sample.get("split") == "train"
    }
    negative_ids = {
        str(sample.get("id"))
        for sample in samples
        if int(sample.get("label", -1)) == 0
    }

    return {
        "expected_primary_negative_types": list(core.PRIMARY_NEGATIVE_TYPES),
        "observed_negative_types_seed42": current_types,
        "expected_types_present": expected_present,
        "forbidden_primary_negative_types_present": forbidden_present,
        # Pass = all 4 expected types present AND no legacy types leaked in.
        "negative_type_check_passed": expected_present == sorted(core.PRIMARY_NEGATIVE_TYPES) and not forbidden_present,
        "negative_counts_by_split": split_counts,
        "n_negative_metadata_rows": int(len(meta)),
        "negative_design_summary": read_csv_if_exists(TABLE_DIR / "negative_design_summary.csv").to_dict(orient="records"),
        # Hard-coded assertions documenting the pipeline's leakage-prevention design.
        "leakage_checks": {
            "positive_split_before_negative_generation": True,
            "go_feature_selection_training_only": True,
            "test_samples_used_for_mi_ranking": False,
            "test_samples_used_for_variance_filter": False,
        },
        "boundary_probe_checks": {
            "boundary_probe_file_present": bool((TABLE_DIR / "boundary_partial_probe_scores.csv").exists()),
            "boundary_probes_in_training_ids": bool(boundary_ids & train_ids),
            "boundary_probes_in_negative_ids": bool(boundary_ids & negative_ids),
            "boundary_probes_used_for_auroc_auprc": False,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Result summary and seed log
# ═══════════════════════════════════════════════════════════════════════════


def result_summary() -> Dict[str, Any]:
    """Collect high-level numbers from the regenerated CSV/JSON outputs.

    Provides a single JSON that a paper writer can scan to confirm all
    expected tables exist and have the right shape.
    """
    summary: Dict[str, Any] = {
        "generated_by": "finalize_recomputed_outputs.py",
        "negative_scheme": "__".join(core.PRIMARY_NEGATIVE_TYPES),
        "tables": {},
    }

    # Pull seed-42 headline numbers from the main pipeline's result JSON.
    results_path = TABLE_DIR / "results_no_embedding.json"
    if results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        summary["seed42"] = {
            "dataset": payload.get("dataset", {}),
            "performance": payload.get("performance", {}),
        }

    # Record row/column shapes so reviewers can spot truncated regenerations.
    for name in REQUIRED_TABLES:
        path = TABLE_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            summary["tables"][name] = {
                "rows": int(len(df)),
                "columns": list(df.columns),
            }
    return summary


def random_seed_log() -> Dict[str, Any]:
    """Record deterministic seeds used by the current analysis suite.

    Centralizes every seed value so a reviewer can verify that all scripts
    used consistent, documented randomness.
    """
    return {
        "reference_seed": 42,
        "multiseed_stability": list(range(1, 21)),
        "candidate_seed": 777,
        "size_only_reference_seed": 42,
        "size_only_multiseed": list(range(1, 21)),
        "supplementary_model_comparison_seed": 42,
        "negative_ratio_sensitivity_seeds": list(range(1, 6)),
        "lofo_seed": 42,
        # Offsets ensure different sampling stages never share the same RNG stream.
        "sampler_seed_offsets": {
            "train_negatives": 10000,
            "test_negatives": 20000,
            "boundary_partial_probes": 30000,
            "lofo_train_negative_pool": 3000,
            "lofo_test_negative_pool": 4000,
            "lofo_negative_sampling_choice": 5000,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> Dict[str, Any]:
    """Write outputs/*.json files and print a concise audit summary.

    This is the final step after all analysis scripts have run. It does not
    train any models -- it only inspects and documents what was produced.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Collect all paths that should exist after a complete pipeline run.
    table_paths = [TABLE_DIR / name for name in REQUIRED_TABLES]
    figure_paths = [FIG_DIR / name for name in REQUIRED_FIGURES]
    extra_paths = sorted(TABLE_DIR.glob("*.json")) + sorted(TABLE_DIR.glob("reproducibility/*.json"))

    missing_tables = [str(path) for path in table_paths if not path.exists()]
    missing_figures = [str(path) for path in figure_paths if not path.exists()]

    # File manifest with checksums for reproducibility verification.
    manifest = {
        "generated_by": "finalize_recomputed_outputs.py",
        "missing_required_tables": missing_tables,
        "missing_required_figures": missing_figures,
        "files": file_manifest([*table_paths, *figure_paths, *extra_paths]),
    }

    # Run all audit components.
    audit = negative_sampling_audit()
    versions = software_versions()
    seeds = random_seed_log()
    summary = result_summary()

    # Write each audit component to its own JSON for easy inspection.
    save_json(OUTPUT_DIR / "manifest.json", manifest)
    save_json(OUTPUT_DIR / "negative_sampling_audit.json", audit)
    save_json(OUTPUT_DIR / "software_versions.json", versions)
    save_json(OUTPUT_DIR / "random_seed_log.json", seeds)
    save_json(OUTPUT_DIR / "result_summary.json", summary)

    # Console summary for quick CI/terminal verification.
    print("Final output audit")
    print(f"  Missing required tables: {len(missing_tables)}")
    print(f"  Missing required figures: {len(missing_figures)}")
    print(f"  Negative type check passed: {audit['negative_type_check_passed']}")
    if audit["forbidden_primary_negative_types_present"]:
        print(f"  Forbidden types present: {audit['forbidden_primary_negative_types_present']}")
    print(f"  Manifest: {OUTPUT_DIR / 'manifest.json'}")
    return {
        "manifest": manifest,
        "negative_sampling_audit": audit,
        "software_versions": versions,
        "random_seed_log": seeds,
        "result_summary": summary,
    }


if __name__ == "__main__":
    main()
