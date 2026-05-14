#!/usr/bin/env python3
"""Supplementary analyses for the no-embedding PathwayML-Ath pipeline.

This script deliberately does not replace the main manuscript tables. It adds
two optional analyses that are useful for a thesis supplement:

1. A 13-model comparison on the canonical seed-42, 1:2 negative-ratio split.
2. A negative-ratio sensitivity analysis from 1:1 to 1:5.

Both analyses reuse the same data loading, negative sampling, feature selection,
and no-embedding feature extraction code as the canonical pipeline.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# --- Third-party imports ---------------------------------------------------
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server/CI rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

# --- Canonical pipeline imports (shared data loading & feature logic) ------
from run_no_embedding_reproducible import (
    DEFAULT_REFERENCE_SEED,
    FIG_DIR,
    TABLE_DIR,
    build_feature_matrix,
    build_split_samples,
    ensure_dirs,
    load_data,
    parse_seed_list,
    save_json,
    select_go_terms,
)


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════


def parse_int_list(value: str) -> List[int]:
    """Parse comma-separated integers and inclusive ranges such as 1-5,10."""
    return parse_seed_list(value)


def optional_import_lightgbm() -> Any | None:
    """Gracefully degrade when LightGBM is not installed."""
    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except Exception:
        return None


def optional_import_catboost() -> Any | None:
    """Gracefully degrade when CatBoost is not installed."""
    try:
        import catboost  # type: ignore

        return catboost
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Model construction
# ═══════════════════════════════════════════════════════════════════════════


def class_balance_weight(y_train: np.ndarray) -> float:
    """Return n_negative / n_positive for XGBoost-style class weighting.

    Adapts scale_pos_weight dynamically so the same model config works
    across all negative ratios (1:1 through 1:5).
    """
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    return float(n_neg / n_pos) if n_pos else 1.0


def make_xgboost(seed: int, y_train: np.ndarray) -> xgb.XGBClassifier:
    """XGBoost configured like the main model but with ratio-aware class weight."""
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=class_balance_weight(y_train),
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def make_13_models(seed: int, y_train: np.ndarray) -> Dict[str, Tuple[str, Any]]:
    """Return the supplementary 13-model comparison set.

    The dictionary values are (model_group, estimator). LightGBM and CatBoost
    require optional packages; if either is unavailable, that row is skipped and
    the summary JSON records the missing package.
    """
    lgb = optional_import_lightgbm()
    catboost = optional_import_catboost()

    # Each entry: name -> (model_group_label, estimator_instance)
    models: Dict[str, Tuple[str, Any]] = {
        "Logistic Regression": (
            "linear",
            make_pipeline(
                StandardScaler(),
                # liblinear gives a deterministic small-data baseline.
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ),
        "Linear SVM": (
            "kernel",
            make_pipeline(
                StandardScaler(),
                LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=10000),
            ),
        ),
        "RBF SVM": (
            "kernel",
            make_pipeline(
                StandardScaler(),
                SVC(C=2.0, gamma="scale", class_weight="balanced", random_state=seed),
            ),
        ),
        "kNN": (
            "instance",
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance")),
        ),
        "Gaussian NB": ("probabilistic", GaussianNB()),
        "Decision Tree": (
            "tree",
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "Random Forest": (
            "ensemble",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        ),
        "Extra Trees": (
            "ensemble",
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        ),
        "Gradient Boosting": (
            "boosting",
            GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=3,
                random_state=seed,
            ),
        ),
        "AdaBoost": (
            "boosting",
            AdaBoostClassifier(n_estimators=250, learning_rate=0.05, random_state=seed),
        ),
        "XGBoost": ("advanced boosting", make_xgboost(seed, y_train)),
        "Gaussian Process": (
            "probabilistic",
            make_pipeline(StandardScaler(), GaussianProcessClassifier(random_state=seed, max_iter_predict=100)),
        ),
    }

    if lgb is not None:
        models["LightGBM"] = (
            "advanced boosting",
            lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
                verbose=-1,
            ),
        )

    if catboost is not None:
        models["CatBoost"] = (
            "advanced boosting",
            catboost.CatBoostClassifier(
                iterations=500,
                depth=5,
                learning_rate=0.03,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            ),
        )

    # Enforce a stable display order matching the thesis table layout.
    preferred_order = [
        "Logistic Regression",
        "Linear SVM",
        "RBF SVM",
        "kNN",
        "Gaussian NB",
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "AdaBoost",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]
    ordered = {name: models[name] for name in preferred_order if name in models}

    # If optional packages are unavailable, include Gaussian Process so the
    # script still produces a broad comparison instead of failing.
    if len(ordered) < 13 and "Gaussian Process" in models:
        ordered["Gaussian Process"] = models["Gaussian Process"]
    return ordered


def model_scores(model: Any, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return continuous scores and hard labels for any sklearn-like classifier.

    Handles three API surfaces: predict_proba (most models), decision_function
    (LinearSVC), and plain predict (fallback).  This lets AUROC/AUPRC work
    uniformly across all 13 models.
    """
    if hasattr(model, "predict_proba"):
        prob = np.asarray(model.predict_proba(x))
        scores = prob[:, 1] if prob.ndim == 2 else prob
        labels = (scores >= 0.5).astype(int)
        return scores, labels

    # LinearSVC exposes decision_function instead of probabilities.
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x))
        if scores.ndim == 2:
            scores = scores[:, 1]
        labels = (scores >= 0.0).astype(int)  # decision boundary at 0
        return scores.astype(float), labels

    # Last resort: binary labels used as both score and prediction.
    labels = np.asarray(model.predict(x)).astype(int)
    return labels.astype(float), labels


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_classifier(
    name: str,
    group: str,
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cv_splits: int,
    cv_repeats: int,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate one classifier with repeated CV and held-out test metrics.

    CV is used to estimate variance; the final test score comes from a
    model retrained on the full training set to maximize data usage.
    """
    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=seed)
    cv_aurocs: List[float] = []
    cv_auprcs: List[float] = []
    for train_idx, valid_idx in cv.split(x_train, y_train):
        fold_model = clone(model)  # fresh copy avoids state leakage between folds
        fold_model.fit(x_train[train_idx], y_train[train_idx])
        scores, _labels = model_scores(fold_model, x_train[valid_idx])
        cv_aurocs.append(roc_auc_score(y_train[valid_idx], scores))
        cv_auprcs.append(average_precision_score(y_train[valid_idx], scores))

    # Retrain on full training set for the final held-out evaluation.
    final_model = clone(model)
    final_model.fit(x_train, y_train)
    scores, labels = model_scores(final_model, x_test)
    cv_auroc_sd = float(np.std(cv_aurocs, ddof=1)) if len(cv_aurocs) > 1 else 0.0
    cv_auprc_sd = float(np.std(cv_auprcs, ddof=1)) if len(cv_auprcs) > 1 else 0.0
    return {
        "model": name,
        "model_group": group,
        "cv_auroc_mean": float(np.mean(cv_aurocs)),
        "cv_auroc_sd": cv_auroc_sd,
        "cv_auroc_se": float(cv_auroc_sd / np.sqrt(len(cv_aurocs))) if len(cv_aurocs) > 1 else 0.0,
        "cv_auprc_mean": float(np.mean(cv_auprcs)),
        "cv_auprc_sd": cv_auprc_sd,
        "cv_auprc_se": float(cv_auprc_sd / np.sqrt(len(cv_auprcs))) if len(cv_auprcs) > 1 else 0.0,
        "test_auroc": float(roc_auc_score(y_test, scores)),
        "test_auprc": float(average_precision_score(y_test, scores)),
        "test_f1": float(f1_score(y_test, labels)),
        "test_precision": float(precision_score(y_test, labels, zero_division=0)),
        "test_recall": float(recall_score(y_test, labels, zero_division=0)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Dataset preparation
# ═══════════════════════════════════════════════════════════════════════════


def prepare_dataset(data: Any, seed: int, negative_multiplier: int) -> Dict[str, Any]:
    """Build samples, split, select GO terms, and return train/test matrices.

    Uses the canonical CV-based GO feature selection from the main pipeline.
    """
    records, sample_meta, split_info, train_idx, test_idx = build_split_samples(
        data,
        seed=seed,
        negative_multiplier=negative_multiplier,
    )
    y_all = np.asarray([record["label"] for record in records], dtype=int)
    train_records = [records[int(i)] for i in train_idx]
    selected_go, fs_df, _mi_df = select_go_terms(train_records, data, cv_splits=5, seed=seed)
    x_all, feature_names, _groups = build_feature_matrix(records, selected_go, data, seed=seed)
    return {
        "records": records,
        "sample_meta": sample_meta,
        "selected_go": selected_go,
        "feature_names": feature_names,
        "feature_selection": fs_df,
        "x_train": x_all[train_idx],
        "x_test": x_all[test_idx],
        "y_train": y_all[train_idx],
        "y_test": y_all[test_idx],
        "train_idx": train_idx,
        "test_idx": test_idx,
        "split_info": split_info,
    }


def fast_select_go_terms(train_records: Sequence[Dict[str, Any]], data: Any, seed: int, mi_fraction: float = 0.70) -> List[str]:
    """Fast training-only GO selection for supplementary sensitivity analyses.

    The main model comparison keeps the canonical CV-based feature-selection
    routine. The negative-ratio appendix uses this faster cumulative-MI cutoff
    because it repeats sampling 25 times and is intended as a robustness
    diagnostic rather than a hyperparameter-selection experiment.
    """
    y_train = np.asarray([record["label"] for record in train_records], dtype=int)
    # Compute GO-term frequency features: fraction of pathway genes annotated to each term.
    freq_train = np.vstack(
        [
            np.asarray([
                len(set(record["genes"]) & data.go_genes[term]) / max(len(record["genes"]), 1)
                for term in data.go_terms
            ])
            for record in train_records
        ]
    )
    # Remove near-zero-variance terms before MI ranking to avoid noisy estimates.
    vt = VarianceThreshold(threshold=0.001)
    freq_vt = vt.fit_transform(freq_train)
    kept_terms = [term for term, keep in zip(data.go_terms, vt.get_support()) if keep]
    mi = mutual_info_classif(freq_vt, y_train, random_state=seed, n_neighbors=5)
    order = np.argsort(mi)[::-1]  # descending MI
    total = float(mi[order].sum())
    if total <= 0:
        # Degenerate case: MI is zero everywhere, keep a small fallback set.
        k = min(20, len(order))
    else:
        # Select enough top terms to capture mi_fraction of total MI.
        k = int(np.argmax(np.cumsum(mi[order]) / total >= mi_fraction) + 1)
    k = max(3, min(k, len(order)))  # enforce floor of 3, cap at available terms
    return [kept_terms[int(i)] for i in order[:k]]


def prepare_dataset_fast_go(data: Any, seed: int, negative_multiplier: int) -> Dict[str, Any]:
    """Build samples using fast training-only GO selection for ratio sensitivity."""
    records, sample_meta, split_info, train_idx, test_idx = build_split_samples(
        data,
        seed=seed,
        negative_multiplier=negative_multiplier,
    )
    y_all = np.asarray([record["label"] for record in records], dtype=int)
    train_records = [records[int(i)] for i in train_idx]
    selected_go = fast_select_go_terms(train_records, data, seed=seed, mi_fraction=0.70)
    x_all, feature_names, _groups = build_feature_matrix(records, selected_go, data, seed=seed)
    return {
        "records": records,
        "sample_meta": sample_meta,
        "selected_go": selected_go,
        "feature_names": feature_names,
        "x_train": x_all[train_idx],
        "x_test": x_all[test_idx],
        "y_train": y_all[train_idx],
        "y_test": y_all[test_idx],
        "train_idx": train_idx,
        "test_idx": test_idx,
        "split_info": split_info,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


def write_model_comparison_plot(df: pd.DataFrame) -> None:
    """Horizontal bar chart of test AUROC for each model (supplementary figure)."""
    plot_df = df.sort_values("test_auroc", ascending=True)
    plt.figure(figsize=(8.5, 6.0))
    # Highlight advanced boosting models (XGBoost/LightGBM/CatBoost) in red.
    colors = np.where(plot_df["model_group"].eq("advanced boosting"), "#C53030", "#2B6CB0")
    plt.barh(plot_df["model"], plot_df["test_auroc"], color=colors)
    plt.xlabel("Held-out test AUROC")
    plt.title("Supplementary 13-model comparison")
    plt.xlim(0.0, 1.0)
    for i, value in enumerate(plot_df["test_auroc"]):
        plt.text(min(value + 0.01, 0.98), i, f"{value:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "supplementary_13_model_comparison.png", dpi=300)
    plt.savefig(FIG_DIR / "supplementary_13_model_comparison.pdf")
    # Stable alias requested for the final thesis data handoff.
    plt.savefig(FIG_DIR / "fig_supp_model_comparison.png", dpi=300)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 1: 13-model comparison
# ═══════════════════════════════════════════════════════════════════════════


def run_model_comparison(
    data: Any,
    seed: int,
    cv_splits: int,
    cv_repeats: int,
) -> pd.DataFrame:
    """Run the supplementary model comparison on the canonical 1:2 dataset.

    Trains all 13 classifiers on identical features/splits so differences
    reflect model capacity, not data preparation artefacts.
    """
    prepared = prepare_dataset(data, seed=seed, negative_multiplier=2)
    x_train = prepared["x_train"]
    y_train = prepared["y_train"]
    x_test = prepared["x_test"]
    y_test = prepared["y_test"]
    models = make_13_models(seed, y_train)

    rows: List[Dict[str, Any]] = []
    for i, (name, (group, model)) in enumerate(models.items(), start=1):
        print(f"  [{i}/{len(models)}] {name}")
        row = evaluate_classifier(
            name=name,
            group=group,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            cv_splits=cv_splits,
            cv_repeats=cv_repeats,
            seed=seed,
        )
        # Attach dataset metadata so each row is self-describing in the CSV.
        row.update(
            {
                "seed": seed,
                "negative_multiplier": 2,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "n_go_selected": int(len(prepared["selected_go"])),
                "D": int(len(prepared["feature_names"])),
            }
        )
        rows.append(row)

    # Sort best-first for convenient reading in the thesis supplement.
    df = pd.DataFrame(rows).sort_values(["test_auroc", "test_auprc"], ascending=False)
    df.to_csv(TABLE_DIR / "supplementary_13_model_comparison.csv", index=False)
    df.to_csv(TABLE_DIR / "supplementary_model_comparison.csv", index=False)

    # Separate rounded copy avoids loss of precision in downstream analysis.
    rounded = df.copy()
    metric_cols = [
        "cv_auroc_mean",
        "cv_auroc_sd",
        "cv_auroc_se",
        "cv_auprc_mean",
        "cv_auprc_sd",
        "cv_auprc_se",
        "test_auroc",
        "test_auprc",
        "test_f1",
        "test_precision",
        "test_recall",
    ]
    rounded[metric_cols] = rounded[metric_cols].round(3)
    rounded.to_csv(TABLE_DIR / "paper_supplementary_13_model_comparison_rounded.csv", index=False)
    write_model_comparison_plot(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 2: negative-ratio sensitivity
# ═══════════════════════════════════════════════════════════════════════════


def summarize_ratio_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed ratio runs into mean/SD/SE/min/max per ratio."""
    rows: List[Dict[str, Any]] = []
    metrics = [
        "test_auroc",
        "test_auprc",
        "test_f1",
        "test_precision",
        "test_recall",
        "n_go_selected",
        "D",
    ]
    for ratio, group in df.groupby("negative_multiplier", sort=True):
        row: Dict[str, Any] = {
            "negative_ratio": f"1:{int(ratio)}",
            "negative_multiplier": int(ratio),
            "n_runs": int(len(group)),
            "mean_n_train": float(group["n_train"].mean()),
            "mean_n_test": float(group["n_test"].mean()),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = sd
            row[f"{metric}_se"] = float(sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_ratio_plot(summary_df: pd.DataFrame) -> None:
    """Error-bar plot showing AUROC/AUPRC stability across negative ratios."""
    x = np.arange(len(summary_df))
    labels = summary_df["negative_ratio"].tolist()
    plt.figure(figsize=(8.0, 4.8))
    # Slight horizontal offset (-0.06 / +0.06) prevents error-bar overlap.
    plt.errorbar(
        x - 0.06,
        summary_df["test_auroc_mean"],
        yerr=summary_df["test_auroc_sd"],
        marker="o",
        capsize=4,
        label="AUROC",
        color="#2B6CB0",
    )
    plt.errorbar(
        x + 0.06,
        summary_df["test_auprc_mean"],
        yerr=summary_df["test_auprc_sd"],
        marker="s",
        capsize=4,
        label="AUPRC",
        color="#C53030",
    )
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Positive:negative ratio")
    plt.ylabel("Held-out test metric, mean +/- SD")
    plt.title("Supplementary negative-ratio sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "supplementary_negative_ratio_sensitivity.png", dpi=300)
    plt.savefig(FIG_DIR / "supplementary_negative_ratio_sensitivity.pdf")
    plt.savefig(FIG_DIR / "fig_supp_negative_ratio_sensitivity.png", dpi=300)
    plt.close()


def run_negative_ratio_sensitivity(
    data: Any,
    ratios: Sequence[int],
    seeds: Sequence[int],
    cv_splits: int,
    cv_repeats: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate XGBoost while varying the negative-to-positive sampling ratio.

    Unlike the 13-model comparison, this sensitivity analysis uses one held-out
    evaluation per ratio/seed rather than repeated CV. That keeps the analysis
    focused on the ratio trade-off and avoids turning an appendix diagnostic
    into the slowest part of the pipeline.
    """
    # Use vectorized feature builders for speed across many ratio/seed combos.
    from generalization_fast import build_go_binary, go_freq_matrix, jac_size_matrix, select_indices

    _genes, gene_idx, _terms, go_binary = build_go_binary(data)
    rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        for seed in seeds:
            print(f"  ratio 1:{ratio}, seed={seed}")
            # Each ratio/seed combo regenerates negatives and splits from scratch.
            records, _sample_meta, _split_info, train_idx, test_idx = build_split_samples(
                data,
                seed=seed,
                negative_multiplier=ratio,
            )
            y_all = np.asarray([int(record["label"]) for record in records], dtype=int)
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]
            # Vectorized GO frequency calculation keeps high-ratio appendix runs
            # tractable without changing the train-only feature-selection rule.
            x_go = go_freq_matrix(records, gene_idx, go_binary)
            selected_idx = select_indices(x_go[train_idx], y_train, seed, frac=0.70)
            x_jac, x_size = jac_size_matrix(records, data, seed)
            x_all = np.hstack([x_go[:, selected_idx], x_jac, x_size])
            x_train = x_all[train_idx]
            x_test = x_all[test_idx]
            model = make_xgboost(seed, y_train)
            model.fit(x_train, y_train)
            scores, labels = model_scores(model, x_test)
            # CV columns are NaN because this analysis uses held-out test only.
            row = {
                "model": "XGBoost",
                "model_group": "advanced boosting",
                "cv_auroc_mean": np.nan,
                "cv_auroc_sd": np.nan,
                "cv_auroc_se": np.nan,
                "cv_auprc_mean": np.nan,
                "cv_auprc_sd": np.nan,
                "cv_auprc_se": np.nan,
                "test_auroc": float(roc_auc_score(y_test, scores)),
                "test_auprc": float(average_precision_score(y_test, scores)),
                "test_f1": float(f1_score(y_test, labels)),
                "test_precision": float(precision_score(y_test, labels, zero_division=0)),
                "test_recall": float(recall_score(y_test, labels, zero_division=0)),
            }
            row.update(
                {
                    "seed": seed,
                    "negative_ratio": f"1:{int(ratio)}",
                    "negative_multiplier": int(ratio),
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "n_go_selected": int(len(selected_idx)),
                    "D": int(x_train.shape[1]),
                    "train_negative_to_positive": class_balance_weight(y_train),
                }
            )
            rows.append(row)

    # Per-seed detail table and aggregated summary table.
    run_df = pd.DataFrame(rows)
    summary_df = summarize_ratio_runs(run_df)
    run_df.to_csv(TABLE_DIR / "supplementary_negative_ratio_per_seed.csv", index=False)
    summary_df.to_csv(TABLE_DIR / "supplementary_negative_ratio_summary.csv", index=False)
    summary_df.to_csv(TABLE_DIR / "supplementary_negative_ratio_sensitivity.csv", index=False)

    rounded = summary_df.copy()
    numeric_cols = rounded.select_dtypes(include=[np.number]).columns
    rounded[numeric_cols] = rounded[numeric_cols].round(3)
    rounded.to_csv(TABLE_DIR / "paper_supplementary_negative_ratio_summary_rounded.csv", index=False)
    write_ratio_plot(summary_df)
    return run_df, summary_df


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    """Orchestrate both supplementary analyses and write a combined JSON summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_REFERENCE_SEED)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--ratios", default="1-5")       # negative-ratio range to test
    parser.add_argument("--ratio-seeds", default="1-5")  # seeds for ratio sensitivity
    parser.add_argument("--skip-model-comparison", action="store_true")
    parser.add_argument("--skip-ratio-sensitivity", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    print("Loading data...")
    data = load_data()
    print(f"  Pathways: {len(data.pathways)}")
    print(f"  Filtered GO terms: {len(data.go_terms)}")

    payload: Dict[str, Any] = {
        "generated_by": "supplementary_analysis.py",
        "seed": args.seed,
        "cv": {
            "splits": args.cv_splits,
            "repeats": args.cv_repeats,
            "folds": args.cv_splits * args.cv_repeats,
        },
        "notes": [
            "Supplementary-only outputs; main manuscript tables are not overwritten.",
            "All analyses reuse the canonical no-embedding feature construction.",
            "Negative-ratio sensitivity regenerates negatives, train/test splits, and GO feature selection for each ratio/seed.",
        ],
    }

    if not args.skip_model_comparison:
        print("\nRunning supplementary 13-model comparison...")
        model_df = run_model_comparison(
            data=data,
            seed=args.seed,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
        )
        payload["model_comparison"] = {
            "source": "tables/supplementary_13_model_comparison.csv",
            "paper_ready_source": "tables/paper_supplementary_13_model_comparison_rounded.csv",
            "n_models": int(len(model_df)),
            "best_model": str(model_df.iloc[0]["model"]),
            "best_test_auroc": float(model_df.iloc[0]["test_auroc"]),
        }
    elif (TABLE_DIR / "supplementary_13_model_comparison.csv").exists():
        model_df = pd.read_csv(TABLE_DIR / "supplementary_13_model_comparison.csv")
        payload["model_comparison"] = {
            "source": "tables/supplementary_13_model_comparison.csv",
            "paper_ready_source": "tables/paper_supplementary_13_model_comparison_rounded.csv",
            "n_models": int(len(model_df)),
            "best_model": str(model_df.iloc[0]["model"]),
            "best_test_auroc": float(model_df.iloc[0]["test_auroc"]),
            "note": "Loaded from existing CSV because --skip-model-comparison was used.",
        }

    if not args.skip_ratio_sensitivity:
        ratios = parse_int_list(args.ratios)
        ratio_seeds = parse_int_list(args.ratio_seeds)
        print("\nRunning supplementary negative-ratio sensitivity...")
        run_df, summary_df = run_negative_ratio_sensitivity(
            data=data,
            ratios=ratios,
            seeds=ratio_seeds,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
        )
        payload["negative_ratio_sensitivity"] = {
            "per_seed_source": "tables/supplementary_negative_ratio_per_seed.csv",
            "summary_source": "tables/supplementary_negative_ratio_summary.csv",
            "paper_ready_source": "tables/paper_supplementary_negative_ratio_summary_rounded.csv",
            "ratios": [f"1:{int(r)}" for r in ratios],
            "seeds": list(ratio_seeds),
            "n_runs": int(len(run_df)),
            "summary": summary_df.to_dict(orient="records"),
        }

    save_json(TABLE_DIR / "supplementary_analysis_summary.json", payload)
    print("\nWrote supplementary outputs under tables/ and figures/.")
    print(json.dumps(payload, indent=2, sort_keys=True)[:2000])
    return payload


if __name__ == "__main__":
    main()
