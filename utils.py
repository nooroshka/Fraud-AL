"""
Utility helpers for the fraud active-learning project.

This module centralizes small, reusable pieces so analysis scripts and learners
stay tidy: metrics, calibration helpers, plotting utilities, diversity selection,
and dataset loading. Functions aim to be simple to call and return plain types.

If you skim one thing: the diversity selectors pick a set of points that are
far apart; metrics include calibration and a few capacity-facing ones.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score, average_precision_score)
from sklearn.metrics.pairwise import pairwise_distances


# Reproducibility

def set_seed(seed: int) -> None:
    """Set NumPy's global RNG seed for reproducible sampling."""
    np.random.seed(seed)


# Metrics / Calibration Helpers

def binary_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Binary entropy H(p) element-wise, with clipping for numerical stability."""
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def recall_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, target_fpr: float = 0.001) -> float:
    """Recall at a given false-positive-rate, via ROC interpolation."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    if target_fpr <= fpr.max():
        return float(np.interp(target_fpr, fpr, tpr))
    return 0.0

def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Precision among the top-k scores (useful for capacity-constrained settings)."""
    if k <= 0:
        return 0.0
    k = min(k, y_scores.shape[0])
    idx = np.argsort(-y_scores)[:k]
    return float(y_true[idx].mean()) if len(idx) > 0 else 0.0

def lift_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Lift at k: precision@k divided by the base rate."""
    base = float(y_true.mean()) if y_true.size else 0.0
    p_at_k = precision_at_k(y_true, y_scores, k)
    return float(p_at_k / (base + 1e-12)) if base > 0 else 0.0

def precision_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, target_fpr=0.001) -> Tuple[float, float]:
    """Return (precision, threshold) at a target FPR, via ROC threshold interpolation."""
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    if target_fpr > fpr.max():
        return 0.0, 0.5
    th = float(np.interp(target_fpr, fpr, thr))
    y_pred = (y_scores >= th).astype(int)
    tp = int(np.count_nonzero((y_pred == 1) & (y_true == 1)))
    fp = int(np.count_nonzero((y_pred == 1) & (y_true == 0)))
    prec = float(tp / max(1, tp + fp))
    return prec, th

def ece_score(y_true: np.ndarray, y_scores: np.ndarray, n_bins=15) -> float:
    """Expected calibration error (ECE) with fixed bins in [0, 1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_scores, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = (idx == b)
        if not np.any(mask):
            continue
        conf = y_scores[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

def expected_profit(y_true: np.ndarray, y_scores: np.ndarray, th: float,
                    cost_fn: float = 100.0, cost_fp: float = 1.0) -> Tuple[float, int, int, int]:
    """Simple profit proxy: reward true positives, penalize false positives.

    Returns a tuple: (profit, tp, fp, fn)
    """
    y_pred = (y_scores >= th).astype(int)
    tp = int(np.count_nonzero((y_pred == 1) & (y_true == 1)))
    fp = int(np.count_nonzero((y_pred == 1) & (y_true == 0)))
    fn = int(np.count_nonzero((y_pred == 0) & (y_true == 1)))
    profit = float(cost_fn * tp - cost_fp * fp)
    return profit, tp, fp, fn


def qbc_js_disagreement(preds: np.ndarray) -> np.ndarray:
    """Committee disagreement: H(mean_p) - mean H(p_i).

    Args:
        preds: Array of shape (n_models, n_samples) with per-model probabilities for class 1.

    Returns:
        A 1D array of length n_samples with higher values meaning more disagreement.
    """
    if preds.ndim != 2:
        preds = np.atleast_2d(preds)
    pbar = preds.mean(axis=0)
    # Use the same binary_entropy function for stability
    return binary_entropy(pbar) - binary_entropy(preds).mean(axis=0)


# Plotting / Saving

def save_curves(y_true: np.ndarray, scores: np.ndarray, outdir: Path, run_name: str) -> None:
    """Save basic ROC and PR curves as PNGs in the output directory.

    Now annotates AUROC/AUPRC and draws chance baselines for readability.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    base_rate = float(np.mean(y_true)) if len(y_true) else 0.0

    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUROC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(outdir / f"{run_name}.roc.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec, label=f"AUPRC={auprc:.3f}")
    if base_rate > 0:
        plt.axhline(base_rate, linestyle="--", color="gray", label=f"chance={base_rate:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    plt.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(outdir / f"{run_name}.pr.png", dpi=150); plt.close()


def plot_reliability_curve(y_true: np.ndarray, scores: np.ndarray, outdir: Path, run_name: str, n_bins: int = 15) -> None:
    """Plot a reliability (calibration) diagram with ECE and Brier annotations."""
    from sklearn.metrics import brier_score_loss
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(scores, bins) - 1
    x, y, w = [], [], []
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m):
            continue
        x.append(np.clip(scores[m].mean(), 0.0, 1.0))
        y.append(np.clip(y_true[m].mean(), 0.0, 1.0))
        w.append(float(np.mean(m)))
    ece = ece_score(y_true, scores, n_bins=n_bins)
    brier = brier_score_loss(y_true, scores)

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    if x:
        plt.plot(x, y, marker="o", label="empirical")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.title(f"Reliability diagram\nECE={ece:.3f}, Brier={brier:.3f}")
    plt.tight_layout()
    out = outdir / f"{run_name}.reliability.png"
    plt.savefig(out, dpi=150)
    plt.close()


def save_operating_point_visuals(y_true: np.ndarray, scores: np.ndarray, outdir: Path, run_name: str,
                                 threshold: float, cost_tp: float = 100.0, cost_fp: float = 1.0) -> None:
    """Save confusion matrix heatmap at threshold and a cost vs threshold curve."""
    # Confusion matrix at given threshold
    y_pred = (scores >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    plt.figure(figsize=(4.2, 4))
    im = plt.imshow(cm, cmap="Blues")
    plt.title("Confusion @ threshold")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"]) ; plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center', color='black')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outdir / f"{run_name}.confusion.png", dpi=150)
    plt.close()

    # Cost vs threshold curve
    ts = np.linspace(0.0, 1.0, 201)
    costs = []
    for th in ts:
        yp = (scores >= th).astype(int)
        tp_i = int(np.sum((yp == 1) & (y_true == 1)))
        fp_i = int(np.sum((yp == 1) & (y_true == 0)))
        costs.append(cost_tp * tp_i - cost_fp * fp_i)
    costs = np.asarray(costs)
    plt.figure(figsize=(6, 4))
    plt.plot(ts, costs)
    plt.axvline(threshold, color='red', linestyle='--', label=f"chosen t={threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Profit (proxy)")
    plt.title("Profit vs threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{run_name}.profit_curve.png", dpi=150)
    plt.close()


def save_metrics_history(history: dict, outdir: Path, run_name: str) -> None:
    """Write the per-round history to CSV and a JSON one-line summary.

    Also emits a few quick-look plots for common metrics vs. labels_used.
    """
    lengths = {k: len(v) for k, v in history.items()}
    if len(set(lengths.values())) != 1:
        # Trim to the shortest length if a metric was recorded less times
        Lmin = min(lengths.values())
        for k in history:
            history[k] = history[k][:Lmin]

    df = pd.DataFrame(history)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{run_name}.metrics.csv", index=False)

    summary = {k: (v[-1] if len(v) else None) for k, v in history.items()}
    (outdir / f"{run_name}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot standard and capacity-facing metrics across rounds
    to_plot = [m for m in [
        "auroc", "auprc", "recall@0.1%fpr", "f1@0.1%fpr"
    ] if m in df.columns]

    # Add all recall@K curves dynamically
    for col in df.columns:
        m = re.match(r"^recall@(\d+)$", str(col))
        if m:
            to_plot.append(col)

    for metric in to_plot:
        plt.figure(figsize=(7, 4))
        plt.plot(df["labels_used"], df[metric], marker="o", markersize=4)
        title = f"{metric.upper()} vs. labels"
        # Clarify actual FPR used if relevant
        if metric.endswith("%fpr") and "target_fpr" in df.columns:
            try:
                fpr_pct = float(df["target_fpr"].iloc[-1]) * 100.0
                title += f" (FPR={fpr_pct:.3f}%)"
            except Exception:
                pass
        plt.title(title)
        plt.xlabel("Number of labeled samples"); plt.ylabel(metric.upper())
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(outdir / f"{run_name}.{metric}.png", dpi=150)
        plt.close()


# Diversity helper

def _fpf_select(X_emb: np.ndarray, n_samples: int, metric: str = "euclidean",
                precomputed_dist: np.ndarray | None = None, random_state: int = 0) -> np.ndarray:
    """Greedy Farthest-Point selection on an embedding or a precomputed distance matrix.

    If precomputed_dist is given, it should be a square matrix of pairwise distances for X_emb.
    Returns indices relative to X_emb rows.
    """
    if len(X_emb) <= n_samples:
        return np.arange(len(X_emb), dtype=int)

    # Choose a deterministic but sensible start: the point with largest norm.
    start = int(np.argmax(np.linalg.norm(X_emb, axis=1)))
    sel = [start]

    if precomputed_dist is not None:
        mind = precomputed_dist[start].astype(float).copy()
        for _ in range(1, n_samples):
            nxt = int(np.argmax(mind)); sel.append(nxt)
            mind = np.minimum(mind, precomputed_dist[nxt])
        return np.array(sel, dtype=int)

    mind = pairwise_distances(X_emb, X_emb[[start]], metric=metric).ravel()
    for _ in range(1, n_samples):
        nxt = int(np.argmax(mind)); sel.append(nxt)
        mind = np.minimum(mind, pairwise_distances(X_emb, X_emb[[nxt]], metric=metric).ravel())
    return np.array(sel, dtype=int)


def farthest_point_diverse_subset(X_candidates: np.ndarray,
                                  n_samples: int,
                                  metric: str = "euclidean",
                                  embed: bool = True,
                                  embed_max_dim: int = 32,
                                  random_state: int = 0) -> np.ndarray:
    """
    Greedy farthest-point selection with flexible distance metrics.

    If the feature space is large, we optionally project candidates with PCA to
    a compact embedding before computing distances. Supports common metrics like
    euclidean, cosine, manhattan, chebyshev.

    Returns indices relative to X_candidates.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    if len(X_candidates) <= n_samples:
        return np.arange(len(X_candidates), dtype=int)

    n_components = min(embed_max_dim, len(X_candidates), X_candidates.shape[1]) if embed else X_candidates.shape[1]
    X_emb = X_candidates
    if embed:
        pca = PCA(n_components=n_components, random_state=random_state)
        X_emb = pca.fit_transform(X_candidates)

    # For cosine distance, normalize vectors (cosine = 1 - dot product of normalized vectors)
    if metric == "cosine":
        X_emb = normalize(X_emb, norm='l2', axis=1)

    return _fpf_select(X_emb, n_samples=n_samples, metric=metric, random_state=random_state)


def advanced_diverse_subset(X_candidates: np.ndarray,
                            probs: np.ndarray,
                            n_samples: int,
                            metric: str = "euclidean",
                            diversity_mode: str = "pure",
                            embed_max_dim: int = 32,
                            random_state: int = 0) -> np.ndarray:
    """
    Enhanced diversity selection with fraud-aware metrics.

    Args:
        X_candidates: Feature matrix (n_samples, n_features)
        probs: Fraud probabilities for each candidate
        n_samples: Number of samples to select
        metric: Distance metric (euclidean, cosine, manhattan, chebyshev)
        diversity_mode:
            - "pure": Standard FPF with chosen metric
            - "probability_weighted": Weight distances by prediction uncertainty
            - "mahalanobis": Use Mahalanobis distance (accounts for feature correlations)
            - "hybrid": Combine spatial diversity (70%) with prediction diversity (30%)
        embed_max_dim: Max dimensions for PCA embedding
        random_state: Random seed for reproducibility

    Returns:
        Array of selected indices
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    if len(X_candidates) <= n_samples:
        return np.arange(len(X_candidates), dtype=int)

    # Embedding with PCA
    n_components = min(embed_max_dim, len(X_candidates), X_candidates.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    X_emb = pca.fit_transform(X_candidates)

    if diversity_mode == "probability_weighted":
        # Weight features by prediction uncertainty (entropy-like)
        uncertainty = probs * (1 - probs)
        Xw = X_emb * uncertainty[:, None]
        if metric == "cosine":
            Xw = normalize(Xw, norm='l2', axis=1)
        return _fpf_select(Xw, n_samples=n_samples, metric=metric, random_state=random_state)

    if diversity_mode == "mahalanobis":
        # Account for feature correlations using Mahalanobis distance; let the distance
        # routine handle the inverse covariance internally to keep things simple.
        try:
            return _fpf_select(X_emb, n_samples=n_samples, metric="mahalanobis", random_state=random_state)
        except Exception:
            # Fallback to euclidean if distance computation fails for any reason
            diversity_mode = "pure"
            metric = "euclidean"

    if diversity_mode == "hybrid":
        # Combine spatial diversity with prediction diversity
        prob_diversity = np.abs(probs[:, None] - probs[None, :])
        if metric == "cosine":
            X_emb_norm = normalize(X_emb, norm='l2', axis=1)
            spatial_dist = pairwise_distances(X_emb_norm, metric=metric)
        else:
            spatial_dist = pairwise_distances(X_emb, metric=metric)

        # Normalize both to [0, 1]
        spatial_dist = (spatial_dist - spatial_dist.min()) / (spatial_dist.max() - spatial_dist.min() + 1e-9)
        prob_diversity = (prob_diversity - prob_diversity.min()) / (prob_diversity.max() - prob_diversity.min() + 1e-9)

        # Combine: 70% spatial diversity + 30% probability diversity
        combined_dist = 0.7 * spatial_dist + 0.3 * prob_diversity

        return _fpf_select(X_emb, n_samples=n_samples, precomputed_dist=combined_dist, random_state=random_state)

    # Default: pure mode with chosen metric
    if metric == "cosine":
        from sklearn.preprocessing import normalize
        X_emb = normalize(X_emb, norm='l2', axis=1)

    return _fpf_select(X_emb, n_samples=n_samples, metric=metric, random_state=random_state)


# Dataset Loaders

def prepare_creditcard(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Minimal preprocessing for the Kaggle Credit Card Fraud dataset.

    Keeps the PCA components and standardizes Amount. Returns (X, y, feature_names, name).
    """
    if "Class" not in df.columns:
        raise ValueError("CreditCard dataset must contain 'Class'.")
    y = df["Class"].values.astype(int)
    X_df = df.drop(columns=[c for c in ["Time", "Class"] if c in df.columns]).copy()
    if "Amount" in X_df.columns:
        X_df["Amount"] = StandardScaler().fit_transform(X_df[["Amount"]]).flatten()
    feature_names = list(X_df.columns)
    return X_df.values, y, feature_names, "creditcard"

def prepare_paysim(df: pd.DataFrame, subsample_if_huge: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Preprocess the PaySim dataset; optionally downsample huge negatives for speed.

    Adds a few intuitive balance-delta features and one-hots the transfer type.
    Returns (X, y, feature_names, name).
    """
    required = {"step","type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFraud","isFlaggedFraud"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"PaySim dataset missing columns: {missing}")

    if subsample_if_huge and len(df) > 1_000_000:
        fraud = df[df["isFraud"] == 1]
        legit = df[df["isFraud"] == 0].sample(n=min(500_000, max(100_000, len(df)//20)), random_state=0)
        df = pd.concat([fraud, legit], axis=0).sample(frac=1.0, random_state=0).reset_index(drop=True)

    y = df["isFraud"].astype(int).values
    X_df = df[["step","type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFlaggedFraud"]].copy()
    X_df["errorBalanceOrig"] = X_df["newbalanceOrig"] + X_df["amount"] - X_df["oldbalanceOrg"]
    X_df["errorBalanceDest"] = X_df["oldbalanceDest"] + X_df["amount"] - X_df["newbalanceDest"]
    X_df["deltaOrig"] = X_df["oldbalanceOrg"] - X_df["newbalanceOrig"]
    X_df["deltaDest"] = X_df["newbalanceDest"] - X_df["oldbalanceDest"]
    eps = 1e-6
    X_df["amt_over_oldOrig"] = X_df["amount"] / (np.abs(X_df["oldbalanceOrg"]) + eps)
    type_ohe = pd.get_dummies(X_df["type"], prefix="type")
    X_df = pd.concat([X_df.drop(columns=["type"]), type_ohe], axis=1)
    to_scale = [c for c in ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest",
                            "errorBalanceOrig","errorBalanceDest","deltaOrig","deltaDest","amt_over_oldOrig","step"]
                if c in X_df.columns]
    X_df[to_scale] = StandardScaler().fit_transform(X_df[to_scale])
    feature_names = list(X_df.columns)
    return X_df.values, y, feature_names, "paysim"

def load_dataset_auto(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Load a CSV and auto-detect dataset type (creditcard vs. PaySim)."""
    df = pd.read_csv(path)
    if "Class" in df.columns:
        return prepare_creditcard(df)
    if "isFraud" in df.columns:
        return prepare_paysim(df)
    raise ValueError("Could not detect dataset type: expected 'Class' (creditcard) or 'isFraud' (PaySim).")
