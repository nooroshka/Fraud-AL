# al_fraud_experiment.py
#qbc sampler-added standard scailing+data leakage fixed
from __future__ import annotations
import argparse, json, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

# --------------------------- Utilities ---------------------------

def set_seed(seed: int):
    np.random.seed(seed)

def binary_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def recall_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, target_fpr: float = 0.001) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    if target_fpr <= fpr.max():
        return float(np.interp(target_fpr, fpr, tpr))
    return 0.0

# Extra diagnostics
from sklearn.metrics import brier_score_loss

def precision_at_k(y_true, y_scores, k: int):
    if k <= 0:
        return 0.0
    k = min(k, y_scores.shape[0])
    idx = np.argsort(-y_scores)[:k]
    return float(y_true[idx].mean()) if len(idx) > 0 else 0.0

def lift_at_k(y_true, y_scores, k: int):
    base = float(y_true.mean()) if y_true.size else 0.0
    p_at_k = precision_at_k(y_true, y_scores, k)
    return float(p_at_k / (base + 1e-12)) if base > 0 else 0.0

def precision_at_fpr(y_true, y_scores, target_fpr=0.001) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    if target_fpr > fpr.max():
        return 0.0, 0.5
    th = float(np.interp(target_fpr, fpr, thr))
    y_pred = (y_scores >= th).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    prec = float(tp / max(1, tp + fp))
    return prec, th

def ece_score(y_true, y_scores, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_scores, bins) - 1
    ece = 0.0
    m = len(y_scores)
    for b in range(n_bins):
        mask = (idx == b)
        if not np.any(mask):
            continue
        conf = y_scores[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

# --------------------------- Base Strategy ---------------------------

class BaseQueryStrategy:
    name: str = "base"
    def select(self, model, X_pool: np.ndarray, batch_size: int, **kwargs) -> np.ndarray:
        raise NotImplementedError

# --------------------------- FRaUD / FRaUD++ ---------------------------

@dataclass
class FRaUDConfig:
    alpha: float = 1.0          # entropy weight
    gamma: float = 2.0          # focal emphasis
    prior: Optional[float] = 0.00172  # prevalence prior (Kaggle ~0.00172)
    diversity_metric: str = "euclidean"
    embed_max_dim: int = 32

    # FRaUD++
    quota: float = 0.25
    lambda_anomaly: float = 0.5
    use_schedules: bool = True
    rarity_k: int = 10
    exploit_topk_factor: float = 5.0
    explore_topk_factor: float = 10.0
    random_state: int = 0
    use_calibration: bool = False  # optional probability calibration

class FRaUDSampler(BaseQueryStrategy):
    """FRaUD base: s0 only (no rarity boost, but with diversity and quotas like FRaUD++ explore branch)."""
    name = "fraud"

    def __init__(self, cfg: FRaUDConfig):
        self.cfg = cfg

    def _diverse_subset(self, X_candidates: np.ndarray, n_samples: int) -> np.ndarray:
        if len(X_candidates) <= n_samples:
            return np.arange(len(X_candidates))
        n_components = min(self.cfg.embed_max_dim, len(X_candidates), X_candidates.shape[1])
        pca = PCA(n_components=n_components, random_state=self.cfg.random_state)
        X_emb = pca.fit_transform(X_candidates)
        start = np.argmax(np.linalg.norm(X_emb, axis=1))
        sel = [start]
        mind = pairwise_distances(X_emb, X_emb[[start]], metric=self.cfg.diversity_metric).ravel()
        for _ in range(1, n_samples):
            nxt = np.argmax(mind)
            sel.append(nxt)
            mind = np.minimum(mind, pairwise_distances(X_emb, X_emb[[nxt]], metric=self.cfg.diversity_metric).ravel())
        return np.array(sel, dtype=int)

    def select(self, model, X_pool: np.ndarray, batch_size: int, **kwargs) -> np.ndarray:
        probs = model.predict_proba(X_pool)[:, 1]
        Xs = StandardScaler().fit_transform(X_pool)
        round_idx = kwargs.get("round_idx", 1)
        y_labeled = kwargs.get("y_labeled", np.array([]))

        # schedules (same as FRaUD++)
        if self.cfg.use_schedules:
            n_frauds = int(y_labeled.sum()) if len(y_labeled) else 0
            quota = 0.15 if round_idx <= 3 else (0.25 if round_idx <= 8 or n_frauds < 20 else 0.35)
        else:
            quota = self.cfg.quota

        # s0
        unc = binary_entropy(probs)
        obs_prior = y_labeled.mean() if len(y_labeled) else 0.0
        prior = max(obs_prior, self.cfg.prior or 0.0, 1e-6)
        focal = (probs / (prior + probs + 1e-12)) ** self.cfg.gamma
        egl = np.linalg.norm(Xs, axis=1) * probs * (1 - probs)
        s0 = (unc ** self.cfg.alpha) * focal * egl

        # quotas
        n_exploit = int(quota * batch_size)
        n_explore = batch_size - n_exploit

        # Exploit: top p(y=1), diversify
        ex_pool = int(max(n_exploit, self.cfg.exploit_topk_factor * n_exploit))
        ex_candidates = np.argsort(-probs)[:ex_pool] if ex_pool > 0 else np.array([], dtype=int)
        if ex_candidates.size > 0:
            ex_sel_rel = self._diverse_subset(Xs[ex_candidates], min(n_exploit, len(ex_candidates)))
            final_ex = ex_candidates[ex_sel_rel]
        else:
            final_ex = np.array([], dtype=int)

        # Explore: top s0, diversify (excluding exploit)
        mask = np.ones(len(X_pool), dtype=bool)
        mask[final_ex] = False
        rem_idx = np.where(mask)[0]
        exr_pool = int(max(n_explore, self.cfg.explore_topk_factor * n_explore))
        if exr_pool > 0 and rem_idx.size > 0:
            cand_sorted = rem_idx[np.argsort(-s0[rem_idx])[:min(exr_pool, rem_idx.size)]]
            exr_sel_rel = self._diverse_subset(Xs[cand_sorted], min(n_explore, len(cand_sorted)))
            final_exr = cand_sorted[exr_sel_rel]
        else:
            final_exr = np.array([], dtype=int)

        return np.concatenate([final_ex, final_exr])

class FRaUDPlusSampler(BaseQueryStrategy):
    """FRaUD++: s0 + rarity boost on candidate subset; dual-branch quota; diversity per branch; schedules."""
    name = "fraudpp"

    def __init__(self, cfg: FRaUDConfig):
        self.cfg = cfg

    def _diverse_subset(self, X_candidates: np.ndarray, n_samples: int) -> np.ndarray:
        if len(X_candidates) <= n_samples:
            return np.arange(len(X_candidates))
        n_components = min(self.cfg.embed_max_dim, len(X_candidates), X_candidates.shape[1])
        pca = PCA(n_components=n_components, random_state=self.cfg.random_state)
        X_emb = pca.fit_transform(X_candidates)
        start = np.argmax(np.linalg.norm(X_emb, axis=1))
        sel = [start]
        mind = pairwise_distances(X_emb, X_emb[[start]], metric=self.cfg.diversity_metric).ravel()
        for _ in range(1, n_samples):
            nxt = np.argmax(mind)
            sel.append(nxt)
            mind = np.minimum(mind, pairwise_distances(X_emb, X_emb[[nxt]], metric=self.cfg.diversity_metric).ravel())
        return np.array(sel, dtype=int)

    def select(self, model, X_pool: np.ndarray, batch_size: int, **kwargs) -> np.ndarray:
        probs = model.predict_proba(X_pool)[:, 1]
        Xs = StandardScaler().fit_transform(X_pool)
        round_idx = kwargs.get("round_idx", 1)
        y_labeled = kwargs.get("y_labeled", np.array([]))

        # schedules
        if self.cfg.use_schedules:
            n_frauds = int(y_labeled.sum()) if len(y_labeled) else 0
            quota = 0.15 if round_idx <= 3 else (0.25 if round_idx <= 8 or n_frauds < 20 else 0.35)
            lambda_anomaly = 0.6 if round_idx <= 3 else (0.4 if round_idx <= 8 else 0.25)
        else:
            quota = self.cfg.quota
            lambda_anomaly = self.cfg.lambda_anomaly

        # s0
        unc = binary_entropy(probs)
        obs_prior = y_labeled.mean() if len(y_labeled) else 0.0
        prior = max(obs_prior, self.cfg.prior or 0.0, 1e-6)
        focal = (probs / (prior + probs + 1e-12)) ** self.cfg.gamma
        egl = np.linalg.norm(Xs, axis=1) * probs * (1 - probs)
        s0 = (unc ** self.cfg.alpha) * focal * egl

        n_pool = len(X_pool)
        # rarity on a candidate subset for performance
        pre_k = int(self.cfg.explore_topk_factor * batch_size * 4)  # 4x exploration candidate size
        pre_k = min(max(pre_k, batch_size), n_pool)
        pre_candidates = np.argsort(-s0)[:pre_k]

        # Guard for very small candidate sets
        knn_k = min(self.cfg.rarity_k, max(1, len(pre_candidates) - 1))
        if len(pre_candidates) >= 2:
            cand_scaled = Xs[pre_candidates]
            nbrs = NearestNeighbors(n_neighbors=knn_k).fit(cand_scaled)
            distances, _ = nbrs.kneighbors(cand_scaled)
            rarity_c = 1.0 / (distances.mean(axis=1) + 1e-9)
            rarity_c = (rarity_c - rarity_c.min()) / (rarity_c.max() - rarity_c.min() + 1e-9)
            final_score = s0.copy()
            final_score[pre_candidates] = s0[pre_candidates] * (1 + lambda_anomaly * rarity_c)
        else:
            final_score = s0

        # quotas
        n_exploit = int(quota * batch_size)
        n_explore = batch_size - n_exploit

        # Exploit: top p(y=1), diversify
        ex_pool = int(max(n_exploit, self.cfg.exploit_topk_factor * n_exploit))
        ex_candidates = np.argsort(-probs)[:ex_pool] if ex_pool > 0 else np.array([], dtype=int)
        if ex_candidates.size > 0:
            ex_sel_rel = self._diverse_subset(Xs[ex_candidates], min(n_exploit, len(ex_candidates)))
            final_ex = ex_candidates[ex_sel_rel]
        else:
            final_ex = np.array([], dtype=int)

        # Explore: top final_score, diversify (excluding exploit)
        mask = np.ones(n_pool, dtype=bool)
        mask[final_ex] = False
        rem_idx = np.where(mask)[0]

        exr_pool = int(max(n_explore, self.cfg.explore_topk_factor * n_explore))
        if exr_pool > 0 and rem_idx.size > 0:
            cand_sorted = rem_idx[np.argsort(-final_score[rem_idx])[:min(exr_pool, rem_idx.size)]]
            exr_sel_rel = self._diverse_subset(Xs[cand_sorted], min(n_explore, len(cand_sorted)))
            final_exr = cand_sorted[exr_sel_rel]
        else:
            final_exr = np.array([], dtype=int)

        return np.concatenate([final_ex, final_exr])

# --------------------------- Baselines ---------------------------

class RandomSampler(BaseQueryStrategy):
    name = "random"
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
    def select(self, _, X_pool, batch_size, **kwargs):
        n = X_pool.shape[0]
        size = min(batch_size, n)
        return self.rng.choice(n, size=size, replace=False)

class EntropySampler(BaseQueryStrategy):
    name = "entropy"
    def select(self, model, X_pool, batch_size, **kwargs):
        probs = model.predict_proba(X_pool)[:, 1]
        return np.argsort(-binary_entropy(probs))[:batch_size]

class MarginSampler(BaseQueryStrategy):
    name = "margin"
    def select(self, model, X_pool, batch_size, **kwargs):
        probs = model.predict_proba(X_pool)
        margin = np.abs(probs[:, 1] - probs[:, 0])
        return np.argsort(margin)[:batch_size]

class CostBalancedEntropySampler(BaseQueryStrategy):
    name = "cost_balanced"
    def __init__(self, fraud_cost: float = 100.0):
        self.fraud_cost = fraud_cost
    def select(self, model, X_pool, batch_size, **kwargs):
        probs = model.predict_proba(X_pool)[:, 1]
        score = binary_entropy(probs) * (probs * self.fraud_cost + (1 - probs))
        return np.argsort(-score)[:batch_size]

class QBCSampler(BaseQueryStrategy):
    name = "qbc"
    def __init__(self, seed: int):
        self.seed = seed
    def select(self, model, X_pool, batch_size, **kwargs):
        X_l = kwargs.get("X_labeled")
        y_l = kwargs.get("y_labeled")
        if X_l is None or y_l is None or len(y_l) < 50:
            return EntropySampler().select(model, X_pool, batch_size)

        committee = [
            LogisticRegression(class_weight="balanced", max_iter=200, random_state=self.seed),
            RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=self.seed),
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=self.seed, early_stopping=True)
        ]
        probs_list = []
        for clf in committee:
            clf.fit(X_l, y_l)
            probs_list.append(clf.predict_proba(X_pool)[:, 1])
        preds = np.vstack(probs_list)
        disagreement = np.std(preds, axis=0)
        return np.argsort(-disagreement)[:batch_size]

# --------------------------- Data loaders (auto-detect) ---------------------------

def prepare_creditcard(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Kaggle creditcard.csv: drop Time, scale Amount."""
    if "Class" not in df.columns:
        raise ValueError("CreditCard dataset must contain 'Class'.")
    y = df["Class"].values.astype(int)
    X_df = df.drop(columns=[c for c in ["Time", "Class"] if c in df.columns]).copy()
    if "Amount" in X_df.columns:
        X_df["Amount"] = StandardScaler().fit_transform(X_df[["Amount"]]).flatten()
    feature_names = list(X_df.columns)
    return X_df.values, y, feature_names, "creditcard"

def prepare_paysim(df: pd.DataFrame, subsample_if_huge: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """
    PaySim log: PS_20174392719_1491204439457_log.csv
    - Drop nameOrig/nameDest (avoid ID leakage & huge cardinality)
    - One-hot encode 'type'
    - Engineer error balances and deltas
    - Keep isFlaggedFraud as a binary feature
    - Optional stratified subsample for very large CSVs
    """
    required = {"step","type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFraud","isFlaggedFraud"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"PaySim dataset missing columns: {missing}")

    # Optional speed-up for huge files
    if subsample_if_huge and len(df) > 1_000_000:
        fraud = df[df["isFraud"] == 1]
        # sample legit but keep at least 100k rows
        legit = df[df["isFraud"] == 0].sample(n=min(500_000, max(100_000, len(df)//20)), random_state=0)
        df = pd.concat([fraud, legit], axis=0).sample(frac=1.0, random_state=0).reset_index(drop=True)

    y = df["isFraud"].astype(int).values

    X_df = df[[
        "step","type","amount","oldbalanceOrg","newbalanceOrig",
        "oldbalanceDest","newbalanceDest","isFlaggedFraud"
    ]].copy()

    # feature engineering
    X_df["errorBalanceOrig"] = X_df["newbalanceOrig"] + X_df["amount"] - X_df["oldbalanceOrg"]
    X_df["errorBalanceDest"] = X_df["oldbalanceDest"] + X_df["amount"] - X_df["newbalanceDest"]
    X_df["deltaOrig"] = X_df["oldbalanceOrg"] - X_df["newbalanceOrig"]
    X_df["deltaDest"] = X_df["newbalanceDest"] - X_df["oldbalanceDest"]
    eps = 1e-6
    X_df["amt_over_oldOrig"] = X_df["amount"] / (np.abs(X_df["oldbalanceOrg"]) + eps)

    # one-hot on 'type'
    type_ohe = pd.get_dummies(X_df["type"], prefix="type")
    X_df = pd.concat([X_df.drop(columns=["type"]), type_ohe], axis=1)

    # scale heavy-tailed numeric columns
    to_scale = [c for c in [
        "amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest",
        "errorBalanceOrig","errorBalanceDest","deltaOrig","deltaDest","amt_over_oldOrig","step"
    ] if c in X_df.columns]
    X_df[to_scale] = StandardScaler().fit_transform(X_df[to_scale])

    feature_names = list(X_df.columns)
    return X_df.values, y, feature_names, "paysim"

def load_dataset_auto(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    df = pd.read_csv(path)
    print(f"Using dataset: {path}")
    if "Class" in df.columns:
        return prepare_creditcard(df)
    if "isFraud" in df.columns:
        return prepare_paysim(df)
    raise ValueError("Could not detect dataset type: expected 'Class' (creditcard) or 'isFraud' (PaySim).")

# --------------------------- Active Learner ---------------------------

@dataclass
class ALConfig:
    batch_size: int = 200
    seed_size: int = 1400
    budget: int = 5000
    test_size: float = 0.2
    model_kind: str = "lgbm"
    random_state: int = 0
    auroc_target: Optional[float] = None  # optional early stop

class ActiveLearner:
    def __init__(self, X, y, cfg: ALConfig, sampler: BaseQueryStrategy, outdir: Path, run_name: str, fraud_cfg: Optional[FRaUDConfig] = None):
        self.cfg, self.sampler, self.outdir, self.run_name = cfg, sampler, outdir, run_name
        self.fraud_cfg = fraud_cfg
        set_seed(cfg.random_state)

        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(
            X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
        )

        # Seed selection: include a few frauds if available
        idx_all = np.arange(self.X_train_full.shape[0])
        fraud_idx = idx_all[self.y_train_full == 1]
        legit_idx = idx_all[self.y_train_full == 0]
        n_seed_frauds = min(len(fraud_idx), max(1, int(cfg.seed_size * max(0.005, y.mean()))))  # ensure a few positives
        n_seed_legit = max(0, cfg.seed_size - n_seed_frauds)
        seed_idx = np.concatenate([
            np.random.choice(fraud_idx, size=n_seed_frauds, replace=False) if n_seed_frauds > 0 else np.array([], dtype=int),
            np.random.choice(legit_idx, size=n_seed_legit, replace=False) if n_seed_legit > 0 else np.array([], dtype=int)
        ])

        self.labeled_mask = np.zeros(self.X_train_full.shape[0], dtype=bool)
        self.labeled_mask[seed_idx] = True

        # track extra diagnostics too
        self.history = {k: [] for k in [
            "labels_used","frauds_found","auroc","auprc","recall@0.1%fpr",
            "brier","ece","precision@0.1%fpr","threshold@0.1%fpr",
            "precision@100","precision@500","lift@500","profit@0.1%fpr",
            "tp@0.1%fpr","fp@0.1%fpr","fn@0.1%fpr","eval_time_s"
        ]}

    def _build_model(self):
        if self.cfg.model_kind == "lgbm" and _HAS_LGBM:
            base = LGBMClassifier(random_state=self.cfg.random_state)
            if self.fraud_cfg and self.fraud_cfg.use_calibration:
                return CalibratedClassifierCV(base, method="sigmoid", cv=3)
            return base
        # fall back to LR
        base = LogisticRegression(class_weight="balanced", max_iter=400, random_state=self.cfg.random_state)
        if self.fraud_cfg and self.fraud_cfg.use_calibration:
            return CalibratedClassifierCV(base, method="sigmoid", cv=3)
        return base

    def _evaluate_and_log(self, model):
        t0 = time.time()
        scores = model.predict_proba(self.X_test)[:, 1]
        auroc = roc_auc_score(self.y_test, scores)
        auprc = average_precision_score(self.y_test, scores)
        rec_1e3 = recall_at_fpr(self.y_test, scores, target_fpr=0.001)

        # extra diagnostics
        brier = brier_score_loss(self.y_test, scores)
        ece   = ece_score(self.y_test, scores)
        prec_fpr, th_fpr = precision_at_fpr(self.y_test, scores, target_fpr=0.001)
        p_at_100 = precision_at_k(self.y_test, scores, 100)
        p_at_500 = precision_at_k(self.y_test, scores, 500)
        lift_500 = lift_at_k(self.y_test, scores, 500)

        # simple profit model
        def expected_profit(y_true, y_scores, th, cost_fn=100.0, cost_fp=1.0):
            y_pred = (y_scores >= th).astype(int)
            tp = ((y_pred==1)&(y_true==1)).sum()
            fp = ((y_pred==1)&(y_true==0)).sum()
            fn = ((y_pred==0)&(y_true==1)).sum()
            return float(cost_fn*tp - cost_fp*fp), int(tp), int(fp), int(fn)
        profit, tp, fp, fn = expected_profit(self.y_test, scores, th_fpr, cost_fn=100.0, cost_fp=1.0)
        eval_time = time.time() - t0

        metrics = {
            "labels_used": int(self.labeled_mask.sum()),
            "frauds_found": int(self.y_train_full[self.labeled_mask].sum()),
            "auroc": float(auroc),
            "auprc": float(auprc),
            "recall@0.1%fpr": float(rec_1e3),
            "brier": float(brier), "ece": float(ece),
            "precision@0.1%fpr": float(prec_fpr), "threshold@0.1%fpr": float(th_fpr),
            "precision@100": float(p_at_100), "precision@500": float(p_at_500),
            "lift@500": float(lift_500),
            "profit@0.1%fpr": float(profit), "tp@0.1%fpr": int(tp), "fp@0.1%fpr": int(fp), "fn@0.1%fpr": int(fn),
            "eval_time_s": float(eval_time)
        }
        for k, v in metrics.items():
            self.history[k].append(v)
        print(f"Round {len(self.history['labels_used'])} | Labels: {metrics['labels_used']} | Frauds: {metrics['frauds_found']} | "
              f"AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f}")

    def run(self):
        print(f"\n--- Running: {self.run_name} ---")
        round_idx = 0
        model = None
        while self.labeled_mask.sum() < self.cfg.budget:
            round_idx += 1
            if not np.any(~self.labeled_mask):
                break

            X_l = self.X_train_full[self.labeled_mask]
            y_l = self.y_train_full[self.labeled_mask]
            X_p = self.X_train_full[~self.labeled_mask]

            model = self._build_model()
            model.fit(X_l, y_l)

            # evaluate (logs exactly one row)
            self._evaluate_and_log(model)

            if self.cfg.auroc_target is not None and self.history["auroc"][-1] >= self.cfg.auroc_target:
                print(f"Reached AUROC target {self.cfg.auroc_target:.3f}. Stopping.")
                break

            # query and update labels
            rel = self.sampler.select(
                model, X_p, self.cfg.batch_size,
                y_labeled=y_l, round_idx=round_idx, X_labeled=X_l
            )
            pool_idx = np.where(~self.labeled_mask)[0]
            abs_idx = pool_idx[rel]
            self.labeled_mask[abs_idx] = True

        # Final evaluation only if current label count wasn't already logged
        if model is None:
            model = self._build_model().fit(self.X_train_full[self.labeled_mask], self.y_train_full[self.labeled_mask])

        current_labels = int(self.labeled_mask.sum())
        already_logged = (len(self.history["labels_used"]) > 0 and int(self.history["labels_used"][-1]) == current_labels)
        if not already_logged:
            self._evaluate_and_log(model)

        self._save_results()

    def _save_results(self):
        # --- Align metric lengths defensively ---
        lengths = {k: len(v) for k, v in self.history.items()}
        if len(set(lengths.values())) != 1:
            Lmin = min(lengths.values())
            print(f"[WARN] Metric length mismatch detected: {lengths}. Truncating all series to the shortest length ({Lmin}).")
            for k in self.history:
                self.history[k] = self.history[k][:Lmin]

        df = pd.DataFrame(self.history)

        self.outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.outdir / f"{self.run_name}.metrics.csv", index=False)

        summary = {k: (v[-1] if len(v) else None) for k, v in self.history.items()}
        with open(self.outdir / f"{self.run_name}.summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Core curves
        for metric in ["auroc", "auprc", "recall@0.1%fpr"]:
            plt.figure(figsize=(7, 4))
            plt.plot(df["labels_used"], df[metric], marker="o", markersize=4)
            plt.title(f"{self.sampler.name.upper()} Performance: {metric.upper()} vs. labels")
            plt.xlabel("Number of labeled samples")
            plt.ylabel(metric.upper())
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(self.outdir / f"{self.run_name}.{metric}.png", dpi=150)
            plt.close()

# --------------------------- Runner ---------------------------

def run_experiment(
    data_path: str,
    strategy_name: str = "fraudpp",
    outdir: str = "results",
    seed: int = 0,
    batch_size: int = 200,
    seed_size: int = 1400,
    budget: int = 5000,
    model_kind: str = "lgbm",
    auroc_target: Optional[float] = None,
    # FRaUD++ defaults
    alpha: float = 1.0, gamma: float = 2.0, prior: float = 0.00172,
    quota: float = 0.25, lambda_anomaly: float = 0.5, use_schedules: bool = True,
    rarity_k: int = 10, exploit_topk_factor: float = 5.0, explore_topk_factor: float = 10.0,
    use_calibration: bool = False
):
    X, y, feature_names, ds_name = load_dataset_auto(data_path)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = ALConfig(batch_size=batch_size, seed_size=seed_size, budget=budget,
                   model_kind=model_kind, random_state=seed, auroc_target=auroc_target)
    fcfg = FRaUDConfig(alpha=alpha, gamma=gamma, prior=prior,
                       quota=quota, lambda_anomaly=lambda_anomaly, use_schedules=use_schedules,
                       rarity_k=rarity_k, exploit_topk_factor=exploit_topk_factor,
                       explore_topk_factor=explore_topk_factor, random_state=seed,
                       use_calibration=use_calibration)

    sampler_map = {
        "random": RandomSampler(seed),
        "entropy": EntropySampler(),
        "margin": MarginSampler(),
        "cost_balanced": CostBalancedEntropySampler(),
        "qbc": QBCSampler(seed),
        "fraud": FRaUDSampler(fcfg),
        "fraudpp": FRaUDPlusSampler(fcfg)
    }
    if strategy_name not in sampler_map:
        raise ValueError(f"Unknown strategy '{strategy_name}'")
    sampler = sampler_map[strategy_name]

    run_name = f"{strategy_name}_seed{seed}_{ds_name}"
    learner = ActiveLearner(X, y, cfg, sampler, outdir, run_name, fraud_cfg=fcfg)
    learner.run()

def main():
    # Defaults (so you can double-click run)
    defaults = dict(
       # data_path = "PS_20174392719_1491204439457_log.csv",  # or # "creditcard.csv"
        data_path="creditcard.csv",
        strategy_name = "fraudpp",
        outdir = "results",
        seed = 0,
        batch_size = 200,
        seed_size = 1400,
        budget = 5000,
        model_kind = "lgbm" if _HAS_LGBM else "lr",
        auroc_target = None,
        # FRaUD++ knobs
        alpha=1.0, gamma=2.0, prior=0.00172,
        quota=0.25, lambda_anomaly=0.5, use_schedules=True,
        rarity_k=10, exploit_topk_factor=5.0, explore_topk_factor=10.0,
        use_calibration=False
    )

    # If you prefer CLI, uncomment this block; otherwise the defaults above are used.
    # p = argparse.ArgumentParser(description="FRaUD++ Active Learning Experiment (auto-detects dataset)")
    # p.add_argument("--data", default=defaults["data_path"])
    # args = p.parse_args()
    # defaults["data_path"] = args.data

    run_experiment(**defaults)

if __name__ == "__main__":
    main()
