"""
 Query strategies for active learning.

This module implements several sampling strategies used during active learning,
including FRaUD/FRaUD++ and a hybrid that mixes FRaUD++ with QBC (committee
uncertainty). To keep things tidy and easier to reason about, shared bits live
in _FraudShared and small top-level helpers replace previously duplicated code.

In plain words:
- FRaUD balances uncertainty with a prior and adds diversity.
- FRaUD++ adds a "rarity" bump to encourage novel-but-promising points.
- Hybrid blends FRaUD++ with a simple Query-by-Committee signal.
- Baselines (random, entropy, margin, cost-balanced, pure QBC) are included.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Any

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# pandas is optional; used only to silence LightGBM warnings when possible
try:
    import pandas as pd
except Exception:
    pd = None

from utils import (
    binary_entropy,
    farthest_point_diverse_subset,
    advanced_diverse_subset,
    precision_at_fpr,
    qbc_js_disagreement,
)


# Small helper to avoid LightGBM "feature names" warning when the model was fitted on a DataFrame
def _ensure_feature_names(model, X):
    """If the trained model remembers feature names, wrap array X into a DataFrame with those names.

    This keeps downstream predict_proba calls quiet for libraries that inspect column names.
    """
    if pd is not None and not hasattr(X, "columns") and hasattr(model, "feature_name_"):
        cols = getattr(model, "feature_name_", None)
        arr = np.asarray(X)
        if cols is not None and len(cols) == arr.shape[1]:
            return pd.DataFrame(arr, columns=list(cols))
    return X


# Committee disagreement (QBC) — de-duplicated helper

def _committee_disagreement(X_l, y_l, X_pool, seed: int) -> np.ndarray:
    """Train a tiny committee on (X_l, y_l) and compute JS-style disagreement on X_pool.

    Returns a 1D array with higher values meaning more disagreement. Falls back to zeros
    if the labeled set is too small to train a meaningful committee.
    """
    if X_l is None or y_l is None or len(y_l) < 50:
        return np.zeros(np.asarray(X_pool).shape[0], dtype=float)

    X_l_np = np.asarray(X_l)
    X_p_np = np.asarray(X_pool)
    scaler: Any = StandardScaler()
    scaler.fit(X_l_np)
    X_l_s = scaler.transform(X_l_np)
    X_p_s = scaler.transform(X_p_np)

    n_l, p = X_l_s.shape
    if p < 64:
        hls = (64, 32)
    elif p < 256:
        hls = (128, 64)
    else:
        hls = (256, 128)

    bs = max(32, min(256, n_l // 10 if n_l >= 100 else n_l))

    committee = [
        LogisticRegression(class_weight="balanced", max_iter=200, random_state=seed),
        RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=seed),
        MLPClassifier(
            hidden_layer_sizes=hls,
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            alpha=0.001,
            batch_size=bs,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            max_iter=400,
            random_state=seed,
            verbose=False,
        ),
    ]
    preds = []
    for clf in committee:
        clf.fit(X_l_s, y_l)
        preds.append(clf.predict_proba(X_p_s)[:, 1])
    preds = np.vstack(preds)
    return qbc_js_disagreement(preds)


# Base

class BaseQueryStrategy:
    """Abstract interface for query strategies."""
    name: str = "base"
    def select(self, model, X_pool: np.ndarray, batch_size: int, **kwargs) -> np.ndarray:
        raise NotImplementedError



class _FraudShared:
    """Helper mixin for FRaUD-family strategies (shared schedules and scoring).

    This class centralizes logic that used to be copied into FRaUD, FRaUD++ and
    the Hybrid. Fewer moving parts means fewer chances to diverge accidentally.
    """
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _minmax(x: np.ndarray, robust: bool = False) -> np.ndarray:
        """Min-max normalize to [0,1]. If robust=True, clip outliers first."""
        x = np.asarray(x, dtype=float)
        if robust and len(x) > 20:
            p5, p95 = np.percentile(x, [5, 95])
            x = np.clip(x, p5, p95)
        lo, hi = np.min(x), np.max(x)
        return (x - lo) / (hi - lo + 1e-12)

    def _schedule_quota_lambda(self, y_labeled: np.ndarray, round_idx: int,
                               budget_used: int = 0, total_budget: int = 5000,
                               current_auroc: float = 0.0):
        """
        Enhanced scheduling with multiple modes:
        - fixed: original hard round-based transitions
        - smooth: continuous interpolation between phases with blending
        - adaptive: based on fraction of budget consumed
        - performance: based on model performance metrics
        """
        if not self.cfg.use_schedules:
            return self.cfg.quota, self.cfg.lambda_anomaly

        schedule_mode = getattr(self.cfg, "schedule_mode", "fixed")

        # Define phase parameters
        quota_vals = [self.cfg.quota_early, self.cfg.quota_mid, self.cfg.quota_late]
        lam_vals = [self.cfg.lam_early, self.cfg.lam_mid, self.cfg.lam_late]

        if schedule_mode == "fixed":
            # Original implementation: hard transitions based on rounds and fraud count
            n_frauds = int(y_labeled.sum()) if len(y_labeled) else 0
            if round_idx <= 3:
                quota, lam = quota_vals[0], lam_vals[0]
            elif round_idx <= 8 or n_frauds < 20:
                quota, lam = quota_vals[1], lam_vals[1]
            else:
                quota, lam = quota_vals[2], lam_vals[2]

        elif schedule_mode == "smooth":
            # Smooth transitions with blending between phases
            n_frauds = int(y_labeled.sum()) if len(y_labeled) else 0
            transition_rounds = getattr(self.cfg, "smooth_transition_rounds", 2)

            if round_idx <= 3:
                phase = 0.0
            elif round_idx <= 3 + transition_rounds:
                # Blend early -> mid
                phase = (round_idx - 3) / max(1, transition_rounds)
            elif round_idx <= 8 or n_frauds < 20:
                phase = 1.0
            elif round_idx <= 8 + transition_rounds and n_frauds >= 20:
                # Blend mid -> late
                phase = 1.0 + (round_idx - 8) / max(1, transition_rounds)
            else:
                phase = 2.0

            # Interpolate between phases
            phase = float(np.clip(phase, 0.0, 2.0))
            if phase < 1.0:
                quota = quota_vals[0] + (quota_vals[1] - quota_vals[0]) * phase
                lam = lam_vals[0] + (lam_vals[1] - lam_vals[0]) * phase
            else:
                quota = quota_vals[1] + (quota_vals[2] - quota_vals[1]) * (phase - 1.0)
                lam = lam_vals[1] + (lam_vals[2] - lam_vals[1]) * (phase - 1.0)

        elif schedule_mode == "adaptive":
            # Based on fraction of budget consumed
            budget_frac = budget_used / max(1, total_budget)
            thresholds = getattr(self.cfg, "adaptive_thresholds", (0.15, 0.35, 0.60))

            if budget_frac < thresholds[0]:
                phase = budget_frac / thresholds[0]  # 0 to 1
                quota = quota_vals[0] + (quota_vals[1] - quota_vals[0]) * phase
                lam = lam_vals[0] + (lam_vals[1] - lam_vals[0]) * phase
            elif budget_frac < thresholds[1]:
                phase = (budget_frac - thresholds[0]) / (thresholds[1] - thresholds[0])
                quota = quota_vals[1] + (quota_vals[2] - quota_vals[1]) * phase
                lam = lam_vals[1] + (lam_vals[2] - lam_vals[1]) * phase
            else:
                # Late phase - continue increasing exploitation
                extra_phase = min(1.0, (budget_frac - thresholds[1]) / (thresholds[2] - thresholds[1]))
                quota = quota_vals[2] + (0.5 - quota_vals[2]) * extra_phase * 0.5
                lam = lam_vals[2] * (1.0 - extra_phase * 0.5)

        elif schedule_mode == "performance":
            # Based on model performance - shift to exploitation when model is confident
            perf_threshold = getattr(self.cfg, "performance_auroc_threshold", 0.85)
            n_frauds = int(y_labeled.sum()) if len(y_labeled) else 0

            # Early rounds: explore
            if round_idx <= 3 or n_frauds < 10:
                quota, lam = quota_vals[0], lam_vals[0]
            # Good performance: exploit more
            elif current_auroc >= perf_threshold:
                quota, lam = quota_vals[2], lam_vals[2]
            # Moderate performance: balanced
            else:
                # Interpolate based on performance
                if current_auroc > 0.0:
                    perf_ratio = min(1.0, current_auroc / perf_threshold)
                    quota = quota_vals[0] + (quota_vals[2] - quota_vals[0]) * perf_ratio
                    lam = lam_vals[0] + (lam_vals[2] - lam_vals[0]) * perf_ratio
                else:
                    quota, lam = quota_vals[1], lam_vals[1]
        else:
            # Fallback to mid values
            quota, lam = quota_vals[1], lam_vals[1]

        return float(quota), float(lam)

    def _schedule_tabp_weight(self, round_idx: int, budget_used: int = 0,
                              total_budget: int = 5000) -> float:
        """
        Enhanced TABP weight scheduling with smooth ramping.
        TABP focuses on decision boundary - ramp it up as model stabilizes.
        """
        if not (self.cfg.use_schedules and self.cfg.tabp_ramp):
            return float(self.cfg.tabp_weight)

        schedule_mode = getattr(self.cfg, "schedule_mode", "fixed")

        if schedule_mode == "fixed":
            # Original hard transitions
            if round_idx <= 3:
                return 0.4
            if round_idx <= 8:
                return 0.7
            return 1.0

        elif schedule_mode == "smooth":
            # Smooth ramp from 0.4 to 1.0 over rounds
            if round_idx <= 3:
                return 0.4
            elif round_idx <= 10:
                # Smooth interpolation from 0.4 to 1.0
                progress = (round_idx - 3) / 7.0  # 7 rounds to ramp
                return 0.4 + 0.6 * progress
            else:
                return 1.0

        elif schedule_mode == "adaptive":
            # Based on budget consumption
            budget_frac = budget_used / max(1, total_budget)
            if budget_frac < 0.20:
                return 0.4
            elif budget_frac < 0.60:
                # Ramp from 0.4 to 1.0
                progress = (budget_frac - 0.20) / 0.40
                return 0.4 + 0.6 * progress
            else:
                return 1.0

        elif schedule_mode == "performance":
            # Ramp based on rounds (simpler for TABP)
            return min(1.0, 0.4 + 0.15 * round_idx)

        return float(self.cfg.tabp_weight)

    def _compute_threshold(
        self,
        y_labeled: np.ndarray,
        probs_labeled: np.ndarray,
        probs_pool: Optional[np.ndarray] = None
    ) -> float:
        # Robust cold start that prefers labeled data when it is informative
        n = len(y_labeled)
        n_neg = int(np.sum(y_labeled == 0)) if n else 0
        if n < 50 or n_neg < 25:
            if probs_pool is not None and probs_pool.size > 0:
                q = np.clip(1.0 - self.cfg.target_fpr, 0.0, 1.0)
                thr = float(np.quantile(probs_pool, q))
                return float(np.clip(thr, 0.1, 0.99))
            return 0.5

        try:
            _, threshold = precision_at_fpr(
                y_labeled,
                probs_labeled,
                target_fpr=self.cfg.target_fpr
            )
            return float(np.clip(float(threshold), 0.1, 0.99))
        except Exception:
            return 0.5

    def _compute_tabp(self, probs: np.ndarray, threshold: float) -> np.ndarray:
        """
        Compute threshold-adaptive boundary proximity (TABP).
        Modes:
          - "inverse": 1 / (|p - t| + 1e-9)  (capped)
          - "gaussian": exp(-d^2 / (2 * tau^2)), tau = tabp_temperature
          - "hinge": max(0, m - d) / m, m = tabp_margin
        """
        d = np.abs(probs - threshold)
        kind = getattr(self.cfg, "tabp_kind", "inverse")

        if kind == "gaussian":
            tau = float(max(1e-9, getattr(self.cfg, "tabp_temperature", 0.1)))
            tabp = np.exp(-(d * d) / (2.0 * tau * tau))
            # Already bounded in [0,1]
            return tabp

        if kind == "hinge":
            m = float(max(1e-9, getattr(self.cfg, "tabp_margin", 0.05)))
            tabp = np.clip((m - d) / m, 0.0, 1.0)
            return tabp

        # default: inverse distance to boundary
        tabp = 1.0 / (d + 1e-9)
        tabp = np.clip(tabp, 0.0, 100.0)
        return tabp

    def _mix_tabp(self, base_score: np.ndarray, tabp: np.ndarray, round_idx: int,
                  budget_used: int = 0, total_budget: int = 5000) -> np.ndarray:
        # Gentle mixing so early noisy thresholds do not dominate
        beta = self._schedule_tabp_weight(round_idx, budget_used, total_budget)
        tabp_n = self._minmax(tabp) + 1e-3
        return base_score * ((1.0 - beta) + beta * tabp_n)

    def _diverse(self, Xs_cand: np.ndarray, n: int, probs: Optional[np.ndarray] = None) -> np.ndarray:
        mode = getattr(self.cfg, "diversity_mode", "pure")
        if mode != "pure" and probs is not None:
            return advanced_diverse_subset(
                Xs_cand,
                probs,
                n_samples=n,
                metric=self.cfg.diversity_metric,
                diversity_mode=mode,
                embed_max_dim=self.cfg.embed_max_dim,
                random_state=self.cfg.random_state,
            )
        return farthest_point_diverse_subset(
            Xs_cand,
            n_samples=n,
            metric=self.cfg.diversity_metric,
            embed=True,
            embed_max_dim=self.cfg.embed_max_dim,
            random_state=self.cfg.random_state,
        )

    # New shared helpers to cut duplication

    def _probs_and_threshold(self, model, X_pool, X_labeled, y_labeled) -> tuple[np.ndarray, float]:
        """Predict probabilities on the pool and compute an operating threshold.

        Threshold prefers using labeled data when available, otherwise backs off to
        a robust percentile of the pool as a cold start.
        """
        X_in = _ensure_feature_names(model, X_pool)
        probs = model.predict_proba(X_in)[:, 1]
        if X_labeled is not None and len(y_labeled) >= 1:
            Xl_in = _ensure_feature_names(model, X_labeled)
            probs_labeled = model.predict_proba(Xl_in)[:, 1]
            threshold = self._compute_threshold(y_labeled, probs_labeled, probs_pool=probs)
        else:
            threshold = 0.5
        return probs, threshold

    def _base_score(self, probs: np.ndarray, y_labeled: np.ndarray, round_idx: int, threshold: float,
                    budget_used: int = 0, total_budget: int = 5000) -> np.ndarray:
        """Compute the FRaUD base score: entropy^alpha * focal(prior, gamma), mixed with TABP."""
        unc = binary_entropy(probs)
        obs_prior = y_labeled.mean() if len(y_labeled) else 0.0
        prior = max(obs_prior, self.cfg.prior or 0.0, 1e-6)
        focal = (probs / (prior + probs + 1e-12)) ** self.cfg.gamma
        base = (unc ** self.cfg.alpha) * focal
        tabp = self._compute_tabp(probs, threshold)
        return self._mix_tabp(base, tabp, round_idx, budget_used, total_budget)

    def _rarity_boost(self, s0: np.ndarray, Xs: np.ndarray, batch_size: int, lambda_anomaly: float) -> np.ndarray:
        """Apply a local rarity/novelty bump to top pre-candidates selected by s0."""
        n_pool = len(Xs)
        pre_k = min(max(int(self.cfg.explore_topk_factor * batch_size * 4), batch_size), n_pool)
        pre_candidates = np.argsort(-s0)[:pre_k]
        final_score = s0.copy()
        if len(pre_candidates) >= 2:
            cand_scaled = Xs[pre_candidates]
            kn = min(self.cfg.rarity_k, max(1, len(pre_candidates) - 1))
            nbrs = NearestNeighbors(n_neighbors=kn + 1).fit(cand_scaled)
            distances, _ = nbrs.kneighbors(cand_scaled)
            distances = distances[:, 1:]  # drop self
            rarity_c = 1.0 / (distances.mean(axis=1) + 1e-9)
            rarity_c = (rarity_c - rarity_c.min()) / (rarity_c.max() - rarity_c.min() + 1e-9)
            final_score[pre_candidates] = s0[pre_candidates] * (1.0 + lambda_anomaly * rarity_c)
        return final_score

    def _compute_graph_features(self, s0: np.ndarray, Xs: np.ndarray, probs: np.ndarray,
                                batch_size: int, k_neighbors: int = 10,
                                hub_weight: float = 0.5, bridge_weight: float = 0.5) -> np.ndarray:
        """
        Compute graph-based features for cluster-aware selection.

        Builds a k-NN graph over the most suspicious points and computes:
        1. Hub score: inverse mean distance to neighbors (cluster density)
        2. Bridge score: variance of neighbors' fraud probabilities (boundary diversity)

        Returns a bonus score array (same shape as s0) to be added to base scores.
        """
        n_pool = len(Xs)

        # Focus on top suspicious candidates (M = topk_factor × batch_size)
        topk_factor = getattr(self.cfg, 'graph_topk_factor', 10)
        M = min(n_pool, max(batch_size * topk_factor, batch_size * 5))

        # Need at least 3 points to build meaningful graph (adaptive k)
        if M < 3:
            return np.zeros_like(s0)

        top_idx = np.argsort(-s0)[:M]
        X_top = Xs[top_idx]

        # Build k-NN graph with adaptive k (never exceeds available points)
        kn = min(k_neighbors, max(2, len(X_top) - 1))
        nbrs = NearestNeighbors(n_neighbors=kn + 1, metric='euclidean').fit(X_top)
        distances, neighbor_ids = nbrs.kneighbors(X_top)

        # Drop self from neighbor sets
        distances = distances[:, 1:]
        neighbor_ids = neighbor_ids[:, 1:]

        # Compute hub score: inverse mean distance (dense clusters have high hub score)
        hub_score = 1.0 / (distances.mean(axis=1) + 1e-9)

        # Compute bridge score: diversity of neighbors' fraud probabilities
        # Combines variance and range to better capture boundary points
        local_probs = probs[top_idx]
        bridge_score = np.zeros(len(top_idx))
        for i in range(len(top_idx)):
            # Include the point itself for more accurate diversity
            neighbor_probs = np.append(local_probs[neighbor_ids[i]], local_probs[i])
            # Combine variance (spread) with range (extremity) for better boundary detection
            variance = np.var(neighbor_probs)
            prob_range = np.ptp(neighbor_probs)  # max - min
            bridge_score[i] = 0.7 * variance + 0.3 * prob_range

        # Normalize both scores to [0, 1]
        def _normalize(v):
            v = v.astype(float)
            v_min, v_max = v.min(), v.max()
            if v_max - v_min < 1e-9:
                return np.zeros_like(v)
            return (v - v_min) / (v_max - v_min)

        hub_normalized = _normalize(hub_score)
        bridge_normalized = _normalize(bridge_score)

        # Combine hub and bridge scores
        graph_bonus_local = hub_weight * hub_normalized + bridge_weight * bridge_normalized

        # Map back to full pool (only top M get bonuses)
        graph_bonus = np.zeros(n_pool, dtype=float)
        graph_bonus[top_idx] = graph_bonus_local

        return graph_bonus

    def _two_stage_pick(self, Xs: np.ndarray, probs: np.ndarray, explore_score: np.ndarray,
                        batch_size: int, quota: float) -> np.ndarray:
        """Two-stage pick: exploit by high probability, then explore by explore_score with diversity.

        Returns absolute indices into the pool.
        """
        n_pool = len(Xs)
        n_exploit = int(quota * batch_size)
        n_explore = batch_size - n_exploit

        # Exploit: pick likely frauds but keep them diverse
        ex_pool = int(max(n_exploit, self.cfg.exploit_topk_factor * max(1, n_exploit)))
        ex_candidates = np.argsort(-probs)[:min(ex_pool, n_pool)] if ex_pool > 0 else np.array([], dtype=int)
        final_ex = np.array([], dtype=int)
        if ex_candidates.size > 0:
            rel = self._diverse(Xs[ex_candidates], min(n_exploit, len(ex_candidates)), probs=probs[ex_candidates])
            final_ex = ex_candidates[rel]

        # Explore: avoid already-picked; use explore_score to rank; keep diversity
        mask = np.ones(n_pool, dtype=bool)
        mask[final_ex] = False
        rem_idx = np.where(mask)[0]
        exr_pool = int(max(n_explore, self.cfg.explore_topk_factor * max(1, n_explore)))
        final_exr = np.array([], dtype=int)
        if exr_pool > 0 and rem_idx.size > 0:
            topk = min(exr_pool, rem_idx.size)
            cand_sorted = rem_idx[np.argsort(-explore_score[rem_idx])[:topk]]
            rel = self._diverse(Xs[cand_sorted], min(n_explore, len(cand_sorted)), probs=probs[cand_sorted])
            final_exr = cand_sorted[rel]

        return np.concatenate([final_ex, final_exr])


class FRaUDSampler(BaseQueryStrategy, _FraudShared):
    """FRaUD sampler: uncertainty + focal prior + diversity with schedules.

    In a sentence: go where the model is unsure, prefer plausible frauds, and
    avoid picking duplicates.
    """
    name = "fraud"
    def __init__(self, cfg):
        _FraudShared.__init__(self, cfg)

    def select(self, model, X_pool: np.ndarray, batch_size: int, **kwargs) -> np.ndarray:
        X_pool_np = np.asarray(X_pool)
        Xs = StandardScaler().fit_transform(X_pool_np)

        round_idx = kwargs.get("round_idx", 1)
        y_labeled = kwargs.get("y_labeled", np.array([]))
        X_labeled = kwargs.get("X_labeled", None)

        probs, threshold = self._probs_and_threshold(model, X_pool, X_labeled, y_labeled)

        # Extract scheduling context
        budget_used = kwargs.get("budget_used", 0)
        total_budget = kwargs.get("total_budget", 5000)
        current_auroc = kwargs.get("current_auroc", 0.0)

        quota, _ = self._schedule_quota_lambda(y_labeled, round_idx, budget_used, total_budget, current_auroc)
        s0 = self._base_score(probs, y_labeled, round_idx, threshold, budget_used, total_budget)

        return self._two_stage_pick(Xs, probs, s0, batch_size, quota)


class FRaUDPlusSampler(BaseQueryStrategy, _FraudShared):
    """FRaUD++ sampler: FRaUD plus a rarity/novelty boost around peaks.

    In practice this often surfaces hard-but-informative cases;
    think "don't just chase the obvious top scores".
    """
    name = "fraudpp"
    def __init__(self, cfg):
        _FraudShared.__init__(self, cfg)

    def select(self, model, X_pool: np.ndarray, batch_size: int, **kwargs) -> np.ndarray:
        X_pool_np = np.asarray(X_pool)
        Xs = StandardScaler().fit_transform(X_pool_np)

        round_idx = kwargs.get("round_idx", 1)
        y_labeled = kwargs.get("y_labeled", np.array([]))
        X_labeled = kwargs.get("X_labeled", None)

        probs, threshold = self._probs_and_threshold(model, X_pool, X_labeled, y_labeled)

        # Extract scheduling context
        budget_used = kwargs.get("budget_used", 0)
        total_budget = kwargs.get("total_budget", 5000)
        current_auroc = kwargs.get("current_auroc", 0.0)

        quota, lambda_anomaly = self._schedule_quota_lambda(y_labeled, round_idx, budget_used, total_budget, current_auroc)
        s0 = self._base_score(probs, y_labeled, round_idx, threshold, budget_used, total_budget)
        final_score = self._rarity_boost(s0, Xs, batch_size, lambda_anomaly)

        return self._two_stage_pick(Xs, probs, final_score, batch_size, quota)


class HybridFraudPPQBCSampler(BaseQueryStrategy, _FraudShared):
    """Hybrid of FRaUD++ ranking with a QBC disagreement boost."""
    name = "fraudpp_hybrid"

    def __init__(self, cfg, seed: int, w_qbc: float = 0.5):
        _FraudShared.__init__(self, cfg)
        self.seed = seed
        self.w_qbc = float(w_qbc)

    def select(self, model, X_pool, batch_size, **kwargs):
        X_pool_np = np.asarray(X_pool)
        Xs = StandardScaler().fit_transform(X_pool_np)

        round_idx = kwargs.get("round_idx", 1)
        y_labeled = kwargs.get("y_labeled", np.array([]))
        X_labeled = kwargs.get("X_labeled", None)

        probs, threshold = self._probs_and_threshold(model, X_pool, X_labeled, y_labeled)

        # Extract scheduling context
        budget_used = kwargs.get("budget_used", 0)
        total_budget = kwargs.get("total_budget", 5000)
        current_auroc = kwargs.get("current_auroc", 0.0)

        quota, lambda_anomaly = self._schedule_quota_lambda(y_labeled, round_idx, budget_used, total_budget, current_auroc)
        s0 = self._base_score(probs, y_labeled, round_idx, threshold, budget_used, total_budget)
        final_score = self._rarity_boost(s0, Xs, batch_size, lambda_anomaly)

        s_qbc = _committee_disagreement(X_labeled, y_labeled, X_pool, self.seed)
        s_mix = self._minmax(final_score) + self.w_qbc * self._minmax(s_qbc)

        return self._two_stage_pick(Xs, probs, s_mix, batch_size, quota)


class GraphFraudHybridSampler(BaseQueryStrategy, _FraudShared):
    """
    Graph-FRaUD Hybrid: FRaUD++ + Graph-based cluster reasoning + QBC disagreement.

    This strategy combines three complementary signals:
    1. FRaUD++ base scoring (uncertainty × focal prior × TABP × rarity)
    2. Graph features (hub scores for cluster coverage + bridge scores for boundary exploration)
    3. QBC disagreement (model uncertainty at decision boundaries)

    The graph reasoning explicitly models fraud clusters via k-NN connectivity,
    identifying both cluster representatives (hubs) and connectors between different
    fraud patterns (bridges). Combined with QBC, this provides comprehensive coverage
    of both fraud topology and model uncertainty.

    Novel contribution: First strategy to explicitly combine graph-theoretic cluster
    analysis with committee-based uncertainty for fraud detection active learning.
    """
    name = "graph_hybrid"

    def __init__(self, cfg, seed: int, w_qbc: float = 0.5,
                 k_neighbors: int = 10, hub_weight: float = 0.5, bridge_weight: float = 0.5,
                 adaptive_qbc: bool = True):
        """
        Initialize Graph-FRaUD Hybrid sampler.

        Args:
            cfg: FRaUDConfig with all base parameters
            seed: Random seed for QBC committee training
            w_qbc: Weight for QBC disagreement signal (default 0.5)
            k_neighbors: Number of neighbors in k-NN graph (default 10)
            hub_weight: Weight for hub score (cluster density, default 0.5)
            bridge_weight: Weight for bridge score (boundary diversity, default 0.5)
            adaptive_qbc: Whether to adapt QBC weight based on training phase (default True)
        """
        _FraudShared.__init__(self, cfg)
        self.seed = seed
        self.w_qbc = float(w_qbc)
        self.k_neighbors = int(k_neighbors)
        self.hub_weight = float(hub_weight)
        self.bridge_weight = float(bridge_weight)
        self.adaptive_qbc = adaptive_qbc

    def _adaptive_qbc_weight(self, base_weight: float, round_idx: int,
                            n_labeled: int, auroc: float) -> float:
        """
        Dynamically adjust QBC weight based on training maturity.

        Rationale:
        - Early rounds: QBC less reliable (small labeled set, unstable models)
        - Mid rounds: Increase QBC as models stabilize
        - Late rounds: High QBC for boundary refinement

        Args:
            base_weight: Configured base QBC weight
            round_idx: Current AL round
            n_labeled: Number of labeled samples
            auroc: Current model AUROC

        Returns:
            Adjusted QBC weight
        """
        if not self.adaptive_qbc:
            return base_weight

        # Factor 1: Round-based maturity (0 to 1.2x)
        round_factor = min(1.2, 0.3 + 0.05 * round_idx)

        # Factor 2: Labeled set size (need enough data for stable committee)
        if n_labeled < 200:
            size_factor = 0.3  # Reduce QBC when data is scarce
        elif n_labeled < 500:
            size_factor = 0.6
        elif n_labeled < 1000:
            size_factor = 0.9
        else:
            size_factor = 1.1  # Boost QBC with large labeled sets

        # Factor 3: Model performance (better model = more reliable disagreement)
        if auroc > 0.88:
            perf_factor = 1.2  # High confidence in boundary
        elif auroc > 0.80:
            perf_factor = 1.0
        else:
            perf_factor = 0.7  # Model still learning, QBC less informative

        # Combine factors (geometric mean for balance)
        adjusted = base_weight * (round_factor * size_factor * perf_factor) ** (1/3)

        # Clamp to reasonable range
        return np.clip(adjusted, 0.1 * base_weight, 2.0 * base_weight)

    def select(self, model, X_pool, batch_size, **kwargs):
        """
        Select batch_size samples using Graph-FRaUD Hybrid strategy.

        Steps:
        1. Compute FRaUD++ base score (uncertainty + focal prior + TABP)
        2. Apply rarity boost (local novelty detection)
        3. Compute graph features (hub + bridge scores over k-NN graph)
        4. Compute QBC disagreement (committee variance)
        5. Combine all signals with weighted sum
        6. Select with diversity-aware two-stage picking
        """
        X_pool_np = np.asarray(X_pool)
        Xs = StandardScaler().fit_transform(X_pool_np)

        round_idx = kwargs.get("round_idx", 1)
        y_labeled = kwargs.get("y_labeled", np.array([]))
        X_labeled = kwargs.get("X_labeled", None)

        probs, threshold = self._probs_and_threshold(model, X_pool, X_labeled, y_labeled)

        # Extract scheduling context
        budget_used = kwargs.get("budget_used", 0)
        total_budget = kwargs.get("total_budget", 5000)
        current_auroc = kwargs.get("current_auroc", 0.0)

        quota, lambda_anomaly = self._schedule_quota_lambda(y_labeled, round_idx, budget_used, total_budget, current_auroc)

        # Step 1-2: FRaUD++ scoring (uncertainty + focal + TABP + rarity)
        s0 = self._base_score(probs, y_labeled, round_idx, threshold, budget_used, total_budget)
        fraudpp_score = self._rarity_boost(s0, Xs, batch_size, lambda_anomaly)

        # Step 3: Graph-based cluster features (hub + bridge)
        graph_bonus = self._compute_graph_features(
            fraudpp_score, Xs, probs, batch_size,
            self.k_neighbors, self.hub_weight, self.bridge_weight
        )

        # Combine FRaUD++ with graph features
        graph_fraud_score = fraudpp_score + graph_bonus

        # Step 4: QBC disagreement
        s_qbc = _committee_disagreement(X_labeled, y_labeled, X_pool, self.seed)

        # Step 5: Weighted combination of all three signals
        # Normalize each component to [0, 1] before combining
        s_mix = self._minmax(graph_fraud_score) + self.w_qbc * self._minmax(s_qbc)

        # Step 6: Two-stage selection with diversity
        return self._two_stage_pick(Xs, probs, s_mix, batch_size, quota)


# Baselines

class RandomSampler(BaseQueryStrategy):
    """Uniform random sampling from the unlabeled pool."""
    name = "random"
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
    def select(self, _, X_pool, batch_size, **kwargs):
        n = np.asarray(X_pool).shape[0]
        return self.rng.choice(n, size=min(batch_size, n), replace=False)

class EntropySampler(BaseQueryStrategy):
    """Select by descending predictive entropy (uncertainty)."""
    name = "entropy"
    def select(self, model, X_pool, batch_size, **kwargs):
        X_in = _ensure_feature_names(model, X_pool)
        probs = model.predict_proba(X_in)[:, 1]
        return np.argsort(-binary_entropy(probs))[:batch_size]

class MarginSampler(BaseQueryStrategy):
    """Select by ascending margin between class probabilities."""
    name = "margin"
    def select(self, model, X_pool, batch_size, **kwargs):
        X_in = _ensure_feature_names(model, X_pool)
        probs = model.predict_proba(X_in)
        margin = np.abs(probs[:, 1] - probs[:, 0])
        return np.argsort(margin)[:batch_size]

class CostBalancedEntropySampler(BaseQueryStrategy):
    """Entropy weighted by an asymmetric fraud cost."""
    name = "cost_balanced"
    def __init__(self, fraud_cost: float = 100.0):
        self.fraud_cost = fraud_cost
    def select(self, model, X_pool, batch_size, **kwargs):
        X_in = _ensure_feature_names(model, X_pool)
        probs = model.predict_proba(X_in)[:, 1]
        score = binary_entropy(probs) * (probs * self.fraud_cost + (1 - probs))
        return np.argsort(-score)[:batch_size]

class QBCSampler(BaseQueryStrategy):
    """Query-by-Committee: pick points where a small committee disagrees most."""
    name = "qbc"
    def __init__(self, seed: int):
        self.seed = seed
    def select(self, model, X_pool, batch_size, **kwargs):
        X_l = kwargs.get("X_labeled")
        y_l = kwargs.get("y_labeled")
        if X_l is None or y_l is None or len(y_l) < 50:
            return EntropySampler().select(model, X_pool, batch_size)
        js = _committee_disagreement(X_l, y_l, X_pool, self.seed)
        return np.argsort(-js)[:batch_size]
