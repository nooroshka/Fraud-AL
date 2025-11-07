from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from utils import binary_entropy

class BaseQueryStrategy:
    name: str = "base"
    def select(self, model, X_pool: np.ndarray, batch_size: int, **kwargs) -> np.ndarray:
        raise NotImplementedError

class RandomSampler(BaseQueryStrategy):
    name = "random"
    def __init__(self, seed: int): self.rng = np.random.default_rng(seed)
    def select(self, _, X_pool, batch_size, **kwargs):
        n = X_pool.shape[0]
        return self.rng.choice(n, size=min(batch_size, n), replace=False)

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
    def __init__(self, fraud_cost: float = 100.0): self.fraud_cost = fraud_cost
    def select(self, model, X_pool, batch_size, **kwargs):
        p = model.predict_proba(X_pool)[:, 1]
        score = binary_entropy(p) * (p * self.fraud_cost + (1 - p))
        return np.argsort(-score)[:batch_size]

class QBCSampler(BaseQueryStrategy):
    name = "qbc"
    def __init__(self, seed: int): self.seed = seed
    def select(self, model, X_pool, batch_size, **kwargs):
        X_l, y_l = kwargs.get("X_labeled"), kwargs.get("y_labeled")
        if X_l is None or y_l is None or len(y_l) < 50:
            return EntropySampler().select(model, X_pool, batch_size)
        scaler = StandardScaler().fit(X_l)
        X_l_s = scaler.transform(X_l)
        X_p_s = scaler.transform(X_pool)
        committee = [
            LogisticRegression(class_weight="balanced", max_iter=200, random_state=self.seed),
            RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=self.seed),
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, early_stopping=True, random_state=self.seed),
        ]
        preds = []
        for clf in committee:
            clf.fit(X_l_s, y_l)
            preds.append(clf.predict_proba(X_p_s)[:, 1])
        preds = np.vstack(preds)
        return np.argsort(-np.std(preds, axis=0))[:batch_size]

# FRaUD / FRaUD++

@dataclass
class FRaUDConfig:
    alpha: float = 1.0
    gamma: float = 2.0
    prior: float = 0.00172
    diversity_metric: str = "euclidean"
    embed_max_dim: int = 32
    quota: float = 0.25
    lambda_anomaly: float = 0.5
    use_schedules: bool = True
    rarity_k: int = 10
    exploit_topk_factor: float = 5.0
    explore_topk_factor: float = 10.0
    random_state: int = 0

def _diverse_subset(X_candidates: np.ndarray, n_samples: int, metric: str, embed_max_dim: int, rs: int) -> np.ndarray:
    if len(X_candidates) <= n_samples: return np.arange(len(X_candidates))
    n_components = min(embed_max_dim, len(X_candidates), X_candidates.shape[1])
    X_emb = PCA(n_components=n_components, random_state=rs).fit_transform(X_candidates)
    start = np.argmax(np.linalg.norm(X_emb, axis=1))
    sel = [start]
    mind = pairwise_distances(X_emb, X_emb[[start]], metric=metric).ravel()
    for _ in range(1, n_samples):
        nxt = int(np.argmax(mind)); sel.append(nxt)
        mind = np.minimum(mind, pairwise_distances(X_emb, X_emb[[nxt]], metric=metric).ravel())
    return np.array(sel, dtype=int)

def _base_score(probs: np.ndarray, y_labeled: np.ndarray, alpha: float, gamma: float, prior_default: float) -> np.ndarray:
    unc = binary_entropy(probs)
    obs_prior = y_labeled.mean() if len(y_labeled) else 0.0
    prior = max(obs_prior, prior_default or 0.0, 1e-6)
    focal = (probs / (prior + probs + 1e-12)) ** gamma
    return (unc ** alpha) * focal

def _rarity_boost(s0: np.ndarray, Xs: np.ndarray, k: int, rarity_k: int, lam: float) -> np.ndarray:
    n = len(Xs); pre_k = min(max(4*k, k), n)
    idx = np.argsort(-s0)[:pre_k]
    if len(idx) < 2: return s0
    nbrs = NearestNeighbors(n_neighbors=min(rarity_k, max(1, len(idx)-1))).fit(Xs[idx])
    d, _ = nbrs.kneighbors(Xs[idx])
    rarity = 1.0 / (d.mean(axis=1) + 1e-9)
    rarity = (rarity - rarity.min()) / (rarity.max() - rarity.min() + 1e-9)
    s = s0.copy(); s[idx] = s0[idx] * (1.0 + lam * rarity)
    return s

class FRaUDSampler(BaseQueryStrategy):
    name = "fraud"
    def __init__(self, cfg: FRaUDConfig): self.cfg = cfg
    def select(self, model, X_pool, batch_size, **kw):
        Xs = StandardScaler().fit_transform(X_pool)
        y_l = kw.get("y_labeled", np.array([])); round_idx = kw.get("round_idx", 1)
        probs = model.predict_proba(X_pool)[:, 1]
        quota = 0.15 if (self.cfg.use_schedules and round_idx <= 3) else (0.25 if self.cfg.use_schedules else self.cfg.quota)
        s0 = _base_score(probs, y_l, self.cfg.alpha, self.cfg.gamma, self.cfg.prior)
        n_exploit = int(quota * batch_size); n_explore = batch_size - n_exploit
        ex_pool = int(max(n_exploit, self.cfg.exploit_topk_factor * max(1, n_exploit)))
        ex_cand = np.argsort(-probs)[:min(ex_pool, len(probs))]
        final_ex = ex_cand[_diverse_subset(Xs[ex_cand], min(n_exploit, len(ex_cand)), self.cfg.diversity_metric, self.cfg.embed_max_dim, self.cfg.random_state)] if ex_cand.size else np.array([], dtype=int)
        mask = np.ones(len(X_pool), bool); mask[final_ex] = False; rem = np.where(mask)[0]
        exr_pool = int(max(n_explore, self.cfg.explore_topk_factor * max(1, n_explore)))
        cand_sorted = rem[np.argsort(-s0[rem])[:min(exr_pool, rem.size)]] if rem.size else np.array([], dtype=int)
        final_exr = cand_sorted[_diverse_subset(Xs[cand_sorted], min(n_explore, len(cand_sorted)), self.cfg.diversity_metric, self.cfg.embed_max_dim, self.cfg.random_state)] if cand_sorted.size else np.array([], dtype=int)
        return np.concatenate([final_ex, final_exr])

class FRaUDPlusSampler(BaseQueryStrategy):
    name = "fraudpp"
    def __init__(self, cfg: FRaUDConfig): self.cfg = cfg
    def select(self, model, X_pool, batch_size, **kw):
        Xs = StandardScaler().fit_transform(X_pool)
        y_l = kw.get("y_labeled", np.array([])); round_idx = kw.get("round_idx", 1)
        probs = model.predict_proba(X_pool)[:, 1]
        quota = 0.15 if (self.cfg.use_schedules and round_idx <= 3) else (0.25 if self.cfg.use_schedules else self.cfg.quota)
        lam = 0.6 if (self.cfg.use_schedules and round_idx <= 3) else (self.cfg.lambda_anomaly)
        s0 = _base_score(probs, y_l, self.cfg.alpha, self.cfg.gamma, self.cfg.prior)
        s_final = _rarity_boost(s0, Xs, batch_size, self.cfg.rarity_k, lam)
        n_exploit = int(quota * batch_size); n_explore = batch_size - n_exploit
        ex_pool = int(max(n_exploit, self.cfg.exploit_topk_factor * max(1, n_exploit)))
        ex_cand = np.argsort(-probs)[:min(ex_pool, len(probs))]
        final_ex = ex_cand[_diverse_subset(Xs[ex_cand], min(n_exploit, len(ex_cand)), self.cfg.diversity_metric, self.cfg.embed_max_dim, self.cfg.random_state)] if ex_cand.size else np.array([], dtype=int)
        mask = np.ones(len(X_pool), bool); mask[final_ex] = False; rem = np.where(mask)[0]
        exr_pool = int(max(n_explore, self.cfg.explore_topk_factor * max(1, n_explore)))
        cand_sorted = rem[np.argsort(-s_final[rem])[:min(exr_pool, rem.size)]] if rem.size else np.array([], dtype=int)
        final_exr = cand_sorted[_diverse_subset(Xs[cand_sorted], min(n_explore, len(cand_sorted)), self.cfg.diversity_metric, self.cfg.embed_max_dim, self.cfg.random_state)] if cand_sorted.size else np.array([], dtype=int)
        return np.concatenate([final_ex, final_exr])
