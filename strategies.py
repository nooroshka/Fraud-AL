# strategies.py
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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
        committee = [
            LogisticRegression(class_weight="balanced", max_iter=200, random_state=self.seed),
            RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=self.seed),
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, early_stopping=True, random_state=self.seed),
        ]
        preds = []
        for clf in committee:
            clf.fit(X_l, y_l)
            preds.append(clf.predict_proba(X_pool)[:, 1])
        preds = np.vstack(preds)
        return np.argsort(-np.std(preds, axis=0))[:batch_size]
