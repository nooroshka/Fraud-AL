# learner.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

# Project imports
from utils import (
    set_seed, load_dataset_auto,
    recall_at_fpr, precision_at_fpr, ece_score
)
from config import load_config, ExperimentConfig

# Strategies (FRaUD/FRaUD++ are optional; handled gracefully if missing)
from strategies import (
    BaseQueryStrategy,
    RandomSampler, EntropySampler, MarginSampler, CostBalancedEntropySampler, QBCSampler
)
try:
    from strategies import FRaUDConfig, FRaUDSampler, FRaUDPlusSampler  # type: ignore
    _HAS_FRAUD = True
except Exception:
    _HAS_FRAUD = False


# -----------------------------
# Config
# -----------------------------
@dataclass
class ALConfig:
    batch_size: int = 200
    seed_size: int = 1400
    budget: int = 5000
    test_size: float = 0.2
    model_kind: str = "lgbm"   # "lgbm" or "logreg"
    random_state: int = 0


# -----------------------------
# Wrapper to enforce DF inputs
# -----------------------------
class _DFCompatModel:
    """
    Wraps an sklearn/LGBM classifier so that predict_proba always receives
    a pandas.DataFrame with the same columns used at fit time. This removes
    the sklearn warning: "X does not have valid feature names, but ... was fitted
    with feature names".
    """
    def __init__(self, base_model):
        self._m = base_model
        self._cols: List[str] = []

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            # if caller gave numpy, create generic column names
            self._cols = [f"f{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self._cols)
        else:
            # DataFrame: remember exact columns
            self._cols = list(X.columns)
        out = self._m.fit(X, y)
        # mirror attributes like classes_ for downstream code
        for attr in ("classes_",):
            if hasattr(self._m, attr):
                setattr(self, attr, getattr(self._m, attr))
        return out

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self._cols)
        else:
            # ensure column order
            X = X[self._cols]
        return self._m.predict_proba(X)

    # pass-through for anything else samplers might call
    def __getattr__(self, name: str):
        return getattr(self._m, name)


# -----------------------------
# Active Learner
# -----------------------------
class ActiveLearner:
    def __init__(self, X, y, cfg: ALConfig, sampler: BaseQueryStrategy, outdir: Path, run_name: str):
        self.cfg, self.sampler, self.outdir, self.run_name = cfg, sampler, outdir, run_name
        set_seed(cfg.random_state)

        # Ensure pandas everywhere (keeps feature names consistent)
        if isinstance(X, np.ndarray):
            cols = [f"f{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=cols)
        if isinstance(y, (np.ndarray, list)):
            y = pd.Series(y)

        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(
            X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
        )

        # Seed labeled set
        n_train = self.X_train_full.shape[0]
        all_idx = np.arange(n_train)
        seed_n = min(cfg.seed_size, n_train)
        seed_idx = np.random.choice(all_idx, size=seed_n, replace=False)

        self.labeled_mask = np.zeros(n_train, dtype=bool)
        self.labeled_mask[seed_idx] = True

        # Metrics history
        self.history: Dict[str, List[float]] = {
            "labels_used": [],
            "frauds_found": [],
            "auroc": [],
            "auprc": [],
            "recall@0.1%fpr": [],
            "precision@0.1%fpr": [],
            "threshold@0.1%fpr": [],
            "ece": [],
            "brier": [],
        }

    def _build_model(self, y_labeled: pd.Series):
        # Dynamic choice: allow LR bootstrap when positives are extremely scarce
        pos = int(y_labeled.sum())
        neg = int(len(y_labeled) - pos)

        if self.cfg.model_kind == "lgbm" and _HAS_LGBM and pos > 0:
            spw = float(neg / max(pos, 1)) if (pos + neg) > 0 else 1.0
            base = LGBMClassifier(
                objective="binary",
                random_state=self.cfg.random_state,
                n_estimators=300,
                num_leaves=64,
                min_child_samples=5,       # help early tiny-positive rounds
                feature_pre_filter=False,  # don't drop all features on tiny seeds
                scale_pos_weight=spw,      # rebalance
                verbosity=-1,              # suppress split spam; model still trains
                n_jobs=-1,
            )
            return _DFCompatModel(base)

        # Fallback: logistic regression
        base = LogisticRegression(
            class_weight="balanced",
            max_iter=400,
            random_state=self.cfg.random_state
        )
        return _DFCompatModel(base)

    def _eval(self, model) -> None:
        scores = model.predict_proba(self.X_test)[:, 1]
        auroc = float(roc_auc_score(self.y_test, scores))
        auprc = float(average_precision_score(self.y_test, scores))
        rec = float(recall_at_fpr(self.y_test, scores, 0.001))
        prec, th = precision_at_fpr(self.y_test, scores, 0.001)
        ece_val = float(ece_score(self.y_test, scores))
        brier = float(brier_score_loss(self.y_test, scores))

        self.history["labels_used"].append(int(self.labeled_mask.sum()))
        self.history["frauds_found"].append(int(self.y_train_full[self.labeled_mask].sum()))
        self.history["auroc"].append(auroc)
        self.history["auprc"].append(auprc)
        self.history["recall@0.1%fpr"].append(rec)
        self.history["precision@0.1%fpr"].append(float(prec))
        self.history["threshold@0.1%fpr"].append(float(th))
        self.history["ece"].append(ece_val)
        self.history["brier"].append(brier)

    def run(self) -> None:
        out = Path(self.outdir)
        out.mkdir(parents=True, exist_ok=True)

        round_idx = 0
        model = None

        while int(self.labeled_mask.sum()) < self.cfg.budget:
            round_idx += 1

            X_l = self.X_train_full[self.labeled_mask]
            y_l = self.y_train_full[self.labeled_mask]
            X_p = self.X_train_full[~self.labeled_mask]
            if X_p.shape[0] == 0:
                break

            # Train and evaluate
            model = self._build_model(y_l).fit(X_l, y_l)
            self._eval(model)

            # Select next batch
            rel = self.sampler.select(
                model, X_p, self.cfg.batch_size,
                X_labeled=X_l, y_labeled=y_l, round_idx=round_idx
            )
            pool_idx = np.where(~self.labeled_mask)[0]
            chosen = pool_idx[np.asarray(rel, dtype=int)]
            self.labeled_mask[chosen] = True

            if int(self.labeled_mask.sum()) >= self.cfg.budget:
                break

        if model is None:
            # edge-case: seed_size >= budget
            X_l = self.X_train_full[self.labeled_mask]
            y_l = self.y_train_full[self.labeled_mask]
            model = self._build_model(y_l).fit(X_l, y_l)
            self._eval(model)

        pd.DataFrame(self.history).to_csv(out / f"{self.run_name}.metrics.csv", index=False)


# -----------------------------
# Runner helpers
# -----------------------------
def _make_sampler(name: str, seed: int) -> BaseQueryStrategy:
    name = name.lower()
    base_map: Dict[str, BaseQueryStrategy] = {
        "random": RandomSampler(seed),
        "entropy": EntropySampler(),
        "margin": MarginSampler(),
        "cost_balanced": CostBalancedEntropySampler(),
        "qbc": QBCSampler(seed),
    }
    if name in base_map:
        return base_map[name]
    if _HAS_FRAUD:
        if name in {"fraud", "fraudpp"}:
            fcfg = FRaUDConfig(random_state=seed)
            return FRaUDSampler(fcfg) if name == "fraud" else FRaUDPlusSampler(fcfg)
    raise ValueError(f"Unknown strategy '{name}'")


def run_single_experiment(
    data_path: str, strategy_name: str, outdir: str,
    seed: int, batch_size: int, seed_size: int, budget: int, model_kind: str
) -> None:
    X, y, _, ds_name = load_dataset_auto(data_path)

    cfg = ALConfig(
        batch_size=batch_size, seed_size=seed_size, budget=budget,
        model_kind=model_kind, random_state=seed
    )
    sampler = _make_sampler(strategy_name, seed)

    run_name = f"{strategy_name}_seed{seed}_{ds_name}"
    ActiveLearner(X, y, cfg, sampler, Path(outdir), run_name).run()


def run_experiment_from_cfg(cfg: ExperimentConfig) -> None:
    for strategy_name in cfg.strategies:
        for seed in cfg.seeds:
            run_single_experiment(
                data_path=cfg.data_path,
                strategy_name=strategy_name,
                outdir=cfg.outdir,
                seed=seed,
                batch_size=cfg.batch_size,
                seed_size=cfg.seed_size,
                budget=cfg.budget,
                model_kind=cfg.model_kind,
            )


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml",
                    help="Path to YAML config. If not found, falls back to single run flags.")
    ap.add_argument("--data_path", type=str, default=None)
    ap.add_argument("--strategy", type=str, default="qbc")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--seed_size", type=int, default=1400)
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--model_kind", type=str, default="lgbm", choices=["lgbm", "logreg"])
    args = ap.parse_args()

    # Try config first; if missing or fails, use flags for a single run
    try:
        cfg = load_config(args.config)
        run_experiment_from_cfg(cfg)
    except Exception:
        if args.data_path is None:
            raise FileNotFoundError(
                "Config load failed and --data_path not provided. "
                "Provide a valid --config or pass flags for a single run."
            )
        run_single_experiment(
            data_path=args.data_path,
            strategy_name=args.strategy,
            outdir=args.outdir,
            seed=args.seed,
            batch_size=args.batch_size,
            seed_size=args.seed_size,
            budget=args.budget,
            model_kind=args.model_kind,
        )


if __name__ == "__main__":
    main()
