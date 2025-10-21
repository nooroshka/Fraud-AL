# al_fraud_experiment.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


# utils
def set_seed(seed: int) -> None:
    np.random.seed(seed)


def binary_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# strategy api
class BaseQueryStrategy:
    name: str = "base"

    def select(self, model, X_pool: np.ndarray, batch_size: int, **kwargs) -> np.ndarray:
        raise NotImplementedError


# simple baselines
class RandomSampler(BaseQueryStrategy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select(self, model, X_pool, batch_size, **kwargs):
        n = X_pool.shape[0]
        k = min(batch_size, n)
        return self.rng.choice(n, size=k, replace=False)


class EntropySampler(BaseQueryStrategy):
    name = "entropy"

    def select(self, model, X_pool, batch_size, **kwargs):
        probs = model.predict_proba(X_pool)[:, 1]
        scores = binary_entropy(probs)
        return np.argsort(-scores)[:batch_size]


# TODO add FRaUD and FRaUD++ classes here
# class FRaUDSampler(BaseQueryStrategy): ...
# class FRaUDPlusSampler(BaseQueryStrategy): ...


# data loading
def prepare_creditcard(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    if "Class" not in df.columns:
        raise ValueError("creditcard dataset must have column named Class")
    y = df["Class"].astype(int).values
    X_df = df.drop(columns=[c for c in ["Time", "Class"] if c in df.columns]).copy()
    if "Amount" in X_df.columns:
        X_df["Amount"] = StandardScaler().fit_transform(X_df[["Amount"]]).ravel()
    return X_df.values, y, list(X_df.columns), "creditcard"


def prepare_paysim(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    required = {
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    }
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"paysim dataset missing columns {missing}")

    y = df["isFraud"].astype(int).values
    X_df = df[
        [
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFlaggedFraud",
        ]
    ].copy()

    X_df["errorBalanceOrig"] = X_df["newbalanceOrig"] + X_df["amount"] - X_df["oldbalanceOrg"]
    X_df["errorBalanceDest"] = X_df["oldbalanceDest"] + X_df["amount"] - X_df["newbalanceDest"]
    X_df["deltaOrig"] = X_df["oldbalanceOrg"] - X_df["newbalanceOrig"]
    X_df["deltaDest"] = X_df["newbalanceDest"] - X_df["oldbalanceDest"]

    X_df = pd.concat([X_df.drop(columns=["type"]), pd.get_dummies(X_df["type"], prefix="type")], axis=1)

    to_scale = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "errorBalanceOrig",
        "errorBalanceDest",
        "deltaOrig",
        "deltaDest",
        "step",
    ]
    X_df[to_scale] = StandardScaler().fit_transform(X_df[to_scale])

    return X_df.values, y, list(X_df.columns), "paysim"


def load_dataset_auto(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    df = pd.read_csv(path)
    if "Class" in df.columns:
        return prepare_creditcard(df)
    if "isFraud" in df.columns:
        return prepare_paysim(df)
    raise ValueError("could not detect dataset kind")


# configs
@dataclass
class ALConfig:
    batch_size: int = 200
    seed_size: int = 1400
    budget: int = 5000
    test_size: float = 0.2
    random_state: int = 0
    model_kind: str = "lr"  # TODO add lgbm option


# learner
class ActiveLearner:
    def __init__(self, X, y, cfg: ALConfig, sampler: BaseQueryStrategy, outdir: Path, run_name: str):
        self.cfg = cfg
        self.sampler = sampler
        self.outdir = outdir
        self.run_name = run_name

        set_seed(cfg.random_state)
        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(
            X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
        )

        idx = np.arange(self.X_train_full.shape[0])
        rng = np.random.default_rng(cfg.random_state)
        seed_idx = rng.choice(idx, size=min(cfg.seed_size, idx.size), replace=False)

        self.labeled_mask = np.zeros(idx.size, dtype=bool)
        self.labeled_mask[seed_idx] = True

    def _build_model(self):
        # simple baseline model
        return LogisticRegression(class_weight="balanced", max_iter=400, random_state=self.cfg.random_state)

    def _evaluate(self, model) -> tuple[float, float]:
        scores = model.predict_proba(self.X_test)[:, 1]
        auroc = roc_auc_score(self.y_test, scores)
        auprc = average_precision_score(self.y_test, scores)
        return float(auroc), float(auprc)

    def run(self):
        out = []
        while self.labeled_mask.sum() < self.cfg.budget and np.any(~self.labeled_mask):
            X_l = self.X_train_full[self.labeled_mask]
            y_l = self.y_train_full[self.labeled_mask]
            X_p = self.X_train_full[~self.labeled_mask]

            model = self._build_model().fit(X_l, y_l)
            auroc, auprc = self._evaluate(model)

            pool_idx = np.where(~self.labeled_mask)[0]
            rel = self.sampler.select(model, X_p, self.cfg.batch_size, X_labeled=X_l, y_labeled=y_l)
            abs_idx = pool_idx[rel]
            self.labeled_mask[abs_idx] = True

            out.append({"labels_used": int(self.labeled_mask.sum()), "auroc": auroc, "auprc": auprc})
            print(f"round {len(out)} labels {out[-1]['labels_used']} auroc {auroc:.4f} auprc {auprc:.4f}")

        df = pd.DataFrame(out)
        self.outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.outdir / f"{self.run_name}.metrics.csv", index=False)
        return df


# runner
def run_experiment(
    data_path: str,
    strategy_name: str = "entropy",
    outdir: str = "results",
    seed: int = 0,
    batch_size: int = 200,
    seed_size: int = 1400,
    budget: int = 5000,
):
    X, y, _, ds_name = load_dataset_auto(data_path)

    cfg = ALConfig(
        batch_size=batch_size,
        seed_size=seed_size,
        budget=budget,
        random_state=seed,
    )

    sampler_map = {
        "random": RandomSampler(seed),
        "entropy": EntropySampler(),
        # "fraud": FRaUDSampler(...),           # TODO
        # "fraudpp": FRaUDPlusSampler(...),     # TODO
    }
    if strategy_name not in sampler_map:
        raise ValueError(f"unknown strategy {strategy_name}")
    sampler = sampler_map[strategy_name]

    run_name = f"{strategy_name}_seed{seed}_{ds_name}"
    learner = ActiveLearner(X, y, cfg, sampler, Path(outdir), run_name)
    learner.run()


def main():
    p = argparse.ArgumentParser(description="active learning fraud skeleton")
    p.add_argument("--data", required=True, help="path to csv file")
    p.add_argument("--strategy", default="entropy", choices=["random", "entropy"])
    p.add_argument("--outdir", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--seed-size", type=int, default=1400)
    p.add_argument("--budget", type=int, default=5000)
    args = p.parse_args()

    run_experiment(
        data_path=args.data,
        strategy_name=args.strategy,
        outdir=args.outdir,
        seed=args.seed,
        batch_size=args.batch_size,
        seed_size=args.seed_size,
        budget=args.budget,
    )


if __name__ == "__main__":
    main()
