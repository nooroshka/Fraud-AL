"""
 Active learning training loop and experiment runner.

This module wires together the dataset loader, query strategy (what to label next),
and the model training loop. It tracks metrics round-by-round, saves quick plots,
and writes a CSV/JSON summary you can compare across runs later.

Nothing fancy under the hood—just clean plumbing so you can focus on the strategy.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# optional LightGBM
try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

from config import load_config, ExperimentConfig
from utils import (
    set_seed, load_dataset_auto, save_curves, save_metrics_history,
    ece_score, precision_at_k, lift_at_k, precision_at_fpr, recall_at_fpr,
    expected_profit,
    # new visual helpers
    plot_reliability_curve, save_operating_point_visuals
)
from strategies import (
    BaseQueryStrategy, RandomSampler, EntropySampler, MarginSampler,
    CostBalancedEntropySampler, QBCSampler,
    FRaUDSampler, FRaUDPlusSampler, HybridFraudPPQBCSampler, GraphFraudHybridSampler
)


# active learner

class ActiveLearner:
    """A small active-learning loop that alternates training and querying.

    State we maintain:
    - a boolean mask over the training set indicating which rows are labeled so far;
    - the query strategy instance;
    - a per-round metrics history (appended after each round).

    The loop is intentionally simple and predictable: fit -> evaluate -> ask for more labels.
    """
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 feature_names: List[str],
                 cfg: ExperimentConfig,
                 sampler: BaseQueryStrategy,
                 outdir: Path,
                 run_name: str):
        self.cfg = cfg
        self.sampler = sampler
        self.outdir = outdir
        self.run_name = run_name
        set_seed(cfg.seed)

        self.feature_names = feature_names

        # A simple holdout split; the test set is never labeled.
        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(
            X, y, test_size=cfg.al.test_size, stratify=y, random_state=cfg.al.random_state
        )

        # Create a separate validation set for early stopping (if enabled)
        # This prevents data leakage from using the test set for model tuning
        self.X_val, self.y_val = None, None
        if cfg.al.use_lgbm_early_stopping and _HAS_LGBM:
            # Take a small validation set from train_full (10% of training data or max 5000 samples)
            val_size = min(5000, int(0.10 * len(self.X_train_full)))
            if val_size >= 100:  # Only create validation set if we have enough data
                val_indices = np.random.choice(
                    len(self.X_train_full), size=val_size, replace=False
                )
                self.X_val = self.X_train_full[val_indices]
                self.y_val = self.y_train_full[val_indices]
                # Remove validation samples from train_full to avoid overlap
                train_mask = np.ones(len(self.X_train_full), dtype=bool)
                train_mask[val_indices] = False
                self.X_train_full = self.X_train_full[train_mask]
                self.y_train_full = self.y_train_full[train_mask]

        # Construct a seed set with a reasonable number of positives for a stable start.
        idx_all = np.arange(self.X_train_full.shape[0])
        fraud_idx = idx_all[self.y_train_full == 1]
        legit_idx = idx_all[self.y_train_full == 0]
        seed_size = int(self.cfg.al.seed_size)
        # Prefer explicit config if provided; otherwise compute a conservative heuristic.
        # Robustly parse seed_pos_frac which may be missing or None in YAML.
        seed_pos_frac_raw = getattr(self.cfg.al, "seed_pos_frac", None)
        try:
            seed_pos_frac = float(seed_pos_frac_raw)
        except (TypeError, ValueError):
            seed_pos_frac = np.nan
        if not np.isfinite(seed_pos_frac):
            base_prev = float(max(1e-6, self.y_train_full.mean()))
            seed_pos_frac = float(min(0.10, max(0.02, 10.0 * base_prev)))
        n_seed_frauds = min(len(fraud_idx), max(1, int(round(seed_size * seed_pos_frac))))
        n_seed_legit = max(0, seed_size - n_seed_frauds)
        seed_idx = np.concatenate([
            np.random.choice(fraud_idx, size=n_seed_frauds, replace=False) if n_seed_frauds > 0 else np.array([], dtype=int),
            np.random.choice(legit_idx, size=n_seed_legit, replace=False) if n_seed_legit > 0 else np.array([], dtype=int)
        ])
        self.labeled_mask = np.zeros(self.X_train_full.shape[0], dtype=bool)
        self.labeled_mask[seed_idx] = True

        # History of per-round metrics; initialize mandatory keys and dynamic containers
        base_keys = [
            "labels_used", "frauds_found", "auroc", "auprc", "recall@0.1%fpr",
            "brier", "ece", "precision@0.1%fpr", "threshold@0.1%fpr",
            "profit@0.1%fpr", "tp@0.1%fpr", "fp@0.1%fpr", "fn@0.1%fpr", "f1@0.1%fpr",
            "eval_time_s", "target_fpr"
        ]
        # legacy fixed capacity keys kept for backwards compatibility (will also add dynamic ones)
        cap_fixed = ["precision@100", "precision@500", "lift@500", "tp@100", "recall@100", "tp@500", "recall@500"]
        dyn_caps = []
        for K in getattr(self.cfg.al, "capacity_checkpoints", [100, 500]):
            dyn_caps += [f"precision@{K}", f"recall@{K}", f"lift@{K}", f"tp@{K}"]
        keys = list(dict.fromkeys(base_keys + cap_fixed + dyn_caps))
        self.history: Dict[str, list] = {k: [] for k in keys}

    # model factory

    def _base_lgbm(self):
        """Return a LightGBM classifier.

        - If cfg.al.lgbm_profile == "original", use a lightweight default.
        - Otherwise, use a slightly stronger profile that tends to work well out-of-the-box.
        """
        if not _HAS_LGBM:
            raise RuntimeError("LightGBM is not available. Switch model_kind to 'lr' in config.")
        if self.cfg.al.lgbm_profile == "original":
            return LGBMClassifier(random_state=self.cfg.al.random_state, verbosity=-1)
        return LGBMClassifier(
            random_state=self.cfg.al.random_state,
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, subsample=0.9, colsample_bytree=0.9, verbosity=-1
        )

    def _build_model(self):
        """Build the base model (LightGBM or a simple LR pipeline) with optional calibration."""
        mk = str(self.cfg.al.model_kind).lower()
        if mk == "lgbm":
            base = self._base_lgbm()
        else:
            base = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced",
                    max_iter=400,
                    random_state=self.cfg.al.random_state
                ))
            ])
        if self.cfg.fraud.use_calibration:
            return CalibratedClassifierCV(base, method="sigmoid", cv=3)
        return base

    # ensure DataFrame with feature names

    def _df(self, X):
        """Wrap a NumPy array in a DataFrame so model feature names line up nicely."""
        return pd.DataFrame(X, columns=self.feature_names)

    # evaluation

    def _evaluate_and_log(self, model):
        """Evaluate on the held-out test set and append metrics to history.

        Note: legacy keys like "recall@0.1%fpr" reflect the original 0.1% FPR target used in
        early experiments. We now compute at the configured target_fpr but keep the key names
        so plotting scripts remain compatible. Additionally, capacity checkpoints are dynamic.
        """
        import time
        t0 = time.time()
        scores = model.predict_proba(self._df(self.X_test))[:, 1]
        from sklearn.metrics import brier_score_loss

        # Use configured FPR target (defaults to 0.001 if missing)
        t_fpr = float(getattr(self.cfg.fraud, "target_fpr", 0.001))
        # Costs from config
        cost_tp = float(getattr(self.cfg.fraud, "cost_tp", 100.0))
        cost_fp = float(getattr(self.cfg.fraud, "cost_fp", 1.0))

        auroc = roc_auc_score(self.y_test, scores)
        auprc = average_precision_score(self.y_test, scores)
        rec_fpr = recall_at_fpr(self.y_test, scores, target_fpr=t_fpr)
        brier = brier_score_loss(self.y_test, scores)
        ece = ece_score(self.y_test, scores)
        prec_fpr, th_fpr = precision_at_fpr(self.y_test, scores, target_fpr=t_fpr)

        # Capacity metrics (dynamic K list)
        pos_total = int(np.count_nonzero(self.y_test == 1))
        capKs = list(dict.fromkeys(getattr(self.cfg.al, "capacity_checkpoints", [100, 500]) + [100, 500]))
        cap_metrics: Dict[str, float | int] = {}
        for K in capKs:
            p_at_k = precision_at_k(self.y_test, scores, int(K))
            tp_at_k = int(round(p_at_k * int(K)))
            rec_at_k = float(tp_at_k / pos_total) if pos_total > 0 else 0.0
            l_at_k = lift_at_k(self.y_test, scores, int(K))
            cap_metrics[f"precision@{K}"] = float(p_at_k)
            cap_metrics[f"tp@{K}"] = int(tp_at_k)
            cap_metrics[f"recall@{K}"] = float(rec_at_k)
            cap_metrics[f"lift@{K}"] = float(l_at_k)

        # Profit proxy and confusion counts at the operating point
        profit, tp, fp, fn = expected_profit(self.y_test, scores, th_fpr, cost_fn=cost_tp, cost_fp=cost_fp)
        den = 2 * tp + fp + fn
        f1_fpr = (2.0 * tp) / den if den > 0 else 0.0

        eval_time = time.time() - t0

        # Base metrics
        metrics = {
            "labels_used": int(self.labeled_mask.sum()),
            "frauds_found": int(self.y_train_full[self.labeled_mask].sum()),
            "auroc": float(auroc),
            "auprc": float(auprc),
            # Legacy keys (values at configured target_fpr)
            "recall@0.1%fpr": float(rec_fpr),
            "brier": float(brier),
            "ece": float(ece),
            "precision@0.1%fpr": float(prec_fpr),
            "threshold@0.1%fpr": float(th_fpr),
            "profit@0.1%fpr": float(profit),
            "tp@0.1%fpr": int(tp),
            "fp@0.1%fpr": int(fp),
            "fn@0.1%fpr": int(fn),
            "f1@0.1%fpr": float(f1_fpr),
            "eval_time_s": float(eval_time),
            "target_fpr": float(t_fpr),
        }
        metrics.update(cap_metrics)

        for k, v in metrics.items():
            # Ensure key exists
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

        # Console summary: quick pulse-check each round
        try:
            fpr_pct = 100.0 * float(metrics.get("target_fpr", float(getattr(self.cfg.fraud, "target_fpr", 0.001))))
        except Exception:
            fpr_pct = 0.1
        # Prefer a common K (use first capacity)
        capK_disp = capKs[0] if capKs else 100
        recK = metrics.get(f"recall@{capK_disp}", 0.0)
        print(
            f"Round {len(self.history['labels_used'])} | Labels: {metrics['labels_used']} "
            f"| Frauds: {metrics['frauds_found']} | AUROC: {metrics['auroc']:.4f} "
            f"| AUPRC: {metrics['auprc']:.4f} | Rec@FPR({fpr_pct:.3f}%): {metrics['recall@0.1%fpr']:.3f} "
            f"| P@FPR: {metrics['precision@0.1%fpr']:.3f} | F1@FPR: {metrics['f1@0.1%fpr']:.3f} "
            f"| Rec@{capK_disp}: {float(recK):.3f}"
        )

    # main loop

    def run(self):
        """Run AL rounds until the labeling budget is exhausted or a target is met."""
        print(f"\nRunning: {self.run_name}")
        round_idx = 0
        model = None

        while self.labeled_mask.sum() < self.cfg.al.budget:
            round_idx += 1
            if not np.any(~self.labeled_mask):
                break

            X_l = self.X_train_full[self.labeled_mask]
            y_l = self.y_train_full[self.labeled_mask]
            X_p = self.X_train_full[~self.labeled_mask]

            model = self._build_model()

            # If using LightGBM without calibration, set scale_pos_weight each round
            if (_HAS_LGBM and self.cfg.al.model_kind == "lgbm"
                    and not self.cfg.fraud.use_calibration):
                pos = int(max(1, int(y_l.sum())))
                neg = int(max(1, len(y_l) - pos))
                spw = float(neg / max(1.0, float(pos)))
                try:
                    model.set_params(scale_pos_weight=spw)
                except Exception:
                    pass

            # Optional: early stopping when using LightGBM without calibration
            fit_kwargs = {}
            if (_HAS_LGBM and self.cfg.al.model_kind == "lgbm"
                    and not self.cfg.fraud.use_calibration
                    and self.cfg.al.use_lgbm_early_stopping):
                # Use validation set if available, otherwise disable early stopping
                if self.X_val is not None and self.y_val is not None:
                    try:
                        fit_kwargs = dict(
                            eval_set=[(self._df(self.X_val), self.y_val)],
                            eval_metric="auc",
                            callbacks=[lgb.early_stopping(50, verbose=False)]
                        )
                    except Exception:
                        fit_kwargs = {}
                # If no validation set, early stopping is disabled (fit_kwargs remains empty)

            model.fit(self._df(X_l), y_l, **fit_kwargs)

            self._evaluate_and_log(model)

            # Early exit if target AUROC is reached
            if self.cfg.al.auroc_target is not None and self.history["auroc"][-1] >= self.cfg.al.auroc_target:
                print(f"Reached AUROC target {self.cfg.al.auroc_target:.3f}. Stopping.")
                break

            # Budget-aware batch size for the next query
            remaining = self.cfg.al.budget - int(self.labeled_mask.sum())
            if remaining <= 0 or (~self.labeled_mask).sum() == 0:
                break
            batch_n = int(min(self.cfg.al.batch_size, remaining, (~self.labeled_mask).sum()))

            # Ask the sampler which pool items to label next
            # Pass scheduling context for adaptive strategies
            current_auroc = self.history["auroc"][ -1] if self.history["auroc"] else 0.0
            rel = self.sampler.select(
                model, X_p, batch_n, y_labeled=y_l, round_idx=round_idx, X_labeled=X_l,
                budget_used=int(self.labeled_mask.sum()),
                total_budget=int(self.cfg.al.budget),
                current_auroc=float(current_auroc)
            )
            rel = np.asarray(rel)[:batch_n]
            pool_idx = np.where(~self.labeled_mask)[0]
            abs_idx = pool_idx[rel]
            self.labeled_mask[abs_idx] = True
            np.save(self.outdir / f"{self.run_name}.round{round_idx:02d}.picked.npy", abs_idx)

        # If the loop ended before we ever trained, fit once on whatever is labeled
        if model is None:
            model = self._build_model().fit(self._df(self.X_train_full[self.labeled_mask]),
                                            self.y_train_full[self.labeled_mask])

        # Log the final state if the last round didn't already do it
        current_labels = int(self.labeled_mask.sum())
        already_logged = (
            len(self.history["labels_used"]) > 0 and int(self.history["labels_used"][-1]) == current_labels
        )
        if not already_logged:
            self._evaluate_and_log(model)

        # Save PR/ROC curves and the per-round metrics table
        scores = model.predict_proba(self._df(self.X_test))[:, 1]
        save_curves(self.y_test, scores, self.outdir, self.run_name)
        save_metrics_history(self.history, self.outdir, self.run_name)

        # Reliability diagram and cost curve/confusion at operating point
        try:
            t_fpr = float(getattr(self.cfg.fraud, "target_fpr", 0.001))
            _, th_fpr = precision_at_fpr(self.y_test, scores, target_fpr=t_fpr)
            plot_reliability_curve(self.y_test, scores, self.outdir, f"{self.run_name}")
            save_operating_point_visuals(self.y_test, scores, self.outdir, f"{self.run_name}", th_fpr,
                                         cost_tp=float(getattr(self.cfg.fraud, 'cost_tp', 100.0)),
                                         cost_fp=float(getattr(self.cfg.fraud, 'cost_fp', 1.0)))
        except Exception:
            pass

        # Optional: dump detailed picks with features for qualitative inspection
        try:
            picks_records = []
            for f in sorted(self.outdir.glob(f"{self.run_name}.round*.picked.npy")):
                m = re.search(r"round(\d+)\.picked\.npy$", f.name)
                r = int(m.group(1)) if m else None
                arr = np.load(f)
                for idx in arr.tolist():
                    rec: Dict[str, Any] = {
                         "round": r,
                         "index": int(idx),
                         "label": int(self.y_train_full[idx]),
                     }
                    # Add feature values for this picked index
                    try:
                        xrow = self.X_train_full[idx]
                        for j, name in enumerate(self.feature_names):
                            rec[name] = float(xrow[j])
                    except Exception:
                        pass
                    picks_records.append(rec)
            if picks_records:
                dfp = pd.DataFrame(picks_records)
                dfp.to_csv(self.outdir / f"{self.run_name}.picks_detailed.csv", index=False)
        except Exception:
            pass


# runner

def build_sampler(strategy_name: str, seed: int, fraud_cfg) -> BaseQueryStrategy:
    """Factory for query strategies by name.

    Tip: try "fraudpp_hybrid" for a strong default, or "random" to sanity-check wiring.
    """
    m = {
        "random": lambda: RandomSampler(seed),
        "entropy": lambda: EntropySampler(),
        "margin": lambda: MarginSampler(),
        "cost_balanced": lambda: CostBalancedEntropySampler(fraud_cost=float(getattr(fraud_cfg, "cost_tp", 100.0))),
         "qbc": lambda: QBCSampler(seed),
        "fraud": lambda: FRaUDSampler(fraud_cfg),
        "fraudpp": lambda: FRaUDPlusSampler(fraud_cfg),
        "fraudpp_hybrid": lambda: HybridFraudPPQBCSampler(
            fraud_cfg, seed, w_qbc=float(getattr(fraud_cfg, "qbc_weight", 0.5))
        ),
        "graph_hybrid": lambda: GraphFraudHybridSampler(
            fraud_cfg,
            seed,
            w_qbc=float(getattr(fraud_cfg, "qbc_weight", 0.5)),
            k_neighbors=int(getattr(fraud_cfg, "graph_k_neighbors", 10)),
            hub_weight=float(getattr(fraud_cfg, "graph_hub_weight", 0.5)),
            bridge_weight=float(getattr(fraud_cfg, "graph_bridge_weight", 0.5)),
            adaptive_qbc=bool(getattr(fraud_cfg, "graph_adaptive_qbc", True))
        ),
    }
    if strategy_name not in m:
        raise ValueError(f"Unknown strategy '{strategy_name}'")
    return m[strategy_name]()

def run_experiment(cfg: ExperimentConfig):
    """Run a single experiment from the given config and save artifacts to disk."""
    X, y, feature_names, ds_name = load_dataset_auto(cfg.data_path)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sampler = build_sampler(cfg.strategy_name, cfg.seed, cfg.fraud)
    run_name = f"{cfg.strategy_name}_seed{cfg.seed}_{ds_name}"
    learner = ActiveLearner(X, y, feature_names, cfg, sampler, outdir, run_name)
    learner.run()

def main():
    p = argparse.ArgumentParser(description="Active Learning for Fraud")
    p.add_argument("--config", "-c", type=str, default="config.yaml",
                   help="Path to YAML or JSON config file")
    args = p.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg)

if __name__ == "__main__":
    main()
