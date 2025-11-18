"""
Report visualizations and model comparison helper.

This script scans a results folder for per-round metrics CSV files, aggregates
them into a single DataFrame, and produces report-ready plots and tables for
comparing active learning strategies.

You can generate learning curves, comparison tables at specific label budgets,
bar/scatter plots at a checkpoint, simple bootstrap-based confidence intervals,
capacity plots, and montages of existing ROC/PR images.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
from typing import Literal, cast
import json

# Optional: allow running experiments before plotting if training modules are available.
try:
    from config import load_config as _load_cfg, ExperimentConfig as _ExpCfg
    from learner import run_experiment as _run_experiment
    _CAN_RUN = True
except Exception:
    _CAN_RUN = False


FILENAME_RE = re.compile(r"^(?P<strategy>.+?)_seed(?P<seed>\d+)(?:_(?P<dataset>[A-Za-z0-9_-]+))?\.metrics\.csv$")


def parse_filename(path: Path) -> Optional[Tuple[str, int, Optional[str]]]:
    """Return (strategy, seed, dataset) parsed from a metrics file name.

    The filename is expected to look like
    ``{strategy}_seed{seed}_{dataset}.metrics.csv`` or
    ``{strategy}_seed{seed}.metrics.csv``. When the dataset is missing in the
    file name, the parent folder name is used as a best-effort guess.
    """
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    strategy = m.group("strategy")
    seed = int(m.group("seed"))
    dataset = m.group("dataset")
    if dataset is None:
        # If the dataset is not encoded in the filename, try to infer it from the
        # parent directory (common for known datasets like "creditcard").
        parent = path.parent.name
        if parent and parent.lower() in ("creditcard", "paysim", "ieee", "ieee-fraud-detection"):
            dataset = parent
    return strategy, seed, dataset


def discover_metrics(results_dir: Path) -> pd.DataFrame:
    """Load all ``*.metrics.csv`` files under ``results_dir`` into one DataFrame.

    Each row is augmented with ``strategy``, ``seed``, ``dataset``, and
    ``source_path`` columns as parsed from the file path. If ``labels_used`` is
    present, it is converted to integer for easier plotting.
    """
    paths = list(results_dir.rglob("*.metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No '*.metrics.csv' files found under {results_dir}")

    dfs: List[pd.DataFrame] = []
    for p in paths:
        meta = parse_filename(p)
        if meta is None:
            continue
        strategy, seed, dataset = meta
        try:
            df = pd.read_csv(p)
        except Exception:
            # Skip unreadable CSVs so that a single corrupt file does not
            # prevent using the rest of the results.
            continue
        df["strategy"] = strategy
        df["seed"] = seed
        df["dataset"] = dataset
        df["source_path"] = str(p)
        if "labels_used" in df.columns:
            df["labels_used"] = pd.to_numeric(df["labels_used"], errors="coerce")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No metrics files could be parsed. Check filename patterns.")

    full = pd.concat(dfs, ignore_index=True)
    # Rows without ``labels_used`` cannot be located on a learning-curve
    # x-axis, so they are dropped for plotting. They still remain available in
    # the raw combined CSV.
    if "labels_used" in full.columns:
        full = full.dropna(subset=["labels_used"]).copy()
        full["labels_used"] = full["labels_used"].astype(int)
    # Replace missing dataset names with a neutral placeholder so that
    # downstream filters always see a string.
    full["dataset"] = full["dataset"].fillna("unknown")
    return full


def ensure_outdir(base_out: Path, dataset: Optional[str]) -> Path:
    """Create and return the output directory.

    If a concrete dataset name is given, results are placed in a dataset
    subfolder under ``base_out``. Otherwise, ``base_out`` is used directly.
    """
    out = base_out if dataset in (None, "unknown") else base_out / dataset
    out.mkdir(parents=True, exist_ok=True)
    return out


# Small utility helpers

def _safe_metric_name(metric: str) -> str:
    """Return a filesystem-friendly version of a metric name for filenames."""
    return metric.replace('%', 'pct').replace('@', 'at_').replace('/', '_').replace(' ', '_')


def _nearest_rows_per_seed(
    df: pd.DataFrame,
    checkpoint: int,
    require_reach: bool = True,
    extra_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Pick one row per ``(strategy, seed)`` nearest to ``checkpoint``.

    The returned DataFrame contains at most one row per strategy/seed pair and
    is useful for building tables or plots at a specific label budget.
    """
    keep_cols = {"strategy", "seed", "labels_used"}
    if extra_columns:
        keep_cols.update([c for c in extra_columns if c in df.columns])

    recs: List[dict] = []
    for (strategy, seed), g in df.groupby(["strategy", "seed"], as_index=False):
        g = g.sort_values("labels_used")
        if require_reach and (g["labels_used"].max() < checkpoint):
            # Ignore runs that never reach the requested checkpoint when
            # ``require_reach`` is enabled.
            continue
        idx = (g["labels_used"] - checkpoint).abs().idxmin()
        row = g.loc[idx]
        rec = {c: row[c] for c in keep_cols if c in row}
        recs.append(rec)

    return pd.DataFrame(recs)


def plot_learning_curves(df: pd.DataFrame, outdir: Path, metrics: List[str], title_suffix: str = "") -> None:
    """Plot learning curves (mean ± SD over seeds) for each strategy.

    For every metric in ``metrics``, a separate line plot of the metric versus
    ``labels_used`` is saved in ``outdir``.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    # Use a consistent color per strategy across all metrics.
    strategies = sorted(df["strategy"].unique())
    palette = sns.color_palette("tab10", n_colors=max(10, len(strategies)))
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(strategies)}

    # Record how many seeds contributed to each strategy for legend labels.
    seeds_by_strategy = df.groupby("strategy")["seed"].nunique().to_dict()

    for metric in metrics:
        if metric not in df.columns:
            print(f"[WARN] Metric '{metric}' not found. Skipping plot.")
            continue
        plt.figure(figsize=(8, 5))
        ax = sns.lineplot(
            data=df,
            x="labels_used",
            y=metric,
            hue="strategy",
            errorbar="sd",
            palette=color_map,
        )
        ax.set_xlabel("Labels used")
        ax.set_ylabel(metric)
        title = f"{metric} vs. labels"
        if title_suffix:
            title += f" ({title_suffix})"
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            new_labels = []
            for lab in labels:
                if lab in seeds_by_strategy:
                    new_labels.append(f"{lab} (n={seeds_by_strategy[lab]})")
                else:
                    new_labels.append(lab)
            ax.legend(handles=handles, labels=new_labels, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        outfile = outdir / f"learning_{_safe_metric_name(metric)}.png"
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved {outfile}")


def summarize_at_checkpoints(
    df: pd.DataFrame,
    checkpoints: List[int],
    metrics: List[str],
) -> pd.DataFrame:
    """Summarize metrics at the rows nearest to each label checkpoint.

    For every ``checkpoint`` and every strategy, the row closest in
    ``labels_used`` is selected per seed and then aggregated (mean and
    standard deviation) across seeds.
    """
    cols = ["strategy", "seed", "labels_used"] + [m for m in metrics if m in df.columns]
    data = df[cols].copy()

    per_cp = []
    for cp in checkpoints:
        nearest = _nearest_rows_per_seed(data, checkpoint=cp, require_reach=False, extra_columns=metrics)
        if nearest.empty:
            continue
        nearest = nearest.rename(columns={"labels_used": "picked_labels"})
        nearest["checkpoint"] = cp
        per_cp.append(nearest)

    if not per_cp:
        return pd.DataFrame()

    sel = pd.concat(per_cp, ignore_index=True)
    agg_funcs = {m: ["mean", "std"] for m in metrics if m in sel.columns}
    summary = sel.groupby(["strategy", "checkpoint"]).agg(agg_funcs)
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()
    return summary


def barplot_at_checkpoint(df: pd.DataFrame, outdir: Path, metric: str, checkpoint: int, require_reach: bool = True) -> None:
    """Create a bar plot of mean ± SD across seeds at a given checkpoint."""
    nearest = _nearest_rows_per_seed(df, checkpoint=checkpoint, require_reach=require_reach, extra_columns=[metric])
    if nearest.empty or metric not in nearest.columns:
        print(f"[INFO] No usable values for '{metric}' at ~{checkpoint}; skipping bar plot")
        return

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=nearest, x="strategy", y=metric, errorbar="sd", estimator="mean", palette="tab10")
    ax.set_title(f"{metric} at ~{checkpoint} labels")
    ax.set_xlabel("Strategy")
    ax.set_ylabel(metric)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_horizontalalignment("right")
    plt.tight_layout()
    out = outdir / f"bar_{_safe_metric_name(metric)}_at_{checkpoint}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


def scatter_auroc_auprc_at_checkpoint(df: pd.DataFrame, outdir: Path, checkpoint: int, require_reach: bool = True) -> None:
    """Scatter plot of AUROC vs. AUPRC at a checkpoint (one point per seed)."""
    nearest = _nearest_rows_per_seed(df, checkpoint=checkpoint, require_reach=require_reach, extra_columns=["auroc", "auprc", "frauds_found"])
    if nearest.empty:
        print(f"[INFO] No runs reached checkpoint {checkpoint}; skipping scatter plot")
        return

    plt.figure(figsize=(7, 6))
    sizes = 50 + 2 * nearest["frauds_found"].fillna(nearest["frauds_found"].median())
    ax = plt.gca()
    for strat, g in nearest.groupby("strategy"):
        ax.scatter(g["auroc"], g["auprc"], s=sizes.loc[g.index], label=strat, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("AUPRC")
    ax.set_title(f"AUROC vs AUPRC at ~{checkpoint} labels")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = outdir / f"scatter_auroc_vs_auprc_at_{checkpoint}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


def delta_vs_baseline_plot(df: pd.DataFrame, outdir: Path, baseline_strategy: str, metric: str) -> None:
    """Plot the mean metric difference versus a baseline strategy over labels."""
    # Compute mean across seeds per strategy x labels
    gmean = df.groupby(["strategy", "labels_used"], as_index=False)[metric].mean()
    base = gmean[gmean["strategy"] == baseline_strategy].loc[:, ["labels_used", metric]].copy()
    base = base.rename(columns={metric: "base"})
    merged = gmean.merge(base, on="labels_used", how="inner")
    merged["delta"] = merged[metric] - merged["base"]
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(data=merged, x="labels_used", y="delta", hue="strategy", palette="tab10")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"Delta {metric} vs {baseline_strategy}")
    ax.set_xlabel("Labels used")
    ax.set_ylabel(f"Δ {metric}")
    plt.tight_layout()
    out = outdir / f"delta_{_safe_metric_name(metric)}_vs_{baseline_strategy}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


def bootstrap_compare(
    df: pd.DataFrame,
    strategies: List[str],
    checkpoint: int,
    metric: str,
    baseline_strategy: Optional[str] = None,
    n_boot: int = 5000,
    random_state: int = 42,
    require_reach: bool = True,
) -> pd.DataFrame:
    """Estimate per-strategy performance and deltas via simple bootstrap.

    For each strategy, values near ``checkpoint`` are collected per seed. When a
    baseline is given, bootstrap resampling is used to approximate confidence
    intervals and a two-sided p-value for the difference in means.
    """
    rng = np.random.default_rng(random_state)

    # Collect per-strategy per-seed values at the checkpoint (nearest row).
    per_seed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for strat in strategies:
        vals = []
        seeds = []
        for seed, g in df[df["strategy"] == strat].groupby("seed"):
            g = g.sort_values("labels_used")
            if require_reach and (g["labels_used"].max() < checkpoint):
                continue
            idx = (g["labels_used"] - checkpoint).abs().idxmin()
            row = g.loc[idx]
            val = row.get(metric, np.nan)
            if pd.notna(val):
                vals.append(val)
                seeds.append(seed)
        per_seed[strat] = (np.array(seeds, dtype=int), np.array(vals, dtype=float))

    def boot_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, Tuple[float, float], float]:
        """Bootstrap the mean difference between two samples.

        Returns the average delta, a 95% CI, and an approximate two-sided
        p-value under the null hypothesis of zero difference.
        """
        if len(x) == 0 or len(y) == 0:
            return np.nan, (np.nan, np.nan), np.nan
        n = min(len(x), len(y))
        diffs_list: List[float] = []
        for _ in range(n_boot):
            bx = x[rng.integers(0, len(x), size=n)]
            by = y[rng.integers(0, len(y), size=n)]
            bx_mean = float(bx.sum()) / float(len(bx)) if len(bx) else 0.0
            by_mean = float(by.sum()) / float(len(by)) if len(by) else 0.0
            diffs_list.append(bx_mean - by_mean)
        diffs = np.array(diffs_list, dtype=float)
        ci_q = np.quantile(diffs, [0.025, 0.975])
        ci = (float(ci_q[0]), float(ci_q[1]))
        p_left = float(np.mean(diffs <= 0))
        p_right = float(np.mean(diffs >= 0))
        p = float(2.0 * min(p_left, p_right))
        return float(np.mean(diffs)), ci, p

    rows = []
    for strat in strategies:
        seeds, vals = per_seed[strat]
        if len(vals) == 0:
            continue
        row = {"strategy": strat, "checkpoint": checkpoint, "metric": metric,
               "mean": float(np.mean(vals)) if len(vals) else np.nan,
               "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
               "n": int(len(vals))}
        if baseline_strategy and baseline_strategy in per_seed:
            _, bvals = per_seed[baseline_strategy]
            dmean, (low, high), p = boot_delta(vals, bvals)
            row.update({"delta_vs_baseline": dmean, "ci_low": low, "ci_high": high, "p_approx": p,
                        "baseline": baseline_strategy})
        rows.append(row)
    return pd.DataFrame(rows)


def save_tables(summary: pd.DataFrame, outdir: Path, basename: str) -> None:
    """Save a summary table to CSV and, if possible, LaTeX with rounded values."""
    csv_path = outdir / f"{basename}.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[OK] Saved {csv_path}")

    def round_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with float columns rounded for nicer LaTeX."""
        df2 = df.copy()
        for c in df2.columns:
            if df2[c].dtype.kind in "fc":
                df2[c] = df2[c].round(4)
        return df2

    try:
        tex_path = outdir / f"{basename}.tex"
        round_cols(summary).to_latex(tex_path, index=False, escape=True)
        print(f"[OK] Saved {tex_path}")
    except Exception as e:
        print(f"[WARN] Could not export LaTeX table: {e}")


def compose_image_montage(
    results_dir: Path,
    outdir: Path,
    dataset: Optional[str],
    strategies: List[str],
    seed: int,
    img_suffix: str,
    fig_title: str,
) -> None:
    """Combine per-strategy images into a single row montage if they exist.

    The function searches for files like ``{strategy}_seed{seed}_{dataset}.{img_suffix}``
    and saves a single montage image for easier side-by-side comparison.
    """
    files = []
    for strat in strategies:
        candidates = []
        if dataset not in (None, "unknown"):
            candidates.append(results_dir / f"{strat}_seed{seed}_{dataset}.{img_suffix}")
        candidates.append(results_dir / f"{strat}_seed{seed}.{img_suffix}")
        if dataset not in (None, "unknown"):
            candidates.append(results_dir / dataset / f"{strat}_seed{seed}.{img_suffix}")
        chosen = None
        for c in candidates:
            if c.exists():
                chosen = c
                break
        if chosen is not None:
            files.append((strat, chosen))
        else:
            print(f"[WARN] Missing image for strategy='{strat}' seed={seed} {img_suffix}")

    if not files:
        print(f"[INFO] No images found for montage: {img_suffix}")
        return

    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    for ax, (strat, fpath) in zip(axes, files):
        try:
            img = mpimg.imread(str(fpath))
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\n{fpath.name}\n{e}", ha="center", va="center")
        ax.set_title(strat)
        ax.axis("off")
    plt.suptitle(fig_title)
    plt.tight_layout()
    out_path = outdir / f"montage_seed{seed}_{img_suffix.replace('.', '_')}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_path}")


def infer_capacity_ks(df: pd.DataFrame) -> List[int]:
    """Infer available capacity ``K`` values from precision/recall/lift/tp@K columns."""
    ks = set()
    for col in df.columns:
        m = re.match(r"^(precision|recall|lift|tp)@(\d+)$", str(col))
        if m:
            ks.add(int(m.group(2)))
    return sorted(ks)


def plot_capacity_curves(df: pd.DataFrame, outdir: Path, checkpoint: int, ks: Optional[List[int]] = None,
                         metrics: Optional[List[str]] = None, require_reach: bool = True) -> None:
    """Plot capacity metrics versus K at the rows nearest to a checkpoint.

    The function looks for columns like ``precision@K`` or ``recall@K`` and
    produces one error-bar plot per metric showing mean ± SD across seeds.
    """
    if ks is None or len(ks) == 0:
        ks = infer_capacity_ks(df)
    if not ks:
        print("[INFO] No capacity metrics found in data; skipping capacity plots")
        return
    if metrics is None:
        metrics = ["precision", "recall", "lift", "tp"]

    nearest = _nearest_rows_per_seed(df, checkpoint=checkpoint, require_reach=require_reach,
                                     extra_columns=[f"{m}@{k}" for m in metrics for k in ks])
    if nearest.empty:
        print(f"[INFO] No rows near checkpoint {checkpoint} for capacity plots")
        return

    for m in metrics:
        plt.figure(figsize=(8, 5))
        for strat, g in nearest.groupby("strategy"):
            means = []
            sds = []
            for k in ks:
                col = f"{m}@{k}"
                if col in g.columns:
                    vals = pd.to_numeric(g[col], errors='coerce').dropna()
                    means.append(vals.mean() if len(vals) else np.nan)
                    sds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
                else:
                    means.append(np.nan)
                    sds.append(0.0)
            plt.errorbar(ks, means, yerr=sds, marker='o', capsize=3, label=strat)
        plt.xlabel("K (analyst capacity)")
        plt.ylabel(m)
        plt.title(f"{m} vs K at ~{checkpoint} labels")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        plt.tight_layout()
        out = outdir / f"capacity_{m}_at_{checkpoint}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved {out}")


def plot_cumulative_frauds(df: pd.DataFrame, outdir: Path, title_suffix: str = "") -> None:
    """Plot cumulative ``frauds_found`` versus ``labels_used`` per strategy."""
    if "frauds_found" not in df.columns:
        print("[INFO] 'frauds_found' not found; skipping cumulative frauds plot")
        return
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    strategies = sorted(df["strategy"].unique())
    palette = sns.color_palette("tab10", n_colors=max(10, len(strategies)))
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(strategies)}

    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        data=df,
        x="labels_used",
        y="frauds_found",
        hue="strategy",
        errorbar="sd",
        palette=color_map,
    )
    ax.set_xlabel("Labels used")
    ax.set_ylabel("Cumulative frauds discovered")
    title = "Frauds discovered vs. labels"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = outdir / "frauds_discovered_vs_labels.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")


def final_performance_table(df: pd.DataFrame, outdir: Path, checkpoint: int, include_labels_used: bool = True) -> pd.DataFrame:
    """Create a comparison table of final performance at a checkpoint.

    The output is sorted by AUPRC and saved as CSV (and LaTeX when possible).
    """
    cols = ["strategy", "auprc", "auroc", "recall@0.1%fpr"]
    if include_labels_used:
        cols.append("labels_used")
    nearest = _nearest_rows_per_seed(df, checkpoint=checkpoint, require_reach=True, extra_columns=cols)
    if nearest.empty:
        print(f"[INFO] No runs reached checkpoint {checkpoint}; skipping final table")
        return pd.DataFrame()
    agg = nearest.groupby("strategy", as_index=False).agg({
        "auprc": "mean",
        "auroc": "mean",
        "recall@0.1%fpr": "mean",
        **({"labels_used": "mean"} if include_labels_used else {})
    })
    agg = agg.sort_values("auprc", ascending=False)
    csv_path = outdir / f"final_performance_at_{checkpoint}.csv"
    agg.to_csv(csv_path, index=False)
    print(f"[OK] Saved {csv_path}")
    try:
        (agg.round(4)).to_latex(outdir / f"final_performance_at_{checkpoint}.tex", index=False)
    except Exception as e:
        print(f"[WARN] Could not export final table LaTeX: {e}")
    return agg


def label_efficiency_table(df: pd.DataFrame, outdir: Path, target_auprc: float, baseline: Optional[str] = None) -> pd.DataFrame:
    """Summarize labels needed to reach a target AUPRC for each strategy."""
    recs = []
    for strat, g in df.groupby("strategy"):
        per_seed = []
        for seed, gs in g.groupby("seed"):
            gs = gs.sort_values("labels_used")
            meet = gs[gs["auprc"] >= target_auprc]
            if meet.empty:
                continue
            lbl = int(meet.iloc[0]["labels_used"])
            per_seed.append(lbl)
        if per_seed:
            recs.append({
                "strategy": strat,
                "labels_needed_mean": float(np.mean(per_seed)),
                "labels_needed_std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan,
                "n_seeds": int(len(per_seed))
            })
    if not recs:
        print(f"[INFO] No strategy reached target AUPRC={target_auprc}; skipping label efficiency table")
        return pd.DataFrame()
    tab = pd.DataFrame(recs).sort_values("labels_needed_mean")
    if baseline and baseline in set(tab["strategy"].unique()):
        base_val = float(tab[tab["strategy"] == baseline]["labels_needed_mean"].iloc[0])
        tab["relative_gain"] = ((base_val - tab["labels_needed_mean"]) / base_val) * 100.0
    csv_path = outdir / f"label_efficiency_target_auprc_{str(target_auprc).replace('.', '_')}.csv"
    tab.to_csv(csv_path, index=False)
    print(f"[OK] Saved {csv_path}")
    try:
        (tab.round(2)).to_latex(outdir / f"label_efficiency_target_auprc_{str(target_auprc).replace('.', '_')}.tex", index=False)
    except Exception as e:
        print(f"[WARN] Could not export label efficiency LaTeX: {e}")
    return tab


def runtime_table(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Summarize average evaluation time per round for each strategy."""
    if "eval_time_s" not in df.columns:
        print("[INFO] 'eval_time_s' not found; skipping runtime table")
        return pd.DataFrame()
    agg = df.groupby("strategy", as_index=False).agg({
        "eval_time_s": ["mean", "std"],
        "labels_used": "max"
    })
    agg.columns = ["strategy", "eval_time_mean", "eval_time_std", "max_labels"]
    csv_path = outdir / "runtime_table.csv"
    agg.to_csv(csv_path, index=False)
    print(f"[OK] Saved {csv_path}")
    try:
        (agg.round(3)).to_latex(outdir / "runtime_table.tex", index=False)
    except Exception as e:
        print(f"[WARN] Could not export runtime table: {e}")
    return agg


def dataset_summary_table(data_path: Optional[str], outdir: Path, dataset_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Create a simple summary of a tabular fraud dataset and save it.

    The function expects a binary label column (``Class`` or ``isFraud``) and
    reports basic counts and the number of features.
    """
    if not data_path:
        print("[INFO] No data_path provided; skipping dataset summary table")
        return None
    p = Path(data_path)
    if not p.exists():
        print(f"[WARN] dataset_summary_table: path not found: {p}")
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[WARN] Could not read dataset '{p}': {e}")
        return None
    label_col = None
    if "Class" in df.columns:
        label_col = "Class"
    elif "isFraud" in df.columns:
        label_col = "isFraud"
    total = len(df)
    frauds = int(df[label_col].sum()) if label_col else np.nan
    legit = int(total - frauds) if label_col else np.nan
    fraud_ratio = (100.0 * frauds / total) if label_col else np.nan
    features = df.drop(columns=[c for c in [label_col, "Time"] if c and c in df.columns])
    feature_count = features.shape[1]

    tab = pd.DataFrame([
        {"Metric": "Total transactions", "Count": total},
        {"Metric": "Fraudulent transactions", "Count": frauds},
        {"Metric": "Legitimate transactions", "Count": legit},
        {"Metric": "Fraud ratio (%)", "Count": round(fraud_ratio, 4) if label_col else None},
        {"Metric": "Feature count", "Count": feature_count},
    ])
    out_csv = outdir / "dataset_summary.csv"
    tab.to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv}")
    try:
        md_lines = ["| Metric | Count |", "| :-- | --: |"] + [f"| {r.Metric} | {r.Count} |" for r in tab.itertuples()]
        (outdir / "dataset_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    except Exception:
        # If Markdown export fails, still return the numeric summary.
        pass
    return tab


def improvement_vs_worst_table_mean(df: pd.DataFrame, outdir: Path, checkpoint: int, metric: str = "auprc") -> pd.DataFrame:
    """Compare each strategy to the worst baseline using mean performance.

    The mean of ``metric`` is computed over all rounds up to ``checkpoint``.
    """
    df_at = df[df["labels_used"] <= checkpoint].copy()
    if df_at.empty:
        print(f"[INFO] No data up to checkpoint {checkpoint}; skipping improvement vs worst (mean) table")
        return pd.DataFrame()

    if metric not in df_at.columns:
        print(f"[WARN] Metric '{metric}' not found in data; skipping improvement vs worst (mean) table")
        return pd.DataFrame()

    agg = df_at.groupby("strategy")[metric].mean()

    if agg.empty:
        print(f"[INFO] No strategy data for improvement vs worst (mean) table")
        return pd.DataFrame()

    worst_baseline = agg.min()
    improvements = ((agg - worst_baseline) / worst_baseline * 100).sort_values(ascending=False)

    result = pd.DataFrame({
        f"Mean {metric.upper()}": agg[improvements.index],
        "Improvement vs Worst (%)": improvements.values,
    })

    csv_path = outdir / f"improvement_vs_worst_mean_{metric}.csv"
    result.to_csv(csv_path)
    print(f"[OK] Saved {csv_path}")

    try:
        result.round(4).to_latex(outdir / f"improvement_vs_worst_mean_{metric}.tex")
    except Exception as e:
        print(f"[WARN] Could not export improvement vs worst (mean) LaTeX: {e}")

    return result


def improvement_vs_worst_table_final(df: pd.DataFrame, outdir: Path, checkpoint: int, metric: str = "auprc") -> pd.DataFrame:
    """Compare each strategy to the worst baseline at a specific checkpoint."""
    df_at = df[df["labels_used"] == checkpoint].copy()
    if df_at.empty:
        print(f"[INFO] No data at checkpoint {checkpoint}; skipping improvement vs worst (final) table")
        return pd.DataFrame()

    if metric not in df_at.columns:
        print(f"[WARN] Metric '{metric}' not found in data; skipping improvement vs worst (final) table")
        return pd.DataFrame()

    agg = df_at.groupby("strategy")[metric].mean()

    if agg.empty:
        print(f"[INFO] No strategy data for improvement vs worst (final) table")
        return pd.DataFrame()

    worst_baseline = agg.min()
    improvements = ((agg - worst_baseline) / worst_baseline * 100).sort_values(ascending=False)

    result = pd.DataFrame({
        f"Final {metric.upper()}": agg[improvements.index],
        "Improvement vs Worst (%)": improvements.values,
    })

    csv_path = outdir / f"improvement_vs_worst_final_{metric}.csv"
    result.to_csv(csv_path)
    print(f"[OK] Saved {csv_path}")

    try:
        result.round(4).to_latex(outdir / f"improvement_vs_worst_final_{metric}.tex")
    except Exception as e:
        print(f"[WARN] Could not export improvement vs worst (final) LaTeX: {e}")

    return result


def statistical_summary_table(df: pd.DataFrame, outdir: Path, checkpoint: int, metrics: List[str] = ["auroc", "auprc"]) -> pd.DataFrame:
    """Compute mean, std, median, min, and max per strategy at a checkpoint."""
    df_at = df[df["labels_used"] == checkpoint].copy()
    if df_at.empty:
        print(f"[INFO] No data at checkpoint {checkpoint}; skipping statistical summary table")
        return pd.DataFrame()

    results = []
    for strategy in sorted(df_at["strategy"].unique()):
        df_s = df_at[df_at["strategy"] == strategy]

        row = {"Strategy": strategy, "n_seeds": len(df_s)}
        for metric in metrics:
            if metric not in df_s.columns:
                continue
            vals = df_s[metric].values
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std() if len(vals) > 1 else 0.0
            row[f"{metric}_median"] = np.median(vals)
            row[f"{metric}_min"] = vals.min()
            row[f"{metric}_max"] = vals.max()

        results.append(row)

    if not results:
        print(f"[INFO] No strategy data for statistical summary table")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    csv_path = outdir / f"statistical_summary_at_{checkpoint}.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"[OK] Saved {csv_path}")

    try:
        result_df.round(4).to_latex(outdir / f"statistical_summary_at_{checkpoint}.tex", index=False)
    except Exception as e:
        print(f"[WARN] Could not export statistical summary LaTeX: {e}")

    return result_df


def plot_improvement_vs_worst(df: pd.DataFrame, outdir: Path, checkpoint: int, metric: str = "auprc",
                              mode: Literal["mean", "final"] = "final") -> None:
    """Create a bar plot showing improvement vs. the worst-performing strategy."""
    if mode == "mean":
        df_at = df[df["labels_used"] <= checkpoint].copy()
        title_suffix = "Mean Across Iterations"
    else:
        df_at = df[df["labels_used"] == checkpoint].copy()
        title_suffix = f"Final (at {checkpoint} labels)"

    if df_at.empty or metric not in df_at.columns:
        return

    agg = df_at.groupby("strategy")[metric].mean()
    if agg.empty:
        return

    worst_baseline = agg.min()
    improvements = ((agg - worst_baseline) / worst_baseline * 100).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(improvements)))
    bars = ax.barh(range(len(improvements)), improvements.values, color=colors)

    ax.set_yticks(range(len(improvements)))
    ax.set_yticklabels(improvements.index)
    ax.set_xlabel("Improvement vs Worst Baseline (%)", fontsize=12)
    ax.set_title(f"Relative Performance Improvement - {metric.upper()}\n{title_suffix}",
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (bar, val) in enumerate(zip(bars, improvements.values)):
        ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    out_path = outdir / f"improvement_vs_worst_{mode}_{metric}.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def plot_statistical_summary_errorbar(df: pd.DataFrame, outdir: Path, checkpoint: int,
                                       metric: str = "auprc") -> None:
    """Plot mean ± std as error bars for each strategy at a checkpoint."""
    df_at = df[df["labels_used"] == checkpoint].copy()
    if df_at.empty or metric not in df_at.columns:
        return

    stats = []
    for strategy in sorted(df_at["strategy"].unique()):
        df_s = df_at[df_at["strategy"] == strategy]
        vals = df_s[metric].values
        if len(vals) > 0:
            stats.append({
                'strategy': strategy,
                'mean': vals.mean(),
                'std': vals.std() if len(vals) > 1 else 0.0,
                'n': len(vals)
            })

    if not stats:
        return

    stats_df = pd.DataFrame(stats).sort_values('mean', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(stats_df))

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(stats_df)))
    ax.barh(y_pos, stats_df['mean'].values, xerr=stats_df['std'].values,
            color=colors, alpha=0.7, capsize=5, ecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats_df['strategy'].values)
    ax.set_xlabel(f"{metric.upper()}", fontsize=12)
    ax.set_title(f"Performance Comparison with Standard Deviation\n{metric.upper()} at {checkpoint} labels",
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (mean, std, n) in enumerate(zip(stats_df['mean'], stats_df['std'], stats_df['n'])):
        label = f'{mean:.4f} ± {std:.4f}' if std > 0 else f'{mean:.4f}'
        ax.text(mean + 0.01, i, label, va='center', fontsize=9)

    plt.tight_layout()
    out_path = outdir / f"statistical_summary_errorbar_{metric}_at_{checkpoint}.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def plot_variance_boxplot(df: pd.DataFrame, outdir: Path, checkpoint: int,
                          metrics: List[str] = ["auprc", "auroc"]) -> None:
    """Create box plots showing per-seed performance distribution per strategy."""
    df_at = df[df["labels_used"] == checkpoint].copy()
    if df_at.empty:
        return

    available_metrics = [m for m in metrics if m in df_at.columns]
    if not available_metrics:
        return

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        strategies = sorted(df_at["strategy"].unique())
        data_by_strategy = []
        labels = []

        for strategy in strategies:
            df_s = df_at[df_at["strategy"] == strategy]
            vals = df_s[metric].values
            if len(vals) > 0:
                data_by_strategy.append(vals)
                labels.append(strategy)

        if not data_by_strategy:
            continue

        ax.boxplot(
            data_by_strategy,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2),
            meanprops=dict(color='green', linewidth=2, linestyle='--'),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )

        ax.set_ylabel(f"{metric.upper()}", fontsize=11)
        ax.set_title(f"{metric.upper()} Distribution Across Seeds", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Median'),
            Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Mean'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.suptitle(f"Performance Variability Across Seeds at {checkpoint} Labels",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = outdir / f"variance_boxplot_at_{checkpoint}.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def main():
    """Parse CLI arguments, optionally run experiments, and build the report.

    This function ties together result discovery, filtering, plotting, table
    generation, bootstrap statistics, and optional ROC/PR montages.
    """
    parser = argparse.ArgumentParser(description="Generate report-ready visualizations and comparisons")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory with results")
    parser.add_argument("--dataset", type=str, default="creditcard", help="Dataset name to include (or 'all')")
    parser.add_argument("--strategies", nargs="*", default=[], help="Strategies to include; default: auto-detect")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds to include; default: all")
    parser.add_argument("--labels_checkpoints", type=str, default="1400,1800,2200,3000,5000",
                        help="Comma-separated label budgets at which to summarize results")
    parser.add_argument("--metrics", nargs="*", default=[
        "frauds_found", "auroc", "auprc", "recall@0.1%fpr", "precision@0.1%fpr", "profit@0.1%fpr"
    ], help="Metrics to include in learning curves and tables")
    parser.add_argument("--out_dir", type=str, default="results_report", help="Output directory for figures/tables")
    parser.add_argument("--important_only", action="store_true", default=False,
                        help="Keep only key strategies (e.g., hybrid, qbc) and exclude clutter")
    parser.add_argument("--exclude_strategies", nargs="*", default=[],
                        help="Strategies to exclude by exact name")
    parser.add_argument("--baseline_strategy", type=str, default="fraudpp_hybrid",
                        help="Baseline strategy for delta plots and stats if present")
    parser.add_argument("--stat_metric", type=str, default="frauds_found",
                        help="Metric to use for statistical analysis")
    parser.add_argument("--stat_checkpoint", type=int, default=5000,
                        help="Checkpoint (labels) used for stats and bar/scatter plots")
    parser.add_argument("--bootstrap_iters", type=int, default=5000,
                        help="Bootstrap iterations for CI/p-values")
    parser.add_argument("--require_full_checkpoint", action="store_true", default=True,
                        help="Only include runs that reach the checkpoint in bar/scatter/stats")
    parser.add_argument("--full_only_all", action="store_true", default=False,
                        help="Drop any (strategy,seed) that did not reach the checkpoint from ALL outputs")
    parser.add_argument("--capacity_list", type=str, default="",
                        help="Comma-separated K values for capacity plots; default: infer from data")
    parser.add_argument("--run", action="store_true", default=False,
                        help="If set, run the selected strategies/seeds to generate metrics before plotting")
    parser.add_argument("--force_run", action="store_true", default=False,
                        help="Re-run even if metrics already exist")
    parser.add_argument("--config_path", type=str, default="config.yaml",
                        help="Base config file to load and override for runs")
    parser.add_argument("--data_path", type=str, default="",
                        help="Explicit data path to use for runs; if empty, inferred from --dataset")
    parser.add_argument("--seeds_run", type=str, default="0",
                        help="Comma-separated seeds to run when --run is set (default: 0)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_base = Path(args.out_dir)

    if args.run:
        if not _CAN_RUN:
            raise RuntimeError("Cannot run experiments: failed to import config/learner.")
        if args.strategies:
            run_strategies = list(args.strategies)
        elif args.important_only:
            run_strategies = ["fraudpp_hybrid", "qbc", "graph_hybrid"]
        else:
            run_strategies = [
                "random", "entropy", "margin", "cost_balanced", "qbc",
                "fraud", "fraudpp", "fraudpp_hybrid", "graph_hybrid"
            ]
        if args.exclude_strategies:
            run_strategies = [s for s in run_strategies if s not in set(args.exclude_strategies)]
        supported = {"random", "entropy", "margin", "cost_balanced", "qbc", "fraud", "fraudpp", "fraudpp_hybrid", "graph_hybrid"}
        run_strategies = [s for s in run_strategies if s in supported]
        if not run_strategies:
            raise RuntimeError("No supported strategies to run after filtering. Supported: " + ", ".join(sorted(supported)))
        seeds_run = [int(s) for s in args.seeds_run.split(',') if s.strip().isdigit()]
        if not seeds_run:
            seeds_run = [0]
        data_path = args.data_path.strip()
        if not data_path:
            ds = args.dataset.strip().lower()
            candidates = []
            if ds and ds != "all":
                candidates = [f"{ds}.csv", f"{ds}.parquet", ds]
            for c in candidates:
                if Path(c).exists():
                    data_path = c
                    break
            if not data_path:
                if ds in ("creditcard", "credit-card", "cc") and Path("creditcard.csv").exists():
                    data_path = "creditcard.csv"
        if not data_path:
            raise FileNotFoundError("Could not infer data path; pass --data_path explicitly.")

        results_dir.mkdir(parents=True, exist_ok=True)

        cfg_base = _load_cfg(args.config_path)
        cfg_base.data_path = data_path
        cfg_base.outdir = str(results_dir)
        try:
            import lightgbm  # type: ignore
            _has_lgbm = True
        except Exception:
            _has_lgbm = False
        if not _has_lgbm:
            try:
                cfg_base.al.model_kind = "lr"
            except Exception:
                pass

        for strat in run_strategies:
            for seed in seeds_run:
                ds_name = Path(data_path).stem
                run_name = f"{strat}_seed{seed}_{ds_name}"
                metrics_path = results_dir / f"{run_name}.metrics.csv"
                if metrics_path.exists() and not args.force_run:
                    print(f"[SKIP] {run_name} (metrics exist). Use --force_run to rerun.")
                    continue
                cfg = _ExpCfg.from_dict(cfg_base.to_dict())
                cfg.strategy_name = cast(Literal["random", "entropy", "margin", "cost_balanced", "qbc", "fraud", "fraudpp", "fraudpp_hybrid", "graph_hybrid"], strat)
                cfg.seed = seed
                cfg.outdir = str(results_dir)
                print(f"[RUN] {run_name}")
                _run_experiment(cfg)

    full = discover_metrics(results_dir)

    ds = args.dataset
    if ds and ds.lower() != "all":
        df = full[full["dataset"].str.lower() == ds.lower()].copy()
        title_suffix = ds
    else:
        df = full.copy()
        title_suffix = "all datasets"

    if df.empty:
        raise RuntimeError(f"No metrics found for dataset filter '{ds}'.")

    if args.strategies:
        keep = set(args.strategies)
        df = df[df["strategy"].isin(keep)].copy()

    if args.exclude_strategies:
        df = df[~df["strategy"].isin(set(args.exclude_strategies))].copy()

    if args.important_only and not args.strategies:
        important = ["fraudpp_hybrid", "qbc", "graph_hybrid"]
        available = [s for s in important if s in set(df["strategy"].unique())]
        if available:
            df = df[df["strategy"].isin(set(available))].copy()

    if args.seeds.strip():
        seeds_filter: List[int] = [int(s) for s in args.seeds.split(',') if s.strip().isdigit()]
        df = df[df["seed"].isin(seeds_filter)].copy()

    strategies = sorted(df["strategy"].unique())
    if not strategies:
        raise RuntimeError("No strategies found after filtering.")

    outdir = ensure_outdir(out_base, None if ds.lower() == "all" else ds)

    combined_path = outdir / "combined_metrics.csv"
    df.to_csv(combined_path, index=False)
    print(f"[OK] Saved {combined_path}")

    checkpoints = [int(x) for x in args.labels_checkpoints.split(',') if x.strip()]

    if args.full_only_all:
        cp_all = args.stat_checkpoint if args.stat_checkpoint else (max(checkpoints) if checkpoints else None)
        if cp_all is not None:
            keep_idx = []
            for (s, seed), g in df.groupby(["strategy", "seed"]):
                if g["labels_used"].max() >= cp_all:
                    keep_idx.append(g.index)
            if keep_idx:
                keep_mask = df.index.isin(np.concatenate(keep_idx))
                dropped = (~keep_mask).sum()
                if dropped:
                    print(f"[INFO] full_only_all: dropping {dropped} rows not reaching {cp_all}")
                df = df.loc[keep_mask].copy()

    plot_learning_curves(df, outdir, args.metrics, title_suffix)

    try:
        plot_cumulative_frauds(df, outdir, title_suffix)
    except Exception as e:
        print(f"[WARN] Could not plot cumulative fraud discovery: {e}")

    summary = summarize_at_checkpoints(df, checkpoints, args.metrics)
    if not summary.empty:
        save_tables(summary, outdir, basename=f"summary_checkpoints")
    else:
        print("[INFO] No summary produced (no checkpoints matched available rows)")

    cp = args.stat_checkpoint if args.stat_checkpoint else (max(checkpoints) if checkpoints else None)
    if cp is not None:
        if args.stat_metric in df.columns:
            barplot_at_checkpoint(df, outdir, metric=args.stat_metric, checkpoint=cp, require_reach=args.require_full_checkpoint)
            delta_vs_baseline_plot(df, outdir, baseline_strategy=args.baseline_strategy, metric=args.stat_metric)
        if "auroc" in df.columns and "auprc" in df.columns:
            scatter_auroc_auprc_at_checkpoint(df, outdir, checkpoint=cp, require_reach=args.require_full_checkpoint)
        ks_cli = [int(x) for x in args.capacity_list.split(',') if x.strip().isdigit()] if args.capacity_list else None
        plot_capacity_curves(df, outdir, checkpoint=cp, ks=ks_cli, require_reach=args.require_full_checkpoint)
        try:
            final_performance_table(df, outdir, checkpoint=cp, include_labels_used=True)
        except Exception as e:
            print(f"[WARN] Could not create final performance table: {e}")

    try:
        target_auprc = float(getattr(args, "target_auprc", 0.0)) if hasattr(args, "target_auprc") else 0.0
    except Exception:
        target_auprc = 0.0
    if target_auprc and "auprc" in df.columns:
        try:
            label_efficiency_table(df, outdir, target_auprc=target_auprc, baseline=args.baseline_strategy)
        except Exception as e:
            print(f"[WARN] Could not compute label efficiency: {e}")

    try:
        runtime_table(df, outdir)
    except Exception as e:
        print(f"[WARN] Could not compute runtime table: {e}")

    try:
        dpath = args.data_path if hasattr(args, "data_path") else ""
        if dpath:
            dataset_summary_table(dpath, outdir, dataset_name=ds)
    except Exception as e:
        print(f"[WARN] Could not generate dataset summary: {e}")

    strategies = sorted(df["strategy"].unique())
    if strategies and cp is not None and args.stat_metric in df.columns:
        stats_tbl = bootstrap_compare(
            df=df,
            strategies=strategies,
            checkpoint=cp,
            metric=args.stat_metric,
            baseline_strategy=args.baseline_strategy if args.baseline_strategy in strategies else None,
            n_boot=args.bootstrap_iters,
            random_state=42,
            require_reach=args.require_full_checkpoint,
        )
        stats_csv = outdir / f"stats_bootstrap_{_safe_metric_name(args.stat_metric)}_at_{cp}.csv"
        stats_tbl.to_csv(stats_csv, index=False)
        print(f"[OK] Saved {stats_csv}")
        try:
            (stats_tbl.round(4)).to_latex(outdir / f"stats_bootstrap_{_safe_metric_name(args.stat_metric)}_at_{cp}.tex", index=False)
        except Exception as e:
            print(f"[WARN] Could not export stats LaTeX: {e}")

    try:
        seeds_for_montage = sorted(df["seed"].unique().tolist())
        if args.seeds.strip():
            seeds_for_montage = [int(s) for s in args.seeds.split(',') if s.strip().isdigit()]
        seeds_for_montage = seeds_for_montage[:3]
        for s in seeds_for_montage:
            compose_image_montage(results_dir, outdir, None if ds.lower() == 'all' else ds, strategies, s, "roc.png", f"ROC (seed={s})")
            compose_image_montage(results_dir, outdir, None if ds.lower() == 'all' else ds, strategies, s, "pr.png", f"PR (seed={s})")
    except Exception as e:
        print(f"[WARN] Could not create montages: {e}")

    print("\nDone. Figures and tables saved under:")
    print(outdir.resolve())


if __name__ == "__main__":
    main()
