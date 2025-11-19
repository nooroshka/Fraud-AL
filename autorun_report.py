import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import List, Literal, cast
import shutil

import numpy as np
import pandas as pd

from config import load_config as _load_cfg, ExperimentConfig as _ExpCfg
from learner import run_experiment as _run_experiment
import report_visualizations as rv


def parse_list_ints(s: str) -> List[int]:
    """Parse a comma-separated string into a list of integers.

    Non-numeric tokens are ignored rather than raising an error.
    """
    vals = []
    for t in s.split(','):
        t = t.strip()
        if t.isdigit():
            vals.append(int(t))
    return vals


def infer_data_path(dataset: str, explicit: str) -> str:
    """Infer the dataset file path from a dataset name and optional explicit path.

    If an explicit path is provided, it is used directly. Otherwise, the
    function tries a few common filename patterns and some special-case
    handling for the credit card dataset.
    """
    if explicit.strip():
        return explicit.strip()
    ds = (dataset or "").strip().lower()
    candidates = []
    if ds and ds != "all":
        candidates = [f"{ds}.csv", f"{ds}.parquet", ds]
    for c in candidates:
        if Path(c).exists():
            return c
    if ds in ("creditcard", "credit-card", "cc") and Path("creditcard.csv").exists():
        return "creditcard.csv"
    raise FileNotFoundError("Could not infer data path; pass --data_path explicitly.")


def clean_run_artifacts(results_dir: Path, ds_name: str, strategies: List[str], seeds: List[int]) -> None:
    """Remove artifacts for selected strategy/seed combinations for a dataset.

    This is used to implement a "fresh" rebuild by deleting known metric and
    diagnostic files, as well as per-round picks, from the results directory.
    Failures to delete individual files are silently ignored.
    """
    for strat in strategies:
        for seed in seeds:
            base = f"{strat}_seed{seed}_{ds_name}"
            # Remove known artifact files for this run.
            for ext in [
                "metrics.csv", "summary.json", "roc.png", "pr.png", "reliability.png",
                "confusion.png", "profit_curve.png", "picks_detailed.csv"
            ]:
                f = results_dir / f"{base}.{ext}"
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
            # Remove any per-round pick files for this run.
            for f in results_dir.glob(f"{base}.round*.picked.npy"):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass


def perform_runs(results_dir: Path, strategies: List[str], seeds: List[int], data_path: str, force: bool) -> None:
    """Execute experiments for the requested strategies and seeds.

    Unsupported strategy names are filtered out. Metrics and other artifacts
    are written under ``results_dir``, and existing runs are skipped unless
    ``force`` is True.
    """
    supported = {"random", "entropy", "margin", "cost_balanced", "qbc", "fraud", "fraudpp", "fraudpp_hybrid", "graph_hybrid"}
    run_strategies = [s for s in strategies if s in supported]
    if not run_strategies:
        raise RuntimeError("No supported strategies to run after filtering. Supported: " + ", ".join(sorted(supported)))
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg_base = _load_cfg("config.yaml")
    cfg_base.data_path = data_path
    cfg_base.outdir = str(results_dir)

    # Prefer LightGBM if available; otherwise fall back to logistic regression.
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

    ds_name = Path(data_path).stem
    for strat in run_strategies:
        for seed in seeds:
            run_name = f"{strat}_seed{seed}_{ds_name}"
            metrics_path = results_dir / f"{run_name}.metrics.csv"
            # Skip runs where metrics already exist, unless forced to rerun.
            if metrics_path.exists() and not force:
                print(f"[SKIP] {run_name} (metrics exist). Use --force to rerun.")
                continue
            cfg = _ExpCfg.from_dict(cfg_base.to_dict())
            cfg.strategy_name = cast(Literal["random", "entropy", "margin", "cost_balanced", "qbc", "fraud", "fraudpp", "fraudpp_hybrid", "graph_hybrid"], strat)
            cfg.seed = seed
            cfg.outdir = str(results_dir)
            print(f"[RUN] {run_name}")
            _run_experiment(cfg)


def _md5_of_file(p: Path) -> str:
    """Compute the MD5 hex digest of a file, reading it in chunks."""
    h = hashlib.md5()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def compute_code_digest(project_root: Path) -> dict:
    """Compute MD5 digests for key code files and return a manifest.

    The manifest contains individual file hashes, a global ``__digest__`` built
    from their concatenation, and a ``__timestamp__`` marking when the digest
    was created.
    """
    key_files = [
        project_root / 'learner.py',
        project_root / 'strategies.py',
        project_root / 'utils.py',
        project_root / 'config.py',
        project_root / 'report_visualizations.py',
    ]
    manifest = {}
    for p in key_files:
        if p.exists():
            manifest[str(p.name)] = _md5_of_file(p)
        else:
            manifest[str(p.name)] = None
    # Compute a global digest by hashing the concatenation of all per-file digests.
    concat = ''.join([manifest[k] or 'MISSING' for k in sorted(manifest.keys())]).encode('utf-8')
    manifest['__digest__'] = hashlib.md5(concat).hexdigest()
    manifest['__timestamp__'] = time.time()
    return manifest


def load_saved_digest(results_dir: Path) -> dict | None:
    """Load a previously saved code digest manifest from the results directory.

    Returns None if the manifest file does not exist or cannot be parsed.
    """
    p = results_dir / '.codehash.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def save_digest(results_dir: Path, manifest: dict) -> None:
    """Persist the current code digest manifest under the results directory.

    Errors during saving are silently ignored so they do not break the run.
    """
    try:
        (results_dir / '.codehash.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    except Exception:
        pass


def main():
    """Entry point for running experiments and generating the summary report.

    This CLI parses configuration options, runs any missing or requested
    experiments, detects code changes to optionally force reruns, and then
    generates metrics, tables, and visualizations for the report.
    """
    ap = argparse.ArgumentParser(description="Autorun experiments then generate report visuals")
    ap.add_argument("--results_dir", type=str, default="results", help="Directory to write metrics and artifacts")
    ap.add_argument("--out_dir", type=str, default="results_report", help="Directory to write report figures/tables")
    ap.add_argument("--dataset", type=str, default="creditcard", help="Dataset name or 'all'")
    ap.add_argument("--data_path", type=str, default="", help="Path to dataset; inferred if empty for known datasets")
    ap.add_argument("--strategies", nargs="*", default=[
        "random", "entropy", "margin", "cost_balanced", "qbc", "fraud", "fraudpp", "fraudpp_hybrid", "graph_hybrid"
    ], help="Strategies to run and include in report")
    ap.add_argument("--seeds", type=str, default="0", help="Comma-separated list of seeds to run")
    ap.add_argument("--full", action="store_true", default=False, help="Run a full clean rebuild (equivalent to --fresh + --force)")
    ap.add_argument("--force", action="store_true", default=False, help="Force rerun even if metrics exist")
    ap.add_argument("--fresh", action="store_true", default=False, help="Delete existing artifacts for selected runs first")
    ap.add_argument("--force_on_code_change", action="store_true", default=True, help="Auto-force rerun if code digest changed since last run")
    ap.add_argument("--check_only", action="store_true", default=False, help="Only check which runs are missing and exit")
    ap.add_argument("--labels_checkpoints", type=str, default="1400,1800,2200,3000,5000")
    ap.add_argument("--stat_checkpoint", type=int, default=5000)
    ap.add_argument("--metrics", nargs="*", default=[
        "frauds_found", "auroc", "auprc", "recall@0.1%fpr", "precision@0.1%fpr", "profit@0.1%fpr"
    ])
    ap.add_argument("--baseline_strategy", type=str, default="fraudpp_hybrid")
    ap.add_argument("--capacity_list", type=str, default="", help="Comma-separated K values for capacity plots (optional)")
    ap.add_argument("--target_auprc", type=float, default=0.0, help="Target AUPRC for label-efficiency table (0 disables)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_base = Path(args.out_dir)
    data_path = infer_data_path(args.dataset, args.data_path)
    seeds = parse_list_ints(args.seeds)
    if not seeds:
        seeds = [0]

    # In "full" mode we always clean old artifacts and force reruns.
    if args.full:
        args.fresh = True
        args.force = True

    # Optionally remove artifacts for the selected dataset/strategy/seed combos.
    if args.fresh:
        clean_run_artifacts(results_dir, Path(data_path).stem, args.strategies, seeds)

    # Detect code changes by comparing the current digest to the last saved one.
    project_root = Path(__file__).parent
    current_digest = compute_code_digest(project_root)
    saved_digest = load_saved_digest(results_dir)
    if saved_digest and saved_digest.get('__digest__') != current_digest.get('__digest__'):
        print("[INFO] Detected code changes since last run.")
        if args.force_on_code_change and not args.force:
            print("[INFO] --force_on_code_change active: forcing rerun for selected combos.")
            args.force = True

    # In check-only mode we only report which runs are missing and then exit.
    if args.check_only:
        ds_name = Path(data_path).stem
        missing = []
        for strat in args.strategies:
            for seed in seeds:
                run_name = f"{strat}_seed{seed}_{ds_name}"
                if not (results_dir / f"{run_name}.metrics.csv").exists():
                    missing.append(run_name)
        if missing:
            print("Missing runs:")
            for m in missing:
                print(" -", m)
        else:
            print("No missing runs. All selected combos present.")
        return

    # Run any requested experiments that are missing or that should be forced.
    perform_runs(results_dir, args.strategies, seeds, data_path, force=args.force)
    # Save an updated code digest so future runs can detect changes.
    save_digest(results_dir, current_digest)

    # Load all metrics from the results directory and filter for the chosen dataset.
    full = rv.discover_metrics(results_dir)
    ds = args.dataset
    if ds and ds.lower() != "all":
        df = full[full["dataset"].str.lower() == ds.lower()].copy()
        title_suffix = ds
    else:
        df = full.copy()
        title_suffix = "all datasets"

    outdir = rv.ensure_outdir(out_base, None if ds.lower() == "all" else ds)

    # Persist the combined metrics used to generate the report.
    (outdir / "combined_metrics.csv").write_text(df.to_csv(index=False))

    # Sanity-check coverage of requested (strategy, seed) pairs against the data.
    desired = {(s, sd) for s in args.strategies for sd in seeds}
    ds_name = Path(data_path).stem
    have = set()
    for (s, sd), g in df.groupby(["strategy", "seed"], as_index=False):
        if isinstance(sd, (int, np.integer)):
            have.add((s, int(sd)))
    missing_pairs = sorted(desired - have)
    if missing_pairs:
        print("[WARN] Some (strategy,seed) combos have no metrics in results:")
        for s, sd in missing_pairs:
            print(f" - {s}_seed{sd}_{ds_name} (missing metrics)")
    else:
        print("[OK] All selected (strategy,seed) combos are present in results.")

    # Core learning-curve and cumulative-fraud plots.
    rv.plot_learning_curves(df, outdir, args.metrics, title_suffix)
    try:
        rv.plot_cumulative_frauds(df, outdir, title_suffix)
    except Exception:
        pass

    # Summaries at selected label checkpoints.
    checkpoints = [int(x) for x in args.labels_checkpoints.split(',') if x.strip()]
    summary = rv.summarize_at_checkpoints(df, checkpoints, args.metrics)
    if not summary.empty:
        rv.save_tables(summary, outdir, basename="summary_checkpoints")

    # Choose a single checkpoint for more detailed statistical analysis.
    cp = args.stat_checkpoint if args.stat_checkpoint else (max(checkpoints) if checkpoints else None)
    if cp is not None:
        if args.metrics and args.metrics[0] in df.columns:
            try:
                rv.barplot_at_checkpoint(df, outdir, metric=args.metrics[0], checkpoint=cp, require_reach=True)
                rv.delta_vs_baseline_plot(df, outdir, baseline_strategy=args.baseline_strategy, metric=args.metrics[0])
            except Exception:
                pass
        if "auroc" in df.columns and "auprc" in df.columns:
            rv.scatter_auroc_auprc_at_checkpoint(df, outdir, checkpoint=cp, require_reach=True)
        ks_cli = [int(x) for x in args.capacity_list.split(',') if x.strip().isdigit()] if args.capacity_list else None
        rv.plot_capacity_curves(df, outdir, checkpoint=cp, ks=ks_cli, require_reach=True)
        try:
            rv.final_performance_table(df, outdir, checkpoint=cp, include_labels_used=True)
        except Exception:
            pass

    # Label-efficiency analysis at a requested target AUPRC, relative to a baseline.
    if float(args.target_auprc) > 0 and "auprc" in df.columns:
        try:
            rv.label_efficiency_table(df, outdir, target_auprc=float(args.target_auprc), baseline=args.baseline_strategy)
        except Exception:
            pass

    # Aggregate runtime statistics for the different strategies.
    try:
        rv.runtime_table(df, outdir)
    except Exception:
        pass

    # Summarize basic dataset-level statistics for the report.
    try:
        rv.dataset_summary_table(data_path, outdir, dataset_name=ds)
    except Exception:
        pass

    # Additional comparative analyses at the main checkpoint, including variance.
    if cp is not None:
        stat_metric = args.metrics[0] if args.metrics and args.metrics[0] in df.columns else "auprc"

        # Improvement vs worst baseline (mean across iterations) - table and plot.
        try:
            rv.improvement_vs_worst_table_mean(df, outdir, checkpoint=cp, metric=stat_metric)
            rv.plot_improvement_vs_worst(df, outdir, checkpoint=cp, metric=stat_metric, mode="mean")
        except Exception as e:
            print(f"[WARN] Could not create improvement vs worst (mean) analysis: {e}")

        # Improvement vs worst baseline (final iteration) - table and plot.
        try:
            rv.improvement_vs_worst_table_final(df, outdir, checkpoint=cp, metric=stat_metric)
            rv.plot_improvement_vs_worst(df, outdir, checkpoint=cp, metric=stat_metric, mode="final")
        except Exception as e:
            print(f"[WARN] Could not create improvement vs worst (final) analysis: {e}")

        # Statistical summary with variance metrics - table and plots.
        try:
            metrics_to_analyze = [m for m in ["auroc", "auprc", "recall@0.1%fpr", "precision@0.1%fpr"] if m in df.columns]
            if metrics_to_analyze:
                rv.statistical_summary_table(df, outdir, checkpoint=cp, metrics=metrics_to_analyze)
                # Error bar plot for the main comparison metric.
                rv.plot_statistical_summary_errorbar(df, outdir, checkpoint=cp, metric=stat_metric)
                # Box plots to visualize variance in key metrics across strategies.
                rv.plot_variance_boxplot(df, outdir, checkpoint=cp,
                                        metrics=[m for m in ["auprc", "auroc"] if m in df.columns])
        except Exception as e:
            print(f"[WARN] Could not create statistical summary analysis: {e}")

    # Compose montage images for a few seeds to compare ROC and PR curves side by side.
    strategies = sorted(df["strategy"].unique())
    try:
        seeds_for_montage = sorted(df["seed"].unique().tolist())[:3]
        for s in seeds_for_montage:
            rv.compose_image_montage(results_dir, outdir, None if ds.lower() == 'all' else ds, strategies, s, "roc.png", f"ROC (seed={s})")
            rv.compose_image_montage(results_dir, outdir, None if ds.lower() == 'all' else ds, strategies, s, "pr.png", f"PR (seed={s})")
    except Exception:
        pass

    print("\nDone. Figures and tables saved under:")
    print(outdir.resolve())


if __name__ == "__main__":
    main()
