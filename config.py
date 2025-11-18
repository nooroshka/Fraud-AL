"""
Experiment configuration schema and loader.

This file defines small dataclasses for the experiment knobs (model, strategy,
and FRaUD-family options) and a loader that reads YAML/JSON into those objects.

If you're new to the codebase, start by opening a config YAML file next to this
module and map the keys to the dataclass fields below—they line up 1:1.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional, Literal, Dict, Any
import json

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


# Dataclass Schemas

@dataclass
class FRaUDConfig:
    """Tuning knobs for FRaUD/FRaUD++ style strategies.

    Most fields have sensible defaults. You can leave the majority as-is and
    tweak quota/lambda to trade off exploitation vs. exploration.
    """
    alpha: float = 1.0
    gamma: float = 2.0
    prior: Optional[float] = 0.00172
    diversity_metric: str = "euclidean"  # euclidean, cosine, manhattan, chebyshev
    diversity_mode: str = "pure"  # pure, probability_weighted, mahalanobis, hybrid
    embed_max_dim: int = 32
    quota: float = 0.25
    lambda_anomaly: float = 0.5
    use_schedules: bool = True
    rarity_k: int = 10
    exploit_topk_factor: float = 5.0
    explore_topk_factor: float = 10.0
    random_state: int = 0
    use_calibration: bool = False
    target_fpr: float = 0.001
    tabp_weight: float = 1.0
    tabp_ramp: bool = True

    # Optional TABP refinements (backwards-compatible defaults)
    #    - "inverse" reproduces previous behavior
    #    - "gaussian" uses exp(-d^2 / (2*tau^2)) with tau = tabp_temperature
    #    - "hinge" uses max(0, margin - d)/margin with margin = tabp_margin
    tabp_kind: Literal["inverse", "gaussian", "hinge"] = "inverse"
    tabp_margin: float = 0.05
    tabp_temperature: float = 0.1

    # For hybrid FRaUD++ + QBC mixing weight
    qbc_weight: float = 0.5

    # Schedule knobs to control exploit vs explore across rounds
    quota_early: float = 0.15   # fraction of batch for exploitation in very early rounds
    quota_mid: float = 0.25     # mid rounds
    quota_late: float = 0.35    # late rounds
    lam_early: float = 0.6      # anomaly rarity weight early
    lam_mid: float = 0.4        # mid
    lam_late: float = 0.25      # late

    # Advanced schedule options
    schedule_mode: Literal["fixed", "smooth", "adaptive", "performance"] = "smooth"
    # smooth = continuous interpolation between phases
    # adaptive = based on labeled data proportion
    # performance = based on model metrics
    smooth_transition_rounds: int = 2  # rounds to blend between phases
    adaptive_thresholds: tuple = (0.15, 0.35, 0.60)  # fraction of budget for phase transitions
    performance_auroc_threshold: float = 0.85  # switch to exploitation when reached

    # Graph-FRaUD parameters (for graph_hybrid strategy)
    graph_k_neighbors: int = 10         # Number of neighbors in k-NN graph
    graph_hub_weight: float = 0.5       # Weight for hub score (cluster density)
    graph_bridge_weight: float = 0.5    # Weight for bridge score (boundary diversity)
    graph_topk_factor: int = 10         # Build graph over top (topk_factor × batch_size) candidates
    graph_adaptive_qbc: bool = True     # Enable adaptive QBC weighting based on training phase


@dataclass
class ALConfig:
    """Active Learning loop settings.

    batch_size is how many labels to request per round; budget is the total
    number of labeled examples we allow ourselves to use.
    """
    batch_size: int = 200
    seed_size: int = 1400
    budget: int = 5000
    test_size: float = 0.2
    model_kind: Literal["lgbm", "lr"] = "lgbm"
    random_state: int = 0
    auroc_target: Optional[float] = None
    lgbm_profile: Literal["original", "tuned"] = "original"
    use_lgbm_early_stopping: bool = False
    # Optional: allow user to request a positive fraction in the seed set
    seed_pos_frac: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Top-level config with dataset path, strategy, and composed sub-configs."""
    data_path: str = "creditcard.csv"
    strategy_name: Literal["random", "entropy", "margin", "cost_balanced", "qbc", "fraud", "fraudpp", "fraudpp_hybrid", "graph_hybrid"] = "fraudpp_hybrid"
    outdir: str = "results"
    seed: int = 0
    # composed configs
    al: ALConfig = field(default_factory=ALConfig)
    fraud: FRaUDConfig = field(default_factory=FRaUDConfig)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentConfig":
        def _filter_kwargs(cls, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
            allowed = {f.name for f in fields(cls)}
            return {k: v for k, v in (cfg_dict or {}).items() if k in allowed}

        al = ALConfig(**_filter_kwargs(ALConfig, d.get("al", {})))
        fr = FRaUDConfig(**_filter_kwargs(FRaUDConfig, d.get("fraud", {})))
        base_keys = {"data_path", "strategy_name", "outdir", "seed"}
        base = {k: d.get(k) for k in base_keys if k in d}
        return ExperimentConfig(al=al, fraud=fr, **base)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        return out

# Loader

def load_config(path: str | Path) -> ExperimentConfig:
    """
     Load configuration from YAML (preferred) or JSON.

    The loader is forgiving: if the file extension is missing or wrong, we'll
    try YAML first (if installed), then JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    text = p.read_text(encoding="utf-8")
    data: Dict[str, Any]

    if _HAS_YAML and p.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(text) or {}
    elif p.suffix.lower() == ".json":
        data = json.loads(text or "{}")
    else:
        # try YAML first if installed, else JSON
        if _HAS_YAML:
            try:
                data = yaml.safe_load(text) or {}
            except Exception:
                data = json.loads(text or "{}")
        else:
            data = json.loads(text or "{}")

    return ExperimentConfig.from_dict(data)
