from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass
class ExperimentConfig:
    data_path: str = "creditcard.csv"
    outdir: str = "results"
    seeds: Tuple[int, ...] = (0,)
    strategies: Tuple[str, ...] = ("qbc",)
    batch_size: int = 200
    seed_size: int = 1400
    budget: int = 5000
    model_kind: str = "lgbm"

def load_config(path: str | None = "config.yaml") -> ExperimentConfig:
    if path and Path(path).exists():
        import yaml
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if "seeds" in raw and isinstance(raw["seeds"], list): raw["seeds"] = tuple(raw["seeds"])
        if "strategies" in raw and isinstance(raw["strategies"], list): raw["strategies"] = tuple(raw["strategies"])
        return ExperimentConfig(**raw)
    return ExperimentConfig()
