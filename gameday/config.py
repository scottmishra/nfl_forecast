"""Central configuration: paths, positions, stats, and model hyperparameters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("GAMEDAY_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("GAMEDAY_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
ARTIFACTS_DIR = Path(os.environ.get("GAMEDAY_ARTIFACTS_DIR", ROOT / "artifacts"))
MODELS_ROOT = ARTIFACTS_DIR / "models"  # bundle installs swap current/previous here
MODELS_DIR = MODELS_ROOT / "current"    # the active model set engines read/write
FORECASTS_DIR = ARTIFACTS_DIR / "forecasts"

# Offensive skill positions we model, and the stat lines forecast for each.
POSITION_STATS: dict[str, list[str]] = {
    "QB": ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "fantasy_points"],
    "RB": ["rushing_yards", "rushing_tds", "receptions", "receiving_yards", "fantasy_points"],
    "WR": ["receptions", "receiving_yards", "receiving_tds", "fantasy_points"],
    "TE": ["receptions", "receiving_yards", "receiving_tds", "fantasy_points"],
}
POSITIONS = list(POSITION_STATS)

# Forecast quantiles: p10/p90 give the floor/ceiling band shown in the UI.
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# Rolling windows (in games) used for player-form and opponent-quality features.
FORM_WINDOWS = (3, 8)

# Packaging gate (`gameday train --gate`): refuse to ship a bundle whose
# backtest shows no skill vs the naive baseline or badly mis-calibrated 80%
# intervals. Lenient on purpose — this catches broken models, not mediocre ones.
GATE_MIN_SKILL = 0.0
GATE_COVERAGE80 = (0.70, 0.90)

# Usage columns exported to artifacts/forecasts/latest_usage.parquet after a
# refresh (filtered to whichever exist in the feature frame) — the dashboard's
# player usage sparklines read them via /api/player/{id}/usage.
USAGE_ARTIFACT_COLS = ["attempts", "carries", "targets", "target_share",
                       "air_yards_share", "wopr", "racr"]


@dataclass
class GBMParams:
    """LightGBM quantile model settings (CPU-friendly; trains in seconds/stat)."""

    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 400
    min_child_samples: int = 30
    subsample: float = 0.9
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0


@dataclass
class NeuralParams:
    """Torch quantile-MLP settings. ~1M params — trivially fits in 8GB VRAM."""

    hidden: tuple[int, ...] = (256, 256, 128)
    dropout: float = 0.15
    lr: float = 1e-3
    batch_size: int = 1024
    epochs: int = 40
    device: str = "cuda"  # falls back to cpu automatically if unavailable


@dataclass
class Settings:
    seasons: list[int] = field(default_factory=lambda: list(range(2016, 2026)))
    gbm: GBMParams = field(default_factory=GBMParams)
    neural: NeuralParams = field(default_factory=NeuralParams)


settings = Settings()


def ensure_dirs() -> None:
    for d in (RAW_DIR, FEATURES_DIR, MODELS_DIR, FORECASTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
