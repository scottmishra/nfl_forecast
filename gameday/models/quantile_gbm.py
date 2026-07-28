"""Per-position, per-stat quantile forecasters built on LightGBM.

One booster per (position, stat, quantile). Boosters are small (~hundreds of
KB) and train in seconds on CPU; the full slate of models trains in a couple
of minutes on a laptop, no GPU required. Artifacts are plain LightGBM text
models plus a JSON manifest recording the feature list AND the train-time
median fill values, so a machine that only ships the artifacts (e.g. the Pi)
predicts with exactly the information the trainer saw.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from gameday.config import MODELS_DIR, POSITION_STATS, QUANTILES, settings

log = logging.getLogger(__name__)


def _model_path(models_dir: Path, position: str, stat: str, q: float) -> Path:
    return models_dir / f"gbm_{position}_{stat}_q{int(q * 100):02d}.txt"


def _manifest_path(models_dir: Path, position: str) -> Path:
    return models_dir / f"manifest_{position}.json"


def train_position(df: pd.DataFrame, position: str, feature_cols: list[str],
                   models_dir: Path = MODELS_DIR) -> dict:
    """Train quantile boosters for every stat of one position.

    `df` must contain only historical (non-NaN target) rows for `position`.
    NaN features are filled with the training medians, which are persisted in
    the manifest as `fill_values` for identical treatment at predict time.
    Returns per-stat validation pinball loss on the most recent season.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    params = settings.gbm
    report: dict[str, float] = {}

    fill_values = df[feature_cols].median(numeric_only=True)
    df = df.copy()
    df[feature_cols] = df[feature_cols].fillna(fill_values)

    last_season = int(df["season"].max())
    train_mask = df["season"] < last_season
    if train_mask.sum() < 500:  # tiny datasets: fall back to random split
        rng = np.random.default_rng(0)
        train_mask = pd.Series(rng.random(len(df)) < 0.85, index=df.index)

    for stat in POSITION_STATS[position]:
        y = df[stat].astype(float)
        X = df[feature_cols].astype(float)
        losses = []
        for q in QUANTILES:
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=q,
                num_leaves=params.num_leaves, learning_rate=params.learning_rate,
                n_estimators=params.n_estimators, min_child_samples=params.min_child_samples,
                subsample=params.subsample, colsample_bytree=params.colsample_bytree,
                reg_lambda=params.reg_lambda, verbose=-1,
            )
            model.fit(X[train_mask], y[train_mask])
            pred = model.predict(X[~train_mask])
            err = y[~train_mask].values - pred
            losses.append(float(np.mean(np.maximum(q * err, (q - 1) * err))))
            model.booster_.save_model(str(_model_path(models_dir, position, stat, q)))
        report[stat] = round(float(np.mean(losses)), 4)
        log.info("%s/%s pinball=%.3f", position, stat, report[stat])

    _manifest_path(models_dir, position).write_text(json.dumps(
        {"position": position, "features": feature_cols, "stats": POSITION_STATS[position],
         "quantiles": QUANTILES, "validation_pinball": report,
         "fill_values": {k: (None if pd.isna(v) else float(v))
                         for k, v in fill_values.items()}}, indent=2))
    return report


def predict_position(df: pd.DataFrame, position: str,
                     models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Quantile predictions for rows of `position`. Adds `{stat}_p{q}` columns.

    NaN features are filled with the manifest's train-time medians, so
    inference on a fresh machine matches inference next to the trainer."""
    manifest = json.loads(_manifest_path(models_dir, position).read_text())
    feature_cols = manifest["features"]
    X = df[feature_cols].astype(float)
    fill_values = manifest.get("fill_values")
    if fill_values:
        X = X.fillna({k: v for k, v in fill_values.items() if v is not None})

    out = df.copy()
    for stat in manifest["stats"]:
        preds = {}
        for q in manifest["quantiles"]:
            booster = lgb.Booster(model_file=str(_model_path(models_dir, position, stat, q)))
            preds[q] = np.clip(booster.predict(X), 0, None)
        # Enforce non-crossing quantiles: sort each row's quantile values.
        stacked = np.sort(np.column_stack([preds[q] for q in manifest["quantiles"]]), axis=1)
        for i, q in enumerate(manifest["quantiles"]):
            out[f"{stat}_p{int(q * 100):02d}"] = np.round(stacked[:, i], 2)
    return out
