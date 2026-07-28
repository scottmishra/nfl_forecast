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
from gameday.features.build import NATIVE_NAN_PATTERN
from gameday.models.calibrate import apply_offsets, conformal_offsets

log = logging.getLogger(__name__)


def _model_path(models_dir: Path, position: str, stat: str, q: float) -> Path:
    return models_dir / f"gbm_{position}_{stat}_q{int(q * 100):02d}.txt"


def _manifest_path(models_dir: Path, position: str) -> Path:
    return models_dir / f"manifest_{position}.json"


def _fit_one(X: pd.DataFrame, y: pd.Series, q: float) -> lgb.LGBMRegressor:
    """One quantile booster with the configured hyperparameters."""
    params = settings.gbm
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=q,
        num_leaves=params.num_leaves, learning_rate=params.learning_rate,
        n_estimators=params.n_estimators, min_child_samples=params.min_child_samples,
        subsample=params.subsample, colsample_bytree=params.colsample_bytree,
        reg_lambda=params.reg_lambda, verbose=-1,
    )
    model.fit(X, y)
    return model


def _fit_all(df: pd.DataFrame, mask: pd.Series, position: str,
             feature_cols: list[str]) -> dict[tuple[str, float], lgb.LGBMRegressor]:
    """Fit every (stat, quantile) booster on the masked rows."""
    X = df.loc[mask, feature_cols].astype(float)
    return {(stat, q): _fit_one(X, df.loc[mask, stat].astype(float), q)
            for stat in POSITION_STATS[position] for q in QUANTILES}


def _prune_features(models: dict, feature_cols: list[str], top_k: int | None) -> list[str]:
    """Top-K features by total gain importance across every booster."""
    if not top_k or top_k >= len(feature_cols):
        return feature_cols
    gain = np.zeros(len(feature_cols))
    for model in models.values():
        gain += model.booster_.feature_importance(importance_type="gain")
    keep = set(np.argsort(gain)[::-1][:top_k])
    return [c for i, c in enumerate(feature_cols) if i in keep]


def train_position(df: pd.DataFrame, position: str, feature_cols: list[str],
                   models_dir: Path = MODELS_DIR) -> dict:
    """Train quantile boosters for every stat of one position.

    `df` must contain only historical (non-NaN target) rows for `position`.
    NaN handling is split by family: classic features are filled with training
    medians (persisted in the manifest as `fill_values` for identical treatment
    at predict time), while the temporal families (EWM/lag/slope/vol, matched
    by NATIVE_NAN_PATTERN) are left as raw NaN — LightGBM routes missing values
    natively, and "no history yet" is signal a median would erase.

    The internal split (train on seasons before the most recent, hold out the
    most recent) doubles as the conformal-calibration split: CQR offsets
    computed on the holdout land in the manifest under "calibration" and are
    applied by predict_position.

    Returns per-stat validation pinball loss on the most recent season
    (raw model output, pre-calibration).
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    stats = POSITION_STATS[position]
    report: dict[str, float] = {}

    fill_cols = [c for c in feature_cols if not NATIVE_NAN_PATTERN.search(c)]
    fill_values = df[fill_cols].median(numeric_only=True)
    df = df.copy()
    df[fill_cols] = df[fill_cols].fillna(fill_values)

    last_season = int(df["season"].max())
    train_mask = df["season"] < last_season
    if train_mask.sum() < 500:  # tiny datasets: fall back to random split
        rng = np.random.default_rng(0)
        train_mask = pd.Series(rng.random(len(df)) < 0.85, index=df.index)

    models = _fit_all(df, train_mask, position, feature_cols)

    # Prune pass: keep the highest-gain features and retrain on that list.
    pruned = _prune_features(models, feature_cols,
                             settings.gbm.top_k_features.get(position))
    if pruned is not feature_cols:
        log.info("%s: pruned %d -> %d features by gain", position,
                 len(feature_cols), len(pruned))
        feature_cols = pruned
        models = _fit_all(df, train_mask, position, feature_cols)

    # Holdout pinball on the most recent season + the calibration frame
    # (clipped, non-crossing — exactly what predict_position would emit raw).
    holdout = df[~train_mask]
    X_hold = holdout[feature_cols].astype(float)
    hold_pred = pd.DataFrame(index=holdout.index)
    for stat in stats:
        y = holdout[stat].astype(float).values
        losses = []
        raw = {q: np.clip(models[(stat, q)].predict(X_hold), 0, None) for q in QUANTILES}
        stacked = np.sort(np.column_stack([raw[q] for q in QUANTILES]), axis=1)
        for i, q in enumerate(QUANTILES):
            err = y - raw[q]
            losses.append(float(np.mean(np.maximum(q * err, (q - 1) * err))))
            hold_pred[f"{stat}_p{int(q * 100):02d}"] = stacked[:, i]
        report[stat] = round(float(np.mean(losses)), 4)
        log.info("%s/%s pinball=%.3f", position, stat, report[stat])

    calibration = (conformal_offsets(holdout[stats].astype(float), hold_pred, stats)
                   if settings.calibrate and not holdout.empty else None)

    # Refit on ALL rows while keeping the holdout's offsets. Strict CQR wants
    # offsets scored against the deployed model; refitting bends that
    # exchangeability slightly — the deployed boosters saw more data than the
    # calibrated ones, so their raw intervals are no worse and the stored
    # offsets err (mildly) conservative. Worth it: the point forecasts get the
    # most recent season's rows instead of losing them to calibration.
    if settings.refit_on_all:
        models = _fit_all(df, pd.Series(True, index=df.index), position, feature_cols)

    for (stat, q), model in models.items():
        model.booster_.save_model(str(_model_path(models_dir, position, stat, q)))

    _manifest_path(models_dir, position).write_text(json.dumps(
        {"position": position, "features": feature_cols, "stats": stats,
         "quantiles": QUANTILES, "validation_pinball": report,
         "calibration": calibration,
         "fill_values": {k: (None if pd.isna(v) else float(v))
                         for k, v in fill_values.items()}}, indent=2))
    return report


def predict_position(df: pd.DataFrame, position: str,
                     models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Quantile predictions for rows of `position`. Adds `{stat}_p{q}` columns.

    NaN features are filled with the manifest's train-time medians (temporal
    families were never filled at train time, so they carry no fill value and
    stay NaN here too). When the manifest holds conformal offsets they are
    applied to the interval endpoints — eval and serving both see the
    CALIBRATED band."""
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
            out[f"{stat}_p{int(q * 100):02d}"] = stacked[:, i]

    calibration = manifest.get("calibration")
    if calibration:
        out = apply_offsets(out, calibration, manifest["stats"])
    for stat in manifest["stats"]:
        for q in manifest["quantiles"]:
            col = f"{stat}_p{int(q * 100):02d}"
            out[col] = np.round(out[col].astype(float), 2)
    return out
