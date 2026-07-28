"""Pre-kickoff usage forecaster: snap share and volume shares per position.

A LightGBM quantile model (q25/50/75) per (position, usage target) predicts
how much opportunity a player gets BEFORE the stat models run — depth chart,
injury report, role priors, and usage form in; predicted snap/carry/target
shares out. The stat engines then consume `pred_{target}_{p25,p50,p75}` and
`pred_{target}_spread` as v3 features.

Leakage contract (the whole point of this module):
  * Inputs never include same-week actual usage — only shift(1) temporal
    forms, current-week availability (published pre-kickoff), and priors.
  * Training rows receive OUT-OF-FOLD predictions (grouped season-fold CV
    inside the training window), so a stat model never trains on a usage
    forecast that was fit on that row's own week. Test/slate rows get the
    final model fit on the full training window.

The role-prior table (features/opportunity.py) is fit here, on the fold's
training rows only, and persisted in the manifest so predict-time rows are
scored with exactly the priors the trainer saw.

Artifacts land in models_dir/usage/: one booster per (position, target,
quantile) plus manifest_usage_{position}.json with features, priors, and CV
accuracy vs the EWM-halflife-5 naive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from gameday.config import MODELS_DIR, settings
from gameday.features.opportunity import (USAGE_SERIES, apply_usage_prior,
                                          usage_prior)
from gameday.features.temporal import temporal_feature_names

log = logging.getLogger(__name__)

USAGE_QUANTILES = [0.25, 0.50, 0.75]

# What each position's usage models forecast. Route participation is absent
# from nflverse NGS, so WR/TE stop at snap and target share.
USAGE_TARGETS: dict[str, list[str]] = {
    "QB": ["snap_pct"],
    "RB": ["snap_pct", "carry_share", "target_share_team"],
    "WR": ["snap_pct", "target_share_team"],
    "TE": ["snap_pct", "target_share_team"],
}

# Non-temporal context the usage models may see (presence-filtered).
_CONTEXT_COLS = [
    "inj_questionable", "inj_doubtful", "inj_out",
    "practice_dnp", "practice_limited", "practice_full",
    "depth_rank", "depth_rank_change", "is_depth_promotion",
    "games_missed_last8", "weeks_since_return", "missed_recent",
    "n_with_team", "usage_w", "has_usage",
    "games_played", "week", "pos_rank_prev", "is_new_team", "is_rookie",
    "draft_number", "years_exp",
]


def _usage_dir(models_dir: Path) -> Path:
    return models_dir / "usage"


def _model_path(models_dir: Path, position: str, target: str, q: float) -> Path:
    return _usage_dir(models_dir) / f"usage_{position}_{target}_q{int(q * 100):02d}.txt"


def _manifest_path(models_dir: Path, position: str) -> Path:
    return _usage_dir(models_dir) / f"manifest_usage_{position}.json"


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Usage-model inputs present in `df` — never a same-week usage actual."""
    cols = temporal_feature_names(USAGE_SERIES) + _CONTEXT_COLS
    cols += [f"{t}_prior" for t in USAGE_SERIES] + [f"{t}_est" for t in USAGE_SERIES]
    return [c for c in dict.fromkeys(cols) if c in df.columns]


def _fit_one(X: pd.DataFrame, y: pd.Series, q: float) -> lgb.LGBMRegressor:
    p = settings.usage_gbm
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=q,
        num_leaves=p.num_leaves, learning_rate=p.learning_rate,
        n_estimators=p.n_estimators, min_child_samples=p.min_child_samples,
        subsample=p.subsample, colsample_bytree=p.colsample_bytree,
        reg_lambda=p.reg_lambda, verbose=-1,
    )
    model.fit(X, y)
    return model


def _attach_preds(df: pd.DataFrame, target: str, preds: dict[float, np.ndarray],
                  index) -> None:
    """Write clipped, non-crossing pred columns for `target` onto df rows."""
    stacked = np.sort(np.column_stack(
        [np.clip(preds[q], 0, 1) for q in USAGE_QUANTILES]), axis=1)
    for i, q in enumerate(USAGE_QUANTILES):
        df.loc[index, f"pred_{target}_p{int(q * 100):02d}"] = np.round(stacked[:, i], 4)
    df.loc[index, f"pred_{target}_spread"] = np.round(stacked[:, -1] - stacked[:, 0], 4)


def train_usage(df: pd.DataFrame, position: str,
                models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Fit priors + usage models on a fold's TRAINING rows; return the rows
    with prior/est/usage_w and OUT-OF-FOLD pred_* columns attached.

    Persists final boosters (fit on all rows) + manifest for predict_usage.
    The returned frame is what the stat engine should train on."""
    _usage_dir(models_dir).mkdir(parents=True, exist_ok=True)
    targets = [t for t in USAGE_TARGETS[position]
               if t in df.columns and df[t].notna().any()]

    table = usage_prior(df, USAGE_SERIES)
    df = apply_usage_prior(df, table, USAGE_SERIES)
    if not targets:
        log.warning("%s: no usage targets present; priors only", position)
        return df

    feats = _feature_cols(df)
    X_all = df[feats].astype(float)
    seasons = sorted(df["season"].unique())
    report: dict[str, dict] = {}

    for target in targets:
        y_all = pd.to_numeric(df[target], errors="coerce")
        labeled = y_all.notna()

        # Out-of-fold predictions: hold out one training season at a time.
        oof: dict[float, np.ndarray] = {q: np.full(len(df), np.nan) for q in USAGE_QUANTILES}
        if len(seasons) >= 3:
            for s in seasons:
                fit_m = labeled & (df["season"] != s)
                out_m = labeled & (df["season"] == s)
                if not fit_m.any() or not out_m.any():
                    continue
                for q in USAGE_QUANTILES:
                    m = _fit_one(X_all[fit_m], y_all[fit_m], q)
                    oof[q][out_m.to_numpy()] = m.predict(X_all[out_m])
        else:  # too little history for season folds: leave OOF NaN (native)
            log.warning("%s/%s: <3 training seasons; OOF usage preds left NaN",
                        position, target)
        _attach_preds(df, target, oof, df.index)

        # Accuracy report on the OOF rows vs the EWM-hl5 naive.
        p50 = df[f"pred_{target}_p50"].to_numpy(dtype=float)
        naive = pd.to_numeric(df.get(f"{target}_ewm5"), errors="coerce").to_numpy()
        scored = labeled.to_numpy() & ~np.isnan(p50) & ~np.isnan(naive)
        if scored.any():
            y = y_all.to_numpy(dtype=float)[scored]
            mae = float(np.mean(np.abs(y - p50[scored])))
            mae_naive = float(np.mean(np.abs(y - naive[scored])))
            report[target] = {
                "n_oof": int(scored.sum()), "mae_oof": round(mae, 4),
                "mae_naive_ewm5": round(mae_naive, 4),
                "skill_vs_ewm5": round(1 - mae / mae_naive, 4) if mae_naive > 0 else None,
            }
            log.info("%s/%s usage OOF mae=%.4f naive=%.4f", position, target, mae, mae_naive)

        # Final models on the full training window (used for test/slate rows).
        for q in USAGE_QUANTILES:
            m = _fit_one(X_all[labeled], y_all[labeled], q)
            m.booster_.save_model(str(_model_path(models_dir, position, target, q)))

    _manifest_path(models_dir, position).write_text(json.dumps(
        {"position": position, "targets": targets, "quantiles": USAGE_QUANTILES,
         "features": feats, "prior_table": table, "cv_report": report}, indent=2))
    return df


def predict_usage(df: pd.DataFrame, position: str,
                  models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Priors + final-model usage predictions for test/slate rows.

    No-op (returns `df` unchanged) when no usage manifest exists — the stat
    models then simply see none of the pred_* columns."""
    path = _manifest_path(models_dir, position)
    if not path.exists():
        return df
    manifest = json.loads(path.read_text())
    df = apply_usage_prior(df, manifest["prior_table"], USAGE_SERIES)
    feats = manifest["features"]
    X = df[[c for c in feats if c in df.columns]].reindex(columns=feats).astype(float)
    for target in manifest["targets"]:
        preds = {}
        for q in manifest["quantiles"]:
            booster = lgb.Booster(model_file=str(_model_path(models_dir, position, target, q)))
            preds[q] = booster.predict(X)
        _attach_preds(df, target, preds, df.index)
    return df
