"""Split-conformal quantile calibration (CQR) for the forecast intervals.

The raw quantile models under-cover (the 2024 replay's p10–p90 band caught
~67% of outcomes against an 80% target), so we widen each interval by the
empirical miss observed on a calibration split the boosters never trained on:

  * lower score  s_lo = q_lo_hat − y   (how far the floor sat above reality)
  * upper score  s_up = y − q_hi_hat   (how far the ceiling sat below it)
  * offsets      the (1 − α/2) empirical quantile of each side's scores, with
                 the finite-sample (n+1) correction, where α = lo + (1 − hi)

Corrections are asymmetric (floor and ceiling widen independently) and, when
the calibration split is deep enough, *group-conditional*: offsets are
computed per (stat × p50-tertile) cell so cheap and expensive forecasts get
their own widths rather than one marginal stretch. Offsets can be negative,
which narrows an over-covered interval — the same arithmetic runs both ways.

Everything returned is JSON-serializable so the engines can persist offsets
in their model manifests and apply them at predict time on any machine.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gameday.config import QUANTILES

log = logging.getLogger(__name__)

# Interval endpoints calibrated as (lower, upper) quantile pairs.
DEFAULT_PAIRS = ((0.10, 0.90), (0.25, 0.75))

# Minimum calibration rows per (stat × tertile) cell before group-conditional
# offsets are trusted; below this the stat falls back to marginal offsets.
MIN_CELL_ROWS = 300


def _q_col(stat: str, q: float) -> str:
    return f"{stat}_p{int(q * 100):02d}"


def _pair_key(lo: float, hi: float) -> str:
    return f"{int(lo * 100):02d}-{int(hi * 100):02d}"


def _finite_sample_quantile(scores: np.ndarray, level: float) -> float:
    """Empirical quantile at `level` with the (n+1) conformal correction."""
    scores = scores[~np.isnan(scores)]
    n = len(scores)
    if n == 0:
        return 0.0
    rank = min(int(np.ceil((n + 1) * level)), n)
    return float(np.sort(scores)[rank - 1])


def _cell_offsets(y: np.ndarray, pred: pd.DataFrame, stat: str,
                  pairs) -> dict:
    """Per-pair asymmetric offsets for one calibration cell."""
    out = {}
    for lo, hi in pairs:
        alpha = lo + (1.0 - hi)
        level = 1.0 - alpha / 2.0
        s_lo = pred[_q_col(stat, lo)].to_numpy() - y
        s_up = y - pred[_q_col(stat, hi)].to_numpy()
        out[_pair_key(lo, hi)] = {
            "lo": round(_finite_sample_quantile(s_lo, level), 4),
            "up": round(_finite_sample_quantile(s_up, level), 4),
        }
    return out


def conformal_offsets(y_true: pd.DataFrame, pred_quantiles_df: pd.DataFrame,
                      stats: list[str], pairs=DEFAULT_PAIRS) -> dict:
    """Compute per-stat CQR offsets from a held-out calibration set.

    `y_true` carries one column of actuals per stat; `pred_quantiles_df` the
    matching `{stat}_pXX` predictions (same row order). Returns, per stat,
    either marginal offsets or per-p50-tertile offsets keyed by the tertile
    boundaries (used to bucket rows at predict time)."""
    offsets: dict = {}
    for stat in stats:
        y = y_true[stat].to_numpy(dtype=float)
        valid = ~np.isnan(y)
        y_s, pred_s = y[valid], pred_quantiles_df.loc[valid]
        p50 = pred_s[_q_col(stat, 0.50)].to_numpy(dtype=float)

        bounds = [float(np.quantile(p50, 1 / 3)), float(np.quantile(p50, 2 / 3))]
        cells = np.searchsorted(bounds, p50, side="right")
        counts = [(cells == i).sum() for i in range(3)]
        if min(counts) >= MIN_CELL_ROWS:
            offsets[stat] = {
                "type": "tertile", "bounds": [round(b, 4) for b in bounds],
                "cells": [dict(_cell_offsets(y_s[cells == i], pred_s[cells == i],
                                             stat, pairs), n=int(counts[i]))
                          for i in range(3)],
            }
        else:  # thin (or degenerate-p50) stats: one marginal cell
            offsets[stat] = {
                "type": "marginal",
                "cells": [dict(_cell_offsets(y_s, pred_s, stat, pairs),
                               n=int(valid.sum()))],
            }
    return offsets


def apply_offsets(pred_df: pd.DataFrame, offsets: dict,
                  stats: list[str]) -> pd.DataFrame:
    """Widen (or narrow) `{stat}_pXX` interval endpoints by the stored offsets.

    p50 is never moved. After shifting, each row's quantiles are re-sorted for
    non-crossing and clipped at zero — the same repair predict_position applies
    to the raw model output."""
    out = pred_df.copy()
    for stat in stats:
        spec = offsets.get(stat)
        if not spec:
            continue
        p50 = out[_q_col(stat, 0.50)].to_numpy(dtype=float)
        if spec["type"] == "tertile":
            cell_idx = np.searchsorted(spec["bounds"], p50, side="right")
        else:
            cell_idx = np.zeros(len(out), dtype=int)
        for pair_key in [k for k in spec["cells"][0] if k != "n"]:
            lo_s, hi_s = pair_key.split("-")
            lo, hi = int(lo_s) / 100, int(hi_s) / 100
            d_lo = np.array([spec["cells"][i][pair_key]["lo"] for i in cell_idx])
            d_up = np.array([spec["cells"][i][pair_key]["up"] for i in cell_idx])
            out[_q_col(stat, lo)] = out[_q_col(stat, lo)].to_numpy(dtype=float) - d_lo
            out[_q_col(stat, hi)] = out[_q_col(stat, hi)].to_numpy(dtype=float) + d_up
        qcols = [_q_col(stat, q) for q in QUANTILES]
        stacked = np.sort(np.clip(out[qcols].to_numpy(dtype=float), 0, None), axis=1)
        out[qcols] = stacked
    return out
