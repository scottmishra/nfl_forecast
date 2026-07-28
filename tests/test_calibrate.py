"""Conformal (CQR) calibration — offsets fix synthetic miscoverage."""

import json

import numpy as np
import pandas as pd
import pytest

from gameday.models.calibrate import apply_offsets, conformal_offsets


def _synthetic(n=4000, seed=11, hetero=False):
    """Under-covered synthetic forecasts: the true spread is wider than the
    predicted band, so raw p10–p90 coverage lands far below 0.80."""
    rng = np.random.default_rng(seed)
    p50 = rng.uniform(20, 80, n) if hetero else np.full(n, 50.0)
    scale = (2 + p50 / 10) if hetero else 10.0
    y = pd.DataFrame({"yards": p50 + rng.normal(0, scale, n)})
    pred = pd.DataFrame({
        "yards_p10": p50 - 5, "yards_p25": p50 - 2, "yards_p50": p50,
        "yards_p75": p50 + 2, "yards_p90": p50 + 5,
    })
    return y, pred


def _coverage(y, pred, lo, hi):
    return float(np.mean((y["yards"].values >= pred[f"yards_p{lo}"].values)
                         & (y["yards"].values <= pred[f"yards_p{hi}"].values)))


def test_offsets_restore_target_coverage():
    y, pred = _synthetic()
    assert _coverage(y, pred, 10, 90) < 0.45          # badly under-covered
    offsets = conformal_offsets(y, pred, ["yards"])
    calibrated = apply_offsets(pred, offsets, ["yards"])
    assert _coverage(y, calibrated, 10, 90) == pytest.approx(0.80, abs=0.03)
    assert _coverage(y, calibrated, 25, 75) == pytest.approx(0.50, abs=0.03)
    # constant p50 cannot support tertiles — falls back to marginal
    assert offsets["yards"]["type"] == "marginal"


def test_group_conditional_tertiles():
    """Heteroscedastic case: each p50-tertile gets its own width, and each
    tertile independently hits ~80% coverage."""
    y, pred = _synthetic(hetero=True)
    offsets = conformal_offsets(y, pred, ["yards"])
    assert offsets["yards"]["type"] == "tertile"
    calibrated = apply_offsets(pred, offsets, ["yards"])
    cells = np.searchsorted(offsets["yards"]["bounds"],
                            pred["yards_p50"].values, side="right")
    for i in range(3):
        m = cells == i
        cov = _coverage(y[m].reset_index(drop=True),
                        calibrated[m].reset_index(drop=True), 10, 90)
        assert cov == pytest.approx(0.80, abs=0.05)


def test_apply_keeps_quantiles_sane():
    y, pred = _synthetic()
    calibrated = apply_offsets(pred, conformal_offsets(y, pred, ["yards"]),
                               ["yards"])
    q = calibrated[["yards_p10", "yards_p25", "yards_p50", "yards_p75", "yards_p90"]].values
    assert (np.diff(q, axis=1) >= -1e-9).all()        # non-crossing
    assert (q >= 0).all()                             # clipped at zero
    # p50 untouched (it was already >= 0 and mid-band)
    assert np.allclose(calibrated["yards_p50"], pred["yards_p50"])


def test_offsets_narrow_overcovered_intervals():
    """Negative scores → negative offsets → the band tightens."""
    rng = np.random.default_rng(5)
    y = pd.DataFrame({"yards": 50 + rng.normal(0, 2.0, 3000)})
    pred = pd.DataFrame({
        "yards_p10": np.full(3000, 30.0), "yards_p25": np.full(3000, 45.0),
        "yards_p50": np.full(3000, 50.0), "yards_p75": np.full(3000, 55.0),
        "yards_p90": np.full(3000, 70.0),
    })
    offsets = conformal_offsets(y, pred, ["yards"])
    calibrated = apply_offsets(pred, offsets, ["yards"])
    width_raw = (pred["yards_p90"] - pred["yards_p10"]).mean()
    width_cal = (calibrated["yards_p90"] - calibrated["yards_p10"]).mean()
    assert width_cal < width_raw
    assert _coverage(y, calibrated, 10, 90) == pytest.approx(0.80, abs=0.03)


def test_offsets_json_serializable():
    y, pred = _synthetic(n=500)
    offsets = conformal_offsets(y, pred, ["yards"])
    round_trip = json.loads(json.dumps(offsets))
    assert round_trip["yards"]["cells"][0]["10-90"]["lo"] > 0


def test_gbm_engine_persists_and_applies_calibration(tmp_path):
    """train_position stores offsets in the manifest; predict_position applies
    them, so downstream eval always scores the calibrated band."""
    from gameday.config import POSITION_STATS
    from gameday.models import quantile_gbm as qg

    rng = np.random.default_rng(0)
    n = 1400
    df = pd.DataFrame({
        "season": np.where(np.arange(n) < 1000, 2023, 2024),
        "f1": rng.uniform(0, 1, n),
    })
    for stat in POSITION_STATS["WR"]:
        df[stat] = np.clip(10 * df["f1"] + rng.normal(0, 3, n), 0, None)

    qg.train_position(df, "WR", ["f1"], models_dir=tmp_path)
    manifest = json.loads((tmp_path / "manifest_WR.json").read_text())
    assert manifest["calibration"] and "receptions" in manifest["calibration"]
    pred = qg.predict_position(df[df["season"] == 2024], "WR", models_dir=tmp_path)
    for stat in POSITION_STATS["WR"]:
        q = pred[[f"{stat}_p10", f"{stat}_p25", f"{stat}_p50",
                  f"{stat}_p75", f"{stat}_p90"]].values
        assert (np.diff(q, axis=1) >= -1e-9).all() and (q >= 0).all()
