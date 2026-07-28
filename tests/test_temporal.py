"""Temporal feature stack — leakage rules, values, and feature-set selection."""

import numpy as np
import pandas as pd
import pytest

from gameday.features.build import feature_columns
from gameday.features.temporal import add_temporal, temporal_feature_names


def _frame(vals, pid="p1"):
    return pd.DataFrame({
        "player_id": [pid] * len(vals),
        "season": [2024] * len(vals),
        "week": list(range(1, len(vals) + 1)),
        "yards": vals,
    })


def test_no_leakage_from_own_row():
    """A row's ewm/lag/slope/vol features must not change when that row's own
    stat value changes — they may only see strictly-prior games."""
    base = _frame([10.0, 20.0, 30.0, 40.0, 50.0])
    tampered = base.copy()
    tampered.loc[3, "yards"] = 999.0
    fb = add_temporal(base, ["yards"])
    ft = add_temporal(tampered, ["yards"])
    new_cols = [c for c in fb.columns if c not in base.columns]
    pd.testing.assert_series_equal(fb.loc[3, new_cols], ft.loc[3, new_cols])
    # ...while the NEXT game does see the tampered value.
    assert ft.loc[4, "yards_ewm2"] != fb.loc[4, "yards_ewm2"]
    assert ft.loc[4, "yards_lag1"] == 999.0


def test_temporal_values():
    df = add_temporal(_frame([10.0, 20.0, 30.0]), ["yards"])
    assert np.isnan(df.loc[0, "yards_ewm2"])          # no history yet
    assert df.loc[1, "yards_ewm2"] == 10.0            # one prior game
    assert df.loc[2, "yards_lag1"] == 20.0 and df.loc[2, "yards_lag2"] == 10.0
    assert np.isnan(df.loc[1, "yards_vol8"])          # std needs 2 priors
    expected = pd.Series([10.0, 20.0]).ewm(halflife=2, min_periods=1).mean().iloc[-1]
    assert df.loc[2, "yards_ewm2"] == pytest.approx(expected)
    assert df.loc[2, "yards_vol8"] == pytest.approx(np.std([10.0, 20.0], ddof=1))
    assert df.loc[2, "yards_slope"] == pytest.approx(
        df.loc[2, "yards_ewm2"] - df.loc[2, "yards_ewm10"])


def test_players_are_isolated():
    """The shift/ewm never crosses a player boundary."""
    both = pd.concat([_frame([10.0, 20.0, 30.0], "a"),
                      _frame([100.0, 200.0, 300.0], "b")], ignore_index=True)
    df = add_temporal(both, ["yards"])
    assert np.isnan(df.loc[3, "yards_ewm5"])          # b's first game
    assert np.isnan(df.loc[3, "yards_lag1"])          # not a's last stat
    assert df.loc[4, "yards_lag1"] == 100.0


def test_missing_columns_skipped():
    """Wish-list columns absent from the frame (demo has no shares) are a no-op."""
    df = add_temporal(_frame([1.0, 2.0]), ["yards", "target_share"])
    assert "yards_ewm2" in df.columns
    assert not any(c.startswith("target_share") for c in df.columns)


def test_feature_set_versions():
    """v1 = pre-temporal columns; v2 adds the temporal families on top."""
    feats = add_temporal(_frame([5.0] * 6).rename(columns={"yards": "receptions"}),
                         ["receptions"])
    v1 = feature_columns("WR", feats, feature_set="v1")
    v2 = feature_columns("WR", feats, feature_set="v2")
    assert set(v1) <= set(v2)
    assert not any("_ewm" in c or "_lag" in c for c in v1)
    assert "receptions_ewm2" in v2 and "receptions_lag1" in v2


def test_share_features_gated_by_position():
    """Usage-share families feed pass-catchers, never the QB model."""
    names = temporal_feature_names(["target_share", "wopr", "racr", "air_yards_share"])
    df = pd.DataFrame({c: [0.0] for c in names})
    assert not set(names) & set(feature_columns("QB", df))
    assert set(feature_columns("WR", df)) >= set(names)
    assert set(feature_columns("RB", df)) >= set(names)
