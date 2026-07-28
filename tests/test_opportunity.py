"""Usage/opportunity features — shares, availability, priors, leakage rules."""

import numpy as np
import pandas as pd
import pytest

from gameday.features.build import feature_columns
from gameday.features.opportunity import (USAGE_SERIES, add_absence_history,
                                          add_availability, add_team_stint,
                                          apply_usage_prior,
                                          compute_usage_shares, usage_prior)
from gameday.features.temporal import add_temporal, temporal_feature_names


def _pw():
    return pd.DataFrame({
        "player_id": ["a", "a", "b", "b"],
        "season": [2024] * 4, "week": [1, 2, 1, 2],
        "team": ["KC", "KC", "KC", "KC"],
        "carries": [10.0, 8.0, 5.0, 12.0],
        "targets": [2.0, 4.0, 6.0, 3.0],
    })


def test_compute_usage_shares_math():
    snaps = pd.DataFrame({
        "player_id": ["a", "b"], "season": [2024, 2024], "week": [1, 1],
        "offense_pct": [0.9, 0.5],
    })
    team_weeks = pd.DataFrame({
        "team": ["KC", "KC"], "season": [2024, 2024], "week": [1, 2],
        "carries": [20.0, 20.0], "targets": [40.0, 30.0],
    })
    out = compute_usage_shares(_pw(), snaps, team_weeks)
    a1 = out[(out["player_id"] == "a") & (out["week"] == 1)].iloc[0]
    assert a1["snap_pct"] == 0.9
    assert a1["carry_share"] == pytest.approx(10 / 20)
    assert a1["target_share_team"] == pytest.approx(2 / 40)
    # week-2 snap missing -> NaN, shares still computed from team totals
    a2 = out[(out["player_id"] == "a") & (out["week"] == 2)].iloc[0]
    assert np.isnan(a2["snap_pct"]) and a2["carry_share"] == pytest.approx(8 / 20)


def test_compute_usage_shares_fallback_totals():
    """Without stats_team, team volume is reconstructed from player rows."""
    snaps = pd.DataFrame(columns=["player_id", "season", "week", "offense_pct"])
    out = compute_usage_shares(_pw(), snaps, None)
    a1 = out[(out["player_id"] == "a") & (out["week"] == 1)].iloc[0]
    assert a1["carry_share"] == pytest.approx(10 / 15)  # 10 of (10 + 5)


def test_availability_onehots_and_promotion():
    df = pd.DataFrame({
        "player_id": ["a"] * 3, "season": [2024] * 3, "week": [1, 2, 3],
        "team": ["KC"] * 3,
    })
    injuries = pd.DataFrame({
        "player_id": ["a"], "season": [2024], "week": [2],
        "report_status": ["Questionable"],
        "practice_status": ["Limited Participation in Practice"],
    })
    depth = pd.DataFrame({
        "player_id": ["a"] * 3, "season": [2024] * 3, "week": [1, 2, 3],
        "team": ["KC"] * 3, "position": ["RB"] * 3, "depth_rank": [2, 2, 1],
    })
    out = add_availability(df, injuries, depth)
    assert out.loc[1, "inj_questionable"] == 1 and out.loc[1, "practice_limited"] == 1
    assert out.loc[0, "inj_questionable"] == 0      # no report row = healthy
    assert out.loc[2, "is_depth_promotion"] == 1    # rank 2 -> 1
    assert out.loc[2, "depth_rank_change"] == 1
    assert out.loc[1, "is_depth_promotion"] == 0


def test_absence_history_counts_team_games():
    games = pd.DataFrame({
        "season": [2024] * 4, "week": [1, 2, 3, 4],
        "home_team": ["KC"] * 4, "away_team": ["BUF"] * 4,
        "home_score": [20] * 4, "away_score": [10] * 4,
    })
    df = pd.DataFrame({  # player misses weeks 2-3, returns week 4
        "player_id": ["a"] * 2, "season": [2024] * 2, "week": [1, 4],
        "team": ["KC"] * 2,
    })
    out = add_absence_history(df, games)
    assert out.loc[1, "missed_recent"] == 2
    assert out.loc[1, "games_missed_last8"] == 2
    assert out.loc[1, "weeks_since_return"] == 1    # first game back
    assert np.isnan(out.loc[0, "weeks_since_return"])  # never missed yet


def test_prior_shrinkage_and_team_reset():
    train = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(10)],
        "position": ["RB"] * 10, "depth_rank": [1] * 10,
        "is_rookie": [0] * 10, "draft_number": [50] * 10,
        "snap_pct": [0.8] * 10, "carry_share": [0.5] * 10,
        "target_share_team": [0.1] * 10,
    })
    table = usage_prior(train, USAGE_SERIES)
    assert table["targets"]["snap_pct"]["cells"]["RB|1|vet"] == pytest.approx(0.8)

    df = pd.DataFrame({
        "player_id": ["x", "x", "x"], "position": ["RB"] * 3,
        "team": ["KC", "KC", "BUF"], "season": [2024] * 3, "week": [1, 2, 3],
        "depth_rank": [1] * 3, "is_rookie": [0] * 3, "draft_number": [50] * 3,
        "snap_pct_ewm5": [np.nan, 0.4, 0.4],
    })
    df = add_team_stint(df)
    assert list(df["n_with_team"]) == [0, 1, 0]     # trade resets the count
    out = apply_usage_prior(df, table, ["snap_pct"])
    assert out.loc[0, "snap_pct_est"] == pytest.approx(0.8)   # no history: prior
    w1 = 1 / (1 + 4.0)
    assert out.loc[1, "snap_pct_est"] == pytest.approx(w1 * 0.4 + (1 - w1) * 0.8)
    assert out.loc[2, "snap_pct_est"] == pytest.approx(0.8)   # reset -> w = 0


def test_v3_features_never_include_sameweek_actuals():
    """Raw same-week usage (snap_pct, carry_share, target_share_team) must
    never be a stat-model input — only shifted forms and forecasts."""
    cols = {c: [0.0] for c in USAGE_SERIES}
    cols.update({c: [0.0] for c in temporal_feature_names(USAGE_SERIES)})
    cols.update({"pred_snap_pct_p50": [0.0], "snap_pct_prior": [0.0]})
    df = pd.DataFrame(cols)
    for pos in ("QB", "RB", "WR", "TE"):
        v3 = feature_columns(pos, df, feature_set="v3")
        assert not set(USAGE_SERIES) & set(v3)
        assert "snap_pct_ewm2" in v3
