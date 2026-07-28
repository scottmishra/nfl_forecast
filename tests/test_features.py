import numpy as np
import pandas as pd
import pytest

from gameday.data import demo
from gameday.features.build import build_features, feature_columns


@pytest.fixture(scope="module")
def league():
    return demo.generate(seasons=[2024, 2025], weeks=6, seed=3)


@pytest.fixture(scope="module")
def feats(league):
    player_weeks, games = league
    return build_features(player_weeks, games)


def test_demo_shapes(league):
    player_weeks, games = league
    assert set(player_weeks["position"]) == {"QB", "RB", "WR", "TE"}
    assert len(games) == 2 * 6 * 16
    upcoming = games[games["home_score"].isna()]
    assert len(upcoming) == 16  # final week is the slate


def test_features_no_leakage(feats, league):
    """Rolling form must exclude the current game: a player's first game has
    NaN form, and week-2 form equals exactly the week-1 stat."""
    player_weeks, _ = league
    p = player_weeks.sort_values(["season", "week"]).iloc[0]["player_id"]
    rows = feats[feats["player_id"] == p].sort_values(["season", "week"])
    first, second = rows.iloc[0], rows.iloc[1]
    assert np.isnan(first["fantasy_points_r3"])
    assert second["fantasy_points_r3"] == pytest.approx(first["fantasy_points"])


def test_upcoming_rows_exist_with_context(feats):
    future = feats[feats["fantasy_points"].isna()]
    assert len(future) > 100  # rosters for 32 teams
    # upcoming rows carry game context and pre-game form
    assert future["game_id"].notna().all()
    assert future["is_home"].notna().all()
    assert future["fantasy_points_r8"].notna().mean() > 0.9


def test_feature_columns_exist(feats):
    for pos in ("QB", "RB", "WR", "TE"):
        cols = feature_columns(pos, feats)
        assert "def_vs_pos_r8" in cols and "wind_kph" in cols
        assert all(c in feats.columns for c in cols)


def test_dome_weather_neutralized(feats):
    dome_rows = feats[feats["indoor"] == 1]
    assert (dome_rows["wind_kph"] == 0).all()
    assert (dome_rows["temp_c"] == 21.0).all()


def test_prior_season_rank_vintage_safe(feats, league):
    """pos_rank_prev is the PREVIOUS season's positional fantasy finish: the
    first demo season has no prior, and 2025 rows carry exactly the 2024
    rank — never influenced by 2025 results."""
    player_weeks, _ = league
    assert feats[feats["season"] == 2024]["pos_rank_prev"].isna().all()
    totals = (player_weeks[player_weeks["season"] == 2024]
              .groupby(["position", "player_id"])["fantasy_points"].sum())
    qb_ranks = totals.loc["QB"].rank(ascending=False, method="min")
    pid = qb_ranks.index[0]
    row = feats[(feats["player_id"] == pid) & (feats["season"] == 2025)].iloc[0]
    assert row["pos_rank_prev"] == qb_ranks.loc[pid]


def test_prior_season_rank_only_in_v2(feats):
    assert "pos_rank_prev" in feature_columns("QB", feats)
    assert "pos_rank_prev" not in feature_columns("QB", feats, feature_set="v1")
