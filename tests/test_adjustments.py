"""Season-to-season player adjustments + historical-replay scorecard."""

import numpy as np
import pandas as pd
import pytest

from gameday import backtest as bt
from gameday.data import demo
from gameday.features.build import ADJUSTMENT_COLS, build_features, feature_columns


@pytest.fixture(scope="module")
def league():
    return demo.generate(seasons=[2024, 2025], weeks=6, seed=3)


def test_adjustment_columns_neutral_without_rosters(league):
    """Demo ids don't match nflverse rosters, so the adjustment columns exist
    but are neutral constants (no signal), keeping demo/tests deterministic."""
    player_weeks, games = league
    feats = build_features(player_weeks, games)  # rosters=None
    for c in ADJUSTMENT_COLS:
        assert c in feats.columns
        assert feats[c].notna().all()
    assert feats["is_rookie"].nunique() == 1
    assert feats["is_new_team"].nunique() == 1


def test_feature_columns_toggle(league):
    player_weeks, games = league
    feats = build_features(player_weeks, games)
    with_adj = feature_columns("WR", feats, include_adjustments=True)
    without = feature_columns("WR", feats, include_adjustments=False)
    assert set(ADJUSTMENT_COLS).issubset(with_adj)
    assert not set(ADJUSTMENT_COLS).intersection(without)
    assert set(without).issubset(set(with_adj))


def test_roster_join_populates_real_features():
    """A rosters frame drives years_exp / age / is_rookie / is_new_team by the
    gsis_id == player_id + season join; undrafted → sentinel draft number."""
    pw = pd.DataFrame({
        "player_id": ["00-1", "00-1", "00-2"],
        "player_display_name": ["A", "A", "B"],
        "position": ["WR", "WR", "RB"],
        "team": ["KC", "BUF", "KC"],
        "opponent_team": ["BUF", "KC", "BUF"],
        "season": [2023, 2024, 2024], "week": [1, 1, 1],
        "fantasy_points": [10.0, 12.0, 8.0],
    })
    games = pd.DataFrame({
        "game_id": ["2023_01_BUF_KC", "2024_01_BUF_KC"],
        "season": [2023, 2024], "week": [1, 1],
        "home_team": ["KC", "KC"], "away_team": ["BUF", "BUF"],
        "home_score": [20, 21], "away_score": [17, 18],
        "home_rest": [7, 7], "away_rest": [7, 7],
        "roof": ["outdoor", "outdoor"], "temp": [15, 15], "wind": [8, 8],
        "gameday": ["2023-09-07", "2024-09-05"], "gametime": ["20:00", "20:00"],
    })
    rosters = pd.DataFrame({
        "gsis_id": ["00-1", "00-1", "00-2"],
        "season": [2023, 2024, 2024],
        "team": ["KC", "BUF", "KC"],
        "years_exp": [3, 4, 0],
        "birth_date": pd.to_datetime(["1998-01-01", "1998-01-01", "2002-01-01"]),
        "rookie_year": [2020, 2020, 2024],
        "draft_number": [50, 50, np.nan],
    })
    feats = build_features(pw, games, rosters=rosters)
    a24 = feats[(feats["player_id"] == "00-1") & (feats["season"] == 2024)].iloc[0]
    assert a24["years_exp"] == 4
    assert a24["is_new_team"] == 1          # KC -> BUF between seasons
    assert a24["age"] == 2024 - 1998
    b = feats[feats["player_id"] == "00-2"].iloc[0]
    assert b["is_rookie"] == 1              # years_exp 0
    assert b["is_new_team"] == 0            # no prior season
    assert b["draft_number"] == 260         # undrafted sentinel


@pytest.fixture(scope="module")
def replay_report():
    return bt.run_backtest(source="demo", seasons=[2024, 2025], weeks=[6],
                           n_sims=0, compare=True, persist_replay=True)


def test_scorecard_has_baseline_and_segments(replay_report):
    sc = replay_report["scorecard"]
    assert "adjusted" in sc and "baseline" in sc
    assert sc["adjusted"]["overall"]["n"] > 0
    for group in ("phase", "experience", "team"):
        assert group in sc["adjusted"]["segments"]


def test_scorecard_named_variants(replay_report):
    """Variant A/B: both named variants are scored, the legacy adjusted/
    baseline keys alias the primary/comparison, and the new interval metrics
    are present."""
    sc = replay_report["scorecard"]
    assert sc["primary"] == "v2"
    assert set(sc["variants"]) == {"v2", "v1"}
    assert sc["adjusted"] == sc["variants"]["v2"]
    assert sc["baseline"] == sc["variants"]["v1"]
    for card in sc["variants"].values():
        assert {"coverage80", "coverage50", "interval_width80", "pinball"} \
            <= set(card["overall"])


def test_variants_parquet_written(replay_report):
    df = pd.read_parquet(bt.REPLAY_DIR / "2025" / "variants.parquet")
    assert set(df["variant"]) == {"v2", "v1"}
    assert set(df["engine"]) == {"gbm"}


def test_players_parquet_stays_single_variant(replay_report):
    """The dashboard contract: one row per player-week (primary variant)."""
    df = pd.read_parquet(bt.REPLAY_DIR / "2025" / "players.parquet")
    assert set(df["variant"]) == {"v2"}
    assert not df.duplicated(["player_id", "season", "week"]).any()


def test_aggregate_reports(replay_report):
    """n-weighted multi-season combination reproduces a repeated season."""
    agg = bt.aggregate_reports([replay_report, replay_report])
    single = replay_report["scorecard"]["variants"]["v2"]["overall"]
    combined = agg["scorecard"]["variants"]["v2"]["overall"]
    assert combined["n"] == 2 * single["n"]
    assert combined["mae_p50"] == pytest.approx(single["mae_p50"], abs=1e-6)
    text = bt.format_aggregate(agg, [replay_report, replay_report])
    assert "MULTI-SEASON" in text and "ALL" in text


def test_replay_artifacts_written(replay_report):
    sdir = bt.REPLAY_DIR / "2025"
    assert (sdir / "players.parquet").exists()
    assert (sdir / "scorecard.json").exists()
    df = pd.read_parquet(sdir / "players.parquet")
    for c in ("player_id", "week", "fantasy_points", "fantasy_points_p50",
              "fantasy_points_r8", "is_rookie", "is_new_team"):
        assert c in df.columns
    assert len(df) > 0


def test_format_scorecard_renders(replay_report):
    text = bt.format_scorecard(replay_report)
    assert "SCORECARD" in text and "rookie" in text
