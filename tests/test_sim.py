import json

import numpy as np
import pytest

from gameday.config import FORECASTS_DIR
from gameday.data import demo
from gameday.sim.calibrate import build_profiles
from gameday.sim.run import simulate_matchup, simulate_slate


@pytest.fixture(scope="module")
def league():
    return demo.generate(seasons=[2024, 2025], weeks=8, seed=3)


@pytest.fixture(scope="module")
def profiles(league):
    return build_profiles(*league)


@pytest.fixture(scope="module")
def matchup(profiles):
    return simulate_matchup(profiles["PHI"], profiles["DAL"], n_sims=200, seed=11)


def test_profiles_cover_all_teams(profiles):
    assert len(profiles) == 32
    p = profiles["PHI"]
    assert 50 <= p.plays_per_game <= 85
    assert 0.35 <= p.pass_rate <= 0.75
    assert any(x.position == "QB" for x in p.players)
    assert abs(sum(x.target_share for x in p.players) - 1.0) < 1e-6


def test_deterministic_under_seed(profiles):
    a = simulate_matchup(profiles["GB"], profiles["CHI"], n_sims=50, seed=5)
    b = simulate_matchup(profiles["GB"], profiles["CHI"], n_sims=50, seed=5)
    assert a == b


def test_probabilities_sum(matchup):
    total = matchup["home_win_prob"] + matchup["away_win_prob"] + matchup["tie_prob"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_realistic_game_shape(matchup):
    for team_agg in matchup["teams"].values():
        assert 45 <= team_agg["plays_mean"] <= 85
        assert 0 <= team_agg["points"]["p10"] <= team_agg["points"]["p50"] <= team_agg["points"]["p90"] <= 70
        assert team_agg["pass_plays_mean"] + team_agg["run_plays_mean"] == pytest.approx(
            team_agg["plays_mean"], abs=0.01)
        # third down should be more pass-heavy than first down
        rates = team_agg["pass_rate_by_down"]
        assert rates["3"] > rates["1"]
        # drive outcomes are shares of drives
        assert 0.9 <= sum(team_agg["drive_outcomes"].values()) <= 1.05


def test_snap_counts_bounded(matchup):
    for team_agg in matchup["teams"].values():
        for p in team_agg["players"]:
            assert p["snaps_mean"] <= team_agg["plays_mean"] + 1e-9
            assert p["snaps_p10"] <= p["snaps_mean"] <= p["snaps_p90"]
        qb = next(p for p in team_agg["players"] if p["position"] == "QB")
        assert qb["snap_share"] == pytest.approx(1.0, abs=0.01)


def test_simulate_slate_writes_artifact(league):
    player_weeks, games = league
    sims = simulate_slate(player_weeks, games, n_sims=25, seed=2)
    upcoming = games[games["home_score"].isna()]
    assert set(sims) == set(upcoming["game_id"])
    on_disk = json.loads((FORECASTS_DIR / "latest_sims.json").read_text())
    assert set(on_disk) == set(sims)
