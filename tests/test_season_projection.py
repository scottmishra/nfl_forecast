"""Season projection composition + the /api/draft endpoint."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gameday.api.server import app
from gameday.config import FORECASTS_DIR
from gameday.season_projection import compose_weeks

client = TestClient(app)


def _future(team, opp, players):
    """Minimal next-week scaffold rows for one team (week 1 vs opp)."""
    return pd.DataFrame([{
        "player_id": pid, "player_display_name": name, "position": pos,
        "team": team, "season": 2026, "week": 1, "opponent_team": opp,
        "is_home": 1, "home_team": team, "game_id": f"2026_01_{opp}_{team}",
        "rest": 7.0, "travel_km": 0.0, "altitude_m": 100, "indoor": 0,
        "temp_c": 15.0, "wind_kph": 8.0, "fantasy_points": np.nan,
        "team_pts_r8": 24.0, "def_pts_allowed_r8": 20.0 if opp == "KC" else 27.0,
        "def_vs_pos_r8": 15.0 if opp == "KC" else 22.0,
    } for pid, name, pos in players])


@pytest.fixture
def frames():
    future = pd.concat([
        _future("SEA", "KC", [("p1", "QB One", "QB"), ("p2", "RB Two", "RB")]),
        _future("KC", "SEA", [("p3", "WR Three", "WR")]),
    ], ignore_index=True)
    games = pd.DataFrame([
        # week 1 (the base slate), then wk2 rematch flipped, wk3 = SEA bye, KC plays DEN
        dict(game_id="2026_01_KC_SEA", season=2026, week=1, home_team="SEA",
             away_team="KC", home_score=np.nan, home_rest=7, away_rest=7, roof="outdoors"),
        dict(game_id="2026_02_SEA_KC", season=2026, week=2, home_team="KC",
             away_team="SEA", home_score=np.nan, home_rest=6, away_rest=8, roof="outdoors"),
        dict(game_id="2026_03_DEN_KC", season=2026, week=3, home_team="KC",
             away_team="DEN", home_score=np.nan, home_rest=7, away_rest=7, roof="outdoors"),
    ])
    return future, games


def test_compose_weeks_swaps_context(frames):
    future, games = frames
    out = compose_weeks(future, games, max_week=18)

    # Base week passes through untouched.
    wk1 = out[out["week"] == 1]
    assert len(wk1) == 3 and (wk1["game_id"].str.contains("2026_01")).all()

    # Week 2: SEA players away at KC, opponent context remapped to KC's frozen values.
    sea2 = out[(out["week"] == 2) & (out["team"] == "SEA")]
    assert len(sea2) == 2
    assert (sea2["opponent_team"] == "KC").all() and (sea2["is_home"] == 0).all()
    assert (sea2["def_pts_allowed_r8"] == 20.0).all()
    assert (sea2["rest"] == 8.0).all()  # away_rest from the schedule row

    # Week 3: SEA on bye -> no rows; KC plays DEN (context frozen for an
    # opponent with no lookup entry -> NaN, deferred to manifest fills).
    assert out[(out["week"] == 3) & (out["team"] == "SEA")].empty
    kc3 = out[(out["week"] == 3) & (out["team"] == "KC")]
    assert len(kc3) == 1 and kc3["opponent_team"].iloc[0] == "DEN"
    assert kc3["def_pts_allowed_r8"].isna().all()

    # Frozen player state must not leak edits across weeks (copies, not views).
    assert (out[out["week"] == 1]["is_home"] == 1).all()


def test_compose_weeks_empty_future(frames):
    _, games = frames
    out = compose_weeks(pd.DataFrame(), games)
    assert out.empty


@pytest.fixture
def season_artifact():
    """A small latest_season.parquet: 2 QBs x 2 weeks, one with a wk3 bye."""
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid, team, base in (("q1", "SEA", 20.0), ("q2", "KC", 15.0)):
        for wk in (1, 2):
            rows.append(dict(
                player_id=pid, player_display_name=f"QB {pid}", position="QB",
                team=team, season=2026, week=wk, opponent_team="X", is_home=1,
                fantasy_points_p10=base - 8, fantasy_points_p25=base - 4,
                fantasy_points_p50=base, fantasy_points_p75=base + 4,
                fantasy_points_p90=base + 8))
    path = FORECASTS_DIR / "latest_season.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    yield path
    path.unlink(missing_ok=True)


def test_draft_endpoint(season_artifact):
    r = client.get("/api/draft?position=QB")
    assert r.status_code == 200
    body = r.json()
    assert body["position"] == "QB" and body["season"] == 2026
    players = body["players"]
    assert [p["player_id"] for p in players] == ["q1", "q2"]  # ranked by VORP
    top = players[0]
    assert top["total_p50"] == 40.0
    assert top["total_floor"] == 32.0 and top["total_ceiling"] == 48.0
    assert top["weeks"] == {"1": 20.0, "2": 20.0}
    # Only 2 QBs -> replacement falls back to the last-ranked player (30.0).
    assert body["replacement"]["QB"] == 30.0
    assert top["vorp"] == 10.0 and players[1]["vorp"] == 0.0


def test_draft_endpoint_unknown_position(season_artifact):
    assert client.get("/api/draft?position=K").status_code == 404


def test_draft_endpoint_missing_artifact():
    (FORECASTS_DIR / "latest_season.parquet").unlink(missing_ok=True)
    r = client.get("/api/draft")
    assert r.status_code == 404
    assert "refresh" in r.json()["detail"]
