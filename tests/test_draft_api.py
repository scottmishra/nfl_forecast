"""/api/draft — the board itself and the ESPN/FFToday/Sleeper cross-reference.

The board predates the market layer and has to keep working without it, so the
no-artifact path is a regression guard, not an edge case.
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gameday.api import server
from gameday.api.server import app
from gameday.config import FORECASTS_DIR

client = TestClient(app)

SEASON = 2026
# vorp order ends up ace > deuce > trey > mule; the market disagrees on purpose.
PLAYERS = [
    # player_id, name, position, team, weekly p50
    ("00-0000001", "Ace Runner", "RB", "DET", 20.0),
    ("00-0000002", "Deuce Catcher", "WR", "LAR", 18.0),
    ("00-0000003", "Trey Blocker", "TE", "SEA", 12.0),
    ("00-0000004", "Mule Backup", "QB", "CHI", 10.0),
]


def _season_frame(first_week=1):
    rows = []
    for pid, name, pos, team, base in PLAYERS:
        for week in range(first_week, 19):
            if week == 7:  # a shared bye, so _draft_rows has one to find
                continue
            rows.append({
                "player_id": pid, "player_display_name": name, "position": pos,
                "team": team, "season": SEASON, "week": week,
                "opponent_team": "GB", "is_home": 1,
                "fantasy_points_p10": base - 6, "fantasy_points_p25": base - 3,
                "fantasy_points_p50": base, "fantasy_points_p75": base + 3,
                "fantasy_points_p90": base + 6,
            })
    return pd.DataFrame(rows)


def _market_frame():
    """Ace is a market darling; Trey is a bargain nobody else likes."""
    return pd.DataFrame([
        {"player_id": "00-0000001", "espn_adp": 1.5, "espn_rank_ppr": 1,
         "espn_auction": 60.0, "espn_proj_pts": 350.0, "fft_proj_ppr": 330.0,
         "fft_bye": 7, "sleeper_rank": 2},
        {"player_id": "00-0000002", "espn_adp": 2.5, "espn_rank_ppr": 2,
         "espn_auction": 55.0, "espn_proj_pts": 300.0, "fft_proj_ppr": 295.0,
         "fft_bye": 7, "sleeper_rank": 4},
        {"player_id": "00-0000003", "espn_adp": 90.0, "espn_rank_ppr": 90,
         "espn_auction": 3.0, "espn_proj_pts": 180.0, "fft_proj_ppr": 170.0,
         "fft_bye": 7, "sleeper_rank": 95},
        # Mule has only a Sleeper rank — the ESPN-less fallback path.
        {"player_id": "00-0000004", "espn_adp": None, "espn_rank_ppr": None,
         "espn_auction": None, "espn_proj_pts": None, "fft_proj_ppr": None,
         "fft_bye": None, "sleeper_rank": 150},
    ])


@pytest.fixture
def board(request):
    """Publish a season artifact (+ optionally a market one) for one test."""
    first_week = getattr(request, "param", {}).get("first_week", 1)
    with_market = getattr(request, "param", {}).get("market", True)
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    season_path = FORECASTS_DIR / "latest_season.parquet"
    market_path = FORECASTS_DIR / "latest_market.parquet"
    meta_path = FORECASTS_DIR / "latest_market.json"
    _season_frame(first_week).to_parquet(season_path, index=False)
    if with_market:
        _market_frame().to_parquet(market_path, index=False)
        meta_path.write_text(json.dumps({
            "fetched_at": "2026-08-01T12:00:00+00:00", "season": SEASON,
            "coverage": {"espn": {"matched": 3, "total": 4, "top100": 3},
                         "fftoday": {"matched": 3, "total": 4, "top100": 3},
                         "sleeper": {"matched": 4, "total": 4, "top100": 4}},
        }))
    else:
        market_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    server._mtime_cache.clear()
    yield
    for path in (season_path, market_path, meta_path):
        path.unlink(missing_ok=True)
    server._mtime_cache.clear()


def get(**params):
    r = client.get("/api/draft", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def by_name(data):
    return {p["name"]: p for p in data["players"]}


# --------------------------------------------------------------------------
# the board without any market data — the pre-existing contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("board", [{"market": False}], indirect=True)
def test_board_renders_without_the_market_artifact(board):
    data = get()
    assert data["market"]["available"] is False
    assert data["market"]["coverage"] == {}
    ace = by_name(data)["Ace Runner"]
    # every pre-existing field is untouched...
    assert ace["total_p50"] == pytest.approx(20.0 * 17)
    assert ace["bye"] == 7 and ace["games"] == 17 and "vorp" in ace
    # ...and the market fields are present but empty, never missing.
    assert ace["espn_adp"] is None and ace["value"] is None
    assert ace["value_tier"] == "unranked" and ace["proj_spread"] is None


# --------------------------------------------------------------------------
# with market data
# --------------------------------------------------------------------------

def test_external_projections_and_adp_attach(board):
    ace = by_name(get())["Ace Runner"]
    assert ace["espn_adp"] == pytest.approx(1.5)
    assert ace["espn_proj_pts"] == pytest.approx(350.0)
    assert ace["fft_proj_ppr"] == pytest.approx(330.0)
    assert ace["sleeper_rank"] == 2
    assert ace["market_source"] == "espn_adp"


def test_value_is_market_rank_minus_our_rank(board):
    players = by_name(get())
    ace, trey = players["Ace Runner"], players["Trey Blocker"]
    assert (ace["our_rank"], ace["market_rank"], ace["value"]) == (1, 1, 0)
    # We have Trey 3rd; the market has him 3rd too by rank, but ADP 90 vs our
    # 3rd means the ordering agrees — value stays small and untiered.
    assert trey["market_rank"] == 3 and trey["value_tier"] in ("", "sleeper")


def test_sleeper_rank_only_fills_in_where_espn_has_no_adp(board):
    mule = by_name(get())["Mule Backup"]
    assert mule["espn_adp"] is None
    assert mule["market_source"] == "sleeper_rank"
    assert mule["market_rank"] == 4  # ranked after every ESPN-ranked player


def test_projection_spread_across_the_three_sources(board):
    ace = by_name(get())["Ace Runner"]
    assert ace["proj_sources"] == 3
    assert ace["proj_min"] == pytest.approx(330.0)
    assert ace["proj_max"] == pytest.approx(350.0)   # ours is 340
    assert ace["proj_spread"] == pytest.approx(20.0)
    assert ace["proj_spread_pct"] == pytest.approx(20.0 / 340.0, abs=1e-3)


def test_spread_needs_two_sources(board):
    mule = by_name(get())["Mule Backup"]
    assert mule["proj_sources"] == 1 and mule["proj_spread"] is None


def test_coverage_is_reported_from_the_sidecar(board):
    market = get()["market"]
    assert market["available"] is True
    assert market["coverage"]["fftoday"]["top100"] == 3
    assert market["fetched_at"].startswith("2026-08-01")
    assert market["ranked_players"] == 4


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------

def test_position_filter_does_not_shift_value(board):
    everyone = by_name(get())["Deuce Catcher"]
    just_wr = by_name(get(position="WR"))["Deuce Catcher"]
    assert just_wr["value"] == everyone["value"]
    assert just_wr["our_rank"] == everyone["our_rank"]
    assert just_wr["market_rank"] == everyone["market_rank"]


def test_tier_flags_stop_at_the_draftable_window(board, monkeypatch):
    """Deep in the pool both ranks are noise, so no flag should fire there."""
    monkeypatch.setattr(server, "DRAFT_POOL_SIZE", 2)
    players = by_name(get())
    # Mule sits at our #4 / market #4 — outside a 2-player draftable window.
    assert players["Mule Backup"]["value_tier"] == ""
    assert players["Mule Backup"]["value"] is not None  # the number still shows
    # Ace is #1 on both boards, comfortably inside it.
    assert players["Ace Runner"]["value"] == 0


def test_tier_filter_returns_only_that_tier(board):
    data = get(tier="sleeper")
    assert all(p["value_tier"] == "sleeper" for p in data["players"])
    assert data["tier"] == "sleeper"


def test_unknown_tier_and_position_are_rejected(board):
    assert client.get("/api/draft", params={"tier": "bargain"}).status_code == 404
    assert client.get("/api/draft", params={"position": "K"}).status_code == 404


# --------------------------------------------------------------------------
# mid-season guard
# --------------------------------------------------------------------------

@pytest.mark.parametrize("board", [{"first_week": 5}], indirect=True)
def test_spread_is_suppressed_mid_season_but_value_survives(board):
    data = get()
    assert data["market"]["partial_season"] is True
    ace = by_name(data)["Ace Runner"]
    # Our total now covers weeks 5-18 only; comparing it to full-season
    # projections would be nonsense, so no spread is offered at all.
    assert ace["proj_spread"] is None and ace["proj_sources"] == 0
    assert ace["espn_adp"] == pytest.approx(1.5)  # ADP is still season-agnostic
    assert ace["value"] is not None


def test_missing_season_artifact_returns_404():
    server._mtime_cache.clear()
    (FORECASTS_DIR / "latest_season.parquet").unlink(missing_ok=True)
    assert client.get("/api/draft").status_code == 404
