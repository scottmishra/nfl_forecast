"""The draft agent's data surface — offline, no SDK, no model, no network.

These are the tests that matter most: the tools decide what the agent believes,
and a wrong number here becomes a confident wrong answer at the draft table.
Artifacts are synthesized the same way `tests/test_draft_api.py` does it.
"""

import asyncio
import json

import pandas as pd
import pytest

from gameday.agent import data
from gameday.agent.data import ToolError
from gameday.api import server
from gameday.config import FORECASTS_DIR

SEASON = 2026
PLAYERS = [
    # id, name, pos, team, weekly p50
    ("00-0000001", "Ace Runner", "RB", "DET", 20.0),
    ("00-0000002", "Marvin Harrison Jr.", "WR", "ARI", 18.0),
    ("00-0000003", "Amon-Ra St. Brown", "WR", "DET", 12.0),
    ("00-0000004", "Mule Backup", "QB", "CHI", 10.0),
]


def _season_frame():
    rows = []
    for pid, name, pos, team, base in PLAYERS:
        for week in range(1, 19):
            if week == 7:  # shared bye
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
    return pd.DataFrame([
        {"player_id": "00-0000001", "espn_adp": 1.5, "espn_rank_ppr": 1,
         "espn_auction": 60.0, "espn_proj_pts": 350.0, "fft_proj_ppr": 330.0,
         "fft_bye": 7, "sleeper_rank": 2},
        {"player_id": "00-0000002", "espn_adp": 40.0, "espn_rank_ppr": 40,
         "espn_auction": 12.0, "espn_proj_pts": 250.0, "fft_proj_ppr": 240.0,
         "fft_bye": 7, "sleeper_rank": 44},
        {"player_id": "00-0000003", "espn_adp": 12.0, "espn_rank_ppr": 12,
         "espn_auction": 30.0, "espn_proj_pts": 210.0, "fft_proj_ppr": 205.0,
         "fft_bye": 7, "sleeper_rank": 15},
    ])


def _usage_frame():
    return pd.DataFrame([
        {"player_id": "00-0000001", "season": 2025, "week": w,
         "targets": 4 + w, "carries": 12 + w, "snap_pct": 0.60 + w / 100,
         "target_share": 0.10 + w / 100, "carry_share": 0.5}
        for w in (10, 11, 12)
    ])


def _replay_frame():
    rows = []
    for week, actual in ((1, 22.0), (2, 15.0), (3, None)):
        rows.append({
            "player_id": "00-0000001", "player_display_name": "Ace Runner",
            "position": "RB", "team": "DET", "opponent_team": "GB",
            "is_home": 1, "season": 2024, "week": week, "game_id": f"g{week}",
            "fantasy_points": actual,
            "fantasy_points_p10": 8.0, "fantasy_points_p25": 14.0,
            "fantasy_points_p50": 18.0, "fantasy_points_p75": 23.0,
            "fantasy_points_p90": 28.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def artifacts(request, tmp_path, monkeypatch):
    """Publish synthetic artifacts; drop the ones a test asks to omit.

    REPLAY_DIR is redirected to a per-test directory: the backtest tests persist
    real replay seasons into the shared artifacts root, and "no replay at all"
    is a case these tests need to be able to construct.
    """
    omit = set(getattr(request, "param", ()) or ())
    monkeypatch.setattr(server, "REPLAY_DIR", tmp_path / "replay")
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    season_p = FORECASTS_DIR / "latest_season.parquet"
    market_p = FORECASTS_DIR / "latest_market.parquet"
    usage_p = FORECASTS_DIR / "latest_usage.parquet"
    replay_dir = server.REPLAY_DIR / "2024"

    _season_frame().to_parquet(season_p, index=False)
    if "market" not in omit:
        _market_frame().to_parquet(market_p, index=False)
    else:
        market_p.unlink(missing_ok=True)
    if "usage" not in omit:
        _usage_frame().to_parquet(usage_p, index=False)
    else:
        usage_p.unlink(missing_ok=True)
    if "replay" not in omit:
        replay_dir.mkdir(parents=True, exist_ok=True)
        _replay_frame().to_parquet(replay_dir / "players.parquet", index=False)
        (replay_dir / "scorecard.json").write_text(json.dumps({
            "adjusted": {"overall": {"n": 100, "mae_p50": 4.5, "skill_vs_naive": 0.02,
                                     "coverage80": 0.79}}}))

    server._mtime_cache.clear()
    yield
    for path in (season_p, market_p, usage_p):
        path.unlink(missing_ok=True)
    for path in (replay_dir / "players.parquet", replay_dir / "scorecard.json"):
        path.unlink(missing_ok=True)
    server._mtime_cache.clear()


# --------------------------------------------------------------------------
# search and resolution
# --------------------------------------------------------------------------

def test_search_matches_partial_names(artifacts):
    assert [m["name"] for m in data.search("Ace")] == ["Ace Runner"]
    assert [m["name"] for m in data.search("runner")] == ["Ace Runner"]


@pytest.mark.parametrize("query", ["Marvin Harrison Jr.", "marvin harrison",
                                   "Harrison", "MARVIN HARRISON JR"])
def test_search_ignores_suffixes_and_punctuation(artifacts, query):
    """The name the user types is rarely the name in the data."""
    assert "Marvin Harrison Jr." in [m["name"] for m in data.search(query)]


def test_search_ranks_exact_match_first(artifacts):
    # "brown" is a substring of "Amon-Ra St. Brown" and nothing else here,
    # but an exact hit must always outrank a partial.
    hits = data.search("Amon-Ra St. Brown")
    assert hits[0]["match"] == "exact"


def test_search_unknown_returns_empty_not_error(artifacts):
    assert data.search("Nobody At All") == []


def test_resolve_accepts_a_player_id(artifacts):
    assert data.resolve("00-0000001")["name"] == "Ace Runner"


def test_resolve_unknown_names_the_fix(artifacts):
    with pytest.raises(ToolError, match="search_players"):
        data.resolve("Nobody At All")


def test_resolve_refuses_to_guess_between_candidates(artifacts):
    """Two players share a name fragment — picking one silently is how a
    draft-day answer goes wrong, so this must raise with the candidates."""
    with pytest.raises(ToolError, match="ambiguous"):
        data.resolve("r")  # substring of several names


# --------------------------------------------------------------------------
# the board and player detail
# --------------------------------------------------------------------------

def test_board_carries_projection_and_market_fields(artifacts):
    players = data.board(limit=5)["players"]
    ace = next(p for p in players if p["name"] == "Ace Runner")
    assert ace["total_p50"] == pytest.approx(20.0 * 17)  # 17 weeks, week 7 bye
    assert ace["bye"] == 7 and ace["games"] == 17
    for field in ("vorp", "our_rank", "value", "value_tier",
                  "espn_adp", "proj_spread"):
        assert field in ace


def test_board_is_sorted_by_vorp_not_raw_points(artifacts):
    """VORP over the positional replacement is what makes cross-position
    comparison valid — it is the ordering the agent reasons about."""
    players = data.board(limit=99)["players"]
    vorps = [p["vorp"] for p in players]
    assert vorps == sorted(vorps, reverse=True)
    # Harrison outscores St. Brown by 102 at the same position, and WR is the
    # only position here with two players, so he sets the top of the board even
    # though Ace Runner projects for more raw points.
    assert players[0]["name"] == "Marvin Harrison Jr."
    assert players[0]["vorp"] > next(
        p for p in players if p["name"] == "Ace Runner")["vorp"]


def test_board_position_filter(artifacts):
    wrs = data.board(position="WR")["players"]
    assert {p["position"] for p in wrs} == {"WR"}


def test_player_detail_matches_the_board_entry(artifacts):
    """A number the agent quotes must be the number the dashboard renders."""
    from_board = next(p for p in data.board(limit=99)["players"]
                      if p["name"] == "Ace Runner")
    assert data.player("Ace Runner") == from_board


def test_compare_requires_two_to_four(artifacts):
    with pytest.raises(ToolError, match="at least two"):
        data.compare(["Ace Runner"])
    with pytest.raises(ToolError, match="at most"):
        data.compare(["Ace Runner"] * 5)


def test_compare_returns_one_row_per_player(artifacts):
    out = data.compare(["Ace Runner", "Amon-Ra St. Brown"])
    assert [p["name"] for p in out["players"]] == ["Ace Runner", "Amon-Ra St. Brown"]
    assert all("vorp" in p and "espn_adp" in p for p in out["players"])


# --------------------------------------------------------------------------
# usage, track record, replay
# --------------------------------------------------------------------------

def test_usage_returns_weeks_oldest_first(artifacts):
    out = data.usage("Ace Runner")
    assert [w["week"] for w in out["weeks"]] == [10, 11, 12]
    assert "snap_pct" in out["weeks"][0]


def test_track_record_exposes_skill_and_coverage(artifacts):
    out = data.track_record()
    assert 2024 in out["seasons_available"]
    assert out["scorecards"]["2024"]["overall"]["skill_vs_naive"] == 0.02


def test_player_replay_scores_only_played_weeks(artifacts):
    out = data.player_history("Ace Runner", 2024)
    assert out["summary"]["weeks_scored"] == 2       # week 3 has no actual
    assert out["summary"]["in_p25_p75"] == 2         # 22.0 and 15.0 both inside
    # actual minus projected: (22-18) + (15-18) = +1 over two weeks
    assert out["summary"]["mean_error"] == pytest.approx(0.5)


def test_player_replay_unknown_season_is_an_error(artifacts):
    with pytest.raises(ToolError):
        data.player_history("Ace Runner", 1999)


# --------------------------------------------------------------------------
# degradation — a missing artifact must be a message, not a stack trace
# --------------------------------------------------------------------------

@pytest.mark.parametrize("artifacts", [("market",)], indirect=True)
def test_board_still_works_without_market_data(artifacts):
    top = data.board(limit=3)["players"][0]
    assert top["espn_adp"] is None
    assert top["value_tier"] == "unranked"


@pytest.mark.parametrize("artifacts", [("usage",)], indirect=True)
def test_missing_usage_names_the_player(artifacts):
    with pytest.raises(ToolError, match="Ace Runner"):
        data.usage("Ace Runner")


@pytest.mark.parametrize("artifacts", [("replay",)], indirect=True)
def test_missing_replay_suggests_the_command(artifacts):
    with pytest.raises(ToolError, match="gameday replay"):
        data.track_record()


def test_data_status_reports_freshness_fields(artifacts):
    status = data.data_status()
    assert status["board"]["season"] == SEASON
    assert "market" in status and "available" in status["market"]
    assert "note" in status  # the agent needs to know what a skip means


# --------------------------------------------------------------------------
# the MCP wrappers (import the SDK; skipped when the extra isn't installed)
# --------------------------------------------------------------------------

sdk = pytest.importorskip("claude_agent_sdk", reason="needs the [agent] extra")


def _call(tool, **kwargs):
    return asyncio.run(tool.handler(kwargs))


def test_every_tool_is_declared_read_only():
    """Read-only annotations are what let the agent batch lookups, and what
    make bypassPermissions defensible — the surface has no write path."""
    from gameday.agent import tools

    assert tools.TOOLS, "no tools registered"
    for tool in tools.TOOLS:
        assert tool.annotations is not None and tool.annotations.readOnlyHint


def test_tool_names_are_namespaced():
    from gameday.agent import tools

    assert all(n.startswith("mcp__gameday__") for n in tools.TOOL_NAMES)
    assert len(tools.TOOL_NAMES) == len(tools.TOOLS)


def test_tool_returns_parsable_json(artifacts):
    from gameday.agent import tools

    result = _call(tools.get_player, player="Ace Runner")
    assert not result.get("is_error")
    assert json.loads(result["content"][0]["text"])["name"] == "Ace Runner"


def test_tool_failure_is_flagged_not_raised(artifacts):
    """A bad lookup must come back as a readable tool result — an exception
    would end the agent's turn instead of letting it try another route."""
    from gameday.agent import tools

    result = _call(tools.get_player, player="Nobody At All")
    assert result["is_error"] is True
    assert "search_players" in result["content"][0]["text"]
