"""The draft-market cross-reference: scoring, parsing, matching, and fallbacks.

All network traffic goes through httpx.MockTransport via market._transport; no
test touches the real network. CACHE_DIR is remapped per-test to tmp_path.

Fixture payloads are trimmed copies of the real 2026 responses, so the shapes
under test are the shapes the live sources actually return.
"""

import json

import httpx
import pandas as pd
import pytest

from gameday.data import market

SEASON = 2026


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

CROSSWALK_CSV = (
    "gsis_id,espn_id,sleeper_id,name,position,team,db_season\n"
    "00-0038542,4429795,9221,Jahmyr Gibbs,RB,DET,2026\n"
    "00-0039337,4426515,9493,Puka Nacua,WR,LAR,2026\n"
    "00-0036963,4241389,7547,Amon-Ra St. Brown,WR,DET,2026\n"
    "00-0039910,4432708,11565,Marvin Harrison Jr.,WR,ARI,2026\n"
    "00-0038120,4373678,8155,Kenneth Walker III,RB,SEA,2026\n"
    "00-0031234,3051392,1234,Backup Quarterback,QB,CHI,2026\n"
)

# One ESPN player with the full stats array, so the (season, 1, 0) selector is
# genuinely exercised rather than accidentally satisfied by array position.
ESPN_JSON = json.dumps({"players": [
    {"player": {
        "id": 4429795, "fullName": "Jahmyr Gibbs",
        "draftRanksByRankType": {"PPR": {"rank": 1, "auctionValue": 57},
                                 "STANDARD": {"rank": 1, "auctionValue": 57}},
        "ownership": {"averageDraftPosition": 1.72, "auctionValueAverage": 63.76},
        "stats": [
            {"seasonId": 2026, "scoringPeriodId": 13, "statSourceId": 1,
             "statSplitTypeId": 1, "appliedTotal": 22.0},      # weekly projection
            {"seasonId": 2025, "scoringPeriodId": 0, "statSourceId": 1,
             "statSplitTypeId": 0, "appliedTotal": 317.3},     # last season's projection
            {"seasonId": 2026, "scoringPeriodId": 0, "statSourceId": 0,
             "statSplitTypeId": 0, "appliedTotal": 0.0},       # actuals, not played yet
            {"seasonId": 2026, "scoringPeriodId": 0, "statSourceId": 1,
             "statSplitTypeId": 0, "appliedTotal": 365.5},     # <- the one we want
        ]}},
    {"player": {
        "id": 4426515, "fullName": "Puka Nacua",
        "draftRanksByRankType": {"PPR": {"rank": 3, "auctionValue": 52}},
        "ownership": {"averageDraftPosition": 3.67, "auctionValueAverage": 58.11},
        "stats": [{"seasonId": 2026, "scoringPeriodId": 0, "statSourceId": 1,
                   "statSplitTypeId": 0, "appliedTotal": 356.6}]}},
]}).encode()


def fft_row(name, player_id, team, bye, cells, shade="#ffffff"):
    tds = "".join(f'<TD class="smallbody" ALIGN="center" BGCOLOR="{shade}">{c}</TD>'
                  for c in cells)
    return (f'<TR>\n<TD class="smallbody" ALIGN="center" BGCOLOR="{shade}">&nbsp;</TD>\n'
            f'<TD class="smallbody" ALIGN="LEFT" BGCOLOR="{shade}">&nbsp;'
            f'<A HREF="/stats/players/{player_id}/{name.replace(" ", "_")}?LeagueID=1">'
            f'{name}</A>  </TD>\n'
            f'<TD class="smallbody" ALIGN="center" BGCOLOR="{shade}">{team}</TD>\n'
            f'<TD class="smallbody" ALIGN="center" BGCOLOR="{shade}">{bye}</TD>\n'
            f"{tds}\n</TR>")


# Real 2026 lines. Gibbs: 279/1422/12 rush, 72/572/4 rec, FFToday FPts 295.4
# (standard scoring — PPR should come out ~72 points higher).
FFT_RB = ("<html><body><table>"
          + fft_row("Jahmyr Gibbs", 18522, "DET", 6,
                    ["279", "1,422", "12", "72", "572", "4", "295.4"])
          + fft_row("Kenneth Walker III", 19001, "SEA", 8,
                    ["220", "950", "8", "40", "300", "1", "180.0"])
          + "</table></body></html>").encode()
FFT_WR = ("<html><body><table>"
          + fft_row("Puka Nacua", 18100, "LAR", 11,
                    ["105", "1,400", "7", "5", "40", "0", "225.4"])
          + fft_row("Amon-Ra St. Brown", 17500, "DET", 6,
                    ["100", "1,200", "9", "2", "10", "0", "205.2"])
          + fft_row("Marvin Harrison Jr.", 18900, "ARI", 9,
                    ["90", "1,150", "8", "0", "0", "0", "191.0"])
          + "</table></body></html>").encode()
FFT_QB = ("<html><body><table>"
          + fft_row("Josh Allen", 12345, "BUF", 7,
                    ["326", "479", "3,787", "26", "9", "113", "567", "12", "422.1"])
          + "</table></body></html>").encode()
FFT_EMPTY = b"<html><body><table></table></body></html>"

SLEEPER_JSON = json.dumps({
    "9221": {"full_name": "Jahmyr Gibbs", "position": "RB", "search_rank": 2},
    "9493": {"full_name": "Puka Nacua", "position": "WR", "search_rank": 5},
    "7547": {"full_name": "Amon-Ra St. Brown", "position": "WR", "search_rank": 8},
    "11565": {"full_name": "Marvin Harrison Jr.", "position": "WR", "search_rank": 40},
    "1234": {"full_name": "Backup Quarterback", "position": "QB", "search_rank": 999},
    "5555": {"full_name": "Some Kicker", "position": "K", "search_rank": 300},
    "6666": {"full_name": "No Rank Guy", "position": "WR", "search_rank": None},
}).encode()


class Origin:
    """Scriptable origin: routes by URL substring and records every request."""

    def __init__(self, **overrides):
        self.status = dict(overrides.pop("status", {}))
        self.requests = []
        self.bodies = {
            "db_playerids.csv": CROSSWALK_CSV.encode(),
            "lm-api-reads": ESPN_JSON,
            "PosID=20": FFT_RB,
            "PosID=30": FFT_WR,
            "PosID=10": FFT_QB,
            "PosID=40": FFT_EMPTY,
            "sleeper": SLEEPER_JSON,
        }
        self.bodies.update(overrides)

    def __call__(self, request):
        url = str(request.url)
        self.requests.append(url)
        for key, code in self.status.items():
            if key in url:
                return httpx.Response(code)
        # Only the first FFToday page carries rows; page 2+ ends the loop.
        if "cur_page=" in url and "cur_page=0" not in url:
            return httpx.Response(200, content=FFT_EMPTY)
        for key, body in self.bodies.items():
            if key in url:
                return httpx.Response(200, content=body)
        return httpx.Response(404)


@pytest.fixture
def origin(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "CACHE_DIR", tmp_path / "market")
    monkeypatch.setattr(market, "FORECASTS_DIR", tmp_path / "forecasts")
    monkeypatch.setattr(market, "MARKET_PATH", tmp_path / "forecasts" / "latest_market.parquet")
    monkeypatch.setattr(market, "MARKET_META_PATH", tmp_path / "forecasts" / "latest_market.json")
    server = Origin()
    monkeypatch.setattr(market, "_transport", httpx.MockTransport(server))
    return server


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def test_ppr_recompute_ignores_fftodays_standard_scoring(origin):
    """FFToday's own FPts column is standard scoring; PPR must add receptions."""
    fft = market.fetch_fftoday(SEASON)
    gibbs = fft[fft["fft_name"] == "Jahmyr Gibbs"].iloc[0]
    # 1422*.1 + 12*6 + 72*1 + 572*.1 + 4*6
    assert gibbs["fft_proj_ppr"] == pytest.approx(367.4)
    assert gibbs["fft_proj_ppr"] != pytest.approx(295.4)  # the published FPts
    assert gibbs["fft_bye"] == 6


def test_qb_layout_scores_passing_at_four_points_per_td(origin):
    """QB columns shift by one (Cmp first) and FFToday pays 6/pass TD — ours is 4."""
    fft = market.fetch_fftoday(SEASON)
    allen = fft[fft["fft_name"] == "Josh Allen"].iloc[0]
    # 3787*.04 + 26*4 - 9*2 + 567*.1 + 12*6
    assert allen["fft_proj_ppr"] == pytest.approx(366.2)


def test_thousands_separators_parse(origin):
    fft = market.fetch_fftoday(SEASON)
    assert fft["fft_proj_ppr"].notna().all()


# --------------------------------------------------------------------------
# ESPN
# --------------------------------------------------------------------------

def test_espn_season_projection_selected_by_filter_not_position(origin):
    espn = market.fetch_espn(SEASON)
    gibbs = espn[espn["espn_id"] == "4429795"].iloc[0]
    assert gibbs["espn_proj_pts"] == pytest.approx(365.5)  # not 317.3 (2025) or 22.0 (weekly)
    assert gibbs["espn_adp"] == pytest.approx(1.72)
    assert gibbs["espn_rank_ppr"] == 1


def test_espn_request_carries_the_ppr_draft_rank_filter(origin):
    market.fetch_espn(SEASON)
    assert any("lm-api-reads" in u and str(SEASON) in u for u in origin.requests)


# --------------------------------------------------------------------------
# Sleeper
# --------------------------------------------------------------------------

def test_sleeper_drops_the_unranked_sentinel_and_other_positions(origin):
    sleeper = market.fetch_sleeper()
    names = set(sleeper["sleeper_name"])
    assert "Jahmyr Gibbs" in names
    assert "Backup Quarterback" not in names  # search_rank 999 sentinel
    assert "Some Kicker" not in names         # not a modeled position
    assert "No Rank Guy" not in names         # no search_rank at all


# --------------------------------------------------------------------------
# name matching
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Marvin Harrison Jr.", "marvinharrison"),
    ("Kenneth Walker III", "kennethwalker"),
    ("Amon-Ra St. Brown", "amonrastbrown"),
    ("D'Andre Swift", "dandreswift"),
    ("Michael Pittman Jr", "michaelpittman"),
])
def test_merge_name_strips_suffixes_and_punctuation(raw, expected):
    assert market.merge_name(raw) == expected


def test_fftoday_matches_players_with_suffixes_and_punctuation(origin):
    built, _ = market.build_market(SEASON)
    matched = built[built["fft_proj_ppr"].notna()]["player_id"]
    # Harrison Jr., Walker III, and St. Brown all join despite the name noise.
    assert {"00-0039910", "00-0038120", "00-0036963"} <= set(matched)


# --------------------------------------------------------------------------
# assembly and degradation
# --------------------------------------------------------------------------

def test_build_market_joins_all_three_sources_onto_gsis_id(origin):
    built, matched = market.build_market(SEASON)
    gibbs = built[built["player_id"] == "00-0038542"].iloc[0]
    assert gibbs["espn_adp"] == pytest.approx(1.72)
    assert gibbs["espn_proj_pts"] == pytest.approx(365.5)
    assert gibbs["fft_proj_ppr"] == pytest.approx(367.4)
    assert gibbs["sleeper_rank"] == 2
    assert matched["espn"] == 2 and matched["sleeper"] == 4


def test_players_with_no_external_signal_are_dropped(origin):
    """The backup QB is in the crosswalk but no source ranks or projects him."""
    built, _ = market.build_market(SEASON)
    assert "00-0031234" not in set(built["player_id"])


def test_one_failing_source_leaves_the_others_intact(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "CACHE_DIR", tmp_path / "market")
    server = Origin(status={"lm-api-reads": 500})
    monkeypatch.setattr(market, "_transport", httpx.MockTransport(server))

    built, matched = market.build_market(SEASON)
    assert matched["espn"] == 0
    assert built["espn_adp"].isna().all()
    assert built["fft_proj_ppr"].notna().any()   # FFToday unaffected
    assert built["sleeper_rank"].notna().any()   # Sleeper unaffected


def test_missing_crosswalk_yields_an_empty_frame_not_an_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "CACHE_DIR", tmp_path / "market")
    server = Origin(status={"db_playerids": 500})
    monkeypatch.setattr(market, "_transport", httpx.MockTransport(server))

    built, matched = market.build_market(SEASON)
    assert built.empty and matched["espn"] == 0


def test_fftoday_layout_change_degrades_to_blanks(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "CACHE_DIR", tmp_path / "market")
    server = Origin(**{"PosID=20": b"<html>totally different markup</html>",
                       "PosID=30": b"<html>totally different markup</html>",
                       "PosID=10": b"<html>totally different markup</html>"})
    monkeypatch.setattr(market, "_transport", httpx.MockTransport(server))

    built, matched = market.build_market(SEASON)
    assert matched["fftoday"] == 0
    assert built["fft_proj_ppr"].isna().all()
    assert built["espn_adp"].notna().any()  # the board still gets ADP


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------

def test_second_call_inside_the_ttl_issues_no_request(origin):
    market.fetch_sleeper()
    first = len(origin.requests)
    market.fetch_sleeper()
    assert len(origin.requests) == first


def test_force_bypasses_the_ttl(origin):
    market.fetch_sleeper()
    first = len(origin.requests)
    market.fetch_sleeper(force=True)
    assert len(origin.requests) == first + 1


def test_stale_cache_is_served_when_the_network_fails(origin, monkeypatch):
    market.fetch_sleeper()
    meta_path = market.CACHE_DIR / "sleeper_players.json.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["fetched_at"] = (market._now() - pd.Timedelta(hours=48)).isoformat()
    meta_path.write_text(json.dumps(meta))

    monkeypatch.setattr(market, "_transport",
                        httpx.MockTransport(Origin(status={"sleeper": 500})))
    assert not market.fetch_sleeper().empty


# --------------------------------------------------------------------------
# coverage + artifact
# --------------------------------------------------------------------------

def test_coverage_counts_matches_against_the_board_pool(origin):
    built, _ = market.build_market(SEASON)
    pool = pd.DataFrame({
        "player_id": ["00-0038542", "00-0039337", "00-0031234"],
        "fantasy_points_p50": [300.0, 280.0, 200.0],
    })
    cov = market.coverage(built, pool)
    assert cov["espn"]["total"] == 3
    assert cov["espn"]["matched"] == 2         # the backup QB has no ADP
    assert cov["espn"]["top100"] == 2
    assert cov["sleeper"]["matched"] == 2


def test_refresh_market_writes_both_artifacts_atomically(origin):
    meta = market.refresh_market(SEASON)
    assert market.MARKET_PATH.exists() and market.MARKET_META_PATH.exists()
    assert json.loads(market.MARKET_META_PATH.read_text())["season"] == SEASON
    assert meta["rows"] == len(pd.read_parquet(market.MARKET_PATH))
    assert not list(market.MARKET_PATH.parent.glob(".tmp-*"))


def test_refresh_market_raises_only_when_everything_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "CACHE_DIR", tmp_path / "market")
    monkeypatch.setattr(market, "FORECASTS_DIR", tmp_path / "forecasts")
    server = Origin(status={"db_playerids": 500})
    monkeypatch.setattr(market, "_transport", httpx.MockTransport(server))
    with pytest.raises(RuntimeError, match="no market data"):
        market.refresh_market(SEASON)
