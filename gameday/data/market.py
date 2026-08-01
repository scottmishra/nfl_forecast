"""External draft-market sources for the draft board cross-reference.

Our season projection is a closed loop: it says who *we* like, but not whether
the room agrees. This module pulls three outside opinions and joins them onto
our `player_id` (= gsis_id) so /api/draft can show where we disagree with the
market — which is the only place a projection creates an edge.

Sources (all public, no auth):

  ESPN     lm-api-reads.fantasy.espn.com kona_player_info on the PPR league
           default. Gives a true ADP (`ownership.averageDraftPosition`), a PPR
           draft rank + auction value, and ESPN's own season projection.
  FFToday  the per-position projection tables (HTML). Gives component stat
           lines, which we re-score to PPR ourselves.
  Sleeper  the full player dump, for `search_rank`.

Two things worth knowing before reading further:

  * **Sleeper publishes no ADP.** `/players/nfl/adp/{season}`, `/v1/adp/nfl/...`
    and friends all 404. `search_rank` is a market-ordered draft-rank proxy —
    useful as a second opinion on *ordering*, but it is not an ADP and is never
    presented as one.
  * **FFToday's `FPts` column is standard scoring, not PPR.** Its 2026 line for
    Jahmyr Gibbs reads 295.4 with 72 receptions uncounted. We ignore that column
    entirely and recompute from the component stats with the same formula behind
    nflverse `fantasy_points_ppr`, so every projection on the board is on one
    scale.

Joining: ESPN and Sleeper carry stable IDs that the DynastyProcess crosswalk
maps to gsis_id, so those joins are exact. FFToday publishes no ID any crosswalk
knows, so it matches on normalized name + position. Anything that fails to match
is dropped rather than guessed, and the miss shows up in the coverage report —
a silently-broken scraper should look broken, not look like consensus.

Caching mirrors `gameday.data.releases`: a sidecar per file under
RAW_DIR/market/, TTL revalidation, stale-cache fallback on network errors, and a
module-level `_transport` hook so tests never touch the network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from gameday.config import (FFTODAY_PAGES, FFTODAY_POS_IDS, FORECASTS_DIR,
                            MARKET_TTL_HOURS, POSITIONS, RAW_DIR)
from gameday.data.teams import normalize_team

log = logging.getLogger(__name__)

# Test hook: set to an httpx.MockTransport to exercise fetch logic offline.
_transport: httpx.BaseTransport | None = None

CACHE_DIR = RAW_DIR / "market"
USER_AGENT = "gameday-forecaster/1.0 (+https://github.com/scottmishra/nfl_forecast)"

CROSSWALK_URL = ("https://raw.githubusercontent.com/dynastyprocess/data/master/"
                 "files/db_playerids.csv")
ESPN_URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/"
            "{season}/segments/0/leaguedefaults/3?view=kona_player_info")
ESPN_FILTER = {"players": {"limit": 400,
                           "sortDraftRanks": {"sortPriority": 1, "sortAsc": True,
                                              "value": "PPR"}}}
FFTODAY_URL = ("https://www.fftoday.com/rankings/playerproj.php"
               "?Season={season}&PosID={pos_id}&LeagueID=1&cur_page={page}")
SLEEPER_URL = "https://api.sleeper.app/v1/players/nfl"

SOURCE_URLS = {"espn": ESPN_URL, "fftoday": FFTODAY_URL,
               "sleeper": SLEEPER_URL, "crosswalk": CROSSWALK_URL}

# The crosswalk uses MFL team codes; FFToday uses JAC. Only used to break
# name-match ties, so a passthrough on anything unrecognized is fine.
_TEAM_FIX = {"GBP": "GB", "KCC": "KC", "LVR": "LV", "NEP": "NE", "NOS": "NO",
             "SFO": "SF", "TBB": "TB", "RAM": "LAR", "SDC": "LAC", "JAC": "JAX"}

# Name suffixes stripped before matching (the crosswalk's merge_name convention).
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _age_hours(meta: dict) -> float:
    try:
        return (_now() - datetime.fromisoformat(meta["fetched_at"])).total_seconds() / 3600.0
    except (KeyError, TypeError, ValueError):
        return float("inf")


def _cached_get(name: str, url: str, ttl_hours: float,
                headers: dict | None = None, force: bool = False) -> bytes | None:
    """Bytes for `url`, cached at RAW_DIR/market/{name} with a TTL.

    Inside the TTL the cache is served without a request. On any network or HTTP
    error a stale cache is served when one exists, else None — every market
    source is optional by construction, so a caller gets blanks, never an
    exception.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / name
    meta_path = CACHE_DIR / (name + ".meta.json")
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        meta = {}

    if cache.exists() and not force and _age_hours(meta) < ttl_hours:
        return cache.read_bytes()

    req_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    try:
        with httpx.Client(follow_redirects=True, timeout=120,
                          transport=_transport) as client:
            log.info("fetching %s", url)
            resp = client.get(url, headers=req_headers)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        if cache.exists():
            log.warning("market fetch failed for %s (%s); serving stale cache", name, exc)
            return cache.read_bytes()
        log.warning("market fetch failed for %s (%s); no cache to fall back on", name, exc)
        return None

    content = resp.content
    tmp = CACHE_DIR / (name + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(cache)
    meta_path.write_text(json.dumps({
        "url": url, "fetched_at": _now().isoformat(),
        "sha256": hashlib.sha256(content).hexdigest(), "status": resp.status_code,
    }, indent=1))
    return content


# --------------------------------------------------------------------------
# name normalization
# --------------------------------------------------------------------------

def merge_name(name: str) -> str:
    """Normalize a display name for cross-source matching.

    Lowercase, strip accents, drop punctuation and generational suffixes, and
    squeeze whitespace out entirely: "Amon-Ra St. Brown" -> "amonrastbrown",
    "Marvin Harrison Jr." -> "marvinharrison". Matches the convention behind the
    crosswalk's own merge_name column.
    """
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    parts = [p for p in text.split() if p not in _SUFFIXES]
    return "".join(parts)


def _fix_team(abbr) -> str | None:
    if not isinstance(abbr, str) or not abbr.strip():
        return None
    abbr = abbr.strip().upper()
    return _TEAM_FIX.get(abbr, normalize_team(abbr))


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def fetch_crosswalk(force: bool = False) -> pd.DataFrame:
    """DynastyProcess ID crosswalk: gsis_id <-> espn_id <-> sleeper_id."""
    raw = _cached_get("db_playerids.csv", CROSSWALK_URL,
                      MARKET_TTL_HOURS["crosswalk"], force=force)
    if raw is None:
        return pd.DataFrame()
    import io

    df = pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False)
    keep = ["gsis_id", "espn_id", "sleeper_id", "name", "position", "team"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df[df["gsis_id"].notna() & df["position"].isin(POSITIONS)]
    df["match_name"] = df["name"].map(merge_name)
    df["team"] = df["team"].map(_fix_team)
    # One row per gsis_id: the crosswalk carries a row per db_season.
    return df.drop_duplicates(subset=["gsis_id"], keep="first").reset_index(drop=True)


def _espn_season_projection(stats: list, season: int) -> float | None:
    """ESPN's projected season total from a player's `stats` array.

    Selected by (seasonId, statSourceId=1 projected, statSplitTypeId=0 season)
    rather than by position — the array order is not stable between responses.
    """
    for entry in stats or []:
        if (entry.get("seasonId") == season and entry.get("statSourceId") == 1
                and entry.get("statSplitTypeId") == 0):
            total = entry.get("appliedTotal")
            return float(total) if total is not None else None
    return None


def fetch_espn(season: int, force: bool = False) -> pd.DataFrame:
    """ESPN ADP, PPR draft rank, auction value, and season projection."""
    raw = _cached_get(f"espn_{season}.json", ESPN_URL.format(season=season),
                      MARKET_TTL_HOURS["espn"],
                      headers={"x-fantasy-filter": json.dumps(ESPN_FILTER)},
                      force=force)
    if raw is None:
        return pd.DataFrame()
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        log.warning("ESPN payload was not JSON (%s)", exc)
        return pd.DataFrame()

    rows = []
    for item in payload.get("players", []):
        p = item.get("player") or {}
        ranks = (p.get("draftRanksByRankType") or {}).get("PPR") or {}
        own = p.get("ownership") or {}
        adp = own.get("averageDraftPosition")
        rows.append({
            "espn_id": str(p.get("id")),
            "espn_name": p.get("fullName"),
            "espn_adp": float(adp) if adp else None,  # 0.0 means "undrafted"
            "espn_rank_ppr": ranks.get("rank") or None,
            "espn_auction": own.get("auctionValueAverage") or ranks.get("auctionValue") or None,
            "espn_proj_pts": _espn_season_projection(p.get("stats"), season),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.dropna(subset=["espn_id"]).drop_duplicates(subset=["espn_id"])


# Per-position FFToday column layouts, verified against the live 2026 tables.
# Cells 0-3 are always Chg / Player / Tm / Bye; the trailing FPts column is
# standard scoring and deliberately unmapped.
_FFT_COLUMNS: dict[str, dict[int, str]] = {
    "QB": {5: "pass_att", 6: "pass_yds", 7: "pass_td", 8: "interceptions",
           9: "rush_att", 10: "rush_yds", 11: "rush_td"},
    "RB": {4: "rush_att", 5: "rush_yds", 6: "rush_td",
           7: "rec", 8: "rec_yds", 9: "rec_td"},
    "WR": {4: "rec", 5: "rec_yds", 6: "rec_td",
           7: "rush_att", 8: "rush_yds", 9: "rush_td"},
    "TE": {4: "rec", 5: "rec_yds", 6: "rec_td"},
}
_FFT_ROW_RE = re.compile(
    r"<TR>\s*((?:<TD class=\"smallbody\".*?</TD>\s*)+)</TR>", re.S | re.I)
_FFT_CELL_RE = re.compile(r"<TD[^>]*>(.*?)</TD>", re.S | re.I)
_FFT_LINK_RE = re.compile(r"/stats/players/(\d+)/", re.I)


def ppr_points(row: dict) -> float:
    """Standard PPR scoring — the formula behind nflverse `fantasy_points_ppr`.

    Fumbles lost and two-point conversions are not published by FFToday, so
    those (small, mostly negative) terms are absent; the resulting total runs a
    touch high relative to a full-season PPR figure.
    """
    def g(key):
        return float(row.get(key) or 0.0)

    return (g("pass_yds") * 0.04 + g("pass_td") * 4 - g("interceptions") * 2
            + g("rush_yds") * 0.1 + g("rush_td") * 6
            + g("rec") * 1.0 + g("rec_yds") * 0.1 + g("rec_td") * 6)


def _parse_fftoday(html: str, position: str) -> list[dict]:
    """Rows from one FFToday projection page."""
    layout = _FFT_COLUMNS[position]
    out = []
    for block in _FFT_ROW_RE.findall(html):
        raw_cells = _FFT_CELL_RE.findall(block)
        cells = [re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").strip()
                 for c in raw_cells]
        if len(cells) <= max(layout):
            continue
        link = _FFT_LINK_RE.search(block)
        row = {
            "fftoday_id": link.group(1) if link else None,
            "fft_name": cells[1],
            "fft_team": _fix_team(cells[2]),
            "fft_bye": pd.to_numeric(cells[3], errors="coerce"),
            "position": position,
        }
        for idx, field in layout.items():
            row[field] = pd.to_numeric(cells[idx].replace(",", ""), errors="coerce")
        row["fft_proj_ppr"] = round(ppr_points(row), 1)
        out.append(row)
    return out


def fetch_fftoday(season: int, force: bool = False) -> pd.DataFrame:
    """FFToday projections re-scored to PPR, across all four positions."""
    rows = []
    for position, pos_id in FFTODAY_POS_IDS.items():
        for page in range(FFTODAY_PAGES):
            raw = _cached_get(
                f"fftoday_{season}_{position}_{page}.html",
                FFTODAY_URL.format(season=season, pos_id=pos_id, page=page),
                MARKET_TTL_HOURS["fftoday"], force=force)
            if raw is None:
                break
            page_rows = _parse_fftoday(raw.decode("utf-8", errors="replace"), position)
            rows.extend(page_rows)
            if len(page_rows) < 50:  # short page = last page
                break
    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("FFToday returned no parsable rows — layout may have changed")
        return df
    df["match_name"] = df["fft_name"].map(merge_name)
    keep = ["fftoday_id", "fft_name", "match_name", "position", "fft_team",
            "fft_bye", "fft_proj_ppr"]
    return df[keep].drop_duplicates(subset=["match_name", "position"], keep="first")


SLEEPER_UNRANKED = 999  # sentinel Sleeper parks undraftable players at


def fetch_sleeper(force: bool = False) -> pd.DataFrame:
    """Sleeper `search_rank` — a coarse draft-order proxy, NOT an ADP.

    Sleeper exposes no public ADP endpoint (`/players/nfl/adp/{season}`,
    `/v1/adp/nfl/{season}` and friends all 404). search_rank is the closest
    public signal, but it is a *search relevance* ordering and it is lumpy:
    ~40% of ranks are duplicated, and it ties across positions (Bijan Robinson
    and Josh Allen both rank 1). It is therefore a fallback ordering for players
    ESPN does not rank — never averaged into a consensus with a real ADP.

    Players parked at the 999 sentinel are dropped as unranked.
    """
    raw = _cached_get("sleeper_players.json", SLEEPER_URL,
                      MARKET_TTL_HOURS["sleeper"], force=force)
    if raw is None:
        return pd.DataFrame()
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        log.warning("Sleeper payload was not JSON (%s)", exc)
        return pd.DataFrame()

    rows = []
    for pid, p in (payload or {}).items():
        if not isinstance(p, dict) or p.get("position") not in POSITIONS:
            continue
        rank = p.get("search_rank")
        if rank is None:
            continue
        rows.append({"sleeper_id": str(pid), "sleeper_name": p.get("full_name"),
                     "sleeper_rank": int(rank)})
    df = pd.DataFrame(rows)
    return df[df["sleeper_rank"] < SLEEPER_UNRANKED] if not df.empty else df


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

MARKET_COLUMNS = ["player_id", "espn_adp", "espn_rank_ppr", "espn_auction",
                  "espn_proj_pts", "fft_proj_ppr", "fft_bye", "sleeper_rank"]


def build_market(season: int, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """All three sources joined onto gsis_id, plus per-source row counts.

    Returns (frame keyed by player_id, {source: rows_matched}). A source that
    failed to fetch contributes no columns and reports 0 — callers should treat
    every external column as optional.
    """
    cross = fetch_crosswalk(force=force)
    if cross.empty:
        log.warning("ID crosswalk unavailable — cannot map external sources to player_id")
        return pd.DataFrame(columns=MARKET_COLUMNS), {k: 0 for k in SOURCE_URLS}

    out = cross[["gsis_id", "match_name", "position", "team"]].rename(
        columns={"gsis_id": "player_id"}).copy()
    matched = {"crosswalk": len(out)}

    espn = fetch_espn(season, force=force)
    if not espn.empty:
        ids = cross[["gsis_id", "espn_id"]].dropna()
        espn = espn.merge(ids, on="espn_id", how="inner")
        out = out.merge(
            espn.drop(columns=["espn_id", "espn_name"]).rename(columns={"gsis_id": "player_id"}),
            on="player_id", how="left")
        matched["espn"] = int(espn["gsis_id"].nunique())
    else:
        matched["espn"] = 0

    fft = fetch_fftoday(season, force=force)
    if not fft.empty:
        # No ID crosswalk covers FFToday, so match on normalized name + position.
        joined = fft.merge(out[["player_id", "match_name", "position", "team"]],
                           on=["match_name", "position"], how="inner")
        # Same normalized name at the same position: keep the one whose team agrees.
        joined["team_match"] = (joined["fft_team"] == joined["team"]).astype(int)
        joined = (joined.sort_values("team_match", ascending=False)
                  .drop_duplicates(subset=["match_name", "position"], keep="first"))
        out = out.merge(joined[["player_id", "fft_proj_ppr", "fft_bye"]],
                        on="player_id", how="left")
        matched["fftoday"] = int(joined["player_id"].nunique())
    else:
        matched["fftoday"] = 0

    sleeper = fetch_sleeper(force=force)
    if not sleeper.empty:
        ids = cross[["gsis_id", "sleeper_id"]].dropna()
        sleeper = sleeper.merge(ids, on="sleeper_id", how="inner")
        out = out.merge(
            sleeper[["gsis_id", "sleeper_rank"]].rename(columns={"gsis_id": "player_id"}),
            on="player_id", how="left")
        matched["sleeper"] = int(sleeper["gsis_id"].nunique())
    else:
        matched["sleeper"] = 0

    for col in MARKET_COLUMNS:
        if col not in out.columns:
            out[col] = None
    # Only carry players some source actually has an opinion about.
    signal = ["espn_adp", "espn_rank_ppr", "espn_proj_pts", "fft_proj_ppr", "sleeper_rank"]
    out = out[out[signal].notna().any(axis=1)]
    return out[MARKET_COLUMNS].reset_index(drop=True), matched


def coverage(market: pd.DataFrame, pool: pd.DataFrame | None) -> dict:
    """Per-source match rate against the draft-board pool.

    {source: {matched, total, top100}} where `pool` is the season-projection
    frame (player_id + fantasy_points_p50) and top100 counts matches inside the
    100 highest-projecting players — the part of the board that gets drafted.
    """
    fields = {"espn": "espn_adp", "fftoday": "fft_proj_ppr", "sleeper": "sleeper_rank"}
    if pool is None or pool.empty or market.empty:
        return {name: {"matched": 0, "total": 0, "top100": 0} for name in fields}

    totals = (pool.groupby("player_id")["fantasy_points_p50"].sum()
              .sort_values(ascending=False))
    ids, top100 = set(totals.index), set(totals.head(100).index)
    have = market[market["player_id"].isin(ids)]
    return {name: {
        "matched": int(have[col].notna().sum()),
        "total": len(ids),
        "top100": int(have[have["player_id"].isin(top100)][col].notna().sum()),
    } for name, col in fields.items()}


# --------------------------------------------------------------------------
# artifact
# --------------------------------------------------------------------------

MARKET_PATH = FORECASTS_DIR / "latest_market.parquet"
MARKET_META_PATH = FORECASTS_DIR / "latest_market.json"


def _atomic_write(path: Path, write) -> None:
    """Same publish pattern as the forecast artifacts: write a pid-tagged temp
    file, then os.replace — a reader never sees a half-written file."""
    tmp = path.with_name(f".tmp-{os.getpid()}-{path.name}")
    write(tmp)
    os.replace(tmp, path)


def refresh_market(season: int, force: bool = False) -> dict:
    """Fetch every source and publish artifacts/forecasts/latest_market.parquet.

    Returns the meta dict that is also written alongside as latest_market.json.
    Raises only if *no* source produced a single row — a partial result is a
    normal outcome and is reported through `coverage`, not through an exception.
    """
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    market, matched = build_market(season, force=force)
    if market.empty:
        raise RuntimeError(
            "no market data: every source failed or nothing matched the crosswalk")

    pool = None
    season_path = FORECASTS_DIR / "latest_season.parquet"
    if season_path.exists():
        try:
            pool = pd.read_parquet(season_path, columns=["player_id", "fantasy_points_p50"])
        except Exception as exc:  # noqa: BLE001 — coverage is reporting, not data
            log.warning("could not read season artifact for coverage (%s)", exc)

    meta = {
        "fetched_at": _now().isoformat(timespec="seconds"),
        "season": season,
        "rows": len(market),
        "matched": matched,
        "coverage": coverage(market, pool),
        "source_urls": SOURCE_URLS,
        "notes": {
            "sleeper_rank": "Sleeper search_rank — a coarse, tie-heavy ordering, not an ADP; "
                            "used only as a fallback where ESPN has no ADP",
            "fft_proj_ppr": "recomputed from FFToday component stats (their FPts is standard scoring)",
            "coverage": "unmatched players in the top 100 are mostly backup QBs that no "
                        "outside source projects — a gap in our pool, not a broken join",
        },
    }
    _atomic_write(MARKET_PATH, lambda p: market.to_parquet(p, index=False))
    _atomic_write(MARKET_META_PATH, lambda p: p.write_text(json.dumps(meta, indent=2)))
    log.info("wrote market artifact: %d players (espn %d / fftoday %d / sleeper %d)",
             len(market), matched.get("espn", 0), matched.get("fftoday", 0),
             matched.get("sleeper", 0))
    return meta
