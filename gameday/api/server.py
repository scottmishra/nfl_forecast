"""FastAPI backend: serves the forecast artifacts and the dashboard.

Endpoints:
  GET /api/slate                upcoming games with venue/weather + team meta
  GET /api/game/{game_id}       both teams' player forecasts for one game
  GET /api/players?q=           fuzzy player search across the slate
  GET /                         the dashboard (static SPA under web/)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from gameday import bundle
from gameday.config import (ARTIFACTS_DIR, DRAFT_POOL_SIZE, FORECASTS_DIR,
                            MODELS_ROOT, POSITION_STATS, QUANTILES,
                            REPLACEMENT_RANK, ROOT, SEASON_MAX_WEEK,
                            VALUE_ROUND_SIZE)
from gameday.data.teams import TEAMS

app = FastAPI(title="Gameday Forecaster", version="0.1.0")

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"
REPLAY_DIR = ARTIFACTS_DIR / "replay"
DEMO_DIR = FORECASTS_DIR / "demo"  # committed fixtures: a fresh clone serves these


def _clean(obj):
    """Recursively convert numpy scalars to native Python and NaN -> None for JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):  # numpy int/float/bool scalar -> python scalar
        obj = obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


# --------------------------------------------------------------------------
# artifact loading — cached per file-mtime so a nightly refresh (which swaps
# artifacts atomically via os.replace) is picked up on the next request with
# no server restart. stat() per request costs microseconds.
# --------------------------------------------------------------------------

_mtime_cache: dict[str, tuple[tuple, object]] = {}


def _mtime_cached(key: str, paths: list[Path], loader):
    stamp = tuple(p.stat().st_mtime_ns if p.exists() else None for p in paths)
    hit = _mtime_cache.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    value = loader()
    _mtime_cache[key] = (stamp, value)
    return value


def _forecast_paths() -> tuple[Path, Path]:
    """Live artifacts when present, else the committed demo fixtures."""
    fpath = FORECASTS_DIR / "latest_forecasts.parquet"
    spath = FORECASTS_DIR / "latest_slate.parquet"
    if fpath.exists() and spath.exists():
        return fpath, spath
    return DEMO_DIR / "latest_forecasts.parquet", DEMO_DIR / "latest_slate.parquet"


def _load():
    fpath, spath = _forecast_paths()
    if not fpath.exists() or not spath.exists():
        raise HTTPException(
            status_code=503,
            detail="No forecasts yet — run `gameday demo` or `gameday forecast` first.",
        )
    return _mtime_cached("forecasts", [fpath, spath],
                         lambda: (pd.read_parquet(fpath), pd.read_parquet(spath)))


def _team_meta(abbr: str) -> dict:
    t = TEAMS.get(abbr, {})
    return {
        "abbr": abbr, "name": t.get("name", abbr), "primary": t.get("primary", "#444"),
        "secondary": t.get("secondary", "#888"), "stadium": t.get("stadium"),
        "city": t.get("city"), "roof": t.get("roof"),
    }


def _player_payload(row: pd.Series) -> dict:
    stats = []
    for stat in POSITION_STATS[row["position"]]:
        qs = {f"p{int(q * 100):02d}": row.get(f"{stat}_p{int(q * 100):02d}") for q in QUANTILES}
        if qs["p50"] is None:
            continue
        stats.append({"stat": stat, **qs})
    return _clean({
        "player_id": row["player_id"],
        "name": row["player_display_name"],
        "position": row["position"],
        "team": row["team"],
        "opponent": row["opponent_team"],
        "is_home": bool(row.get("is_home", 0) == 1),
        "buzz": row.get("buzz", 0.0),
        "forecasts": stats,
    })


def _forecast_meta() -> dict | None:
    """latest_meta.json (live, else demo fixture), minus the bulky data_versions."""
    live = FORECASTS_DIR / "latest_meta.json"
    demo = DEMO_DIR / "latest_meta.json"

    def loader():
        path = live if live.exists() else demo
        if not path.exists():
            return None
        meta = json.loads(path.read_text())
        meta.pop("data_versions", None)
        return meta

    return _mtime_cached("meta", [live, demo], loader)


@app.get("/api/slate")
def slate():
    forecasts, games = _load()
    out = []
    for _, g in games.iterrows():
        headliners = forecasts[forecasts["game_id"] == g.game_id]
        top = (headliners.sort_values("fantasy_points_p50", ascending=False)
               .head(3)[["player_display_name", "position", "team", "fantasy_points_p50"]]
               .to_dict("records")) if not headliners.empty else []
        out.append(_clean({
            "game_id": g.game_id, "season": int(g.season), "week": int(g.week),
            "gameday": g.get("gameday"), "gametime": g.get("gametime"),
            "home": _team_meta(g.home_team), "away": _team_meta(g.away_team),
            "stadium": g.get("stadium"), "city": g.get("city"), "roof": g.get("roof"),
            "temp_c": None if pd.isna(g.get("temp")) else float(g.get("temp")),
            "wind_kph": None if pd.isna(g.get("wind")) else float(g.get("wind")),
            "headliners": top,
        }))
    return {"games": out, "meta": _forecast_meta()}


@app.get("/api/game/{game_id}")
def game_detail(game_id: str):
    forecasts, games = _load()
    match = games[games["game_id"] == game_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"unknown game_id {game_id}")
    g = match.iloc[0]
    rows = forecasts[forecasts["game_id"] == game_id]

    def team_block(abbr: str) -> dict:
        players = rows[rows["team"] == abbr].copy()
        players = players.sort_values("fantasy_points_p50", ascending=False)
        return {"team": _team_meta(abbr), "players": [_player_payload(r) for _, r in players.iterrows()]}

    return _clean({
        "game_id": game_id, "gameday": g.get("gameday"), "gametime": g.get("gametime"),
        "stadium": g.get("stadium"), "city": g.get("city"), "roof": g.get("roof"),
        "temp_c": None if pd.isna(g.get("temp")) else float(g.get("temp")),
        "wind_kph": None if pd.isna(g.get("wind")) else float(g.get("wind")),
        "home": team_block(g.home_team), "away": team_block(g.away_team),
    })


def _load_sims() -> dict:
    path = FORECASTS_DIR / "latest_sims.json"
    if not path.exists():
        path = DEMO_DIR / "latest_sims.json"
    return _mtime_cached("sims", [path],
                         lambda: json.loads(path.read_text()) if path.exists() else {})


@app.get("/api/game/{game_id}/sim")
def game_sim(game_id: str):
    sims = _load_sims()
    if game_id not in sims:
        raise HTTPException(
            status_code=404,
            detail="No simulation for this game — rerun the pipeline with --sims > 0.",
        )
    return sims[game_id]


@app.get("/api/players")
def players(q: str = ""):
    forecasts, _ = _load()
    rows = forecasts
    if q:
        rows = rows[rows["player_display_name"].str.contains(q, case=False, na=False)]
    rows = rows.sort_values("fantasy_points_p50", ascending=False).head(50)
    return {"players": [_player_payload(r) for _, r in rows.iterrows()]}


@app.get("/api/health")
def health():
    """Liveness plus deployment introspection: which models are installed,
    what the served forecasts are, and how the last refresh went. Every block
    is null on a fresh clone; "ok" stays a plain liveness bool."""
    manifest = _mtime_cached(
        "health:model", [MODELS_ROOT / "current" / "manifest.json"],
        lambda: bundle.installed_manifest(MODELS_ROOT))
    model = None
    if manifest:
        model = {k: manifest.get(k) for k in
                 ("version", "engine", "git_sha", "created_at", "train_seasons", "metrics")}

    forecasts = None
    meta = _forecast_meta()
    if meta:
        forecasts = {k: meta.get(k) for k in
                     ("generated_at", "as_of", "season", "week",
                      "n_players", "n_games", "model_version")}

    status_path = FORECASTS_DIR / "refresh_status.json"
    status = _mtime_cached(
        "health:refresh", [status_path],
        lambda: json.loads(status_path.read_text()) if status_path.exists() else None)
    refresh = None
    if status:
        refresh = {k: status.get(k) for k in
                   ("started_at", "finished_at", "ok", "skipped_reason",
                    "stage_failed", "error")}

    return {"ok": True, "model": model, "forecasts": forecasts, "refresh": refresh}


@app.get("/api/player/{player_id}/usage")
def player_usage(player_id: str):
    """Recent usage rows (volume/share columns) for one slate player — long
    format, one row per (season, week). Feeds the dashboard sparklines."""
    path = FORECASTS_DIR / "latest_usage.parquet"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No usage artifact yet — it appears after the next `gameday refresh`.",
        )
    df = _mtime_cached("usage", [path], lambda: pd.read_parquet(path))
    rows = df[df["player_id"] == player_id].sort_values(["season", "week"])
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"no usage rows for player {player_id}")
    cols = [c for c in rows.columns if c != "player_id"]
    return {"player_id": player_id, "weeks": _clean(rows[cols].to_dict("records"))}


# --------------------------------------------------------------------------
# draft board — season-long projections by position (artifacts latest_season)
# --------------------------------------------------------------------------

def _load_season() -> pd.DataFrame:
    path = FORECASTS_DIR / "latest_season.parquet"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No season projection yet — it appears after the next `gameday refresh`.",
        )
    return _mtime_cached("season", [path], lambda: pd.read_parquet(path))


def _draft_rows(df: pd.DataFrame) -> list[dict]:
    """One entry per player: season totals, floor/ceiling envelope, weekly
    medians, and bye week (the team's missing regular-season week)."""
    team_weeks = df.groupby("team")["week"].agg(set)
    all_weeks = set(range(int(df["week"].min()), SEASON_MAX_WEEK + 1))
    out = []
    for pid, rows in df.groupby("player_id"):
        rows = rows.sort_values("week")
        first = rows.iloc[0]
        byes = sorted(all_weeks - team_weeks.get(first["team"], set()))
        out.append({
            "player_id": pid,
            "name": first["player_display_name"],
            "position": first["position"],
            "team": first["team"],
            "bye": byes[0] if len(byes) == 1 else (byes or None),
            "games": int(len(rows)),
            "total_p50": round(float(rows["fantasy_points_p50"].sum()), 1),
            # Envelope, not a true quantile of the season sum: summing weekly
            # p25/p75 assumes perfectly correlated weeks, so it brackets wider
            # than reality — fine as a draft-day floor/ceiling visual.
            "total_floor": round(float(rows["fantasy_points_p25"].sum()), 1),
            "total_ceiling": round(float(rows["fantasy_points_p75"].sum()), 1),
            "weeks": {int(w): round(float(p), 1) for w, p in
                      zip(rows["week"], rows["fantasy_points_p50"])},
        })
    return out


MARKET_FIELDS = ["espn_adp", "espn_rank_ppr", "espn_auction", "espn_proj_pts",
                 "fft_proj_ppr", "sleeper_rank"]


def _load_market() -> tuple[dict, dict]:
    """({player_id: {external fields}}, meta) from the market artifact.

    Absent or unreadable artifact -> ({}, {}). The draft board predates this
    cross-reference and must keep working without it.
    """
    path = FORECASTS_DIR / "latest_market.parquet"
    meta_path = FORECASTS_DIR / "latest_market.json"
    if not path.exists():
        return {}, {}

    def load():
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001 — a bad artifact must not 500 the board
            return {}, {}
        cols = [c for c in MARKET_FIELDS if c in df.columns]
        rows = {str(r["player_id"]): {c: r[c] for c in cols}
                for _, r in df.iterrows()}
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except (OSError, ValueError):
            meta = {}
        return rows, meta

    return _mtime_cached("market", [path, meta_path], load)


def _attach_market(players: list[dict], partial_season: bool) -> dict:
    """Add external projections, a market rank, value-vs-market, and spread.

    Ranks are computed over the players *some* source has an opinion about.
    Our own board carries backup QBs that no outside source projects (the
    season projection scaffolds every rostered player), and letting them
    consume rank slots would shift every real player's value. Players with no
    market signal keep null ranks and a value_tier of "unranked".
    """
    market, meta = _load_market()
    for p in players:
        ext = market.get(str(p["player_id"]), {})
        for field in MARKET_FIELDS:
            value = ext.get(field)
            p[field] = None if value is None or pd.isna(value) else float(value)
        p.update(market_rank=None, our_rank=None, value=None, value_tier="unranked",
                 proj_spread=None, proj_spread_pct=None, proj_sources=0,
                 proj_min=None, proj_max=None, market_source=None)

    # ESPN ADP is the only real ADP here; Sleeper's search_rank is a coarse,
    # tie-heavy ordering, so it only orders players ESPN doesn't rank at all.
    def market_key(p):
        if p["espn_adp"]:
            return (0, p["espn_adp"])
        return (1, p["sleeper_rank"])

    ranked = [p for p in players if p["espn_adp"] or p["sleeper_rank"]]
    for rank, p in enumerate(sorted(ranked, key=market_key), start=1):
        p["market_rank"] = rank
        p["market_source"] = "espn_adp" if p["espn_adp"] else "sleeper_rank"
    for rank, p in enumerate(sorted(ranked, key=lambda p: -p["vorp"]), start=1):
        p["our_rank"] = rank
        p["value"] = p["market_rank"] - rank
        # Only flag players at least one board considers draftable. Deeper than
        # that, ESPN floor-clamps ADP into a ~170 pileup and our own ordering is
        # ranking players nobody takes, so the gap between them is not a signal.
        draftable = min(rank, p["market_rank"]) <= DRAFT_POOL_SIZE
        p["value_tier"] = "" if not draftable else (
            "sleeper" if p["value"] >= VALUE_ROUND_SIZE else
            "reach" if p["value"] <= -VALUE_ROUND_SIZE else "")

    # Projection spread across the three point projections. Our total covers
    # weeks first_week..18 only, so mid-season it is not comparable with the
    # full-season numbers ESPN and FFToday publish — omit rather than mislead.
    if not partial_season:
        for p in players:
            projections = [v for v in (p["total_p50"], p["espn_proj_pts"],
                                       p["fft_proj_ppr"]) if v]
            p["proj_sources"] = len(projections)
            if len(projections) < 2:
                continue
            low, high = min(projections), max(projections)
            mean = sum(projections) / len(projections)
            p.update(proj_min=round(low, 1), proj_max=round(high, 1),
                     proj_spread=round(high - low, 1),
                     proj_spread_pct=round((high - low) / mean, 3) if mean else None)

    return {
        "available": bool(market),
        "fetched_at": meta.get("fetched_at"),
        "coverage": meta.get("coverage", {}),
        "partial_season": partial_season,
        "ranked_players": len(ranked),
        "round_size": VALUE_ROUND_SIZE,
        "draft_pool_size": DRAFT_POOL_SIZE,
    }


@app.get("/api/draft")
def draft_board(position: str = "ALL", tier: str = "", limit: int = 300):
    """Season-long draft board: per-player weekly medians, season totals, VORP,
    and the ESPN/FFToday/Sleeper cross-reference (value vs market, spread)."""
    df = _load_season()
    players = _draft_rows(df)

    # Replacement baselines come from the full pool regardless of the filter.
    baselines = {}
    for pos, rank in REPLACEMENT_RANK.items():
        totals = sorted((p["total_p50"] for p in players if p["position"] == pos),
                        reverse=True)
        baselines[pos] = totals[rank - 1] if len(totals) >= rank else (totals[-1] if totals else 0.0)
    for p in players:
        p["vorp"] = round(p["total_p50"] - baselines.get(p["position"], 0.0), 1)

    first_week = int(df["week"].min())
    # Ranks and value are computed across the whole board before any filter, so
    # clicking a position pill never changes a player's numbers.
    market_meta = _attach_market(players, partial_season=first_week > 1)

    position = position.upper()
    if position != "ALL":
        if position not in POSITION_STATS:
            raise HTTPException(status_code=404, detail=f"unknown position {position}")
        players = [p for p in players if p["position"] == position]
    if tier:
        if tier not in ("sleeper", "reach"):
            raise HTTPException(status_code=404, detail=f"unknown tier {tier}")
        players = [p for p in players if p["value_tier"] == tier]
    players.sort(key=lambda p: p["vorp"], reverse=True)

    meta = _forecast_meta() or {}
    return _clean({
        "position": position,
        "tier": tier,
        "season": int(df["season"].iloc[0]),
        "first_week": first_week,
        "last_week": int(df["week"].max()),
        "replacement": baselines,
        "model_version": meta.get("model_version"),
        "generated_at": meta.get("generated_at"),
        "market": market_meta,
        "players": players[:limit],
    })


# --------------------------------------------------------------------------
# historical replay — forecast-vs-actual for a past season (artifacts/replay)
# --------------------------------------------------------------------------

def _load_replay(season: int) -> pd.DataFrame:
    path = REPLAY_DIR / str(season) / "players.parquet"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"No replay for {season} yet — run `gameday replay --season {season}`.",
        )
    return _mtime_cached(f"replay:{season}", [path], lambda: pd.read_parquet(path))


def _replay_player_payload(row: pd.Series) -> dict:
    """Like _player_payload but with the actual result (and naive baseline)
    overlaid on each stat, plus the season-to-season adjustment context."""
    stats = []
    for stat in POSITION_STATS[row["position"]]:
        p50 = row.get(f"{stat}_p50")
        if p50 is None or (isinstance(p50, float) and math.isnan(p50)):
            continue
        entry = {"stat": stat, "actual": row.get(stat), "r8": row.get(f"{stat}_r8")}
        for q in QUANTILES:
            entry[f"p{int(q * 100):02d}"] = row.get(f"{stat}_p{int(q * 100):02d}")
        stats.append(entry)
    return _clean({
        "player_id": row["player_id"],
        "name": row["player_display_name"],
        "position": row["position"],
        "team": row["team"],
        "opponent": row["opponent_team"],
        "is_home": bool(row.get("is_home", 0) == 1),
        "season": int(row["season"]),
        "week": int(row["week"]),
        "years_exp": row.get("years_exp"),
        "age": row.get("age"),
        "is_rookie": bool(row.get("is_rookie", 0) == 1),
        "is_new_team": bool(row.get("is_new_team", 0) == 1),
        "forecasts": stats,
    })


@app.get("/api/replay/seasons")
def replay_seasons():
    """Seasons (and their weeks) with a persisted replay artifact."""
    out = []
    if REPLAY_DIR.exists():
        for sdir in sorted(REPLAY_DIR.iterdir(), reverse=True):
            pf = sdir / "players.parquet"
            if not sdir.is_dir() or not pf.exists() or not sdir.name.isdigit():
                continue
            weeks = sorted(int(w) for w in pd.read_parquet(pf, columns=["week"])["week"].dropna().unique())
            out.append({"season": int(sdir.name), "weeks": weeks})
    return {"seasons": out}


@app.get("/api/replay/{season}/scorecard")
def replay_scorecard(season: int):
    """Season-to-season scorecard (adjusted, and baseline when compared)."""
    path = REPLAY_DIR / str(season) / "scorecard.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"No scorecard for {season}.")
    return json.loads(path.read_text())


@app.get("/api/replay/{season}/{week}")
def replay_week(season: int, week: int):
    """Every game that week as home/away blocks of forecast-vs-actual players."""
    df = _load_replay(season)
    rows = df[df["week"] == week]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No replay rows for {season} week {week}.")
    out_games = []
    for game_id, gdf in rows.groupby("game_id"):
        teams = sorted(gdf["team"].unique())
        blocks = {}
        for abbr, tdf in gdf.groupby("team"):
            tdf = tdf.sort_values("fantasy_points_p50", ascending=False)
            blocks[abbr] = {"team": _team_meta(abbr),
                            "players": [_replay_player_payload(r) for _, r in tdf.iterrows()]}
        home_rows = gdf[gdf["is_home"] == 1]
        home = home_rows["team"].iloc[0] if not home_rows.empty else teams[0]
        away = next((t for t in teams if t != home), teams[-1])
        out_games.append(_clean({"game_id": game_id,
                                 "home": blocks.get(home), "away": blocks.get(away)}))
    return {"season": season, "week": week, "games": out_games}


@app.get("/api/replay/{season}/{week}/player/{player_id}")
def replay_player(season: int, week: int, player_id: str):
    """One player's forecast-vs-actual detail for a replayed week."""
    df = _load_replay(season)
    rows = df[(df["week"] == week) & (df["player_id"] == player_id)]
    if rows.empty:
        raise HTTPException(status_code=404, detail="player not in this replay week")
    return _replay_player_payload(rows.iloc[0])


class RevalidatingStatics(StaticFiles):
    """Static files that must be revalidated on every load.

    Without this, StaticFiles sends an ETag and Last-Modified but no
    Cache-Control, so browsers apply a heuristic freshness window and keep
    serving a stale app.js after a deploy — the dashboard silently runs old
    code until someone hard-reloads. `no-cache` still allows the cache to be
    used, it just forces a revalidation first, so the common case is a cheap
    304 rather than a re-download.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


if WEB_DIR.exists():
    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html",
                            headers={"Cache-Control": "no-cache"})

    app.mount("/static", RevalidatingStatics(directory=WEB_DIR), name="static")
