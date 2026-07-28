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
from gameday.config import (ARTIFACTS_DIR, FORECASTS_DIR, MODELS_ROOT,
                            POSITION_STATS, QUANTILES, ROOT)
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


if WEB_DIR.exists():
    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
