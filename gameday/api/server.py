"""FastAPI backend: serves the forecast artifacts and the dashboard.

Endpoints:
  GET /api/slate                upcoming games with venue/weather + team meta
  GET /api/game/{game_id}       both teams' player forecasts for one game
  GET /api/players?q=           fuzzy player search across the slate
  GET /                         the dashboard (static SPA under web/)
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from gameday.config import FORECASTS_DIR, POSITION_STATS, QUANTILES, ROOT
from gameday.data.teams import TEAMS

app = FastAPI(title="Gameday Forecaster", version="0.1.0")

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


def _clean(obj):
    """Recursively convert NaN -> None for JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


@lru_cache(maxsize=1)
def _load():
    fpath = FORECASTS_DIR / "latest_forecasts.parquet"
    spath = FORECASTS_DIR / "latest_slate.parquet"
    if not fpath.exists() or not spath.exists():
        raise HTTPException(
            status_code=503,
            detail="No forecasts yet — run `gameday demo` or `gameday forecast` first.",
        )
    return pd.read_parquet(fpath), pd.read_parquet(spath)


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
    return {"games": out}


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


@lru_cache(maxsize=1)
def _load_sims() -> dict:
    path = FORECASTS_DIR / "latest_sims.json"
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text())


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
    return {"ok": True}


if WEB_DIR.exists():
    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
