"""Historical NFL data from nflverse public releases.

Pulls weekly player stat lines and game schedules directly from the
nflverse-data GitHub releases (same source `nfl_data_py` wraps), cached
locally as parquet so repeat runs are instant and offline-friendly.
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd

from gameday.config import RAW_DIR, ensure_dirs

log = logging.getLogger(__name__)

PLAYER_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/player_stats_{season}.parquet"
)
SCHEDULES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"

# Columns we keep from the weekly stat lines (superset across positions).
STAT_COLUMNS = [
    "player_id", "player_display_name", "position", "recent_team", "season", "week",
    "opponent_team", "completions", "attempts", "passing_yards", "passing_tds",
    "interceptions", "carries", "rushing_yards", "rushing_tds", "receptions",
    "targets", "receiving_yards", "receiving_tds", "fantasy_points_ppr",
]


def fetch_player_weeks(seasons: list[int], force: bool = False) -> pd.DataFrame:
    """Weekly per-player stat lines for the given seasons, cached under data/raw."""
    ensure_dirs()
    frames = []
    for season in seasons:
        cache = RAW_DIR / f"player_stats_{season}.parquet"
        if cache.exists() and not force:
            frames.append(pd.read_parquet(cache))
            continue
        url = PLAYER_STATS_URL.format(season=season)
        log.info("downloading %s", url)
        with httpx.Client(follow_redirects=True, timeout=120) as client:
            resp = client.get(url)
            resp.raise_for_status()
            cache.write_bytes(resp.content)
        frames.append(pd.read_parquet(cache))
    df = pd.concat(frames, ignore_index=True)
    keep = [c for c in STAT_COLUMNS if c in df.columns]
    df = df[keep].rename(columns={"fantasy_points_ppr": "fantasy_points", "recent_team": "team"})
    return df[df["position"].isin(["QB", "RB", "WR", "TE"])].reset_index(drop=True)


def fetch_schedules(seasons: list[int], force: bool = False) -> pd.DataFrame:
    """Game schedules/results with kickoff time, roof state, and rest days."""
    ensure_dirs()
    cache = RAW_DIR / "games.csv"
    if force or not cache.exists():
        with httpx.Client(follow_redirects=True, timeout=120) as client:
            resp = client.get(SCHEDULES_URL)
            resp.raise_for_status()
            cache.write_bytes(resp.content)
    games = pd.read_csv(cache)
    games = games[games["season"].isin(seasons)]
    keep = [
        "game_id", "season", "week", "gameday", "gametime", "home_team", "away_team",
        "home_score", "away_score", "home_rest", "away_rest", "roof", "temp", "wind",
    ]
    return games[[c for c in keep if c in games.columns]].reset_index(drop=True)
