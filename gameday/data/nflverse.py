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
from gameday.data.teams import normalize_team

log = logging.getLogger(__name__)

# nflverse froze the old `player_stats` release at 2024 and moved current
# weekly stats to the `stats_player` release (regular + postseason per week).
PLAYER_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.parquet"
)
SCHEDULES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"

# Columns we keep from the weekly stat lines (superset across positions). The
# stats_player schema renamed a couple of fields vs the old release.
STAT_COLUMNS = [
    "player_id", "player_display_name", "position", "team", "season", "week",
    "opponent_team", "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "carries", "rushing_yards", "rushing_tds", "receptions",
    "targets", "receiving_yards", "receiving_tds", "fantasy_points_ppr",
    "target_share", "air_yards_share", "wopr", "racr",
]


def fetch_player_weeks(seasons: list[int], force: bool = False) -> pd.DataFrame:
    """Weekly per-player stat lines for the given seasons, cached under data/raw."""
    ensure_dirs()
    frames = []
    for season in seasons:
        cache = RAW_DIR / f"stats_player_week_{season}.parquet"
        if cache.exists() and not force:
            frames.append(pd.read_parquet(cache))
            continue
        url = PLAYER_STATS_URL.format(season=season)
        log.info("downloading %s", url)
        try:
            with httpx.Client(follow_redirects=True, timeout=120) as client:
                resp = client.get(url)
                resp.raise_for_status()
                cache.write_bytes(resp.content)
        except httpx.HTTPStatusError as exc:
            # A not-yet-started season (e.g. the upcoming one in the offseason)
            # has a schedule but no weekly stats yet — skip it so the upcoming
            # slate can still be forecast from prior seasons + current rosters.
            if exc.response.status_code == 404:
                log.warning("no player stats for %s yet; skipping", season)
                continue
            raise
        frames.append(pd.read_parquet(cache))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    keep = [c for c in STAT_COLUMNS if c in df.columns]
    df = df[keep].rename(columns={"fantasy_points_ppr": "fantasy_points",
                                  "passing_interceptions": "interceptions"})
    for col in ("team", "opponent_team"):
        if col in df.columns:
            df[col] = df[col].map(normalize_team)
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
    games = games[[c for c in keep if c in games.columns]].reset_index(drop=True)
    for col in ("home_team", "away_team"):
        if col in games.columns:
            games[col] = games[col].map(normalize_team)
    return games
