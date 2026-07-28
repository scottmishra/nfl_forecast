"""Historical NFL data from nflverse public releases.

Weekly player stat lines and game schedules, fetched through the generic
release layer (`gameday.data.releases`) which owns caching, conditional GETs,
404-skipping for unpublished seasons, and stale-cache fallbacks. This module
keeps the project-facing normalization: column renames, team-code
canonicalization, and the skill-position filter.
"""

from __future__ import annotations

import logging

import pandas as pd

from gameday.data import releases
from gameday.data.teams import normalize_team

log = logging.getLogger(__name__)

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
    df = releases.fetch_frame("player_weeks", seasons, force=force, columns=STAT_COLUMNS)
    if df.empty:
        return pd.DataFrame()
    keep = [c for c in STAT_COLUMNS if c in df.columns]
    df = df[keep].rename(columns={"fantasy_points_ppr": "fantasy_points",
                                  "passing_interceptions": "interceptions"})
    for col in ("team", "opponent_team"):
        if col in df.columns:
            df[col] = df[col].map(normalize_team)
    return df[df["position"].isin(["QB", "RB", "WR", "TE"])].reset_index(drop=True)


def fetch_schedules(seasons: list[int], force: bool = False) -> pd.DataFrame:
    """Game schedules/results with kickoff time, roof state, and rest days."""
    games = releases.fetch_frame("schedules", force=force)
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
