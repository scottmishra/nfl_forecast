"""Feature matrix construction.

One row per player-game. Every rolling feature is shifted by one game so a
row only sees information available *before* kickoff (no leakage). Rows for
games without results yet (the upcoming slate) get the same features and NaN
targets — they become the inference set.

Feature groups:
  * player form   — trailing 3/8-game means of each stat + usage (targets/carries)
  * opponent      — rolling fantasy points the defense allows to this position,
                    rolling points allowed overall
  * team          — rolling points scored by the player's own offense
  * game context  — home/away, rest days, travel distance, altitude
  * weather       — temp, wind, precipitation proxy, indoor flag
  * buzz          — social-media signal from the configured provider
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gameday.config import POSITION_STATS, FORM_WINDOWS
from gameday.data.teams import TEAMS, travel_km
from gameday.data import social

log = logging.getLogger(__name__)

USAGE_COLS = ["attempts", "carries", "targets"]


def _upcoming_scaffold(player_weeks: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Rows for players in not-yet-played games, based on each team's most
    recent roster (players seen in that team's last 4 played weeks)."""
    upcoming = games[games["home_score"].isna()]
    if upcoming.empty:
        return pd.DataFrame()

    rows = []
    for _, g in upcoming.iterrows():
        for team, opp in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
            hist = player_weeks[player_weeks["team"] == team]
            if hist.empty:
                continue
            recent_weeks = (
                hist[["season", "week"]].drop_duplicates()
                .sort_values(["season", "week"]).tail(4)
            )
            roster = hist.merge(recent_weeks, on=["season", "week"])
            roster = roster.sort_values(["season", "week"]).groupby("player_id").tail(1)
            for _, p in roster.iterrows():
                rows.append(dict(
                    player_id=p.player_id, player_display_name=p.player_display_name,
                    position=p.position, team=team, season=int(g.season), week=int(g.week),
                    opponent_team=opp,
                ))
    return pd.DataFrame(rows)


def _team_week_grid(games: pd.DataFrame) -> pd.DataFrame:
    """Long-format team schedule: one row per team-game with scores and context."""
    home = games.rename(columns={
        "home_team": "team", "away_team": "opp",
        "home_score": "points_for", "away_score": "points_against",
        "home_rest": "rest"})
    home["is_home"] = 1
    away = games.rename(columns={
        "away_team": "team", "home_team": "opp",
        "away_score": "points_for", "home_score": "points_against",
        "away_rest": "rest"})
    away["is_home"] = 0
    cols = ["game_id", "season", "week", "team", "opp", "points_for", "points_against",
            "rest", "is_home", "roof", "temp", "wind", "gameday", "gametime", "home_team"]
    grid = pd.concat([home, away], ignore_index=True)
    grid["home_team"] = np.where(grid["is_home"] == 1, grid["team"], grid["opp"])
    return grid[[c for c in cols if c in grid.columns]]


def _roll(g: pd.Series, window: int) -> pd.Series:
    """Trailing mean over `window` games, excluding the current one."""
    return g.shift(1).rolling(window, min_periods=1).mean()


def build_features(
    player_weeks: pd.DataFrame,
    games: pd.DataFrame,
    buzz_provider: str = "neutral",
) -> pd.DataFrame:
    """Return the full feature matrix (historical rows + upcoming-slate rows)."""
    all_stats = sorted({s for stats in POSITION_STATS.values() for s in stats})

    scaffold = _upcoming_scaffold(player_weeks, games)
    df = pd.concat([player_weeks, scaffold], ignore_index=True)
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # --- player form ---------------------------------------------------
    grp = df.groupby("player_id", sort=False)
    for stat in all_stats + USAGE_COLS:
        if stat not in df.columns:
            continue
        for w in FORM_WINDOWS:
            df[f"{stat}_r{w}"] = grp[stat].transform(lambda s, w=w: _roll(s, w))
    df["games_played"] = grp.cumcount()

    # --- team / opponent context ----------------------------------------
    grid = _team_week_grid(games).sort_values(["team", "season", "week"])
    tg = grid.groupby("team", sort=False)
    grid["team_pts_r8"] = tg["points_for"].transform(lambda s: _roll(s, 8))
    grid["opp_pts_allowed_r8"] = tg["points_against"].transform(lambda s: _roll(s, 8))

    df = df.merge(
        grid[["season", "week", "team", "game_id", "is_home", "rest", "roof",
              "temp", "wind", "team_pts_r8", "home_team"]],
        on=["season", "week", "team"], how="left",
    )
    df = df.merge(
        grid[["season", "week", "team", "opp_pts_allowed_r8"]]
        .rename(columns={"team": "opponent_team", "opp_pts_allowed_r8": "def_pts_allowed_r8"}),
        on=["season", "week", "opponent_team"], how="left",
    )

    # --- defense vs position ---------------------------------------------
    # Fantasy points each defense concedes to each position, rolled over its
    # trailing 8 games. Built on the full team-week grid so upcoming weeks
    # inherit the shifted value cleanly.
    fp_allowed = (
        player_weeks.groupby(["opponent_team", "season", "week", "position"], as_index=False)
        ["fantasy_points"].sum()
        .rename(columns={"opponent_team": "team", "fantasy_points": "fp_allowed"})
    )
    dvp = grid[["team", "season", "week"]].merge(
        pd.DataFrame({"position": list(POSITION_STATS)}), how="cross")
    dvp = dvp.merge(fp_allowed, on=["team", "season", "week", "position"], how="left")
    dvp = dvp.sort_values(["team", "position", "season", "week"])
    dvp["def_vs_pos_r8"] = (
        dvp.groupby(["team", "position"], sort=False)["fp_allowed"]
        .transform(lambda s: _roll(s, 8))
    )
    df = df.merge(
        dvp[["team", "season", "week", "position", "def_vs_pos_r8"]]
        .rename(columns={"team": "opponent_team"}),
        on=["opponent_team", "season", "week", "position"], how="left",
    )

    # --- venue / travel / weather -----------------------------------------
    df["travel_km"] = np.where(
        df["is_home"] == 1, 0.0,
        [travel_km(h, t) if h in TEAMS and t in TEAMS else 0.0
         for h, t in zip(df["home_team"].fillna(""), df["team"].fillna(""))],
    )
    df["altitude_m"] = df["home_team"].map(lambda t: TEAMS.get(t, {}).get("altitude", 100))
    df["indoor"] = df["roof"].isin(["dome", "retractable"]).astype(int)
    df["temp_c"] = np.where(df["indoor"] == 1, 21.0, pd.to_numeric(df["temp"], errors="coerce"))
    df["wind_kph"] = np.where(df["indoor"] == 1, 0.0, pd.to_numeric(df["wind"], errors="coerce"))
    df["temp_c"] = df["temp_c"].fillna(15.0)
    df["wind_kph"] = df["wind_kph"].fillna(8.0)

    # --- buzz ---------------------------------------------------------------
    provider = social.get_provider(buzz_provider)
    df["buzz"] = 0.0
    upcoming_mask = df["game_id"].notna() & df[all_stats[0]].isna() if all_stats[0] in df else pd.Series(False, index=df.index)
    if buzz_provider != "neutral" and upcoming_mask.any():
        target = df[upcoming_mask]
        season, week = int(target["season"].iloc[0]), int(target["week"].iloc[0])
        df.loc[upcoming_mask, "buzz"] = provider(target, season, week).values

    df["rest"] = pd.to_numeric(df["rest"], errors="coerce").fillna(7)
    return df


def feature_columns(position: str, df: pd.DataFrame) -> list[str]:
    """Model inputs for a position: its stat-form columns + shared context."""
    stats = POSITION_STATS[position]
    cols: list[str] = []
    for stat in stats + USAGE_COLS:
        for w in FORM_WINDOWS:
            c = f"{stat}_r{w}"
            if c in df.columns:
                cols.append(c)
    cols += [
        "games_played", "is_home", "rest", "travel_km", "altitude_m",
        "indoor", "temp_c", "wind_kph", "week",
        "team_pts_r8", "def_pts_allowed_r8", "def_vs_pos_r8", "buzz",
    ]
    return [c for c in dict.fromkeys(cols) if c in df.columns]
