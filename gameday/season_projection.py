"""Season-long projections for the draft board: every remaining regular-season
week for every slate player, from one prepared feature frame.

The weekly models forecast a single week from pre-kickoff state. At draft time
no future games have been played, so a player's form/usage state is *frozen* —
the only thing that varies from week to week is schedule context. We therefore
compose "frozen player state x per-week schedule context":

  * base state  — the next-week scaffold rows from build_features (all form,
    usage, availability, and prior features exactly as the weekly model sees
    them);
  * per-week    — opponent, home/away, rest, travel, venue, and the opponent
    context features (def_pts_allowed_r8 / def_vs_pos_r8) remapped to each
    future week's actual opponent, with defense quality frozen at its latest
    trailing value.

Weather for far-future weeks is unknowable, so outdoor games get the same
climatology defaults build_features uses for missing readings (15 C, 8 km/h);
domes stay 21 C / 0. Bye weeks fall out naturally (no game, no row).

Output: one row per (player, week) with the fantasy-point quantile columns —
the artifact behind /api/draft.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gameday.config import MODELS_DIR, POSITIONS, SEASON_MAX_WEEK
from gameday.data.teams import TEAMS, travel_km
from gameday.models import usage_forecast
from gameday.pipeline import model_engine

log = logging.getLogger(__name__)

# Climatology defaults for unknowable future weather (mirrors build_features).
_OUTDOOR_TEMP_C, _OUTDOOR_WIND_KPH = 15.0, 8.0

KEEP_COLS = ["player_id", "player_display_name", "position", "team",
             "season", "week", "opponent_team", "is_home",
             "fantasy_points_p10", "fantasy_points_p25", "fantasy_points_p50",
             "fantasy_points_p75", "fantasy_points_p90"]


def _frozen_context(future: pd.DataFrame):
    """Per-team / per-(team, position) defense+offense context, frozen at the
    trailing values the next-week scaffold rows carry."""
    def_pts = (future.dropna(subset=["def_pts_allowed_r8"])
               .groupby("opponent_team")["def_pts_allowed_r8"].first())
    dvp = (future.dropna(subset=["def_vs_pos_r8"])
           .groupby(["opponent_team", "position"])["def_vs_pos_r8"].first())
    return def_pts, dvp


def compose_weeks(future: pd.DataFrame, full_games: pd.DataFrame,
                  max_week: int = SEASON_MAX_WEEK) -> pd.DataFrame:
    """Base-week rows + one context-swapped copy per later remaining week."""
    if future.empty:
        return future
    season = int(future["season"].iloc[0])
    base_week = int(future["week"].min())
    remaining = full_games[
        (pd.to_numeric(full_games["season"], errors="coerce") == season)
        & full_games["home_score"].isna()
        & (pd.to_numeric(full_games["week"], errors="coerce") > base_week)
        & (pd.to_numeric(full_games["week"], errors="coerce") <= max_week)]
    def_pts, dvp = _frozen_context(future)

    frames = [future[future["week"] == base_week].copy()]
    for _, g in remaining.iterrows():
        week = int(g["week"])
        for team, opp, is_home, rest_col in (
                (g["home_team"], g["away_team"], 1, "home_rest"),
                (g["away_team"], g["home_team"], 0, "away_rest")):
            rows = future[(future["team"] == team) & (future["week"] == base_week)].copy()
            if rows.empty:
                continue
            venue = TEAMS.get(g["home_team"], {})
            roof = g.get("roof") if pd.notna(g.get("roof")) else venue.get("roof")
            indoor = int(roof in ("dome", "retractable"))
            rows["season"], rows["week"] = season, week
            rows["game_id"] = g.get("game_id")
            rows["opponent_team"], rows["is_home"] = opp, is_home
            rows["home_team"] = g["home_team"]
            rest = pd.to_numeric(g.get(rest_col), errors="coerce")
            rows["rest"] = 7.0 if pd.isna(rest) else float(rest)
            rows["travel_km"] = 0.0 if is_home else (
                travel_km(g["home_team"], team)
                if g["home_team"] in TEAMS and team in TEAMS else 0.0)
            rows["altitude_m"] = venue.get("altitude", 100)
            rows["indoor"] = indoor
            rows["temp_c"] = 21.0 if indoor else _OUTDOOR_TEMP_C
            rows["wind_kph"] = 0.0 if indoor else _OUTDOOR_WIND_KPH
            rows["def_pts_allowed_r8"] = def_pts.get(opp)
            rows["def_vs_pos_r8"] = rows["position"].map(
                lambda p, o=opp: dvp.get((o, p)))
            frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def project_season(feats: pd.DataFrame, full_games: pd.DataFrame,
                   engine: str = "gbm", models_dir: Path = MODELS_DIR,
                   max_week: int = SEASON_MAX_WEEK) -> pd.DataFrame:
    """Quantile fantasy-point projections for every remaining regular-season
    (player, week). Returns an empty frame when there is no upcoming slate."""
    future = feats[feats["fantasy_points"].isna()]
    if future.empty:
        return pd.DataFrame(columns=KEEP_COLS)
    composed = compose_weeks(future, full_games, max_week=max_week)
    m = model_engine(engine)
    out = []
    for position in POSITIONS:
        rows = composed[composed["position"] == position].copy()
        if rows.empty:
            continue
        rows = usage_forecast.predict_usage(rows, position, models_dir=models_dir)
        pred = m.predict_position(rows, position, models_dir=models_dir)
        out.append(pred[[c for c in KEEP_COLS if c in pred.columns]])
    if not out:
        return pd.DataFrame(columns=KEEP_COLS)
    result = pd.concat(out, ignore_index=True)
    log.info("season projection: %d player-weeks across weeks %d-%d",
             len(result), int(result["week"].min()), int(result["week"].max()))
    return result
