"""Usage/context player-week feeds from nflverse: snaps, injuries, depth, pace, NGS.

Data contracts — the column names emitted here are consumed as-is by feature
code, so they are part of the interface:

- fetch_snap_counts(seasons) ->
    [player_id, season, week, team, offense_snaps, offense_pct]
- fetch_injuries(seasons) ->
    [player_id, season, week, report_status, practice_status]
- fetch_depth_charts(seasons) ->
    [player_id, season, week, team, position, depth_rank]
- fetch_team_weeks(seasons) ->
    TEAM_WEEK_COLUMNS (team/season/week/opponent_team + volume, EPA, CPOE)
- fetch_ngs(kind, seasons=None) ->
    [player_id, season, season_type, week, team, + that kind's NGS metrics]

`player_id` is always the gsis id — the same id as stats_player `player_id`
and roster `gsis_id`, so every frame joins the modeling tables directly.
Snap counts key on pfr ids upstream and are crosswalked through the players
dataset; the join rate on skill-position rows is logged, and a rate under
70% raises (a silently bad join would poison training). `fetch_ngs` keeps
`season_type` because pre-2021 postseason week numbers can collide with
regular-season weeks.

Depth charts changed schema in 2025: the old feed is one row per
(week, formation slot) with string `depth_team` as the rank; the new feed is
a daily full-team snapshot keyed by a `dt` timestamp with per-position
`pos_rank` and no week column. Both eras normalize to the same frame:
`depth_rank` is the player's integer order at his position (1 = starter;
co-starters in multi-WR sets become 1, 2, 3 by slot), and new-era snapshots
map to the week whose games they precede, keeping the latest snapshot per
team-week.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gameday.data import releases
from gameday.data.teams import normalize_team

log = logging.getLogger(__name__)

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]

DEPTH_COLUMNS = ["player_id", "season", "week", "team", "position", "depth_rank"]
INJURY_COLUMNS = ["player_id", "season", "week", "report_status", "practice_status"]
SNAP_COLUMNS = ["player_id", "season", "week", "team", "offense_snaps", "offense_pct"]

# Kept from stats_team weekly files (133 columns upstream): volume, efficiency,
# and EPA — enough to derive pace/plays (attempts + carries + sacks_suffered)
# and opponent-quality features downstream.
TEAM_WEEK_COLUMNS = [
    "team", "season", "week", "opponent_team", "season_type",
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered", "passing_air_yards",
    "passing_yards_after_catch", "passing_first_downs", "passing_epa",
    "passing_cpoe", "carries", "rushing_yards", "rushing_tds",
    "rushing_first_downs", "rushing_epa", "receptions", "targets",
    "receiving_yards", "receiving_tds", "receiving_air_yards",
    "receiving_first_downs", "receiving_epa", "penalties", "penalty_yards",
]

NGS_KINDS = ("passing", "receiving", "rushing")

# Identity/name columns dropped from NGS frames (player_id + team carry them).
_NGS_DROP = [
    "player_display_name", "player_first_name", "player_last_name",
    "player_jersey_number", "player_short_name", "player_position",
]

# New-era (2025+) depth charts: offensive formation slots. No abbreviation
# overlaps with the defense/special-teams groups, so this set alone selects
# the offense. LT/RT and LG/RG collapse to roster-style T/G.
_OFF_SLOTS = {"QB", "RB", "WR", "TE", "FB", "C", "LG", "RG", "LT", "RT"}
_SLOT_TO_POS = {"LT": "T", "RT": "T", "LG": "G", "RG": "G"}


def _crosswalk_pfr_to_gsis() -> pd.DataFrame:
    """pfr_id -> gsis_id map from the nflverse players dataset."""
    players = releases.fetch_frame("players", columns=["gsis_id", "pfr_id"])
    players = players.dropna(subset=["gsis_id", "pfr_id"])
    return players.drop_duplicates("pfr_id")[["pfr_id", "gsis_id"]]


def fetch_snap_counts(seasons: list[int]) -> pd.DataFrame:
    """Offensive snap counts per player-week, crosswalked from pfr to gsis ids."""
    df = releases.fetch_frame("snap_counts", seasons, columns=[
        "season", "week", "player", "pfr_player_id", "position", "team",
        "offense_snaps", "offense_pct"])
    if df.empty:
        return pd.DataFrame(columns=SNAP_COLUMNS)
    df = df.merge(_crosswalk_pfr_to_gsis(), how="left",
                  left_on="pfr_player_id", right_on="pfr_id")
    skill = df[df["position"].isin(SKILL_POSITIONS)]
    rate = float(skill["gsis_id"].notna().mean()) if len(skill) else 1.0
    log.info("snap-count pfr->gsis join rate: %.1f%% of %d skill rows",
             100 * rate, len(skill))
    if rate < 0.70:
        raise ValueError(
            f"snap-count pfr->gsis join rate {rate:.1%} < 70% — players "
            "crosswalk looks broken; refusing to emit a poisoned frame")
    if rate < 0.90:
        log.warning("snap-count join rate %.1f%% below the expected 90%%", 100 * rate)
    df = df[df["gsis_id"].notna()].rename(columns={"gsis_id": "player_id"})
    df["team"] = df["team"].map(normalize_team)
    return df[SNAP_COLUMNS].reset_index(drop=True)


def fetch_injuries(seasons: list[int]) -> pd.DataFrame:
    """Weekly injury-report rows (optional feed: empty frame when unavailable)."""
    df = releases.fetch_frame("injuries", seasons, columns=[
        "season", "week", "gsis_id", "report_status", "practice_status"])
    if df.empty:
        return pd.DataFrame(columns=INJURY_COLUMNS)
    df = df[df["gsis_id"].notna()].rename(columns={"gsis_id": "player_id"})
    return df[INJURY_COLUMNS].reset_index(drop=True)


def fetch_team_weeks(seasons: list[int]) -> pd.DataFrame:
    """Team offense per week — volume, EPA, and CPOE for pace/quality features."""
    df = releases.fetch_frame("team_weeks", seasons, columns=TEAM_WEEK_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=TEAM_WEEK_COLUMNS)
    for col in ("team", "opponent_team"):
        if col in df.columns:
            df[col] = df[col].map(normalize_team)
    return df.reset_index(drop=True)


def fetch_ngs(kind: str, seasons: list[int] | None = None) -> pd.DataFrame:
    """Next Gen Stats player-weeks for `kind` in passing/receiving/rushing."""
    if kind not in NGS_KINDS:
        raise ValueError(f"kind must be one of {NGS_KINDS}, not {kind!r}")
    df = releases.fetch_frame(f"ngs_{kind}")
    if df.empty:
        return pd.DataFrame()
    df = df[df["week"] >= 1]  # week 0 rows are season-to-date aggregates
    if seasons is not None:
        df = df[df["season"].isin(seasons)]
    df = df.rename(columns={"player_gsis_id": "player_id", "team_abbr": "team"})
    df = df[df["player_id"].notna()]
    df["team"] = df["team"].map(normalize_team)
    df = df.drop(columns=[c for c in _NGS_DROP if c in df.columns])
    lead = ["player_id", "season", "season_type", "week", "team"]
    return df[lead + [c for c in df.columns if c not in lead]].reset_index(drop=True)


def fetch_depth_charts(seasons: list[int]) -> pd.DataFrame:
    """Offensive depth-chart order per player-week, normalized across both eras."""
    frames = []
    for season in seasons:
        path = releases.fetch_asset(releases.CATALOG["depth_charts"], season)
        if path is None:
            continue
        raw = pd.read_parquet(path)
        if "formation" in raw.columns:
            frames.append(_depth_pre2025(raw))
        else:
            frames.append(_depth_2025(raw, season))
    if not frames:
        return pd.DataFrame(columns=DEPTH_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _depth_pre2025(raw: pd.DataFrame) -> pd.DataFrame:
    """Pre-2025 era: one row per (week, formation slot), depth_team = rank."""
    df = raw[raw["formation"] == "Offense"].copy()
    df["depth_rank"] = pd.to_numeric(df["depth_team"], errors="coerce")
    df = df.dropna(subset=["gsis_id", "week", "depth_rank"])
    # Best (lowest-ranked) slot wins when a player appears in several; then
    # co-starters across slots (LWR/RWR/SWR all depth_team=1) densify to
    # 1, 2, 3 — the same per-position ordering the 2025+ era emits natively.
    df = df.sort_values(["season", "week", "club_code", "position",
                         "depth_rank", "depth_position"], kind="stable")
    df = df.drop_duplicates(["gsis_id", "season", "week"], keep="first")
    df["depth_rank"] = df.groupby(["season", "week", "club_code", "position"]).cumcount() + 1
    out = pd.DataFrame({
        "player_id": df["gsis_id"],
        "season": df["season"].astype(int),
        "week": df["week"].astype(int),
        "team": df["club_code"].map(normalize_team),
        "position": df["position"],
        "depth_rank": df["depth_rank"],
    })
    return out.reset_index(drop=True)


def _depth_2025(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    """2025+ era: daily whole-team snapshots with per-position pos_rank.

    Snapshots carry no week — each one maps to the week whose games it
    precedes (via the schedule), keeping the latest snapshot per team-week
    and dropping offseason snapshots past the season's last game.
    """
    df = raw[raw["pos_abb"].isin(_OFF_SLOTS)].copy()
    df = df.dropna(subset=["gsis_id", "pos_rank"])
    week_ends, week_labels = _week_windows(season)
    snap_day = pd.to_datetime(df["dt"], utc=True).dt.tz_localize(None).dt.normalize()
    idx = np.searchsorted(week_ends, snap_day.to_numpy(), side="left")
    in_season = idx < len(week_labels)
    df, idx = df[in_season], idx[in_season]
    df["week"] = week_labels[idx]
    latest = df.groupby(["team", "week"])["dt"].transform("max")
    df = df[df["dt"] == latest]
    df = df.sort_values("pos_rank").drop_duplicates(["gsis_id", "week"], keep="first")
    out = pd.DataFrame({
        "player_id": df["gsis_id"],
        "season": season,
        "week": df["week"].astype(int),
        "team": df["team"].map(normalize_team),
        "position": df["pos_abb"].map(lambda a: _SLOT_TO_POS.get(a, a)),
        "depth_rank": df["pos_rank"].astype(int),
    })
    return out.reset_index(drop=True)


def _week_windows(season: int) -> tuple[np.ndarray, np.ndarray]:
    """(sorted week-end dates, matching week numbers) for a season's schedule."""
    sched = releases.fetch_frame("schedules")
    s = sched[(sched["season"] == season) & sched["gameday"].notna()]
    if s.empty:
        raise ValueError(f"no schedule rows for season {season}; can't place snapshots")
    ends = s.groupby("week")["gameday"].max()
    end_dates = pd.to_datetime(ends.to_numpy())
    order = np.argsort(end_dates.to_numpy())
    return end_dates.to_numpy()[order], ends.index.to_numpy()[order]
