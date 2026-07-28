"""Usage/opportunity features: snaps, team shares, availability, role priors.

Consumes the gameday.data.usage feeds (gsis-keyed, columns documented there)
and turns them into modeling features. Three stages:

  * compute_usage_shares — per player-week `snap_pct` (offensive snap share),
    `carry_share` and `target_share_team` (player volume over the team's
    total from stats_team, with a player-week sum fallback). These series
    then get the temporal treatment (EWM/lag/slope/vol via add_temporal) in
    build_features — same shift(1) discipline as every other form feature.
  * add_availability — CURRENT-week injury-report and depth-chart state.
    This is the one deliberate exception to the shift(1) rule: the week's
    injury report and depth chart are published before kickoff, so joining
    them onto the same week's row is pre-kickoff-legal by construction.
    History features (games_missed_last8, weeks_since_return) count games
    the player skipped on his own team's schedule.
  * usage_prior / apply_usage_prior — empirical mean usage by role cell
    (position × capped depth rank, rookies split by draft capital), fit on
    TRAINING rows only. Leakage rule: the prior table must be fit inside the
    walk-forward fold (models/usage_forecast.py fits it on the fold's train
    rows and persists it in the usage manifest); build_features never bakes
    priors in globally. Shrinkage: usage_est = w·player_ewm + (1−w)·prior
    with w = n_recent/(n_recent+4), where n_recent counts games with the
    CURRENT team — a trade or signing resets the role evidence while the
    efficiency EWMs (yards, catch rate...) deliberately keep their history.

Route participation was considered for WR/TE but nflverse NGS receiving
carries no route data (cushion/separation/YAC only), so it is omitted.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# The usage series that receive the temporal treatment in build_features and
# whose priors/forecasts feed the v3 stat models.
USAGE_SERIES = ["snap_pct", "carry_share", "target_share_team"]

# Availability features emitted by add_availability (all pre-kickoff-legal).
AVAILABILITY_COLS = [
    "inj_questionable", "inj_doubtful", "inj_out",
    "practice_dnp", "practice_limited", "practice_full",
    "depth_rank", "depth_rank_change", "is_depth_promotion",
    "games_missed_last8", "weeks_since_return",
]

# Depth ranks beyond this collapse into one bucket when building role priors.
_DEPTH_CAP = 4

# Draft-capital buckets for rookie prior cells (round-1, day-2, rest).
_DRAFT_EDGES = [32.5, 100.5]


def fetch_usage_feeds(seasons: list[int]) -> dict[str, pd.DataFrame] | None:
    """Fetch snaps/injuries/depth/team-week feeds, or None when unavailable.

    Degrades gracefully: any failure (offline, feed outage) logs and returns
    None so build_features falls back to v2 behavior instead of dying."""
    from gameday.data import usage as usage_data

    try:
        feeds = {
            "snaps": usage_data.fetch_snap_counts(seasons),
            "injuries": usage_data.fetch_injuries(seasons),
            "depth": usage_data.fetch_depth_charts(seasons),
            "team_weeks": usage_data.fetch_team_weeks(seasons),
        }
    except Exception as exc:  # noqa: BLE001 — any feed failure degrades, never kills
        log.warning("usage feeds unavailable (%s); features fall back to v2", exc)
        return None
    if feeds["snaps"].empty:
        log.warning("snap counts empty for %s; usage stage skipped", seasons)
        return None
    return feeds


def compute_usage_shares(player_weeks: pd.DataFrame, snaps: pd.DataFrame,
                         team_weeks: pd.DataFrame) -> pd.DataFrame:
    """Per player-week usage shares, keyed (player_id, season, week).

    snap_pct comes straight from snap_counts offense_pct (0–1). carry_share
    and target_share_team divide the player's carries/targets by the team's
    total for that week — from stats_team when available, else summed from
    the player-week rows themselves."""
    out = player_weeks[["player_id", "season", "week", "team",
                        "carries", "targets"]].copy()
    out = out.merge(
        snaps[["player_id", "season", "week", "offense_pct"]]
        .rename(columns={"offense_pct": "snap_pct"}),
        on=["player_id", "season", "week"], how="left")

    if team_weeks is not None and not team_weeks.empty:
        totals = team_weeks[["team", "season", "week", "carries", "targets"]] \
            .rename(columns={"carries": "team_carries", "targets": "team_targets"})
    else:  # fallback: reconstruct team volume from the player-week rows
        totals = (player_weeks.groupby(["team", "season", "week"], as_index=False)
                  [["carries", "targets"]].sum()
                  .rename(columns={"carries": "team_carries", "targets": "team_targets"}))
    out = out.merge(totals, on=["team", "season", "week"], how="left")
    out["carry_share"] = out["carries"] / out["team_carries"].replace(0, np.nan)
    out["target_share_team"] = out["targets"] / out["team_targets"].replace(0, np.nan)
    return out[["player_id", "season", "week"] + USAGE_SERIES]


def add_availability(df: pd.DataFrame, injuries: pd.DataFrame,
                     depth: pd.DataFrame) -> pd.DataFrame:
    """Attach current-week injury/depth state + absence history to `df`.

    Absence of an injury row means "not on the report" (healthy), so the
    one-hots default to 0 rather than NaN. Depth features stay NaN where a
    player-week has no chart row — that missingness is itself informative
    and the GBM routes it natively."""
    df = df.copy()

    if injuries is not None and not injuries.empty:
        inj = injuries.drop_duplicates(["player_id", "season", "week"], keep="last")
        df = df.merge(inj, on=["player_id", "season", "week"], how="left")
        status = df.pop("report_status").fillna("")
        practice = df.pop("practice_status").fillna("")
    else:
        status = practice = pd.Series("", index=df.index)
    df["inj_questionable"] = (status == "Questionable").astype(int)
    df["inj_doubtful"] = (status == "Doubtful").astype(int)
    df["inj_out"] = (status == "Out").astype(int)
    df["practice_dnp"] = practice.str.startswith("Did Not").astype(int)
    df["practice_limited"] = practice.str.startswith("Limited").astype(int)
    df["practice_full"] = practice.str.startswith("Full").astype(int)

    if depth is not None and not depth.empty:
        dep = depth.drop_duplicates(["player_id", "season", "week"], keep="last")
        df = df.merge(dep[["player_id", "season", "week", "depth_rank"]],
                      on=["player_id", "season", "week"], how="left")
    else:
        df["depth_rank"] = np.nan
    grp = df.groupby("player_id", sort=False)
    prev_rank = grp["depth_rank"].shift(1)
    df["depth_rank_change"] = prev_rank - df["depth_rank"]  # + = moved up
    df["is_depth_promotion"] = (df["depth_rank_change"] > 0).astype(int)
    return df


def add_absence_history(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """games_missed_last8 / weeks_since_return from the team-week schedule.

    A player's absences are the team games between his consecutive stat rows
    (t-index gaps on his CURRENT team's played schedule — cross-team gaps
    after a trade are approximate and left NaN when the map fails).
    weeks_since_return counts games since the player's most recent absence
    (1 = first game back); NaN = never missed a game in the sample."""
    df = df.copy()
    played = games[games["home_score"].notna()]
    grid = pd.concat([
        played[["season", "week", "home_team"]].rename(columns={"home_team": "team"}),
        played[["season", "week", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True).drop_duplicates()
    grid = grid.sort_values(["team", "season", "week"])
    grid["t_idx"] = grid.groupby("team").cumcount()
    df = df.merge(grid, on=["team", "season", "week"], how="left")

    grp = df.groupby("player_id", sort=False)
    gap = (df["t_idx"] - grp["t_idx"].shift(1) - 1).clip(lower=0)
    df["missed_recent"] = gap
    df["games_missed_last8"] = grp["missed_recent"].transform(
        lambda s: s.rolling(8, min_periods=1).sum())

    # weeks_since_return: appearances since the last gap >= 1 (1 = the
    # return game itself). Rows before any absence stay NaN.
    def _since_return(s: pd.Series) -> pd.Series:
        out, count = [], np.nan
        for g in s.to_numpy():
            count = 1.0 if (g >= 1) else (count + 1.0 if count == count else np.nan)
            out.append(count)
        return pd.Series(out, index=s.index)

    df["weeks_since_return"] = grp["missed_recent"].transform(_since_return)
    return df.drop(columns=["t_idx"])


def add_team_stint(df: pd.DataFrame) -> pd.DataFrame:
    """n_with_team: games played for the player's current team before this one.

    Feeds the prior-shrinkage weight — a team change resets role evidence."""
    df = df.copy()
    grp = df.groupby("player_id", sort=False)
    new_stint = (df["team"] != grp["team"].shift(1)).astype(int)
    stint_id = new_stint.groupby(df["player_id"], sort=False).cumsum()
    df["n_with_team"] = df.groupby([df["player_id"], stint_id], sort=False).cumcount()
    return df


def usage_prior(train_df: pd.DataFrame, targets: list[str]) -> dict:
    """Empirical mean usage per role cell, fit on TRAINING rows only.

    Cell key = (position, depth_rank capped at 4, cohort) where cohort splits
    rookies by draft capital (round-1 / day-2 / rest) and lumps veterans.
    Returns a JSON-serializable table {target: {cell: mean}} plus fallbacks
    per (position, cohort='*') and per position for rows with no chart row."""
    cells = _prior_cells(train_df)
    table: dict = {"targets": {}, "n": int(len(train_df))}
    for t in targets:
        if t not in train_df.columns:
            continue
        vals = pd.to_numeric(train_df[t], errors="coerce")
        by_cell = vals.groupby(cells).mean().dropna()
        by_pos = vals.groupby(train_df["position"]).mean().dropna()
        table["targets"][t] = {
            "cells": {k: round(float(v), 4) for k, v in by_cell.items()},
            "position": {k: round(float(v), 4) for k, v in by_pos.items()},
        }
    return table


def apply_usage_prior(df: pd.DataFrame, table: dict, targets: list[str],
                      shrink_k: float = 4.0) -> pd.DataFrame:
    """Attach {t}_prior / {t}_est and the shared shrinkage weight usage_w.

    usage_est = w·EWM(halflife 5, shifted) + (1−w)·prior with
    w = n_with_team/(n_with_team + shrink_k): a player new to his team leans
    on the role prior; an established player leans on his own recent usage."""
    df = df.copy()
    cells = _prior_cells(df)
    n_recent = pd.to_numeric(df.get("n_with_team"), errors="coerce").fillna(0)
    w = n_recent / (n_recent + shrink_k)
    df["usage_w"] = w.round(4)
    for t in targets:
        spec = table.get("targets", {}).get(t)
        if not spec:
            continue
        prior = cells.map(spec["cells"])
        prior = prior.fillna(df["position"].map(spec["position"]))
        ewm = pd.to_numeric(df.get(f"{t}_ewm5"), errors="coerce")
        w_eff = w.where(ewm.notna(), 0.0)  # no usage history -> pure prior
        df[f"{t}_prior"] = pd.to_numeric(prior, errors="coerce").round(4)
        df[f"{t}_est"] = (w_eff * ewm.fillna(0) + (1 - w_eff) * df[f"{t}_prior"]).round(4)
    return df


def _prior_cells(df: pd.DataFrame) -> pd.Series:
    """Role-cell key per row: 'POS|rank|cohort' (JSON-safe string)."""
    rank = pd.to_numeric(df.get("depth_rank"), errors="coerce") \
        .clip(upper=_DEPTH_CAP).fillna(0).astype(int)  # 0 = no chart row
    rookie = pd.to_numeric(df.get("is_rookie"), errors="coerce").fillna(0) == 1
    draft = pd.to_numeric(df.get("draft_number"), errors="coerce")
    bucket = np.searchsorted(_DRAFT_EDGES, draft.fillna(300))
    cohort = np.where(rookie, [f"r{b}" for b in bucket], "vet")
    return (df["position"].astype(str) + "|" + rank.astype(str) + "|"
            + pd.Series(cohort, index=df.index))
