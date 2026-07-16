"""Monte Carlo over the DES engine + aggregation into slate artifacts."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from gameday.config import FORECASTS_DIR
from gameday.sim.calibrate import TeamProfile, build_profiles
from gameday.sim.engine import TeamResult, simulate_game

log = logging.getLogger(__name__)

DEFAULT_SIMS = 500


def simulate_matchup(home: TeamProfile, away: TeamProfile,
                     n_sims: int = DEFAULT_SIMS, seed: int = 7) -> dict:
    """Run n replicates of one matchup and aggregate."""
    rng = np.random.default_rng(seed)
    home_pts, away_pts = [], []
    home_results: list[TeamResult] = []
    away_results: list[TeamResult] = []

    for _ in range(n_sims):
        result = simulate_game(home, away, rng)
        home_pts.append(result.home.points)
        away_pts.append(result.away.points)
        home_results.append(result.home)
        away_results.append(result.away)

    home_pts, away_pts = np.array(home_pts), np.array(away_pts)
    return {
        "n_sims": n_sims,
        "home_win_prob": round(float((home_pts > away_pts).mean()), 3),
        "away_win_prob": round(float((away_pts > home_pts).mean()), 3),
        "tie_prob": round(float((home_pts == away_pts).mean()), 3),
        "teams": {
            home.team: _aggregate_team(home, home_results, home_pts),
            away.team: _aggregate_team(away, away_results, away_pts),
        },
    }


def _aggregate_team(profile: TeamProfile, results: list[TeamResult],
                    pts: np.ndarray) -> dict:
    n = len(results)
    plays = np.array([r.plays for r in results])
    pass_plays = np.array([r.pass_plays for r in results])
    run_plays = np.array([r.run_plays for r in results])

    pass_rate_by_down = {}
    for down in (1, 2, 3, 4):
        called = sum(r.plays_by_down[down] for r in results)
        passed = sum(r.pass_by_down[down] for r in results)
        pass_rate_by_down[str(down)] = round(passed / called, 3) if called else None

    total_drives = sum(r.drives for r in results) or 1
    drive_outcomes = {
        k: round(sum(r.drive_outcomes[k] for r in results) / total_drives, 3)
        for k in results[0].drive_outcomes
    }

    players = []
    for p in profile.players:
        boxes = [r.box[p.player_id] for r in results]
        snaps = np.array([b.snaps for b in boxes])
        entry = {
            "player_id": p.player_id, "name": p.name, "position": p.position,
            "snaps_mean": round(float(snaps.mean()), 1),
            "snaps_p10": int(np.percentile(snaps, 10)),
            "snaps_p90": int(np.percentile(snaps, 90)),
            "snap_share": round(float(snaps.mean() / max(plays.mean(), 1)), 3),
            "carries_mean": round(float(np.mean([b.carries for b in boxes])), 1),
            "targets_mean": round(float(np.mean([b.targets for b in boxes])), 1),
            "receptions_mean": round(float(np.mean([b.receptions for b in boxes])), 1),
            "touches_mean": round(float(np.mean(
                [b.carries + b.receptions for b in boxes])), 1),
            "rush_yards_mean": round(float(np.mean([b.rush_yards for b in boxes])), 1),
            "rec_yards_mean": round(float(np.mean([b.rec_yards for b in boxes])), 1),
            "tds_mean": round(float(np.mean(
                [b.rush_tds + b.rec_tds for b in boxes])), 2),
        }
        if p.position == "QB":
            entry["pass_yards_mean"] = round(float(np.mean([b.pass_yards for b in boxes])), 1)
            entry["pass_tds_mean"] = round(float(np.mean([b.pass_tds for b in boxes])), 2)
            entry["interceptions_mean"] = round(float(np.mean([b.interceptions for b in boxes])), 2)
        players.append(entry)
    players.sort(key=lambda e: -e["snaps_mean"])

    return {
        "points": {
            "mean": round(float(pts.mean()), 1),
            "p10": int(np.percentile(pts, 10)),
            "p50": int(np.percentile(pts, 50)),
            "p90": int(np.percentile(pts, 90)),
        },
        "plays_mean": round(float(plays.mean()), 1),
        "pass_plays_mean": round(float(pass_plays.mean()), 1),
        "run_plays_mean": round(float(run_plays.mean()), 1),
        "pass_rate_by_down": pass_rate_by_down,
        "drives_mean": round(float(np.mean([r.drives for r in results])), 1),
        "drive_outcomes": drive_outcomes,
        "players": players,
    }


def simulate_slate(player_weeks: pd.DataFrame, games: pd.DataFrame,
                   n_sims: int = DEFAULT_SIMS, seed: int = 7) -> dict:
    """Simulate every upcoming game; write latest_sims.json; return the dict."""
    profiles = build_profiles(player_weeks, games)
    upcoming = games[games["home_score"].isna()]

    sims: dict[str, dict] = {}
    for i, (_, g) in enumerate(upcoming.iterrows()):
        if g.home_team not in profiles or g.away_team not in profiles:
            log.warning("no profile for %s or %s; skipping %s",
                        g.home_team, g.away_team, g.game_id)
            continue
        sims[g.game_id] = simulate_matchup(
            profiles[g.home_team], profiles[g.away_team],
            n_sims=n_sims, seed=seed + i)
        log.info("simulated %s (%d replicates)", g.game_id, n_sims)

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    (FORECASTS_DIR / "latest_sims.json").write_text(json.dumps(sims))
    return sims
