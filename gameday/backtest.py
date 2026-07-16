"""Walk-forward backtesting on historical games.

Replays a past season as if each week were an upcoming slate:

  * Player models train ONLY on seasons before the test season, then predict
    every test-week player row. Features are already leakage-safe (each row
    sees only pre-kickoff information), so this is a true out-of-sample test.
  * Game sims re-calibrate team profiles per week from strictly-prior games
    only, simulate each real matchup, and score the results against what
    actually happened.

Scored against actuals:
  players — MAE of the median forecast, MAE of a naive trailing-8-game
            baseline (skill = improvement over naive), p10–p90 interval
            coverage (target ≈ 0.80), and mean pinball loss
  sims    — Brier score and reliability bins for home win probability,
            favorite accuracy, per-team and total points MAE, play-count and
            pass-rate MAE vs (approximate) actual play counts

Backtest model artifacts go to artifacts/backtest/models so they never
clobber production models; the report lands in artifacts/backtest/report.json.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from gameday.config import ARTIFACTS_DIR, POSITION_STATS, POSITIONS, QUANTILES
from gameday.features.build import build_features, feature_columns
from gameday.pipeline import load_data
from gameday.sim.calibrate import build_profiles
from gameday.sim.run import simulate_matchup

log = logging.getLogger(__name__)

BACKTEST_DIR = ARTIFACTS_DIR / "backtest"


def run_backtest(
    source: str = "nflverse",
    seasons: list[int] | None = None,
    test_season: int | None = None,
    weeks: list[int] | None = None,
    n_sims: int = 200,
    engine: str = "gbm",
    seed: int = 7,
) -> dict:
    player_weeks, games = load_data(source, seasons)
    played = games[games["home_score"].notna()]
    test_season = test_season or int(played["season"].max())
    test_games = played[played["season"] == test_season]
    if weeks:
        test_games = test_games[test_games["week"].isin(weeks)]
    weeks = sorted(test_games["week"].unique().tolist())
    log.info("backtest: season %d, weeks %s, %d games", test_season, weeks, len(test_games))

    feats = build_features(player_weeks, games)

    report = {
        "source": source, "test_season": test_season, "weeks": weeks,
        "n_games": int(len(test_games)), "engine": engine, "n_sims": n_sims,
        "players": _backtest_players(feats, test_season, weeks, engine),
        "sims": _backtest_sims(player_weeks, games, test_games, n_sims, seed),
    }

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    (BACKTEST_DIR / "report.json").write_text(json.dumps(report, indent=2))
    log.info("wrote %s", BACKTEST_DIR / "report.json")
    return report


# --------------------------------------------------------------------------
# player stat-line evaluation
# --------------------------------------------------------------------------

def _model_engine(engine: str):
    if engine == "neural":
        from gameday.models import neural as m
    else:
        from gameday.models import quantile_gbm as m
    return m


def _backtest_players(feats: pd.DataFrame, test_season: int,
                      weeks: list[int], engine: str) -> dict:
    m = _model_engine(engine)
    # Redirect artifacts so a backtest never clobbers production models;
    # restored afterward so later pipeline runs in-process are unaffected.
    orig_models_dir = m.MODELS_DIR
    m.MODELS_DIR = BACKTEST_DIR / "models"
    m.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return _score_positions(m, feats, test_season, weeks)
    finally:
        m.MODELS_DIR = orig_models_dir


def _score_positions(m, feats: pd.DataFrame, test_season: int,
                     weeks: list[int]) -> dict:
    out: dict[str, dict] = {}

    for position in POSITIONS:
        pos = feats[(feats["position"] == position) & feats["fantasy_points"].notna()]
        pos = pos[pos["games_played"] >= 2]
        cols = feature_columns(position, pos)

        train = pos[pos["season"] < test_season].copy()
        test = pos[(pos["season"] == test_season) & pos["week"].isin(weeks)].copy()
        if train.empty or test.empty:
            log.warning("%s: no train or test rows; skipping", position)
            continue
        medians = train[cols].median(numeric_only=True)
        train[cols] = train[cols].fillna(medians)
        test[cols] = test[cols].fillna(medians)

        log.info("training %s on %d rows (< %d), scoring %d test rows",
                 position, len(train), test_season, len(test))
        m.train_position(train, position, cols)
        pred = m.predict_position(test, position)

        stats_report = {}
        for stat in POSITION_STATS[position]:
            y = pred[stat].astype(float).values
            p50 = pred[f"{stat}_p50"].values
            p10, p90 = pred[f"{stat}_p10"].values, pred[f"{stat}_p90"].values
            naive = pred[f"{stat}_r8"].fillna(0).values  # trailing 8-game mean

            pinball = float(np.mean([
                np.mean(np.maximum(q * (y - pred[f"{stat}_p{int(q*100):02d}"].values),
                                   (q - 1) * (y - pred[f"{stat}_p{int(q*100):02d}"].values)))
                for q in QUANTILES
            ]))
            mae = float(np.mean(np.abs(y - p50)))
            mae_naive = float(np.mean(np.abs(y - naive)))
            stats_report[stat] = {
                "n": int(len(y)),
                "mae_p50": round(mae, 3),
                "mae_naive": round(mae_naive, 3),
                "skill_vs_naive": round(1.0 - mae / mae_naive, 3) if mae_naive > 0 else None,
                "coverage80": round(float(np.mean((y >= p10) & (y <= p90))), 3),
                "pinball": round(pinball, 3),
            }
        out[position] = stats_report
    return out


# --------------------------------------------------------------------------
# game simulation evaluation
# --------------------------------------------------------------------------

def _truncate_before(player_weeks: pd.DataFrame, games: pd.DataFrame,
                     season: int, week: int):
    """History strictly before (season, week) — what was knowable pre-kickoff."""
    pw_mask = (player_weeks["season"] < season) | (
        (player_weeks["season"] == season) & (player_weeks["week"] < week))
    g_mask = (games["season"] < season) | (
        (games["season"] == season) & (games["week"] < week))
    return player_weeks[pw_mask], games[g_mask]


def _actual_team_stats(player_weeks: pd.DataFrame, team: str,
                       season: int, week: int) -> dict | None:
    rows = player_weeks[(player_weeks["team"] == team)
                        & (player_weeks["season"] == season)
                        & (player_weeks["week"] == week)]
    if rows.empty:
        return None
    attempts, carries = rows["attempts"].sum(), rows["carries"].sum()
    return {
        "plays": float(attempts + carries + 6.0),  # + est. sacks/kneels
        "pass_rate": float(attempts / max(attempts + carries, 1)),
    }


def _backtest_sims(player_weeks: pd.DataFrame, games: pd.DataFrame,
                   test_games: pd.DataFrame, n_sims: int, seed: int) -> dict:
    briers, fav_hits, pts_errs, total_errs, plays_errs, pr_errs = [], [], [], [], [], []
    probs_actuals = []

    for wk, wk_games in test_games.groupby("week"):
        season = int(wk_games["season"].iloc[0])
        pw_hist, g_hist = _truncate_before(player_weeks, games, season, int(wk))
        if pw_hist.empty:
            continue
        profiles = build_profiles(pw_hist, g_hist)

        for i, (_, g) in enumerate(wk_games.iterrows()):
            if g.home_team not in profiles or g.away_team not in profiles:
                continue
            agg = simulate_matchup(profiles[g.home_team], profiles[g.away_team],
                                   n_sims=n_sims, seed=seed + int(wk) * 100 + i)
            p_home = agg["home_win_prob"] + 0.5 * agg["tie_prob"]
            home_won = float(g.home_score > g.away_score) + 0.5 * float(g.home_score == g.away_score)

            briers.append((p_home - home_won) ** 2)
            probs_actuals.append((p_home, home_won))
            if p_home != 0.5:
                fav_hits.append(float((p_home > 0.5) == (home_won > 0.5)))

            sim_h = agg["teams"][g.home_team]
            sim_a = agg["teams"][g.away_team]
            pts_errs += [abs(sim_h["points"]["p50"] - g.home_score),
                         abs(sim_a["points"]["p50"] - g.away_score)]
            total_errs.append(abs((sim_h["points"]["p50"] + sim_a["points"]["p50"])
                                  - (g.home_score + g.away_score)))
            for team, sim_t in ((g.home_team, sim_h), (g.away_team, sim_a)):
                actual = _actual_team_stats(player_weeks, team, season, int(wk))
                if actual:
                    plays_errs.append(abs(sim_t["plays_mean"] - actual["plays"]))
                    sim_pr = sim_t["pass_plays_mean"] / max(
                        sim_t["pass_plays_mean"] + sim_t["run_plays_mean"], 1)
                    pr_errs.append(abs(sim_pr - actual["pass_rate"]))
        log.info("simulated backtest week %s (%d games)", wk, len(wk_games))

    # Reliability: bucket forecast probs, compare with realized frequencies.
    bins = []
    if probs_actuals:
        arr = np.array(probs_actuals)
        for lo in np.arange(0.0, 1.0, 0.2):
            mask = (arr[:, 0] >= lo) & (arr[:, 0] < lo + 0.2 + (lo >= 0.8) * 1e-9)
            if mask.sum():
                bins.append({
                    "bin": f"{lo:.1f}-{lo + 0.2:.1f}", "n": int(mask.sum()),
                    "forecast": round(float(arr[mask, 0].mean()), 3),
                    "realized": round(float(arr[mask, 1].mean()), 3),
                })

    def _m(x):
        return round(float(np.mean(x)), 3) if x else None

    return {
        "n_games": len(briers),
        "brier": _m(briers),
        "brier_coinflip": 0.25,
        "favorite_accuracy": _m(fav_hits),
        "points_mae": _m(pts_errs),
        "total_points_mae": _m(total_errs),
        "plays_mae": _m(plays_errs),
        "pass_rate_mae": _m(pr_errs),
        "reliability": bins,
    }


# --------------------------------------------------------------------------
# console rendering
# --------------------------------------------------------------------------

def format_report(report: dict) -> str:
    lines = [
        f"\nBACKTEST — season {report['test_season']}, weeks {report['weeks'][0]}–{report['weeks'][-1]}"
        f" · {report['n_games']} games · engine={report['engine']} · {report['n_sims']} sims/game",
        "\nPLAYER FORECASTS (vs naive trailing-8-game average)",
        f"{'pos':4} {'stat':17} {'n':>5} {'MAE p50':>8} {'naive':>7} {'skill':>7} {'cov80':>6} {'pinball':>8}",
    ]
    for pos, stats in report["players"].items():
        for stat, r in stats.items():
            skill = f"{r['skill_vs_naive']:+.1%}" if r["skill_vs_naive"] is not None else "—"
            lines.append(
                f"{pos:4} {stat:17} {r['n']:>5} {r['mae_p50']:>8.2f} {r['mae_naive']:>7.2f}"
                f" {skill:>7} {r['coverage80']:>6.0%} {r['pinball']:>8.3f}")

    s = report["sims"]
    lines += [
        "\nGAME SIMULATIONS",
        f"  games scored        {s['n_games']}",
        f"  Brier (home win)    {s['brier']}   (coin flip = 0.250)",
        f"  favorite accuracy   {s['favorite_accuracy']}",
        f"  points MAE / team   {s['points_mae']}",
        f"  total points MAE    {s['total_points_mae']}",
        f"  plays MAE / team    {s['plays_mae']}",
        f"  pass-rate MAE       {s['pass_rate_mae']}",
        "  win-prob reliability (forecast → realized):",
    ]
    for b in s["reliability"]:
        lines.append(f"    {b['bin']}  n={b['n']:<4} {b['forecast']:.2f} → {b['realized']:.2f}")
    return "\n".join(lines)
