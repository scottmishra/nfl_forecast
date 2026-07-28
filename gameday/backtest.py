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
from gameday.data import rosters
from gameday.features.build import build_features, feature_columns
from gameday.pipeline import load_data
from gameday.sim.calibrate import build_profiles
from gameday.sim.run import simulate_matchup

log = logging.getLogger(__name__)

BACKTEST_DIR = ARTIFACTS_DIR / "backtest"
REPLAY_DIR = ARTIFACTS_DIR / "replay"


def run_backtest(
    source: str = "nflverse",
    seasons: list[int] | None = None,
    test_season: int | None = None,
    weeks: list[int] | None = None,
    n_sims: int = 200,
    engine: str = "gbm",
    seed: int = 7,
    compare: bool = False,
    persist_replay: bool = True,
) -> dict:
    """Replay a past season out-of-sample.

    `persist_replay` writes per-player forecast-vs-actual rows + a scenario
    scorecard to artifacts/replay/{season} (consumed by the dashboard Replay
    view). `compare` additionally scores the player model WITHOUT the
    season-to-season adjustment features, so the scorecard shows the before/after
    lift per segment.
    """
    player_weeks, games = load_data(source, seasons)
    roster_df = None if source == "demo" else rosters.fetch_rosters(
        sorted(int(s) for s in pd.to_numeric(games["season"], errors="coerce").dropna().unique()))
    played = games[games["home_score"].notna()]
    test_season = test_season or int(played["season"].max())
    test_games = played[played["season"] == test_season]
    if weeks:
        test_games = test_games[test_games["week"].isin(weeks)]
    weeks = sorted(test_games["week"].unique().tolist())
    log.info("backtest: season %d, weeks %s, %d games", test_season, weeks, len(test_games))

    feats = build_features(player_weeks, games, rosters=roster_df)

    players_report, scorecard = _backtest_players(
        feats, test_season, weeks, engine, compare=compare, persist_replay=persist_replay)

    report = {
        "source": source, "test_season": test_season, "weeks": weeks,
        "n_games": int(len(test_games)), "engine": engine, "n_sims": n_sims,
        "players": players_report,
        "scorecard": scorecard,
        "sims": (_backtest_sims(player_weeks, games, test_games, n_sims, seed)
                 if n_sims > 0 else {"skipped": True}),
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


def _backtest_players(feats: pd.DataFrame, test_season: int, weeks: list[int],
                      engine: str, compare: bool = False,
                      persist_replay: bool = True):
    """Train per-position models on <test_season and predict the test weeks.

    Returns (per-position/stat aggregate report, scenario scorecard). Persists
    per-player replay rows + the scorecard to artifacts/replay when asked."""
    m = _model_engine(engine)
    # Backtest models land in their own directory so they never clobber
    # production models.
    models_dir = BACKTEST_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    report, adj_preds = _predict_positions(
        m, feats, test_season, weeks, include_adjustments=True, models_dir=models_dir)
    base_preds = None
    if compare:
        log.info("scoring baseline (no season-to-season adjustments) for comparison")
        _, base_preds = _predict_positions(
            m, feats, test_season, weeks, include_adjustments=False, models_dir=models_dir)
    scorecard = _scenario_scorecard(adj_preds, base_preds)
    if persist_replay and not adj_preds.empty:
        _persist_replay(test_season, adj_preds, scorecard)
    return report, scorecard


def _predict_positions(m, feats: pd.DataFrame, test_season: int, weeks: list[int],
                       include_adjustments: bool, models_dir=None):
    """Train each position on <test_season, predict the test weeks.

    Returns (per-position/stat aggregate report, concatenated per-player
    prediction frame with actuals, quantiles, the naive baseline, and the
    adjustment features all carried through from `predict_position`).

    NaN handling lives in the engines: train_position stores its train-time
    medians in the manifest and predict_position applies them."""
    models_dir = models_dir or (BACKTEST_DIR / "models")
    report: dict[str, dict] = {}
    preds = []
    for position in POSITIONS:
        pos = feats[(feats["position"] == position) & feats["fantasy_points"].notna()]
        pos = pos[pos["games_played"] >= 2]
        cols = feature_columns(position, pos, include_adjustments=include_adjustments)

        train = pos[pos["season"] < test_season].copy()
        test = pos[(pos["season"] == test_season) & pos["week"].isin(weeks)].copy()
        if train.empty or test.empty:
            log.warning("%s: no train or test rows; skipping", position)
            continue

        log.info("training %s on %d rows (< %d), scoring %d test rows%s",
                 position, len(train), test_season, len(test),
                 "" if include_adjustments else " [baseline]")
        m.train_position(train, position, cols, models_dir=models_dir)
        pred = m.predict_position(test, position, models_dir=models_dir)
        report[position] = _position_stats_report(pred, position)
        preds.append(pred)

    pred_all = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return report, pred_all


def _position_stats_report(pred: pd.DataFrame, position: str) -> dict:
    """Per-stat MAE / skill-vs-naive / coverage / pinball for one position."""
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
    return stats_report


# --------------------------------------------------------------------------
# scenario scorecard — fantasy-points accuracy sliced by season-transition case
# --------------------------------------------------------------------------

def _segment_metrics(df: pd.DataFrame | None) -> dict | None:
    """Fantasy-points MAE / skill-vs-naive / coverage for a slice of rows."""
    if df is None or df.empty or "fantasy_points" not in df:
        return None
    y = pd.to_numeric(df["fantasy_points"], errors="coerce").to_numpy()
    p50 = pd.to_numeric(df["fantasy_points_p50"], errors="coerce").to_numpy()
    p10 = pd.to_numeric(df["fantasy_points_p10"], errors="coerce").to_numpy()
    p90 = pd.to_numeric(df["fantasy_points_p90"], errors="coerce").to_numpy()
    naive = pd.to_numeric(df["fantasy_points_r8"], errors="coerce").fillna(0).to_numpy()
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        return None
    y, p50, p10, p90, naive = y[mask], p50[mask], p10[mask], p90[mask], naive[mask]
    mae = float(np.mean(np.abs(y - p50)))
    mae_naive = float(np.mean(np.abs(y - naive)))
    return {
        "n": int(mask.sum()),
        "mae_p50": round(mae, 3),
        "mae_naive": round(mae_naive, 3),
        "skill_vs_naive": round(1.0 - mae / mae_naive, 3) if mae_naive > 0 else None,
        "coverage80": round(float(np.mean((y >= p10) & (y <= p90))), 3),
    }


def _segments(df: pd.DataFrame):
    """Yield (group, label, subframe) slices used by the scorecard."""
    wk = pd.to_numeric(df.get("week"), errors="coerce")
    yield "phase", "early (wk1-4)", df[wk <= 4]
    yield "phase", "mid (wk5-13)", df[(wk >= 5) & (wk <= 13)]
    yield "phase", "late (wk14+)", df[wk >= 14]
    exp = pd.to_numeric(df.get("years_exp"), errors="coerce")
    rook = pd.to_numeric(df.get("is_rookie"), errors="coerce").fillna(0)
    yield "experience", "rookie", df[rook == 1]
    yield "experience", "2nd year", df[(rook == 0) & (exp == 1)]
    yield "experience", "veteran (3+)", df[(rook == 0) & (exp >= 2)]
    newt = pd.to_numeric(df.get("is_new_team"), errors="coerce").fillna(0)
    yield "team", "changed team", df[newt == 1]
    yield "team", "same team", df[newt == 0]


def _scorecard_for(df: pd.DataFrame) -> dict:
    card: dict = {"overall": _segment_metrics(df), "segments": {}}
    for group, label, sub in _segments(df):
        card["segments"].setdefault(group, {})[label] = _segment_metrics(sub)
    return card


def _scenario_scorecard(adjusted_preds, baseline_preds) -> dict:
    """Fantasy-points scorecard, per segment, for the adjusted model (always)
    and the baseline model (when a comparison run was requested)."""
    out = {"adjusted": _scorecard_for(adjusted_preds)}
    if baseline_preds is not None and not baseline_preds.empty:
        out["baseline"] = _scorecard_for(baseline_preds)
    return out


# --------------------------------------------------------------------------
# per-player replay artifact (consumed by the dashboard Replay view)
# --------------------------------------------------------------------------

def _replay_columns(df: pd.DataFrame) -> pd.DataFrame:
    ident = ["player_id", "player_display_name", "position", "team", "opponent_team",
             "is_home", "season", "week", "game_id",
             "years_exp", "age", "is_rookie", "is_new_team"]
    all_stats = sorted({s for v in POSITION_STATS.values() for s in v})
    stat_cols = [f"{stat}{suf}" for stat in all_stats
                 for suf in ("", "_p10", "_p25", "_p50", "_p75", "_p90", "_r8")]
    keep = [c for c in ident + stat_cols if c in df.columns]
    return df[keep].copy()


def _persist_replay(season: int, preds: pd.DataFrame, scorecard: dict) -> None:
    sdir = REPLAY_DIR / str(season)
    sdir.mkdir(parents=True, exist_ok=True)
    _replay_columns(preds).to_parquet(sdir / "players.parquet", index=False)
    (sdir / "scorecard.json").write_text(json.dumps(scorecard, indent=2))
    log.info("wrote replay artifacts to %s", sdir)


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
    if s.get("skipped"):
        lines.append("\nGAME SIMULATIONS  (skipped — run with --sims N)")
        return "\n".join(lines)
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


def format_scorecard(report: dict) -> str:
    """Render the season-to-season scorecard (fantasy points, sliced by segment;
    adjusted vs baseline when a comparison run was requested)."""
    sc = report.get("scorecard") or {}
    adj = sc.get("adjusted")
    if not adj:
        return ""
    base = sc.get("baseline")
    header = (f"\nSEASON-TO-SEASON SCORECARD — season {report['test_season']} · fantasy points"
              + ("  (adjusted vs baseline)" if base else "  (adjusted model)"))
    cols = f"{'segment':16} {'n':>5} {'MAE':>7} {'skill':>8}"
    if base:
        cols += f" {'base skill':>11} {'Δskill':>8}"
    lines = [header, cols]

    def row(label, a, b):
        if not a:
            return f"{label:16} {'—':>5}"
        skill = f"{a['skill_vs_naive']:+.1%}" if a["skill_vs_naive"] is not None else "—"
        out = f"{label:16} {a['n']:>5} {a['mae_p50']:>7.2f} {skill:>8}"
        if base is not None:
            if b and b["skill_vs_naive"] is not None and a["skill_vs_naive"] is not None:
                out += f" {b['skill_vs_naive']:>+10.1%} {a['skill_vs_naive'] - b['skill_vs_naive']:>+8.1%}"
            else:
                out += f" {'—':>11} {'—':>8}"
        return out

    lines.append(row("OVERALL", adj.get("overall"), base.get("overall") if base else None))
    for group in ("phase", "experience", "team"):
        lines.append(f"— {group} —")
        aseg = adj.get("segments", {}).get(group, {})
        bseg = base.get("segments", {}).get(group, {}) if base else {}
        for label in aseg:
            lines.append(row(label, aseg.get(label), bseg.get(label)))
    return "\n".join(lines)
