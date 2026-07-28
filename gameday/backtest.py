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
    variants: list[dict] | None = None,
) -> dict:
    """Replay a past season out-of-sample.

    `persist_replay` writes per-player forecast-vs-actual rows + a scenario
    scorecard to artifacts/replay/{season} (consumed by the dashboard Replay
    view).

    `variants` names the model configurations to score side by side. Each is
    dict(name, engine, feature_set[, include_adjustments]); the first is the
    *primary* whose rows feed the dashboard. Defaults to the current feature
    schema vs the previous one — [v2, v1] — so the scorecard always shows the
    new-vs-old lift per segment. (`compare` is retained for callers of the old
    adjusted/no-adjustments A/B; the default variants supersede it.)
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

    players_reports, scorecard, primary = _backtest_players(
        feats, test_season, weeks, engine, persist_replay=persist_replay,
        variants=variants)

    report = {
        "source": source, "test_season": test_season, "weeks": weeks,
        "n_games": int(len(test_games)), "engine": engine, "n_sims": n_sims,
        "primary_variant": primary,
        "players": players_reports.get(primary, {}),
        "players_by_variant": players_reports,
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


def _default_variants(engine: str) -> list[dict]:
    """New feature schema vs old, on the requested engine. First = primary."""
    return [dict(name="v2", engine=engine, feature_set="v2"),
            dict(name="v1", engine=engine, feature_set="v1")]


def _backtest_players(feats: pd.DataFrame, test_season: int, weeks: list[int],
                      engine: str, persist_replay: bool = True,
                      variants: list[dict] | None = None):
    """Train and score every variant on <test_season, predict the test weeks.

    Returns ({variant: per-position/stat report}, scenario scorecard, primary
    variant name). Persists per-player replay rows + the scorecard to
    artifacts/replay when asked."""
    variants = variants or _default_variants(engine)
    primary = variants[0]["name"]
    reports: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}
    for var in variants:
        name, var_engine = var["name"], var.get("engine", engine)
        m = _model_engine(var_engine)
        # Backtest models land in their own per-variant directory so variants
        # never clobber each other's manifests — or production models.
        models_dir = BACKTEST_DIR / "models" / name
        models_dir.mkdir(parents=True, exist_ok=True)
        log.info("scoring variant %s (engine=%s, feature_set=%s)",
                 name, var_engine, var.get("feature_set", "v2"))
        rep, preds = _predict_positions(
            m, feats, test_season, weeks,
            include_adjustments=var.get("include_adjustments", True),
            models_dir=models_dir, feature_set=var.get("feature_set", "v2"))
        if not preds.empty:
            preds["variant"] = name
            preds["engine"] = var_engine
        reports[name] = rep
        frames[name] = preds
    scorecard = _scenario_scorecard(frames, primary)
    if persist_replay and not frames[primary].empty:
        _persist_replay(test_season, frames, primary, scorecard)
    return reports, scorecard, primary


def _predict_positions(m, feats: pd.DataFrame, test_season: int, weeks: list[int],
                       include_adjustments: bool, models_dir=None,
                       feature_set: str = "v2"):
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
        cols = feature_columns(position, pos, include_adjustments=include_adjustments,
                               feature_set=feature_set)

        train = pos[pos["season"] < test_season].copy()
        test = pos[(pos["season"] == test_season) & pos["week"].isin(weeks)].copy()
        if train.empty or test.empty:
            log.warning("%s: no train or test rows; skipping", position)
            continue

        log.info("training %s on %d rows (< %d), scoring %d test rows [%s]",
                 position, len(train), test_season, len(test), feature_set)
        m.train_position(train, position, cols, models_dir=models_dir)
        pred = m.predict_position(test, position, models_dir=models_dir)
        report[position] = _position_stats_report(pred, position)
        preds.append(pred)

    pred_all = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return report, pred_all


def _pinball_by_q(y: np.ndarray, pred: pd.DataFrame, stat: str) -> dict[str, float]:
    """Per-quantile pinball loss, keyed p10..p90."""
    out = {}
    for q in QUANTILES:
        err = y - pred[f"{stat}_p{int(q * 100):02d}"].values
        out[f"p{int(q * 100):02d}"] = round(
            float(np.mean(np.maximum(q * err, (q - 1) * err))), 3)
    return out


def _position_stats_report(pred: pd.DataFrame, position: str) -> dict:
    """Per-stat MAE / skill-vs-naive / coverage / pinball for one position."""
    stats_report = {}
    for stat in POSITION_STATS[position]:
        y = pred[stat].astype(float).values
        p50 = pred[f"{stat}_p50"].values
        p10, p90 = pred[f"{stat}_p10"].values, pred[f"{stat}_p90"].values
        p25, p75 = pred[f"{stat}_p25"].values, pred[f"{stat}_p75"].values
        naive = pred[f"{stat}_r8"].fillna(0).values  # trailing 8-game mean

        by_q = _pinball_by_q(y, pred, stat)
        mae = float(np.mean(np.abs(y - p50)))
        mae_naive = float(np.mean(np.abs(y - naive)))
        stats_report[stat] = {
            "n": int(len(y)),
            "mae_p50": round(mae, 3),
            "mae_naive": round(mae_naive, 3),
            "skill_vs_naive": round(1.0 - mae / mae_naive, 3) if mae_naive > 0 else None,
            "coverage80": round(float(np.mean((y >= p10) & (y <= p90))), 3),
            "coverage50": round(float(np.mean((y >= p25) & (y <= p75))), 3),
            "interval_width80": round(float(np.mean(p90 - p10)), 3),
            "pinball": round(float(np.mean(list(by_q.values()))), 3),
            "pinball_by_q": by_q,
        }
    return stats_report


# --------------------------------------------------------------------------
# scenario scorecard — fantasy-points accuracy sliced by season-transition case
# --------------------------------------------------------------------------

def _segment_metrics(df: pd.DataFrame | None) -> dict | None:
    """Fantasy-points MAE / skill-vs-naive / coverage / pinball for a slice."""
    if df is None or df.empty or "fantasy_points" not in df:
        return None
    y = pd.to_numeric(df["fantasy_points"], errors="coerce").to_numpy()
    quants = {q: pd.to_numeric(df[f"fantasy_points_p{int(q * 100):02d}"],
                               errors="coerce").to_numpy() for q in QUANTILES}
    naive = pd.to_numeric(df["fantasy_points_r8"], errors="coerce").fillna(0).to_numpy()
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        return None
    y, naive = y[mask], naive[mask]
    quants = {q: v[mask] for q, v in quants.items()}
    p10, p25, p50, p75, p90 = (quants[q] for q in QUANTILES)
    mae = float(np.mean(np.abs(y - p50)))
    mae_naive = float(np.mean(np.abs(y - naive)))
    pinball = float(np.mean([
        np.mean(np.maximum(q * (y - v), (q - 1) * (y - v))) for q, v in quants.items()]))
    return {
        "n": int(mask.sum()),
        "mae_p50": round(mae, 3),
        "mae_naive": round(mae_naive, 3),
        "skill_vs_naive": round(1.0 - mae / mae_naive, 3) if mae_naive > 0 else None,
        "coverage80": round(float(np.mean((y >= p10) & (y <= p90))), 3),
        "coverage50": round(float(np.mean((y >= p25) & (y <= p75))), 3),
        "interval_width80": round(float(np.mean(p90 - p10)), 3),
        "pinball": round(pinball, 3),
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


def _scenario_scorecard(frames: dict[str, pd.DataFrame], primary: str) -> dict:
    """Fantasy-points scorecard, per segment, for every scored variant.

    The primary variant is also exposed under the legacy "adjusted" key and
    the first comparison variant under "baseline", so the dashboard Replay
    tab keeps rendering without changes."""
    out: dict = {"primary": primary, "variants": {}}
    for name, df in frames.items():
        if df is not None and not df.empty:
            out["variants"][name] = _scorecard_for(df)
    if primary in out["variants"]:
        out["adjusted"] = out["variants"][primary]
    others = [n for n in out["variants"] if n != primary]
    if others:
        out["baseline"] = out["variants"][others[0]]
    return out


# --------------------------------------------------------------------------
# per-player replay artifact (consumed by the dashboard Replay view)
# --------------------------------------------------------------------------

def _replay_columns(df: pd.DataFrame) -> pd.DataFrame:
    ident = ["player_id", "player_display_name", "position", "team", "opponent_team",
             "is_home", "season", "week", "game_id", "engine", "variant",
             "years_exp", "age", "is_rookie", "is_new_team"]
    all_stats = sorted({s for v in POSITION_STATS.values() for s in v})
    stat_cols = [f"{stat}{suf}" for stat in all_stats
                 for suf in ("", "_p10", "_p25", "_p50", "_p75", "_p90", "_r8")]
    keep = [c for c in ident + stat_cols if c in df.columns]
    return df[keep].copy()


def _persist_replay(season: int, frames: dict[str, pd.DataFrame], primary: str,
                    scorecard: dict) -> None:
    """players.parquet keeps its one-row-per-player-week contract (primary
    variant only, tagged with engine/variant columns) so the replay API and
    dashboard need no changes; the full multi-variant frame lands beside it
    in variants.parquet for offline comparison."""
    sdir = REPLAY_DIR / str(season)
    sdir.mkdir(parents=True, exist_ok=True)
    _replay_columns(frames[primary]).to_parquet(sdir / "players.parquet", index=False)
    scored = [f for f in frames.values() if f is not None and not f.empty]
    if len(scored) > 1:
        pd.concat([_replay_columns(f) for f in scored], ignore_index=True) \
            .to_parquet(sdir / "variants.parquet", index=False)
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
        f"{'pos':4} {'stat':17} {'n':>5} {'MAE p50':>8} {'naive':>7} {'skill':>7}"
        f" {'cov80':>6} {'cov50':>6} {'pinball':>8}",
    ]
    for pos, stats in report["players"].items():
        for stat, r in stats.items():
            skill = f"{r['skill_vs_naive']:+.1%}" if r["skill_vs_naive"] is not None else "—"
            lines.append(
                f"{pos:4} {stat:17} {r['n']:>5} {r['mae_p50']:>8.2f} {r['mae_naive']:>7.2f}"
                f" {skill:>7} {r['coverage80']:>6.0%} {r.get('coverage50', float('nan')):>6.0%}"
                f" {r['pinball']:>8.3f}")

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
    """Render the scenario scorecard (fantasy points, sliced by segment;
    primary variant vs the first comparison variant when one was scored)."""
    sc = report.get("scorecard") or {}
    adj = sc.get("adjusted")
    if not adj:
        return ""
    base = sc.get("baseline")
    primary = sc.get("primary", "adjusted")
    names = [n for n in sc.get("variants", {}) if n != primary]
    versus = f"  ({primary} vs {names[0]})" if base and names else \
        ("  (adjusted vs baseline)" if base else f"  ({primary})")
    header = f"\nSCENARIO SCORECARD — season {report['test_season']} · fantasy points{versus}"
    cols = f"{'segment':16} {'n':>5} {'MAE':>7} {'skill':>8} {'cov80':>6} {'cov50':>6}"
    if base:
        cols += f" {'base skill':>11} {'Δskill':>8}"
    lines = [header, cols]

    def row(label, a, b):
        if not a:
            return f"{label:16} {'—':>5}"
        skill = f"{a['skill_vs_naive']:+.1%}" if a["skill_vs_naive"] is not None else "—"
        out = f"{label:16} {a['n']:>5} {a['mae_p50']:>7.2f} {skill:>8}"
        out += f" {a.get('coverage80', float('nan')):>6.0%} {a.get('coverage50', float('nan')):>6.0%}"
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


# --------------------------------------------------------------------------
# multi-season aggregation (the CLI `replay --test-seasons ...` loop)
# --------------------------------------------------------------------------

def _combine_metrics(metrics: list[dict | None]) -> dict | None:
    """n-weighted mean of segment/stat metric dicts; skill recomputed from the
    combined MAEs (a mean of ratios would over-weight small seasons)."""
    ms = [m for m in metrics if m]
    if not ms:
        return None
    n = sum(m["n"] for m in ms)
    out: dict = {"n": n}
    keys = [k for k in ms[0] if k not in ("n", "skill_vs_naive", "pinball_by_q")]
    for k in keys:
        vals = [(m[k], m["n"]) for m in ms if isinstance(m.get(k), (int, float))]
        if vals:
            out[k] = round(sum(v * w for v, w in vals) / sum(w for _, w in vals), 3)
    if out.get("mae_naive"):
        out["skill_vs_naive"] = round(1.0 - out["mae_p50"] / out["mae_naive"], 3)
    if all("pinball_by_q" in m for m in ms):
        out["pinball_by_q"] = {
            qk: round(sum(m["pinball_by_q"][qk] * m["n"] for m in ms) / n, 3)
            for qk in ms[0]["pinball_by_q"]}
    return out


def aggregate_reports(reports: list[dict]) -> dict:
    """Combine several seasons' replay reports into one (n-weighted) report:
    per-variant scorecards (overall + segments) and per-position stat tables."""
    seasons = [r["test_season"] for r in reports]
    variant_names = list(dict.fromkeys(
        n for r in reports for n in r.get("scorecard", {}).get("variants", {})))
    primary = reports[0].get("primary_variant") or (variant_names[0] if variant_names else None)

    scorecards: dict = {}
    for name in variant_names:
        cards = [r["scorecard"]["variants"].get(name) for r in reports
                 if name in r.get("scorecard", {}).get("variants", {})]
        combined = {"overall": _combine_metrics([c.get("overall") for c in cards]),
                    "segments": {}}
        for card in cards:
            for group, seg in card.get("segments", {}).items():
                for label in seg:
                    combined["segments"].setdefault(group, {}).setdefault(label, [])
        for group, labels in combined["segments"].items():
            for label in labels:
                combined["segments"][group][label] = _combine_metrics(
                    [c.get("segments", {}).get(group, {}).get(label) for c in cards])
        scorecards[name] = combined

    players: dict = {}
    for name in variant_names:
        by_pos: dict = {}
        for r in reports:
            for pos, stats in r.get("players_by_variant", {}).get(name, {}).items():
                for stat, m in stats.items():
                    by_pos.setdefault(pos, {}).setdefault(stat, []).append(m)
        players[name] = {pos: {stat: _combine_metrics(ms) for stat, ms in stats.items()}
                         for pos, stats in by_pos.items()}

    return {"test_seasons": seasons, "primary_variant": primary,
            "scorecard": {"primary": primary, "variants": scorecards},
            "players": players}


def format_aggregate(agg: dict, reports: list[dict]) -> str:
    """Render the combined multi-season table: per-variant fantasy-points
    metrics per season plus the n-weighted aggregate."""
    seasons = agg["test_seasons"]
    lines = [
        f"\nMULTI-SEASON REPLAY — seasons {seasons[0]}–{seasons[-1]} · fantasy points (overall)",
        f"{'variant':8} {'season':>7} {'n':>6} {'MAE':>7} {'skill':>8} {'cov80':>6} {'cov50':>6} {'pinball':>8}",
    ]

    def row(name, season, m):
        if not m:
            return f"{name:8} {season:>7} {'—':>6}"
        skill = f"{m['skill_vs_naive']:+.1%}" if m.get("skill_vs_naive") is not None else "—"
        return (f"{name:8} {season:>7} {m['n']:>6} {m['mae_p50']:>7.2f} {skill:>8}"
                f" {m.get('coverage80', float('nan')):>6.0%}"
                f" {m.get('coverage50', float('nan')):>6.0%}"
                f" {m.get('pinball', float('nan')):>8.3f}")

    for name in agg["scorecard"]["variants"]:
        for r in reports:
            card = r.get("scorecard", {}).get("variants", {}).get(name)
            lines.append(row(name, r["test_season"], card and card.get("overall")))
        lines.append(row(name, "ALL", agg["scorecard"]["variants"][name]["overall"]))
        lines.append("")
    return "\n".join(lines)
