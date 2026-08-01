"""Gameday CLI.

  gameday demo                 offline end-to-end run on synthetic seasons
  gameday forecast             real data: ingest -> features -> train -> forecast
  gameday serve                start the API + dashboard
"""

from __future__ import annotations

import logging

import typer

from gameday.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = typer.Typer(help="NFL game-day forecaster", no_args_is_help=True)


def _parse_weeks(weeks: str) -> list[int] | None:
    """Parse a weeks option: '1-8' -> [1..8], '1,3,5' -> [1,3,5], '' -> None."""
    if not weeks:
        return None
    if "-" in weeks:
        lo, hi = weeks.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(w) for w in weeks.split(",")]


@app.command()
def demo(
    engine: str = typer.Option("gbm", help="gbm | neural"),
    sims: int = typer.Option(500, help="game-sim replicates per matchup (0 disables)"),
):
    """Generate synthetic seasons, train, and forecast — no network needed."""
    from gameday import pipeline

    pipeline.run(source="demo", engine=engine, n_sims=sims)
    typer.echo("Demo artifacts written. Run `gameday serve` and open http://localhost:8000")


@app.command()
def forecast(
    seasons: str = typer.Option("", help="comma-separated, e.g. 2020,2021,2022"),
    engine: str = typer.Option("gbm", help="gbm | neural"),
    buzz: str = typer.Option("neutral", help="buzz provider: neutral | reddit"),
    live_weather: bool = typer.Option(True, help="pull Open-Meteo forecasts for the slate"),
    sims: int = typer.Option(500, help="game-sim replicates per matchup (0 disables)"),
):
    """Full pipeline against real nflverse data."""
    from gameday import pipeline

    season_list = [int(s) for s in seasons.split(",") if s] or settings.seasons
    pipeline.run(source="nflverse", seasons=season_list, engine=engine,
                 buzz_provider=buzz, live_weather=live_weather, n_sims=sims)


@app.command()
def backtest(
    source: str = typer.Option("nflverse", help="nflverse | demo"),
    seasons: str = typer.Option("", help="history to load, e.g. 2019,2020,...,2024"),
    test_season: int = typer.Option(0, help="season to replay (default: latest played)"),
    weeks: str = typer.Option("", help="weeks to replay, e.g. 1-8 or 1,3,5 (default: all)"),
    sims: int = typer.Option(200, help="sim replicates per historical game"),
    engine: str = typer.Option("gbm", help="gbm | neural"),
):
    """Walk-forward backtest: replay a past season and score vs actuals."""
    from gameday import backtest as bt

    season_list = [int(s) for s in seasons.split(",") if s] or None
    week_list = _parse_weeks(weeks)

    report = bt.run_backtest(
        source=source, seasons=season_list,
        test_season=test_season or None, weeks=week_list,
        n_sims=sims, engine=engine,
    )
    typer.echo(bt.format_report(report))


@app.command()
def replay(
    season: int = typer.Option(0, help="season to replay (default: latest played)"),
    test_seasons: str = typer.Option("", help="comma-separated seasons to replay in turn, e.g. 2022,2023,2024,2025 (overrides --season)"),
    seasons: str = typer.Option("", help="history to train on, e.g. 2019,2020,...,2024 (default: config seasons)"),
    weeks: str = typer.Option("", help="weeks to replay, e.g. 1-8 or 1,3,5 (default: all)"),
    engine: str = typer.Option("gbm", help="gbm | neural"),
    sims: int = typer.Option(0, help="game-sim replicates per historical game (0 = skip sims, faster)"),
    compare: bool = typer.Option(True, help="also score the previous feature schema (v2) so the scorecard shows the new-vs-old lift"),
):
    """Historical replay + model-variant validation.

    Replays past seasons out-of-sample and writes per-player forecast-vs-actual
    rows and a scenario scorecard to artifacts/replay/{season} for the dashboard
    Replay view. By default the current feature schema (v3) is scored against
    the previous one (v2); --no-compare scores v3 alone (faster). With
    --test-seasons the replay loops several seasons and prints an n-weighted
    combined report at the end; each season's artifacts persist as usual.
    """
    from gameday import backtest as bt

    hist = [int(s) for s in seasons.split(",") if s] or None
    targets = [int(s) for s in test_seasons.split(",") if s] or [season or None]
    variants = None if compare else [dict(name="v3", engine=engine, feature_set="v3")]

    reports = []
    for ts in targets:
        report = bt.run_backtest(
            source="nflverse", seasons=hist, test_season=ts,
            weeks=_parse_weeks(weeks), n_sims=sims, engine=engine,
            persist_replay=True, variants=variants,
        )
        typer.echo(bt.format_scorecard(report))
        typer.echo(bt.format_report(report))
        reports.append(report)

    if len(reports) > 1:
        typer.echo(bt.format_aggregate(bt.aggregate_reports(reports), reports))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Serve the API and dashboard."""
    import uvicorn

    uvicorn.run("gameday.api.server:app", host=host, port=port, reload=reload)


@app.command()
def train(
    seasons: str = typer.Option("", help="train seasons, e.g. 2022,2023,2024 (default: config)"),
    engine: str = typer.Option("gbm", help="gbm | neural"),
    gate: bool = typer.Option(True, help="backtest the latest completed season before packaging"),
    package: bool = typer.Option(True, help="pack a deployable bundle after (gated) training"),
    bundle_dir: str = typer.Option("", help="bundle output dir (default: artifacts/bundles)"),
):
    """Train production models (the train-big machine) and package a bundle.

    Trains into artifacts/models/current, optionally gates on a quick
    out-of-sample backtest of the latest completed season (bounds in config:
    GATE_MIN_SKILL / GATE_COVERAGE80), and packs a model_bundle-*.tar.gz
    ready for `gameday models publish` / a Pi's `gameday models sync`.
    """
    from pathlib import Path

    from gameday import backtest as bt
    from gameday import bundle, pipeline
    from gameday.config import (ARTIFACTS_DIR, GATE_COVERAGE80, GATE_MIN_SKILL,
                                MODELS_DIR)
    from gameday.data import releases

    season_list = [int(s) for s in seasons.split(",") if s] or settings.seasons
    releases.revalidate_current()  # current-season files may sit inside their TTL
    data = pipeline.prepare(source="nflverse", seasons=season_list)
    reports = pipeline.train_models(data.feats, engine=engine, models_dir=MODELS_DIR)
    typer.echo(f"trained {len(reports)} positions -> {MODELS_DIR}")

    metrics = None
    if gate:
        report = bt.run_backtest(source="nflverse", seasons=season_list, n_sims=0,
                                 engine=engine, persist_replay=False)
        overall = ((report.get("scorecard") or {}).get("adjusted") or {}).get("overall") or {}
        metrics = {"test_season": report["test_season"], **overall}
        skill, cov = overall.get("skill_vs_naive"), overall.get("coverage80")
        lo, hi = GATE_COVERAGE80
        problems = []
        if skill is None or skill < GATE_MIN_SKILL:
            problems.append(f"skill_vs_naive={skill} < {GATE_MIN_SKILL}")
        if cov is None or not lo <= cov <= hi:
            problems.append(f"coverage80={cov} outside [{lo}, {hi}]")
        if problems:
            typer.echo(f"GATE FAILED ({report['test_season']}): " + "; ".join(problems))
            if package:
                typer.echo("refusing to package a failing model set")
                raise typer.Exit(1)
        else:
            typer.echo(f"gate passed on {report['test_season']}: "
                       f"skill={skill:+.1%}, coverage80={cov:.0%}")
    if package:
        out = Path(bundle_dir) if bundle_dir else ARTIFACTS_DIR / "bundles"
        path = bundle.pack(MODELS_DIR, out, meta={
            "engine": engine, "train_seasons": season_list, "metrics": metrics})
        typer.echo(f"bundle: {path}")


@app.command()
def refresh(
    horizon_days: int = typer.Option(8, help="skip the run when no game is within this many days"),
    sims: int = typer.Option(300, help="game-sim replicates per matchup (0 disables)"),
    sync_models: bool = typer.Option(True, help="sync the model bundle from the deploy pointer first"),
    pointer: str = typer.Option("", help="deploy pointer path (default: deploy/models.json)"),
    live_weather: bool = typer.Option(False, help="refresh slate weather from Open-Meteo"),
):
    """Nightly Pi refresh: gate -> sync -> fetch -> predict -> persist + status file."""
    from pathlib import Path

    from gameday import refresh as refresh_mod

    code = refresh_mod.run_refresh(
        horizon_days=horizon_days, sims=sims, sync=sync_models,
        pointer=Path(pointer) if pointer else refresh_mod.DEFAULT_POINTER,
        live_weather=live_weather)
    raise typer.Exit(code)


@app.command()
def market(
    season: int = typer.Option(0, help="season to pull (default: the current NFL season)"),
    force: bool = typer.Option(False, help="ignore the per-source cache TTLs"),
):
    """Refresh the draft-board market cross-reference (ESPN / FFToday / Sleeper).

    Deliberately independent of `gameday refresh`, which self-gates in the
    offseason — August is exactly when drafts happen and when the refresh gate
    is closed.
    """
    from gameday.data import market as market_mod
    from gameday.data.releases import current_nfl_season

    season = season or current_nfl_season()
    try:
        meta = market_mod.refresh_market(season, force=force)
    except RuntimeError as exc:
        typer.echo(f"market refresh failed: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"{meta['rows']} players -> {market_mod.MARKET_PATH}")
    for name, cov in meta["coverage"].items():
        typer.echo(f"  {name:<8} {cov['matched']:>4}/{cov['total']} matched "
                   f"· {cov['top100']}/100 of the draftable top 100")


models_app = typer.Typer(help="Model bundle ops: status, publish, sync, rollback",
                         no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("status")
def models_status():
    """Show the installed current/previous bundle versions and gate metrics."""
    import json

    from gameday import bundle
    from gameday.config import MODELS_ROOT

    for which in ("current", "previous"):
        m = bundle.installed_manifest(MODELS_ROOT, which)
        if not m:
            typer.echo(f"{which:9} —")
            continue
        typer.echo(f"{which:9} {m.get('version')}  engine={m.get('engine')}  "
                   f"created={m.get('created_at')}  git={m.get('git_sha')}  "
                   f"positions={','.join(m.get('positions') or {})}")
        if m.get("metrics"):
            typer.echo(f"{'':9} metrics: {json.dumps(m['metrics'])}")


@models_app.command("publish")
def models_publish(
    bundle_path: str = typer.Argument("", help="bundle tar.gz (default: newest in artifacts/bundles)"),
    repo: str = typer.Option("scottmishra/nfl_forecast", help="GitHub repo for the release"),
    dry_run: bool = typer.Option(True, help="print the publish plan without executing"),
):
    """Show how a bundle WOULD be published (gh release + pointer update).

    STUB: prints the exact commands and pointer JSON but never executes them —
    actual publishing happens manually until it's explicitly enabled.
    """
    import datetime as _dt
    import hashlib
    import json
    from pathlib import Path

    from gameday import bundle
    from gameday.config import ARTIFACTS_DIR, ROOT

    if bundle_path:
        path = Path(bundle_path)
    else:
        candidates = sorted((ARTIFACTS_DIR / "bundles").glob("model_bundle-*.tar.gz"))
        if not candidates:
            typer.echo("no bundles under artifacts/bundles — run `gameday train` first")
            raise typer.Exit(1)
        path = candidates[-1]
    manifest = bundle.verify(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    version = manifest["version"]
    pointer = {
        "schema": 1,
        "version": version,
        "url": f"https://github.com/{repo}/releases/download/{version}/{path.name}",
        "sha256": sha,
        "published_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    typer.echo(f"bundle   {path}  ({path.stat().st_size / 1e6:.1f} MB, verified)")
    typer.echo("would run:")
    typer.echo(f"  gh release create {version} \"{path}\" --repo {repo} "
               f"--title \"{version}\" --notes \"engine={manifest.get('engine')} "
               f"train_seasons={manifest.get('train_seasons')}\"")
    typer.echo(f"would write {ROOT / 'deploy' / 'models.json'}:")
    typer.echo(json.dumps(pointer, indent=2))
    typer.echo("then: git add deploy/models.json && git commit && git push")
    if dry_run:
        typer.echo("(dry run — nothing executed)")
    else:
        typer.echo("publish is stubbed: run the printed commands manually "
                   "(requires explicit approval to automate)")
        raise typer.Exit(1)


@models_app.command("sync")
def models_sync(
    pointer: str = typer.Option("", help="pointer file (default: deploy/models.json)"),
):
    """Install the bundle the deploy pointer names (no-op when already current)."""
    from pathlib import Path

    from gameday import bundle
    from gameday.config import MODELS_ROOT
    from gameday.refresh import DEFAULT_POINTER

    version = bundle.sync_from_pointer(
        Path(pointer) if pointer else DEFAULT_POINTER, MODELS_ROOT)
    typer.echo(f"current models: {version}")


@models_app.command("rollback")
def models_rollback():
    """Swap the previous bundle back in as current."""
    from gameday import bundle
    from gameday.config import MODELS_ROOT

    manifest = bundle.rollback(MODELS_ROOT)
    typer.echo(f"rolled back; current is now {manifest.get('version', 'unversioned')}")


if __name__ == "__main__":
    app()
