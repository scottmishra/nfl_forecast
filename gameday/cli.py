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
    week_list: list[int] | None = None
    if weeks:
        if "-" in weeks:
            lo, hi = weeks.split("-")
            week_list = list(range(int(lo), int(hi) + 1))
        else:
            week_list = [int(w) for w in weeks.split(",")]

    report = bt.run_backtest(
        source=source, seasons=season_list,
        test_season=test_season or None, weeks=week_list,
        n_sims=sims, engine=engine,
    )
    typer.echo(bt.format_report(report))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Serve the API and dashboard."""
    import uvicorn

    uvicorn.run("gameday.api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
