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
def demo(engine: str = typer.Option("gbm", help="gbm | neural")):
    """Generate synthetic seasons, train, and forecast — no network needed."""
    from gameday import pipeline

    pipeline.run(source="demo", engine=engine)
    typer.echo("Demo artifacts written. Run `gameday serve` and open http://localhost:8000")


@app.command()
def forecast(
    seasons: str = typer.Option("", help="comma-separated, e.g. 2020,2021,2022"),
    engine: str = typer.Option("gbm", help="gbm | neural"),
    buzz: str = typer.Option("neutral", help="buzz provider: neutral | reddit"),
    live_weather: bool = typer.Option(True, help="pull Open-Meteo forecasts for the slate"),
):
    """Full pipeline against real nflverse data."""
    from gameday import pipeline

    season_list = [int(s) for s in seasons.split(",") if s] or settings.seasons
    pipeline.run(source="nflverse", seasons=season_list, engine=engine,
                 buzz_provider=buzz, live_weather=live_weather)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Serve the API and dashboard."""
    import uvicorn

    uvicorn.run("gameday.api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
