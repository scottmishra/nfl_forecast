"""End-to-end orchestration: data -> features -> models -> forecasts.

Artifacts land under artifacts/forecasts:
  latest_forecasts.parquet  one row per player in the upcoming slate, with
                            {stat}_p10..p90 quantile columns
  latest_slate.parquet      the upcoming games with venue + weather context
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from gameday.config import (FEATURES_DIR, FORECASTS_DIR, POSITIONS, RAW_DIR,
                            ensure_dirs, settings)
from gameday.data import demo as demo_data
from gameday.data import nflverse, rosters, weather
from gameday.data.teams import TEAMS
from gameday.features.build import build_features, feature_columns
from gameday.models import quantile_gbm

log = logging.getLogger(__name__)


def load_data(source: str = "nflverse", seasons: list[int] | None = None):
    if source == "demo":
        return demo_data.generate(seasons)
    seasons = seasons or settings.seasons
    return nflverse.fetch_player_weeks(seasons), nflverse.fetch_schedules(seasons)


def run(source: str = "nflverse", seasons: list[int] | None = None,
        buzz_provider: str = "neutral", engine: str = "gbm",
        live_weather: bool = False, n_sims: int = 500) -> pd.DataFrame:
    """Train on history, forecast the upcoming slate, persist artifacts."""
    ensure_dirs()
    log.info("loading data (source=%s)", source)
    player_weeks, games = load_data(source, seasons)
    if source != "demo":
        games = _next_week_slate(games)
    roster_df = None if source == "demo" else rosters.fetch_rosters(seasons or settings.seasons)

    if live_weather:
        games = _refresh_upcoming_weather(games)

    log.info("building features (%d player-weeks)", len(player_weeks))
    feats = build_features(player_weeks, games, buzz_provider=buzz_provider, rosters=roster_df)
    feats.to_parquet(FEATURES_DIR / "features.parquet", index=False)

    if engine == "neural":
        from gameday.models import neural as model_engine
    else:
        model_engine = quantile_gbm

    forecasts = []
    for position in POSITIONS:
        pos_df = feats[feats["position"] == position]
        cols = feature_columns(position, pos_df)
        target0 = "fantasy_points"
        hist = pos_df[pos_df[target0].notna()].copy()
        future = pos_df[pos_df[target0].isna()].copy()
        hist = hist[hist["games_played"] >= 2]  # need some form signal
        hist[cols] = hist[cols].fillna(hist[cols].median(numeric_only=True))
        log.info("training %s on %d rows / %d features", position, len(hist), len(cols))
        report = model_engine.train_position(hist, position, cols)
        log.info("%s validation: %s", position, report)

        if future.empty:
            continue
        future[cols] = future[cols].fillna(hist[cols].median(numeric_only=True))
        forecasts.append(model_engine.predict_position(future, position))

    if not forecasts:
        log.warning("no upcoming games found; nothing to forecast")
        return pd.DataFrame()

    result = pd.concat(forecasts, ignore_index=True)
    slate = games[games["home_score"].isna()].copy()
    slate["stadium"] = slate["home_team"].map(lambda t: TEAMS.get(t, {}).get("stadium"))
    slate["city"] = slate["home_team"].map(lambda t: TEAMS.get(t, {}).get("city"))

    result.to_parquet(FORECASTS_DIR / "latest_forecasts.parquet", index=False)
    slate.to_parquet(FORECASTS_DIR / "latest_slate.parquet", index=False)
    log.info("wrote %d player forecasts across %d games", len(result), len(slate))

    if n_sims > 0:
        from gameday.sim.run import simulate_slate

        log.info("running game simulations (%d replicates/game)", n_sims)
        simulate_slate(player_weeks, games, n_sims=n_sims)
    return result


def _next_week_slate(games: pd.DataFrame) -> pd.DataFrame:
    """Restrict the live slate to the next unplayed week only — keep every played
    game (for form/opponent context) but drop upcoming weeks beyond the earliest,
    so the dashboard shows one week rather than the whole remaining schedule."""
    upcoming = games[games["home_score"].isna()]
    if upcoming.empty:
        return games
    nxt = upcoming.sort_values(["season", "week"]).iloc[0]
    keep = games["home_score"].notna() | (
        (games["season"] == nxt["season"]) & (games["week"] == nxt["week"]))
    return games[keep].reset_index(drop=True)


def _refresh_upcoming_weather(games: pd.DataFrame) -> pd.DataFrame:
    """Overwrite temp/wind for upcoming games with live Open-Meteo forecasts."""
    games = games.copy()
    for idx, g in games[games["home_score"].isna()].iterrows():
        try:
            kickoff = dt.datetime.fromisoformat(f"{g.gameday}T{g.get('gametime') or '13:00'}")
        except (TypeError, ValueError):
            kickoff = dt.datetime.now() + dt.timedelta(days=3)
        wx = weather.game_weather(g.home_team, kickoff)
        games.loc[idx, ["temp", "wind"]] = wx["temp_c"], wx["wind_kph"]
    return games
