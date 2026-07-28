"""End-to-end orchestration: data -> features -> models -> forecasts.

The pipeline is decomposed into four composable steps so training and
inference can run on different machines ("train big, deploy small"):

  prepare()            data -> features (writes data/features/features.parquet)
  train_models()       fit per-position engines into a models dir
  predict_slate()      predict the upcoming slate with already-trained models
  persist_forecasts()  atomically publish forecast artifacts + latest_meta.json

`run()` composes all four — the original fused dev path.

Artifacts land under artifacts/forecasts:
  latest_forecasts.parquet  one row per player in the upcoming slate, with
                            {stat}_p10..p90 quantile columns
  latest_slate.parquet      the upcoming games with venue + weather context
  latest_meta.json          provenance: when, which season/week, which engine
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from gameday.config import (FEATURES_DIR, FORECASTS_DIR, MODELS_DIR, POSITIONS,
                            ensure_dirs, settings)
from gameday.data import demo as demo_data
from gameday.data import nflverse, rosters, weather
from gameday.data.teams import TEAMS
from gameday.features.build import build_features, feature_columns

log = logging.getLogger(__name__)


def load_data(source: str = "nflverse", seasons: list[int] | None = None):
    if source == "demo":
        return demo_data.generate(seasons)
    seasons = seasons or settings.seasons
    return nflverse.fetch_player_weeks(seasons), nflverse.fetch_schedules(seasons)


@dataclass
class PipelineData:
    """Everything prepare() assembles for the downstream steps."""

    feats: pd.DataFrame
    games: pd.DataFrame
    player_weeks: pd.DataFrame
    source: str
    seasons: list[int] | None = None
    meta: dict = field(default_factory=dict)


def model_engine(engine: str):
    """Resolve an engine name to its module (two-function contract:
    train_position / predict_position emitting {stat}_p10..p90)."""
    if engine == "neural":
        from gameday.models import neural as m
        return m
    from gameday.models import quantile_gbm as m
    return m


def prepare(source: str = "nflverse", seasons: list[int] | None = None,
            buzz_provider: str = "neutral", live_weather: bool = False) -> PipelineData:
    """Load data, restrict to the next unplayed week, build features."""
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
    return PipelineData(feats=feats, games=games, player_weeks=player_weeks,
                        source=source, seasons=seasons)


def train_models(feats: pd.DataFrame, engine: str = "gbm",
                 models_dir: Path = MODELS_DIR) -> dict:
    """Fit one engine per position on all historical rows. Returns reports."""
    m = model_engine(engine)
    reports: dict[str, dict] = {}
    for position in POSITIONS:
        pos_df = feats[feats["position"] == position]
        cols = feature_columns(position, pos_df)
        hist = pos_df[pos_df["fantasy_points"].notna()].copy()
        hist = hist[hist["games_played"] >= 2]  # need some form signal
        if hist.empty:
            log.warning("no historical rows for %s; skipping", position)
            continue
        log.info("training %s on %d rows / %d features", position, len(hist), len(cols))
        reports[position] = m.train_position(hist, position, cols, models_dir=models_dir)
        log.info("%s validation: %s", position, reports[position])
    return reports


def predict_slate(feats: pd.DataFrame, engine: str = "gbm",
                  models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Predict every upcoming-slate row using already-trained models."""
    m = model_engine(engine)
    forecasts = []
    for position in POSITIONS:
        pos_df = feats[feats["position"] == position]
        future = pos_df[pos_df["fantasy_points"].isna()].copy()
        if future.empty:
            continue
        forecasts.append(m.predict_position(future, position, models_dir=models_dir))
    if not forecasts:
        return pd.DataFrame()
    return pd.concat(forecasts, ignore_index=True)


def persist_forecasts(result: pd.DataFrame, games: pd.DataFrame,
                      meta: dict | None = None,
                      forecasts_dir: Path = FORECASTS_DIR) -> pd.DataFrame:
    """Atomically publish forecast artifacts (tmp file + os.replace per file),
    so a reader never sees a half-written parquet. Returns the slate frame."""
    slate = games[games["home_score"].isna()].copy()
    slate["stadium"] = slate["home_team"].map(lambda t: TEAMS.get(t, {}).get("stadium"))
    slate["city"] = slate["home_team"].map(lambda t: TEAMS.get(t, {}).get("city"))

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    full_meta = {
        "generated_at": now,
        "as_of": now,
        "season": int(slate["season"].iloc[0]) if not slate.empty else None,
        "week": int(slate["week"].iloc[0]) if not slate.empty else None,
        "n_players": int(len(result)),
        "n_games": int(len(slate)),
        **(meta or {}),
    }

    forecasts_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(forecasts_dir / "latest_forecasts.parquet",
                  lambda p: result.to_parquet(p, index=False))
    _atomic_write(forecasts_dir / "latest_slate.parquet",
                  lambda p: slate.to_parquet(p, index=False))
    _atomic_write(forecasts_dir / "latest_meta.json",
                  lambda p: p.write_text(json.dumps(full_meta, indent=2)))
    log.info("wrote %d player forecasts across %d games", len(result), len(slate))
    return slate


def _atomic_write(path: Path, writer) -> None:
    """Write via a sibling tmp file, then os.replace (atomic on one filesystem)."""
    tmp = path.with_name(f".tmp-{os.getpid()}-{path.name}")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def run(source: str = "nflverse", seasons: list[int] | None = None,
        buzz_provider: str = "neutral", engine: str = "gbm",
        live_weather: bool = False, n_sims: int = 500) -> pd.DataFrame:
    """Train on history, forecast the upcoming slate, persist artifacts."""
    data = prepare(source, seasons, buzz_provider=buzz_provider, live_weather=live_weather)
    train_models(data.feats, engine=engine)
    result = predict_slate(data.feats, engine=engine)
    if result.empty:
        log.warning("no upcoming games found; nothing to forecast")
        return result

    persist_forecasts(result, data.games, meta={"engine": engine, "source": source})

    if n_sims > 0:
        from gameday.sim.run import simulate_slate

        log.info("running game simulations (%d replicates/game)", n_sims)
        simulate_slate(data.player_weeks, data.games, n_sims=n_sims)
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
