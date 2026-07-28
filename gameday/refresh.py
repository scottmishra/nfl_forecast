"""Nightly refresh — the Pi's entry point for keeping forecasts current.

Runs, in order: an offseason gate (skip everything when no game is close),
best-effort model-bundle sync from the deploy pointer, data refresh through
the TTL-caching release layer, slate prediction with the installed bundle's
engine, optional game sims, and atomic artifact publication.

Every run (except lock contention, where another refresh owns the file)
finishes by atomically writing artifacts/forecasts/refresh_status.json:
{started_at, finished_at, ok, stage_failed, error, skipped_reason,
model_version, next_slate} — the dashboard and cron alerting read it.
run_refresh returns a process exit code (non-zero on failure).

Concurrency: a pid+timestamp lock file at artifacts/.refresh.lock, created
with O_CREAT|O_EXCL so it works on Windows and Linux alike (no fcntl).
Locks older than 2h are presumed crashed and broken.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path

import pandas as pd

from gameday import bundle, pipeline
from gameday.config import (ARTIFACTS_DIR, FORECASTS_DIR, MODELS_ROOT, ROOT,
                            USAGE_ARTIFACT_COLS, ensure_dirs, settings)
from gameday.data import nflverse, releases

log = logging.getLogger(__name__)

LOCK_PATH = ARTIFACTS_DIR / ".refresh.lock"
STATUS_PATH = FORECASTS_DIR / "refresh_status.json"
STALE_LOCK_HOURS = 2.0
DEFAULT_POINTER = ROOT / "deploy" / "models.json"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _acquire_lock() -> bool:
    """True when this process now holds the refresh lock."""
    for attempt in (1, 2):
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {_now_iso()}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age_h = (dt.datetime.now(dt.timezone.utc).timestamp()
                         - LOCK_PATH.stat().st_mtime) / 3600.0
            except OSError:  # holder just released it — retry once
                continue
            if age_h < STALE_LOCK_HOURS or attempt == 2:
                return False
            log.warning("breaking stale refresh lock (%.1fh old)", age_h)
            LOCK_PATH.unlink(missing_ok=True)
    return False


def _write_status(status: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_name(f".tmp-{os.getpid()}-{STATUS_PATH.name}")
    tmp.write_text(json.dumps(status, indent=2))
    os.replace(tmp, STATUS_PATH)


def _next_slate(games: pd.DataFrame) -> dict | None:
    """{season, week, first_game} for the earliest unplayed game, if any."""
    upcoming = games[games["home_score"].isna()].dropna(subset=["gameday"])
    if upcoming.empty:
        return None
    nxt = upcoming.sort_values(["season", "week", "gameday"]).iloc[0]
    return {"season": int(nxt["season"]), "week": int(nxt["week"]),
            "first_game": str(nxt["gameday"])}


def _offseason_reason(games: pd.DataFrame, horizon_days: int) -> str | None:
    """A skip reason when no unplayed game falls within the horizon, else None."""
    upcoming = games[games["home_score"].isna()].dropna(subset=["gameday"])
    if upcoming.empty:
        return "no unplayed games on the schedule"
    days = pd.to_datetime(upcoming["gameday"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    window = days[(days >= today - pd.Timedelta(days=1))
                  & (days <= today + pd.Timedelta(days=horizon_days))]
    if window.empty:
        return f"no game within {horizon_days} days"
    return None


def _emit_season_artifact(feats: pd.DataFrame, full_games: pd.DataFrame,
                          engine: str) -> None:
    """Best-effort: season-long per-(player, week) fantasy quantiles ->
    latest_season.parquet for the draft board. Never fails the refresh."""
    try:
        from gameday.season_projection import project_season
        out = project_season(feats, full_games, engine=engine)
        if out.empty:
            return
        path = FORECASTS_DIR / "latest_season.parquet"
        tmp = path.with_name(f".tmp-{os.getpid()}-{path.name}")
        out.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        log.info("wrote season artifact: %d player-weeks / %d players",
                 len(out), out["player_id"].nunique())
    except Exception as exc:  # noqa: BLE001 — draft board is non-critical
        log.warning("season artifact skipped (%s)", exc)


def _emit_usage_artifact(feats: pd.DataFrame, result: pd.DataFrame) -> None:
    """Best-effort: each slate player's last-8-played-weeks usage rows
    (USAGE_ARTIFACT_COLS, filtered to those present) -> latest_usage.parquet
    for the dashboard sparklines. Never fails the refresh."""
    try:
        cols = [c for c in USAGE_ARTIFACT_COLS if c in feats.columns]
        if not cols or result.empty:
            return
        hist = feats[feats["fantasy_points"].notna()
                     & feats["player_id"].isin(result["player_id"])]
        hist = hist.sort_values(["player_id", "season", "week"]).groupby("player_id").tail(8)
        out = hist[["player_id", "season", "week"] + cols].reset_index(drop=True)
        path = FORECASTS_DIR / "latest_usage.parquet"
        tmp = path.with_name(f".tmp-{os.getpid()}-{path.name}")
        out.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        log.info("wrote usage artifact: %d rows / %d players",
                 len(out), out["player_id"].nunique())
    except Exception as exc:  # noqa: BLE001 — cosmetic artifact only
        log.warning("usage artifact skipped (%s)", exc)


def run_refresh(horizon_days: int = 8, sims: int = 300, sync: bool = True,
                pointer: Path = DEFAULT_POINTER, live_weather: bool = False) -> int:
    """One full refresh cycle; returns a process exit code (0 = success/skip)."""
    ensure_dirs()
    status = {"started_at": _now_iso(), "finished_at": None, "ok": False,
              "stage_failed": None, "error": None, "skipped_reason": None,
              "model_version": None, "next_slate": None}
    if not _acquire_lock():
        log.error("another refresh appears to be running (%s); aborting", LOCK_PATH)
        return 1
    stage = "schedules"
    try:
        # settings.seasons may lag the calendar; the slate needs this season.
        seasons = sorted(set(settings.seasons) | {releases.current_nfl_season()})
        games = nflverse.fetch_schedules(seasons)
        status["next_slate"] = _next_slate(games)
        reason = _offseason_reason(games, horizon_days)
        if reason:
            log.info("refresh skipped: %s", reason)
            status.update(ok=True, skipped_reason=reason)
            return 0

        if sync:
            stage = "sync"
            try:
                bundle.sync_from_pointer(pointer, MODELS_ROOT)
            except Exception as exc:
                # never fail the refresh over sync when installed models exist
                if bundle.installed_manifest(MODELS_ROOT):
                    log.warning("model sync failed (%s); using installed bundle", exc)
                else:
                    raise
        manifest = bundle.installed_manifest(MODELS_ROOT) or {}
        engine = manifest.get("engine", "gbm")
        status["model_version"] = manifest.get("version")

        stage = "prepare"
        data = pipeline.prepare(source="nflverse", seasons=seasons,
                                live_weather=live_weather)
        stage = "predict"
        result = pipeline.predict_slate(data.feats, engine=engine)
        if result.empty:
            raise RuntimeError("no upcoming slate rows to forecast")
        stage = "sims"
        if sims > 0:
            from gameday.sim.run import simulate_slate
            simulate_slate(data.player_weeks, data.games, n_sims=sims)
        stage = "persist"
        pipeline.persist_forecasts(result, data.games, meta={
            "engine": engine,
            "model_version": manifest.get("version"),
            "data_versions": releases.data_versions(),
        })
        _emit_usage_artifact(data.feats, result)  # best-effort, never fatal
        _emit_season_artifact(data.feats, games, engine)  # best-effort too
        status["ok"] = True
        log.info("refresh complete: %d forecasts, models %s",
                 len(result), manifest.get("version"))
        return 0
    except Exception as exc:  # noqa: BLE001 — cron boundary: report, don't crash
        log.exception("refresh failed at stage %s", stage)
        status.update(stage_failed=stage, error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        status["finished_at"] = _now_iso()
        _write_status(status)
        LOCK_PATH.unlink(missing_ok=True)
