"""Refresh orchestration — offseason gate, lock file, status file, failures.

Schedules are monkeypatched frames; nothing here touches the network. The
artifact paths live under the temp dirs conftest.py set up.
"""

import json
import os
import time

import pandas as pd
import pytest

from gameday import refresh


def _sched(days_out: float, played: bool = False) -> pd.DataFrame:
    today = pd.Timestamp.now().normalize()
    return pd.DataFrame({
        "game_id": ["2026_01_BUF_KC"], "season": [2026], "week": [1],
        "gameday": [(today + pd.Timedelta(days=days_out)).strftime("%Y-%m-%d")],
        "gametime": ["13:00"], "home_team": ["KC"], "away_team": ["BUF"],
        "home_score": [21.0 if played else float("nan")],
        "away_score": [14.0 if played else float("nan")],
    })


@pytest.fixture(autouse=True)
def clean_lock():
    refresh.LOCK_PATH.unlink(missing_ok=True)
    yield
    refresh.LOCK_PATH.unlink(missing_ok=True)


def _status() -> dict:
    return json.loads(refresh.STATUS_PATH.read_text())


def test_offseason_gate_skips_cleanly(monkeypatch):
    monkeypatch.setattr(refresh.nflverse, "fetch_schedules",
                        lambda seasons, force=False: _sched(days_out=45))
    assert refresh.run_refresh(sims=0, sync=False) == 0
    status = _status()
    assert status["ok"] and status["skipped_reason"] == "no game within 8 days"
    assert status["next_slate"] == {"season": 2026, "week": 1,
                                    "first_game": status["next_slate"]["first_game"]}
    assert status["stage_failed"] is None
    assert not refresh.LOCK_PATH.exists()  # released


def test_no_unplayed_games_skips(monkeypatch):
    monkeypatch.setattr(refresh.nflverse, "fetch_schedules",
                        lambda seasons, force=False: _sched(days_out=-30, played=True))
    assert refresh.run_refresh(sims=0, sync=False) == 0
    status = _status()
    assert status["ok"] and status["skipped_reason"] == "no unplayed games on the schedule"
    assert status["next_slate"] is None


def test_schedule_failure_writes_status_and_fails(monkeypatch):
    def boom(seasons, force=False):
        raise RuntimeError("nflverse unreachable")

    monkeypatch.setattr(refresh.nflverse, "fetch_schedules", boom)
    assert refresh.run_refresh(sims=0, sync=False) == 1
    status = _status()
    assert not status["ok"]
    assert status["stage_failed"] == "schedules"
    assert "nflverse unreachable" in status["error"]
    assert not refresh.LOCK_PATH.exists()  # released even on failure


def test_lock_contention_aborts(monkeypatch):
    monkeypatch.setattr(refresh.nflverse, "fetch_schedules",
                        lambda seasons, force=False: _sched(days_out=45))
    refresh.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    refresh.LOCK_PATH.write_text("99999 2026-07-27T00:00:00+00:00")
    assert refresh.run_refresh(sims=0, sync=False) == 1
    assert refresh.LOCK_PATH.exists()  # not ours; left alone


def test_stale_lock_is_broken(monkeypatch):
    monkeypatch.setattr(refresh.nflverse, "fetch_schedules",
                        lambda seasons, force=False: _sched(days_out=45))
    refresh.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    refresh.LOCK_PATH.write_text("99999 stale")
    old = time.time() - 3 * 3600
    os.utime(refresh.LOCK_PATH, (old, old))
    assert refresh.run_refresh(sims=0, sync=False) == 0  # proceeded to the gate
    assert _status()["skipped_reason"] == "no game within 8 days"


def test_sync_failure_tolerated_with_installed_models(monkeypatch, tmp_path):
    # a game inside the horizon forces the run past the gate into sync
    monkeypatch.setattr(refresh.nflverse, "fetch_schedules",
                        lambda seasons, force=False: _sched(days_out=3))
    monkeypatch.setattr(refresh.bundle, "sync_from_pointer",
                        lambda pointer, root: (_ for _ in ()).throw(ValueError("bad pointer")))
    # no installed manifest -> sync failure is fatal, reported at its stage
    monkeypatch.setattr(refresh.bundle, "installed_manifest",
                        lambda root, which="current": None)
    assert refresh.run_refresh(sims=0, sync=True, pointer=tmp_path / "nope.json") == 1
    assert _status()["stage_failed"] == "sync"

    # with an installed bundle the same failure is tolerated; the run then
    # proceeds and fails later at prepare (empty frames from our stub)
    monkeypatch.setattr(refresh.bundle, "installed_manifest",
                        lambda root, which="current": {"version": "models-x", "engine": "gbm"})
    monkeypatch.setattr(refresh.pipeline, "prepare",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("stub stop")))
    assert refresh.run_refresh(sims=0, sync=True, pointer=tmp_path / "nope.json") == 1
    status = _status()
    assert status["stage_failed"] == "prepare"  # sync failure didn't kill it
    assert status["model_version"] == "models-x"
