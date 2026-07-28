"""Health/slate-meta/usage endpoints — null-safe on a fresh clone, populated
from fixture artifacts written into the temp dirs conftest.py set up."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gameday.api import server
from gameday.api.server import app
from gameday.config import FORECASTS_DIR, MODELS_ROOT

client = TestClient(app)


@pytest.fixture
def clean_artifacts():
    """Remove the fixture files this module writes (other suites share the
    temp artifacts dir, so only clear what we own)."""
    paths = [FORECASTS_DIR / "latest_meta.json",
             FORECASTS_DIR / "refresh_status.json",
             FORECASTS_DIR / "latest_usage.parquet",
             MODELS_ROOT / "current" / "manifest.json"]
    for p in paths:
        p.unlink(missing_ok=True)
    yield
    for p in paths:
        p.unlink(missing_ok=True)


def test_health_null_safe_on_fresh_clone(clean_artifacts):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True  # liveness contract unchanged
    assert body["model"] is None
    assert body["refresh"] is None
    assert "forecasts" in body  # None (no meta written by other suites) or dict


def test_health_populated_from_artifacts(clean_artifacts):
    (MODELS_ROOT / "current").mkdir(parents=True, exist_ok=True)
    (MODELS_ROOT / "current" / "manifest.json").write_text(json.dumps({
        "version": "models-test", "engine": "gbm", "git_sha": "abc1234",
        "created_at": "2026-07-28T00:00:00+00:00", "train_seasons": [2024],
        "metrics": {"skill_vs_naive": 0.05, "coverage80": 0.78},
        "files": {}}))
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    (FORECASTS_DIR / "latest_meta.json").write_text(json.dumps({
        "generated_at": "2026-07-28T01:00:00+00:00", "as_of": "2026-07-28T01:00:00+00:00",
        "season": 2026, "week": 1, "n_players": 910, "n_games": 16,
        "engine": "gbm", "model_version": "models-test",
        "data_versions": {"player_weeks": {}}}))
    (FORECASTS_DIR / "refresh_status.json").write_text(json.dumps({
        "started_at": "s", "finished_at": "f", "ok": True, "stage_failed": None,
        "error": None, "skipped_reason": "no game within 8 days",
        "model_version": "models-test", "next_slate": None}))

    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["model"]["version"] == "models-test"
    assert body["model"]["metrics"]["coverage80"] == 0.78
    assert "files" not in body["model"]  # bulky manifest fields not leaked
    assert body["forecasts"]["week"] == 1 and body["forecasts"]["n_players"] == 910
    assert "data_versions" not in body["forecasts"]
    assert body["refresh"]["ok"] is True
    assert body["refresh"]["skipped_reason"] == "no game within 8 days"


def test_slate_carries_meta(clean_artifacts):
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    (FORECASTS_DIR / "latest_meta.json").write_text(json.dumps({
        "generated_at": "2026-07-28T01:00:00+00:00", "season": 2026, "week": 1,
        "n_players": 2, "n_games": 1, "engine": "gbm",
        "model_version": "models-test"}))
    (FORECASTS_DIR / "latest_slate.parquet").unlink(missing_ok=True)
    (FORECASTS_DIR / "latest_forecasts.parquet").unlink(missing_ok=True)
    slate = pd.DataFrame({
        "game_id": ["2026_01_BUF_KC"], "season": [2026], "week": [1],
        "gameday": ["2026-09-09"], "gametime": ["20:20"],
        "home_team": ["KC"], "away_team": ["BUF"],
        "home_score": [float("nan")], "away_score": [float("nan")],
        "stadium": ["GEHA Field at Arrowhead"], "city": ["Kansas City, MO"],
        "roof": ["outdoor"], "temp": [22.0], "wind": [10.0]})
    forecasts = pd.DataFrame({
        "game_id": ["2026_01_BUF_KC"] * 2, "player_id": ["00-1", "00-2"],
        "player_display_name": ["QB One", "WR Two"], "position": ["QB", "WR"],
        "team": ["KC", "BUF"], "opponent_team": ["BUF", "KC"],
        "fantasy_points_p50": [21.5, 14.2]})
    slate.to_parquet(FORECASTS_DIR / "latest_slate.parquet", index=False)
    forecasts.to_parquet(FORECASTS_DIR / "latest_forecasts.parquet", index=False)
    try:
        body = client.get("/api/slate").json()
        assert body["games"], "existing games shape intact"
        assert body["games"][0]["home"]["abbr"] == "KC"
        assert body["meta"]["model_version"] == "models-test"
        assert body["meta"]["week"] == 1
    finally:
        (FORECASTS_DIR / "latest_slate.parquet").unlink(missing_ok=True)
        (FORECASTS_DIR / "latest_forecasts.parquet").unlink(missing_ok=True)


def test_usage_endpoint_404_then_happy_path(clean_artifacts):
    r = client.get("/api/player/00-1/usage")
    assert r.status_code == 404
    assert "usage artifact" in r.json()["detail"]

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "player_id": ["00-1"] * 3 + ["00-2"],
        "season": [2025] * 4, "week": [16, 17, 18, 18],
        "targets": [7.0, 9.0, 6.0, 4.0], "carries": [1.0, 0.0, 2.0, 11.0],
        "target_share": [0.21, 0.27, 0.18, 0.12],
    }).to_parquet(FORECASTS_DIR / "latest_usage.parquet", index=False)

    body = client.get("/api/player/00-1/usage").json()
    assert body["player_id"] == "00-1"
    assert [w["week"] for w in body["weeks"]] == [16, 17, 18]  # sorted
    assert body["weeks"][1]["targets"] == 9.0

    assert client.get("/api/player/00-9/usage").status_code == 404
