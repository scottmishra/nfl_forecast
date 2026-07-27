"""Replay API endpoints — including numpy-scalar JSON serialization."""

import pytest
from fastapi.testclient import TestClient

from gameday import backtest as bt
from gameday.api.server import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _replay_artifacts():
    # writes artifacts/replay/2025 under the temp dir set by conftest
    bt.run_backtest(source="demo", seasons=[2024, 2025], weeks=[6],
                    n_sims=0, compare=True, persist_replay=True)


def test_seasons_lists_2025():
    r = client.get("/api/replay/seasons")
    assert r.status_code == 200
    assert any(s["season"] == 2025 and 6 in s["weeks"] for s in r.json()["seasons"])


def test_week_endpoint_serializes():
    r = client.get("/api/replay/2025/6")
    assert r.status_code == 200          # numpy scalars must encode cleanly
    games = r.json()["games"]
    assert games and games[0]["home"]["players"]


def test_scorecard_endpoint():
    r = client.get("/api/replay/2025/scorecard")
    assert r.status_code == 200
    assert "adjusted" in r.json()


def test_player_endpoint_has_actual():
    game = client.get("/api/replay/2025/6").json()["games"][0]
    pid = game["home"]["players"][0]["player_id"]
    r = client.get(f"/api/replay/2025/6/player/{pid}")
    assert r.status_code == 200
    f = r.json()["forecasts"]
    assert f and "actual" in f[0]


def test_missing_replay_returns_503():
    assert client.get("/api/replay/1999/1").status_code == 503
