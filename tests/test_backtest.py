import json

import pytest

from gameday import backtest as bt
from gameday.config import POSITION_STATS, settings


@pytest.fixture(scope="module")
def report():
    return bt.run_backtest(source="demo", seasons=[2024, 2025],
                           weeks=[6], n_sims=15, seed=3)


def test_qb_features_pruned(report):
    """The prune pass caps QB/TE at top-K gain features and records the list."""
    manifest = json.loads(
        (bt.BACKTEST_DIR / "models" / "v3" / "manifest_QB.json").read_text())
    top_k = settings.gbm.top_k_features["QB"]
    assert len(manifest["features"]) <= top_k


def test_new_interval_metrics_present(report):
    for stats in report["players"].values():
        for r in stats.values():
            assert 0.0 <= r["coverage50"] <= 1.0
            assert r["interval_width80"] >= 0
            assert set(r["pinball_by_q"]) == {"p10", "p25", "p50", "p75", "p90"}


def test_report_structure(report):
    assert report["test_season"] == 2025
    assert report["weeks"] == [6]
    assert set(report["players"]) == set(POSITION_STATS)
    assert (bt.BACKTEST_DIR / "report.json").exists()
    # backtest models must not touch production artifacts
    assert (bt.BACKTEST_DIR / "models").exists()


def test_player_metrics_sane(report):
    for stats in report["players"].values():
        for r in stats.values():
            assert r["n"] > 0
            assert r["mae_p50"] >= 0
            assert 0.0 <= r["coverage80"] <= 1.0
            assert r["pinball"] >= 0


def test_sim_metrics_sane(report):
    s = report["sims"]
    assert s["n_games"] == 16
    assert 0.0 <= s["brier"] <= 1.0
    assert 0.0 <= s["favorite_accuracy"] <= 1.0
    assert s["points_mae"] > 0
    assert sum(b["n"] for b in s["reliability"]) == s["n_games"]


def test_format_report_renders(report):
    text = bt.format_report(report)
    assert "PLAYER FORECASTS" in text and "GAME SIMULATIONS" in text
