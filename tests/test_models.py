import numpy as np
import pytest

from gameday import pipeline
from gameday.config import FORECASTS_DIR, POSITION_STATS


@pytest.fixture(scope="module")
def forecasts(tmp_path_factory, monkeypatch_module=None):
    return pipeline.run(source="demo", seasons=[2024, 2025])


def test_pipeline_produces_forecasts(forecasts):
    assert not forecasts.empty
    assert (FORECASTS_DIR / "latest_forecasts.parquet").exists()
    assert (FORECASTS_DIR / "latest_slate.parquet").exists()


def test_quantiles_ordered_and_nonnegative(forecasts):
    for pos, stats in POSITION_STATS.items():
        rows = forecasts[forecasts["position"] == pos]
        if rows.empty:
            continue
        for stat in stats:
            p10, p50, p90 = (rows[f"{stat}_p10"], rows[f"{stat}_p50"], rows[f"{stat}_p90"])
            assert (p10 <= p50 + 1e-9).all() and (p50 <= p90 + 1e-9).all()
            assert (p10 >= 0).all()


def test_forecasts_have_signal(forecasts):
    """Median QB passing yards should vary meaningfully across players."""
    qb = forecasts[forecasts["position"] == "QB"]["passing_yards_p50"]
    assert qb.std() > 10
    assert qb.mean() > 100
