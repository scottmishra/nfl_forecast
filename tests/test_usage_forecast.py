"""Usage forecaster — OOF discipline, manifests, and graceful no-ops."""

import json

import numpy as np
import pandas as pd
import pytest

from gameday.features.opportunity import USAGE_SERIES, add_team_stint
from gameday.features.temporal import add_temporal
from gameday.models import usage_forecast as uf


@pytest.fixture(scope="module")
def usage_df():
    """Synthetic 4-season RB frame with learnable usage structure: snap and
    carry share follow depth rank with noise, so the model beats chance."""
    rng = np.random.default_rng(7)
    rows = []
    for season in (2021, 2022, 2023, 2024):
        for pid in range(30):
            rank = 1 + pid % 3
            base = {1: 0.75, 2: 0.45, 3: 0.2}[rank]
            for week in range(1, 11):
                rows.append(dict(
                    player_id=f"p{pid}", season=season, week=week, team="KC",
                    position="RB", depth_rank=rank, is_rookie=0,
                    draft_number=60.0, is_new_team=0, years_exp=3.0,
                    games_played=week, pos_rank_prev=10.0 + pid,
                    inj_questionable=0, inj_doubtful=0, inj_out=0,
                    practice_dnp=0, practice_limited=0, practice_full=0,
                    depth_rank_change=0.0, is_depth_promotion=0,
                    games_missed_last8=0.0, weeks_since_return=np.nan,
                    missed_recent=0.0, has_usage=1,
                    snap_pct=np.clip(base + rng.normal(0, 0.05), 0, 1),
                    carry_share=np.clip(base * 0.6 + rng.normal(0, 0.05), 0, 1),
                    target_share_team=np.clip(base * 0.2 + rng.normal(0, 0.03), 0, 1),
                ))
    df = pd.DataFrame(rows).sort_values(["player_id", "season", "week"])
    df = df.reset_index(drop=True)
    df = add_temporal(df, USAGE_SERIES)
    return add_team_stint(df)


def test_oof_predictions_differ_from_final_model(usage_df, tmp_path):
    """The pred_* columns attached to TRAINING rows are out-of-fold: rerunning
    the persisted final model (fit on all training rows) over the same rows
    must produce different numbers — proof the stat engine never trains on
    an in-fold usage forecast."""
    train = uf.train_usage(usage_df.copy(), "RB", models_dir=tmp_path)
    assert train["pred_snap_pct_p50"].notna().mean() > 0.9
    refit = uf.predict_usage(usage_df.copy(), "RB", models_dir=tmp_path)
    diff = (train["pred_snap_pct_p50"] - refit["pred_snap_pct_p50"]).abs()
    assert diff.mean() > 1e-6


def test_quantiles_ordered_and_manifest_reports(usage_df, tmp_path):
    train = uf.train_usage(usage_df.copy(), "RB", models_dir=tmp_path)
    for t in ("snap_pct", "carry_share", "target_share_team"):
        p25, p50, p75 = (train[f"pred_{t}_p{q}"] for q in (25, 50, 75))
        ok = p25.notna()
        assert (p25[ok] <= p50[ok] + 1e-9).all() and (p50[ok] <= p75[ok] + 1e-9).all()
        assert (train.loc[ok, f"pred_{t}_spread"] >= -1e-9).all()
    manifest = json.loads((tmp_path / "usage" / "manifest_usage_RB.json").read_text())
    assert manifest["cv_report"]["snap_pct"]["n_oof"] > 0
    assert manifest["prior_table"]["targets"]["snap_pct"]["cells"]
    # the model input list never contains a same-week usage actual
    assert not set(USAGE_SERIES) & set(manifest["features"])


def test_predict_usage_noop_without_manifest(usage_df, tmp_path):
    out = uf.predict_usage(usage_df.copy(), "WR", models_dir=tmp_path)
    assert "pred_snap_pct_p50" not in out.columns
