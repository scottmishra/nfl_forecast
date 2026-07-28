"""Optional GPU path: a multi-quantile MLP trained with pinball loss.

One network per position predicts all (stat, quantile) heads jointly, which
lets stats share representation (a windy-day signal learned for passing
yards transfers to passing TDs). At ~1M parameters and batch training it
uses well under 1GB of VRAM — an 8GB card is plenty; it also runs fine on
CPU for small data. Install with `pip install gameday[gpu]`.

This is deliberately a strong-but-simple architecture. Swapping in a
temporal model (PatchTST / TFT via `neuralforecast` or
`pytorch-forecasting`) is a drop-in replacement behind the same
train/predict interface if you want sequence modeling over raw game logs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gameday.config import MODELS_DIR, POSITION_STATS, QUANTILES, settings

log = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the neural model: pip install 'gameday[gpu]'"
        ) from exc


def _paths(models_dir: Path, position: str) -> tuple[Path, Path]:
    return models_dir / f"mlp_{position}.pt", models_dir / f"mlp_manifest_{position}.json"


def _build_net(torch, n_features: int, n_outputs: int):
    import torch.nn as nn

    cfg = settings.neural
    layers: list = []
    prev = n_features
    for h in cfg.hidden:
        layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(cfg.dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_outputs))
    return nn.Sequential(*layers)


def train_position(df: pd.DataFrame, position: str, feature_cols: list[str],
                   models_dir: Path = MODELS_DIR) -> dict:
    torch = _require_torch()
    cfg = settings.neural
    device = cfg.device if torch.cuda.is_available() else "cpu"
    stats = POSITION_STATS[position]

    fill_values = df[feature_cols].median(numeric_only=True)
    X = df[feature_cols].fillna(fill_values).astype(float).values
    Y = df[stats].astype(float).values
    mu, sigma = X.mean(0), X.std(0) + 1e-8
    Xn = (X - mu) / sigma

    net = _build_net(torch, len(feature_cols), len(stats) * len(QUANTILES)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    qs = torch.tensor(QUANTILES, dtype=torch.float32, device=device)

    ds = torch.utils.data.TensorDataset(
        torch.tensor(Xn, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    net.train()
    for epoch in range(cfg.epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = net(xb).view(-1, len(stats), len(QUANTILES))
            err = yb.unsqueeze(-1) - pred          # (B, stats, Q)
            loss = torch.maximum(qs * err, (qs - 1) * err).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(xb)
        if epoch % 10 == 0:
            log.info("%s epoch %d pinball=%.4f", position, epoch, total / len(ds))

    model_path, manifest_path = _paths(models_dir, position)
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), model_path)
    manifest_path.write_text(json.dumps({
        "position": position, "features": feature_cols, "stats": stats,
        "quantiles": QUANTILES, "norm_mu": mu.tolist(), "norm_sigma": sigma.tolist(),
        "fill_values": {k: (None if pd.isna(v) else float(v))
                        for k, v in fill_values.items()},
    }, indent=2))
    return {"final_pinball": round(total / len(ds), 4)}


def predict_position(df: pd.DataFrame, position: str,
                     models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    torch = _require_torch()
    model_path, manifest_path = _paths(models_dir, position)
    manifest = json.loads(manifest_path.read_text())
    stats, quantiles = manifest["stats"], manifest["quantiles"]

    net = _build_net(torch, len(manifest["features"]), len(stats) * len(quantiles))
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    X = df[manifest["features"]].astype(float)
    fills = manifest.get("fill_values")
    if fills:
        X = X.fillna({k: v for k, v in fills.items() if v is not None})
    X = X.values
    Xn = (X - np.array(manifest["norm_mu"])) / np.array(manifest["norm_sigma"])
    with torch.no_grad():
        pred = net(torch.tensor(Xn, dtype=torch.float32)).view(-1, len(stats), len(quantiles)).numpy()
    pred = np.clip(np.sort(pred, axis=-1), 0, None)  # non-crossing, non-negative

    out = df.copy()
    for i, stat in enumerate(stats):
        for j, q in enumerate(quantiles):
            out[f"{stat}_p{int(q * 100):02d}"] = np.round(pred[:, i, j], 2)
    return out
