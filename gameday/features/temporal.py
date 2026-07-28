"""Temporal player-form features: EWMs, lags, momentum, and volatility.

Everything here obeys the same leakage rule as the rolling-form features in
build.py: each series is computed per player on `shift(1)`, so a row only
carries information available *before* its own kickoff. A row's temporal
features never depend on that row's own stat value.

Families added per input column:
  * `{col}_ewm{h}`  exponentially-weighted trailing mean, halflife `h` games —
                    faster-reacting than the flat r3/r8 windows
  * `{col}_lag{k}`  the raw stat `k` games back
  * `{col}_slope`   short EWM minus long EWM (positive = heating up)
  * `{col}_vol{w}`  trailing standard deviation over `w` games (boom/bust)

Columns missing from the frame (e.g. usage shares on demo data) are skipped,
so callers can pass the full wish-list unconditionally.
"""

from __future__ import annotations

import pandas as pd


def add_temporal(
    df: pd.DataFrame,
    cols: list[str],
    halflives: tuple[int, ...] = (2, 5, 10),
    lags: tuple[int, ...] = (1, 2, 3),
    vol_window: int = 8,
    group_col: str = "player_id",
) -> pd.DataFrame:
    """Return `df` with the temporal feature families added for each of `cols`.

    Assumes `df` is sorted by (`group_col`, season, week) — the same contract
    the rolling-form loop in build_features relies on. New columns keep NaN
    where a player lacks history (first game, short careers); downstream
    engines treat those NaNs natively rather than median-filling them.

    `{col}_slope` is the first-halflife EWM minus the last-halflife EWM
    (ewm2 − ewm10 with the defaults): a momentum read of recent form against
    the longer baseline.
    """
    grp = df.groupby(group_col, sort=False)
    out: dict[str, pd.Series] = {}
    for col in cols:
        if col not in df.columns:
            continue
        shifted = grp[col].shift(1)  # pre-kickoff view of the stat
        sg = shifted.groupby(df[group_col], sort=False)
        for h in halflives:
            out[f"{col}_ewm{h}"] = sg.transform(
                lambda s, h=h: s.ewm(halflife=h, min_periods=1).mean())
        for k in lags:
            out[f"{col}_lag{k}"] = grp[col].shift(k)
        out[f"{col}_slope"] = out[f"{col}_ewm{halflives[0]}"] - out[f"{col}_ewm{halflives[-1]}"]
        out[f"{col}_vol{vol_window}"] = sg.transform(
            lambda s: s.rolling(vol_window, min_periods=2).std())
    if not out:
        return df
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def temporal_feature_names(
    cols: list[str],
    halflives: tuple[int, ...] = (2, 5, 10),
    lags: tuple[int, ...] = (1, 2, 3),
    vol_window: int = 8,
) -> list[str]:
    """The column names add_temporal would emit for `cols` (whether or not the
    inputs exist) — lets feature selection stay in sync with construction."""
    names: list[str] = []
    for col in cols:
        names += [f"{col}_ewm{h}" for h in halflives]
        names += [f"{col}_lag{k}" for k in lags]
        names += [f"{col}_slope", f"{col}_vol{vol_window}"]
    return names
