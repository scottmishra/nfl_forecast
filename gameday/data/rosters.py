"""NFL player rosters from nflverse public releases.

Per-season roster snapshots — age, experience, draft position, and team — used
for season-to-season player adjustments (a rookie vs a 12-year vet, a player who
just changed teams, draft capital as a prior for players with little history).
Same nflverse-data GitHub release source as the weekly stats (`nflverse.py`),
cached locally as parquet. The roster `gsis_id` is the same id as the weekly
`player_id`, so the two join directly.
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd

from gameday.config import RAW_DIR, ensure_dirs

log = logging.getLogger(__name__)

ROSTER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "rosters/roster_{season}.parquet"
)

# Columns kept from each per-season roster. `gsis_id` joins to weekly player_id.
ROSTER_COLUMNS = [
    "gsis_id", "season", "team", "position", "full_name", "years_exp",
    "birth_date", "entry_year", "rookie_year", "draft_number", "status",
]

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


def fetch_rosters(seasons: list[int], force: bool = False) -> pd.DataFrame:
    """One row per (player, season) with experience/age/draft/team, cached under data/raw.

    Missing seasons (e.g. a not-yet-published year) are skipped rather than
    fatal, so a caller can request a superset of seasons safely.
    """
    ensure_dirs()
    frames = []
    for season in seasons:
        cache = RAW_DIR / f"roster_{season}.parquet"
        if cache.exists() and not force:
            frames.append(pd.read_parquet(cache))
            continue
        url = ROSTER_URL.format(season=season)
        log.info("downloading %s", url)
        try:
            with httpx.Client(follow_redirects=True, timeout=120) as client:
                resp = client.get(url)
                resp.raise_for_status()
                cache.write_bytes(resp.content)
        except httpx.HTTPError as exc:  # a season not published yet, or transient
            log.warning("roster fetch failed for %s (%s); skipping", season, exc)
            continue
        frames.append(pd.read_parquet(cache))

    if not frames:
        return pd.DataFrame(columns=ROSTER_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df = df[[c for c in ROSTER_COLUMNS if c in df.columns]].copy()
    df = df[df["gsis_id"].notna()]  # a rare row has a null id; it can't join
    if "birth_date" in df.columns:  # date-typed in rosters, str in players — normalize
        df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    # Collapse to one row per (player, season) — the season file is already
    # ~one-per-player, but guard against any weekly duplicates.
    df = (
        df.sort_values(["gsis_id", "season"])
        .drop_duplicates(["gsis_id", "season"], keep="last")
        .reset_index(drop=True)
    )
    return df


def current_roster(season: int, force: bool = False) -> pd.DataFrame:
    """Active skill-position players per team for `season` — used to scaffold a
    season opener from the *current* roster (offseason signings + rookies)
    instead of last year's final roster."""
    df = fetch_rosters([season], force=force)
    if df.empty:
        return df
    df = df[df["position"].isin(SKILL_POSITIONS)]
    if "status" in df.columns:  # drop cut / practice-squad where marked
        df = df[df["status"].isin(["ACT", "RES", "DEV"]) | df["status"].isna()]
    return df.reset_index(drop=True)
