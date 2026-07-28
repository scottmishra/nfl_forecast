"""Generic nflverse release-asset fetcher with polite HTTP caching.

Every nflverse feed this project uses is an asset on a GitHub release of
nflverse/nflverse-data. This module is the single download path for all of
them: a small CATALOG describes each dataset (release tag, asset filename
pattern, freshness TTL) and `fetch_asset`/`fetch_frame` handle the rest —

- past seasons are immutable: once cached they are never re-requested;
- current-season files revalidate after a TTL via conditional GET
  (If-None-Match / If-Modified-Since), rewriting only when bytes changed;
- unpublished seasons (404) are negative-cached so offseason runs don't
  hammer GitHub re-asking for files that aren't there yet;
- network errors fall back to stale cache when one exists; optional feeds
  (`required=False`) never raise toward the caller.

Cache layout: RAW_DIR/{tag}/{filename}, with a JSON sidecar
RAW_DIR/{tag}/{filename}.meta.json recording url, etag, last_modified,
fetched_at, sha256, and status. Legacy flat files from the pre-catalog
layout (e.g. RAW_DIR/stats_player_week_2023.parquet) are adopted into the
tag directory as cache seeds on first touch, so existing deployments don't
refetch history.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import pyarrow.parquet as pq

from gameday.config import RAW_DIR

log = logging.getLogger(__name__)

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/{tag}/{asset}"

# How long a 404 (season not published yet) is trusted before re-checking.
NEGATIVE_TTL_HOURS = 6.0

# Test hook: set to an httpx.MockTransport to exercise fetch logic offline.
_transport: httpx.BaseTransport | None = None


@dataclass(frozen=True)
class Dataset:
    """One nflverse release-asset family."""

    key: str                  # catalog name, e.g. "player_weeks"
    tag: str                  # GitHub release tag, e.g. "stats_player"
    asset: str                # filename pattern, may contain {season}
    per_season: bool = True   # one file per season vs a single all-years file
    ttl_hours: float = 24.0   # current-season revalidation window
    required: bool = True     # optional feeds must never fail a caller
    skip_404: bool = True     # unpublished seasons return None quietly


# Asset filenames verified against the live release pages (2026-07). NGS files
# are single all-years files; pfr weekly advstats start in 2018, snap counts in
# 2012, injuries in 2009.
CATALOG: dict[str, Dataset] = {ds.key: ds for ds in [
    Dataset("player_weeks", "stats_player", "stats_player_week_{season}.parquet", ttl_hours=6),
    Dataset("schedules", "schedules", "games.csv", per_season=False, ttl_hours=6),
    Dataset("rosters", "rosters", "roster_{season}.parquet"),
    Dataset("roster_weekly", "weekly_rosters", "roster_weekly_{season}.parquet"),
    Dataset("snap_counts", "snap_counts", "snap_counts_{season}.parquet", ttl_hours=6),
    Dataset("depth_charts", "depth_charts", "depth_charts_{season}.parquet", ttl_hours=6),
    Dataset("injuries", "injuries", "injuries_{season}.parquet", ttl_hours=6, required=False),
    Dataset("team_weeks", "stats_team", "stats_team_week_{season}.parquet", ttl_hours=6),
    Dataset("ngs_passing", "nextgen_stats", "ngs_passing.parquet", per_season=False, ttl_hours=6),
    Dataset("ngs_receiving", "nextgen_stats", "ngs_receiving.parquet", per_season=False, ttl_hours=6),
    Dataset("ngs_rushing", "nextgen_stats", "ngs_rushing.parquet", per_season=False, ttl_hours=6),
    Dataset("pfr_advstats_week_pass", "pfr_advstats", "advstats_week_pass_{season}.parquet", required=False),
    Dataset("pfr_advstats_week_rush", "pfr_advstats", "advstats_week_rush_{season}.parquet", required=False),
    Dataset("pfr_advstats_week_rec", "pfr_advstats", "advstats_week_rec_{season}.parquet", required=False),
    Dataset("pbp", "pbp", "play_by_play_{season}.parquet", required=False),
    Dataset("players", "players", "players.parquet", per_season=False),
]}


def current_nfl_season(today: date | None = None) -> int:
    """The season currently being played/prepared (league year opens in March).

    Anything earlier is immutable: nflverse never rewrites a finished season's
    files, so a cached copy is served without ever re-checking the network.
    """
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _age_hours(meta: dict) -> float:
    """Hours since the sidecar's fetched_at (inf when missing or garbled)."""
    try:
        fetched = datetime.fromisoformat(meta["fetched_at"])
    except (KeyError, TypeError, ValueError):
        return float("inf")
    return (_now() - fetched).total_seconds() / 3600.0


def _touch(meta_path: Path, meta: dict) -> None:
    meta["fetched_at"] = _now().isoformat()
    meta_path.write_text(json.dumps(meta, indent=1))


def _adopt_legacy(cache: Path, filename: str) -> None:
    """Adopt a pre-catalog flat cache file under RAW_DIR as a seed.

    Earlier versions cached e.g. RAW_DIR/stats_player_week_2023.parquet at the
    top level; move it into the release-tag directory with a synthesized
    sidecar (fetched_at = file mtime) so past seasons stay immutable and the
    current season revalidates on its normal TTL instead of refetching.
    """
    legacy = RAW_DIR / filename
    if cache.exists() or not legacy.is_file() or legacy == cache:
        return
    mtime = datetime.fromtimestamp(legacy.stat().st_mtime, tz=timezone.utc)
    sha = hashlib.sha256(legacy.read_bytes()).hexdigest()
    legacy.replace(cache)
    (cache.parent / (filename + ".meta.json")).write_text(json.dumps({
        "url": None, "etag": None, "last_modified": None,
        "fetched_at": mtime.isoformat(), "sha256": sha, "status": 200,
    }, indent=1))
    log.info("adopted legacy cache %s -> %s", legacy.name, cache)


def fetch_asset(ds: Dataset, season: int | None = None, *,
                force: bool = False, offline: bool = False) -> Path | None:
    """Path to the locally cached asset, downloading/revalidating as needed.

    Returns None when the asset isn't published (404 on a skip_404 dataset)
    or an optional feed is unreachable with no cache to fall back on.
    """
    if ds.per_season and season is None:
        raise ValueError(f"{ds.key} is per-season; pass season=")
    filename = ds.asset.format(season=season) if ds.per_season else ds.asset
    cache = RAW_DIR / ds.tag / filename
    meta_path = cache.parent / (filename + ".meta.json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    _adopt_legacy(cache, filename)
    meta = _read_meta(meta_path)
    url = BASE_URL.format(tag=ds.tag, asset=filename)

    if offline:
        if cache.exists():
            return cache
        if ds.required:
            raise FileNotFoundError(f"offline with no cache for {ds.key}: {cache}")
        return None

    if cache.exists() and not force:
        immutable = ds.per_season and season < current_nfl_season()
        if immutable or _age_hours(meta) < ds.ttl_hours:
            return cache
    if not cache.exists() and not force and meta.get("status") == 404 \
            and _age_hours(meta) < NEGATIVE_TTL_HOURS:
        return None  # negative-cached: wasn't published last time we looked

    headers = {}
    if cache.exists() and not force:
        if meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]
        if meta.get("last_modified"):
            headers["If-Modified-Since"] = meta["last_modified"]
    try:
        with httpx.Client(follow_redirects=True, timeout=120,
                          transport=_transport) as client:
            log.info("checking %s", url)
            resp = client.get(url, headers=headers)
        if resp.status_code == 304:  # unchanged upstream — restart the TTL
            _touch(meta_path, meta)
            return cache
        if resp.status_code == 404 and ds.skip_404:
            if cache.exists():
                # transient: nflverse deletes+reuploads assets during refreshes
                log.warning("%s briefly 404 upstream; serving cache", filename)
                _touch(meta_path, meta)
                return cache
            log.warning("%s not published yet (404); negative-caching", filename)
            meta_path.write_text(json.dumps(
                {"url": url, "status": 404, "fetched_at": _now().isoformat()}, indent=1))
            return None
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        if cache.exists():
            log.warning("fetch failed for %s (%s); serving stale cache", url, exc)
            return cache
        if ds.required:
            raise
        log.warning("fetch failed for optional %s (%s); skipping", ds.key, exc)
        return None

    sha = hashlib.sha256(resp.content).hexdigest()
    if not (cache.exists() and meta.get("sha256") == sha):
        # Atomic: never leave a half-written parquet at the cache path. (Also
        # the fallback when a redirect hop dropped our conditional headers —
        # identical bytes skip the rewrite.)
        tmp = cache.parent / (filename + ".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(cache)
        log.info("cached %s (%.1f MB)", cache.name, len(resp.content) / 1e6)
    meta_path.write_text(json.dumps({
        "url": url,
        "etag": resp.headers.get("etag"),
        "last_modified": resp.headers.get("last-modified"),
        "fetched_at": _now().isoformat(),
        "sha256": sha,
        "status": 200,
    }, indent=1))
    return cache


def _load(path: Path, columns: list[str] | None) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return df[[c for c in columns if c in df.columns]] if columns else df
    if columns:  # subset the read itself; schemas drift across seasons
        have = set(pq.ParquetFile(path).schema_arrow.names)
        return pd.read_parquet(path, columns=[c for c in columns if c in have])
    return pd.read_parquet(path)


def fetch_frame(key: str, seasons: list[int] | None = None, *,
                force: bool = False, offline: bool = False,
                columns: list[str] | None = None) -> pd.DataFrame:
    """Load a CATALOG dataset as one DataFrame, concatenated across seasons.

    Unpublished seasons and unreachable optional feeds are skipped; an empty
    DataFrame means nothing was available. `columns` subsets the read,
    silently ignoring names a given file doesn't have.
    """
    ds = CATALOG[key]
    if ds.per_season:
        if not seasons:
            raise ValueError(f"{key} is per-season; pass seasons=")
        paths = [fetch_asset(ds, s, force=force, offline=offline) for s in seasons]
    else:
        paths = [fetch_asset(ds, force=force, offline=offline)]
    frames = [_load(p, columns) for p in paths if p is not None]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def revalidate_current(keys: tuple[str, ...] = ("player_weeks", "schedules", "rosters")) -> None:
    """Force-refresh the current season's files for the given datasets.

    Used by `gameday train` to guarantee training sees this week's data even
    inside the TTL window. Past seasons stay immutable; a 404 (offseason,
    stats not published yet) is negative-cached and quietly skipped.
    """
    cur = current_nfl_season()
    for key in keys:
        ds = CATALOG[key]
        fetch_asset(ds, cur if ds.per_season else None, force=True)


def data_versions() -> dict[str, dict]:
    """Cache freshness per catalog dataset, summarized from the sidecars.

    {key: {"seasons_cached": [...], "newest_fetched_at": iso | None}} —
    single-file datasets report seasons_cached=[] and rely on newest_fetched_at.
    Feeds /api/health and refresh-status style endpoints.
    """
    out: dict[str, dict] = {}
    for key, ds in CATALOG.items():
        pattern = re.escape(ds.asset).replace(re.escape("{season}"), r"(\d{4})")
        rx = re.compile(rf"^{pattern}\.meta\.json$")
        seasons: list[int] = []
        newest = None
        tag_dir = RAW_DIR / ds.tag
        for meta_path in sorted(tag_dir.glob("*.meta.json")) if tag_dir.is_dir() else []:
            m = rx.match(meta_path.name)
            if not m:
                continue
            meta = _read_meta(meta_path)
            if meta.get("status") != 200:
                continue  # negative-cached 404s aren't data
            if m.groups():
                seasons.append(int(m.group(1)))
            fetched = meta.get("fetched_at")
            if fetched and (newest is None or fetched > newest):
                newest = fetched
        out[key] = {"seasons_cached": sorted(seasons), "newest_fetched_at": newest}
    return out
