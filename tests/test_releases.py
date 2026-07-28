"""The generic nflverse release fetcher — caching, conditional GETs, fallbacks.

All network traffic goes through httpx.MockTransport via releases._transport;
no test touches the real network. RAW_DIR is remapped per-test to tmp_path.
"""

import json

import httpx
import pandas as pd
import pytest

from gameday.data import releases
from gameday.data import usage
from gameday.data.releases import Dataset

CUR = releases.current_nfl_season()
PAST = CUR - 2

DS = Dataset("test_pw", "stats_player", "stats_player_week_{season}.parquet",
             ttl_hours=6)
OPT = Dataset("test_opt", "injuries", "injuries_{season}.parquet", required=False)

BODY = b"parquet-bytes-v1"


class Server:
    """Scriptable origin: records requests, honors (or drops) conditionals."""

    def __init__(self, body=BODY, etag='"abc"', status=200, honor_conditional=True):
        self.body, self.etag, self.status = body, etag, status
        self.honor_conditional = honor_conditional
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if self.status == 404:
            return httpx.Response(404)
        if self.honor_conditional and request.headers.get("If-None-Match") == self.etag:
            return httpx.Response(304)
        return httpx.Response(200, content=self.body, headers={
            "ETag": self.etag, "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT"})


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(releases, "RAW_DIR", tmp_path)
    return tmp_path


def serve(monkeypatch, server) -> None:
    monkeypatch.setattr(releases, "_transport", httpx.MockTransport(server))


def age_sidecar(raw_dir, ds, filename, hours) -> None:
    """Backdate a sidecar's fetched_at so the TTL window has lapsed."""
    path = raw_dir / ds.tag / (filename + ".meta.json")
    meta = json.loads(path.read_text())
    fetched = releases._now() - pd.Timedelta(hours=hours)
    meta["fetched_at"] = fetched.isoformat()
    path.write_text(json.dumps(meta))


def test_fresh_fetch_writes_cache_and_sidecar(raw_dir, monkeypatch):
    server = Server()
    serve(monkeypatch, server)
    path = releases.fetch_asset(DS, CUR)
    assert path.read_bytes() == BODY
    assert path == raw_dir / "stats_player" / f"stats_player_week_{CUR}.parquet"
    meta = json.loads((raw_dir / "stats_player" / (path.name + ".meta.json")).read_text())
    assert meta["status"] == 200 and meta["etag"] == '"abc"'
    assert meta["url"].endswith(path.name) and meta["sha256"]
    assert len(server.requests) == 1


def test_ttl_hit_serves_cache_without_network(raw_dir, monkeypatch):
    server = Server()
    serve(monkeypatch, server)
    releases.fetch_asset(DS, CUR)
    path = releases.fetch_asset(DS, CUR)
    assert path.read_bytes() == BODY
    assert len(server.requests) == 1  # second call never hit the transport


def test_expired_ttl_revalidates_with_304(raw_dir, monkeypatch):
    server = Server()
    serve(monkeypatch, server)
    path = releases.fetch_asset(DS, CUR)
    age_sidecar(raw_dir, DS, path.name, hours=10)
    assert releases.fetch_asset(DS, CUR) == path
    assert len(server.requests) == 2
    assert server.requests[1].headers["If-None-Match"] == '"abc"'
    # the 304 restarted the TTL, so a third call stays local
    releases.fetch_asset(DS, CUR)
    assert len(server.requests) == 2


def test_identical_body_skips_rewrite(raw_dir, monkeypatch):
    # a redirect hop that drops conditional semantics returns 200 + full body;
    # identical sha256 must not rewrite the cache file
    server = Server(honor_conditional=False)
    serve(monkeypatch, server)
    path = releases.fetch_asset(DS, CUR)
    mtime = path.stat().st_mtime_ns
    age_sidecar(raw_dir, DS, path.name, hours=10)
    server.etag = '"def"'  # upstream re-upload, same bytes
    assert releases.fetch_asset(DS, CUR) == path
    assert path.stat().st_mtime_ns == mtime  # not rewritten
    meta = json.loads((path.parent / (path.name + ".meta.json")).read_text())
    assert meta["etag"] == '"def"'  # but validators refreshed


def test_404_negative_cache_suppresses_retries(raw_dir, monkeypatch):
    server = Server(status=404)
    serve(monkeypatch, server)
    filename = DS.asset.format(season=CUR)
    assert releases.fetch_asset(DS, CUR) is None
    meta = json.loads((raw_dir / DS.tag / (filename + ".meta.json")).read_text())
    assert meta["status"] == 404
    assert releases.fetch_asset(DS, CUR) is None
    assert len(server.requests) == 1  # within the negative-cache window
    age_sidecar(raw_dir, DS, filename, hours=7)
    assert releases.fetch_asset(DS, CUR) is None
    assert len(server.requests) == 2  # window lapsed -> re-asked


def test_network_error_serves_stale_cache(raw_dir, monkeypatch):
    server = Server()
    serve(monkeypatch, server)
    path = releases.fetch_asset(DS, CUR)
    age_sidecar(raw_dir, DS, path.name, hours=10)

    def boom(request):
        raise httpx.ConnectError("network down")

    serve(monkeypatch, boom)
    assert releases.fetch_asset(DS, CUR) == path  # stale beats nothing


def test_network_error_without_cache(raw_dir, monkeypatch):
    def boom(request):
        raise httpx.ConnectError("network down")

    serve(monkeypatch, boom)
    with pytest.raises(httpx.ConnectError):
        releases.fetch_asset(DS, CUR)          # required -> surfaces
    assert releases.fetch_asset(OPT, CUR) is None  # optional -> quiet


def test_offline_mode_never_touches_network(raw_dir, monkeypatch):
    server = Server()
    serve(monkeypatch, server)
    path = releases.fetch_asset(DS, CUR)
    age_sidecar(raw_dir, DS, path.name, hours=1000)
    assert releases.fetch_asset(DS, CUR, offline=True) == path
    assert len(server.requests) == 1  # the initial priming fetch only
    with pytest.raises(FileNotFoundError):
        releases.fetch_asset(DS, PAST, offline=True)
    assert releases.fetch_asset(OPT, CUR, offline=True) is None


def test_past_season_immutable(raw_dir, monkeypatch):
    server = Server()
    serve(monkeypatch, server)
    path = releases.fetch_asset(DS, PAST)
    age_sidecar(raw_dir, DS, path.name, hours=10_000)
    assert releases.fetch_asset(DS, PAST) == path
    assert len(server.requests) == 1  # finished seasons never revalidate


def test_legacy_flat_cache_adopted_without_network(raw_dir, monkeypatch):
    legacy = raw_dir / DS.asset.format(season=PAST)
    legacy.write_bytes(BODY)

    def fail(request):  # any request would be a bug
        raise AssertionError("network touched for an adopted legacy file")

    serve(monkeypatch, fail)
    path = releases.fetch_asset(DS, PAST)
    assert path == raw_dir / DS.tag / legacy.name and path.read_bytes() == BODY
    assert not legacy.exists()  # moved, not copied
    meta = json.loads((path.parent / (path.name + ".meta.json")).read_text())
    assert meta["status"] == 200 and meta["sha256"]


def test_data_versions_summarizes_sidecars(raw_dir, monkeypatch):
    server = Server()
    serve(monkeypatch, server)
    ds = releases.CATALOG["player_weeks"]
    releases.fetch_asset(ds, PAST)
    releases.fetch_asset(ds, CUR)
    versions = releases.data_versions()
    assert versions["player_weeks"]["seasons_cached"] == [PAST, CUR]
    assert versions["player_weeks"]["newest_fetched_at"]
    assert versions["schedules"]["seasons_cached"] == []


def _snap_frames(n_players, n_crosswalked):
    snaps = pd.DataFrame({
        "season": 2024, "week": 1, "player": [f"P{i}" for i in range(n_players)],
        "pfr_player_id": [f"Pfr{i:02d}" for i in range(n_players)],
        "position": "WR", "team": "ARI",
        "offense_snaps": 30.0, "offense_pct": 0.5,
    })
    players = pd.DataFrame({
        "gsis_id": [f"00-{i:07d}" for i in range(n_crosswalked)],
        "pfr_id": [f"Pfr{i:02d}" for i in range(n_crosswalked)],
    })
    return snaps, players


def test_snap_crosswalk_join_rate(monkeypatch):
    snaps, players = _snap_frames(20, 19)  # 95% joins

    def fake_fetch_frame(key, seasons=None, **kw):
        return {"snap_counts": snaps, "players": players}[key]

    monkeypatch.setattr(releases, "fetch_frame", fake_fetch_frame)
    out = usage.fetch_snap_counts([2024])
    assert list(out.columns) == usage.SNAP_COLUMNS
    assert len(out) == 19  # unjoined rows dropped


def test_snap_crosswalk_low_join_rate_raises(monkeypatch):
    snaps, players = _snap_frames(20, 10)  # 50% joins -> poisoned

    def fake_fetch_frame(key, seasons=None, **kw):
        return {"snap_counts": snaps, "players": players}[key]

    monkeypatch.setattr(releases, "fetch_frame", fake_fetch_frame)
    with pytest.raises(ValueError, match="join rate"):
        usage.fetch_snap_counts([2024])
