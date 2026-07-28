"""Model bundles: pack trained artifacts into a portable tar.gz, verify,
install with an atomic-ish swap, and roll back.

"Train big, deploy small": training runs on a workstation and produces a
bundle — manifest.json at the root plus models/ with every trained artifact —
that a Pi installs under artifacts/models/ as current/, keeping the ousted
set as previous/ for one-command rollback.

Integrity: the bundle manifest lists every packed file with its sha256, and
`bundle_sha256` is the sha256 of the sorted "relpath:sha256" lines — chosen
over hashing the tar itself so the fingerprint is independent of tar/gzip
metadata (mtimes, compression level). verify() re-hashes members without
extracting and rejects path traversal and links; install() extracts to
incoming/ and only swaps directories after integrity + compatibility checks
pass, so a failed install never touches current/.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from gameday.features import build as feature_build

log = logging.getLogger(__name__)

SCHEMA = 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundle_fingerprint(files: dict[str, str]) -> str:
    lines = "\n".join(f"{path}:{sha}" for path, sha in sorted(files.items()))
    return _sha256(lines.encode())


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True,
            text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _positions_summary(models_dir: Path) -> dict:
    """{pos: {stats, quantiles, n_features}} from the per-position manifests."""
    out: dict[str, dict] = {}
    for mp in sorted(models_dir.glob("manifest_*.json")):
        m = json.loads(mp.read_text())
        out[m["position"]] = {
            "stats": m.get("stats"),
            "quantiles": m.get("quantiles"),
            "n_features": len(m.get("features", [])),
        }
    return out


def pack(models_dir: Path, out_dir: Path, meta: dict | None = None) -> Path:
    """Package models_dir into out_dir/model_bundle-{version}.tar.gz.

    `meta` may carry engine, train_seasons, and metrics (from the train gate);
    everything else in the bundle manifest is derived here. A bundle-level
    manifest.json already sitting in models_dir (an installed bundle being
    re-packed) is excluded from the payload.
    """
    meta = meta or {}
    models_dir = Path(models_dir)
    now = dt.datetime.now(dt.timezone.utc)
    version = f"models-{now:%Y%m%d-%H%M%S}"
    # Recursive: the usage forecaster lives in a usage/ subdirectory and must
    # ship with the stat boosters (a v3 bundle without it fails at predict).
    model_files = sorted(
        p for p in models_dir.rglob("*")
        if p.is_file() and p.relative_to(models_dir).as_posix() != "manifest.json")
    if not model_files:
        raise FileNotFoundError(f"no model files to pack in {models_dir}")
    files = {f"models/{p.relative_to(models_dir).as_posix()}": _sha256(p.read_bytes())
             for p in model_files}
    manifest = {
        "schema": SCHEMA,
        "version": version,
        "created_at": now.isoformat(timespec="seconds"),
        "engine": meta.get("engine", "gbm"),
        "git_sha": _git_sha(),
        "train_seasons": meta.get("train_seasons"),
        "feature_schema_version": getattr(feature_build, "FEATURE_SCHEMA_VERSION", 1),
        "positions": _positions_summary(models_dir),
        "metrics": meta.get("metrics"),
        "files": files,
        "bundle_sha256": _bundle_fingerprint(files),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / f"model_bundle-{version}.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        payload = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size, info.mtime = len(payload), int(now.timestamp())
        tar.addfile(info, io.BytesIO(payload))
        for p in model_files:
            tar.add(p, arcname=f"models/{p.relative_to(models_dir).as_posix()}")
    log.info("packed %d files (%.1f MB) -> %s",
             len(model_files), bundle_path.stat().st_size / 1e6, bundle_path)
    return bundle_path


def verify(bundle_path: Path) -> dict:
    """Integrity-check a bundle without extracting it; returns its manifest.

    Rejects unsafe member paths (absolute, .., links), a missing/mismatched
    file list, per-file sha256 mismatches, and a bad bundle fingerprint.
    """
    with tarfile.open(bundle_path, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            parts = Path(m.name).parts
            if m.name.startswith(("/", "\\")) or ".." in parts or ":" in m.name:
                raise ValueError(f"unsafe path in bundle: {m.name}")
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"unsupported member type in bundle: {m.name}")
        mf = tar.extractfile("manifest.json")
        if mf is None:
            raise ValueError("bundle has no manifest.json")
        manifest = json.loads(mf.read())
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"unsupported bundle schema: {manifest.get('schema')}")
        files = manifest.get("files") or {}
        present = {m.name for m in members if m.isfile() and m.name != "manifest.json"}
        if set(files) != present:
            raise ValueError(
                f"bundle file list mismatch: manifest lists {len(files)}, tar has {len(present)}")
        for relpath, want in files.items():
            if _sha256(tar.extractfile(relpath).read()) != want:
                raise ValueError(f"sha256 mismatch for {relpath}")
        if manifest.get("bundle_sha256") != _bundle_fingerprint(files):
            raise ValueError("bundle_sha256 does not match the file list")
    return manifest


def verify_compatible(manifest: dict) -> None:
    """Raise when a bundle can't run on this installation.

    Checks the feature schema version and engine dependencies only — feature
    presence is validated naturally at predict time by the per-position
    manifests.
    """
    local = getattr(feature_build, "FEATURE_SCHEMA_VERSION", 1)
    got = manifest.get("feature_schema_version")
    if got != local:
        raise ValueError(
            f"bundle feature_schema_version={got} != local {local}; "
            "retrain against this code or upgrade the deployment")
    if isinstance(got, int) and got >= 3 and not any(
            f.startswith("models/usage/") for f in (manifest.get("files") or {})):
        raise ValueError(
            "feature_schema_version>=3 bundle has no models/usage/ forecaster "
            "files — predict would fail on the pred_*/_est features; repack "
            "from a models dir that includes usage/")
    if manifest.get("engine") == "neural":
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("bundle engine=neural but torch is not installed") from exc


def install(bundle_path: Path, models_root: Path) -> dict:
    """Verify a bundle and swap it in as models_root/current.

    Extracts to models_root/incoming (flattened: the engines read model files
    directly from the current/ root), then swaps current -> previous ->
    deleted. Any failure before the swap leaves current/ untouched.
    """
    models_root = Path(models_root)
    models_root.mkdir(parents=True, exist_ok=True)
    manifest = verify(bundle_path)
    verify_compatible(manifest)
    incoming = models_root / "incoming"
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir()
    try:
        with tarfile.open(bundle_path, "r:gz") as tar:
            (incoming / "manifest.json").write_bytes(tar.extractfile("manifest.json").read())
            for relpath in manifest["files"]:
                # Strip the leading "models/" but keep any deeper structure
                # (usage/ forecaster files must land at current/usage/...).
                dest = incoming.joinpath(*Path(relpath).parts[1:])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(relpath).read())
        _swap(models_root, incoming)
    except BaseException:
        shutil.rmtree(incoming, ignore_errors=True)
        raise
    log.info("installed bundle %s as %s", manifest["version"], models_root / "current")
    return manifest


def _swap(models_root: Path, incoming: Path) -> None:
    current, previous = models_root / "current", models_root / "previous"
    if previous.exists():
        shutil.rmtree(previous)
    demoted = False
    if current.exists():
        os.rename(current, previous)
        demoted = True
    try:
        os.rename(incoming, current)
    except BaseException:
        if demoted:
            os.rename(previous, current)  # restore the old current
        raise


def rollback(models_root: Path) -> dict:
    """Swap previous back in as current (the ousted current becomes previous,
    so rolling back twice toggles between the two installed sets)."""
    models_root = Path(models_root)
    current, previous = models_root / "current", models_root / "previous"
    if not previous.exists():
        raise FileNotFoundError(f"no previous bundle under {models_root} to roll back to")
    tmp = models_root / ".swap"
    if tmp.exists():
        shutil.rmtree(tmp)
    if current.exists():
        os.rename(current, tmp)
    os.rename(previous, current)
    if tmp.exists():
        os.rename(tmp, previous)
    manifest = installed_manifest(models_root) or {}
    log.info("rolled back to %s", manifest.get("version", "an unversioned model set"))
    return manifest


def installed_manifest(models_root: Path, which: str = "current") -> dict | None:
    """The bundle manifest of the installed current/previous set, if any."""
    try:
        return json.loads((Path(models_root) / which / "manifest.json").read_text())
    except (OSError, ValueError):
        return None


def sync_from_pointer(pointer_path: Path, models_root: Path) -> str:
    """Install whatever bundle the deploy pointer names; no-op when current.

    The pointer's `url` may be an https URL or a filesystem path (relative
    paths resolve against the pointer file's directory — handy for offline
    tests and LAN pulls). The download's sha256 must match the pointer.
    Returns the installed (or already-current) version.
    """
    pointer_path = Path(pointer_path)
    ptr = json.loads(pointer_path.read_text())
    version, url, want_sha = ptr.get("version"), ptr.get("url"), ptr.get("sha256")
    if not version or not url:
        raise ValueError(f"pointer {pointer_path} has no published bundle yet")
    cur = installed_manifest(models_root)
    if cur and cur.get("version") == version:
        log.info("models already at %s; nothing to sync", version)
        return version
    if str(url).startswith(("http://", "https://")):
        import httpx

        from gameday.data import releases
        with httpx.Client(follow_redirects=True, timeout=300,
                          transport=releases._transport) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.content
    else:
        src = Path(url)
        if not src.is_absolute():
            src = pointer_path.parent / src
        data = src.read_bytes()
    if want_sha and _sha256(data) != want_sha:
        raise ValueError(f"bundle sha256 mismatch for {url} — refusing to install")
    models_root = Path(models_root)
    models_root.mkdir(parents=True, exist_ok=True)
    tmp = models_root / f".download-{version}.tar.gz"
    tmp.write_bytes(data)
    try:
        manifest = install(tmp, models_root)
    finally:
        tmp.unlink(missing_ok=True)
    if manifest.get("version") != version:
        log.warning("pointer says %s but bundle manifest says %s",
                    version, manifest.get("version"))
    return manifest.get("version", version)
