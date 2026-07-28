"""Model bundles — pack/verify/install/rollback/sync, all in tmp dirs.

No real trained models needed: a fake models dir with dummy artifacts and a
per-position manifest exercises the whole packaging path.
"""

import hashlib
import io
import json
import tarfile

import pytest

from gameday import bundle
from gameday.features import build as feature_build

LOCAL_SCHEMA = getattr(feature_build, "FEATURE_SCHEMA_VERSION", 1)

QB_TXT = "gbm_QB_fantasy_points_q50.txt"
RB_TXT = "gbm_RB_fantasy_points_q50.txt"


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    (d / QB_TXT).write_text("tree data QB")
    (d / RB_TXT).write_text("tree data RB")
    (d / "manifest_QB.json").write_text(json.dumps({
        "position": "QB", "features": ["a", "b", "c"],
        "stats": ["fantasy_points"], "quantiles": [0.5], "fill_values": {}}))
    return d


def test_pack_verify_roundtrip(models_dir, tmp_path):
    path = bundle.pack(models_dir, tmp_path / "bundles",
                       meta={"engine": "gbm", "train_seasons": [2022, 2023],
                             "metrics": {"skill_vs_naive": 0.12}})
    assert path.name.startswith("model_bundle-models-") and path.suffixes[-2:] == [".tar", ".gz"]
    manifest = bundle.verify(path)
    assert manifest["schema"] == 1
    assert manifest["engine"] == "gbm" and manifest["train_seasons"] == [2022, 2023]
    assert manifest["positions"]["QB"]["n_features"] == 3
    assert set(manifest["files"]) == {
        f"models/{QB_TXT}", f"models/{RB_TXT}", "models/manifest_QB.json"}
    assert manifest["feature_schema_version"] == LOCAL_SCHEMA  # stamps the live constant


def test_verify_rejects_tampered_member(models_dir, tmp_path):
    path = bundle.pack(models_dir, tmp_path / "bundles")
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(path) as src, tarfile.open(tampered, "w:gz") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read()
            if m.name.endswith(QB_TXT):
                data = b"evil payload"
                m.size = len(data)
            dst.addfile(m, io.BytesIO(data))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        bundle.verify(tampered)


def test_verify_rejects_path_traversal(tmp_path):
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        data = b"{}"
        info = tarfile.TarInfo("../escape.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="unsafe path"):
        bundle.verify(evil)


def test_install_swap_and_rollback(models_dir, tmp_path):
    root = tmp_path / "models"
    b1 = bundle.pack(models_dir, tmp_path / "bundles")
    bundle.install(b1, root)
    assert (root / "current" / "manifest.json").exists()
    assert (root / "current" / QB_TXT).read_text() == "tree data QB"

    (models_dir / QB_TXT).write_text("tree data QB v2")
    b2 = bundle.pack(models_dir, tmp_path / "bundles2")
    bundle.install(b2, root)
    assert (root / "current" / QB_TXT).read_text() == "tree data QB v2"
    assert (root / "previous" / QB_TXT).read_text() == "tree data QB"
    assert not (root / "incoming").exists()

    bundle.rollback(root)  # v1 back in charge, v2 kept as previous
    assert (root / "current" / QB_TXT).read_text() == "tree data QB"
    assert (root / "previous" / QB_TXT).read_text() == "tree data QB v2"


def test_rollback_without_previous_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bundle.rollback(tmp_path / "models")


def test_incompatible_feature_schema_rejected(models_dir, tmp_path, monkeypatch):
    b1 = bundle.pack(models_dir, tmp_path / "bundles")  # stamped with LOCAL_SCHEMA
    manifest = bundle.verify(b1)
    manifest["feature_schema_version"] = 999
    with pytest.raises(ValueError, match="feature_schema_version"):
        bundle.verify_compatible(manifest)

    # install() must refuse and leave nothing behind when local code moved on
    root = tmp_path / "models"
    monkeypatch.setattr(feature_build, "FEATURE_SCHEMA_VERSION",
                        LOCAL_SCHEMA + 1, raising=False)
    with pytest.raises(ValueError, match="feature_schema_version"):
        bundle.install(b1, root)
    assert not (root / "current").exists()
    assert not (root / "incoming").exists()


def test_sync_from_local_pointer(models_dir, tmp_path):
    b1 = bundle.pack(models_dir, tmp_path / "bundles")
    manifest = bundle.verify(b1)
    pointer = tmp_path / "deploy" / "models.json"
    pointer.parent.mkdir()
    pointer.write_text(json.dumps({
        "schema": 1, "version": manifest["version"], "url": str(b1),
        "sha256": hashlib.sha256(b1.read_bytes()).hexdigest(),
        "published_at": "2026-07-27T00:00:00+00:00"}))

    root = tmp_path / "models"
    assert bundle.sync_from_pointer(pointer, root) == manifest["version"]
    assert (root / "current" / QB_TXT).read_text() == "tree data QB"

    b1.unlink()  # second sync must no-op on version match, not re-read the file
    assert bundle.sync_from_pointer(pointer, root) == manifest["version"]


def test_sync_relative_path_and_sha_mismatch(models_dir, tmp_path):
    b1 = bundle.pack(models_dir, tmp_path / "deploy")
    pointer = tmp_path / "deploy" / "models.json"
    pointer.write_text(json.dumps({
        "schema": 1, "version": "models-x", "url": b1.name,  # relative to pointer dir
        "sha256": "0" * 64, "published_at": None}))
    root = tmp_path / "models"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        bundle.sync_from_pointer(pointer, root)
    assert not (root / "current").exists()


def test_sync_unpublished_pointer_raises(tmp_path):
    pointer = tmp_path / "models.json"
    pointer.write_text(json.dumps({
        "schema": 1, "version": None, "url": None, "sha256": None,
        "published_at": None}))
    with pytest.raises(ValueError, match="no published bundle"):
        bundle.sync_from_pointer(pointer, tmp_path / "models")
