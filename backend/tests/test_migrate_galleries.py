"""Keep old user-gallery archives recoverable while switching encoder versions."""

import json
import numpy as np
import pytest
from PIL import Image

from ml.migrate_galleries import inspect, migrate


def test_migration_preserves_old_archive_and_source(tmp_path):
    folder = tmp_path / ("a" * 32)
    images = folder / "images"
    images.mkdir(parents=True)
    image_key = "b" * 32 + ".png"
    Image.new("RGB", (20, 10), "gray").save(images / image_key)
    old_archive = folder / "gallery_0001.npz"
    np.savez_compressed(old_archive, gallery_ids=np.asarray(["car"]),
                        embeddings=np.ones((1, 1024), dtype=np.float32),
                        model_version=np.asarray("old"), model_sha256=np.asarray("old-sha"))
    meta = {"gallery_id": folder.name, "name": "Test", "state": "collecting",
            "gallery_file": old_archive.name, "generation": 1, "processed": 0,
            "job_id": None, "pending": [], "images": [{"gallery_id": "car",
            "image_key": image_key, "bbox": [0, 0, 20, 10]}]}
    (folder / "meta.json").write_text(json.dumps(meta))
    found, original, status = inspect(folder, "new", "new-sha")
    assert status == "migration_needed"

    class FakeRuntime:
        model_version = "new"
        model_sha256 = "new-sha"

        def embed(self, image, bbox):
            assert image.size == (20, 10) and bbox == (0, 0, 20, 10)
            return np.r_[1.0, np.zeros(1023, dtype=np.float32)]

        def _read_gallery(self, path):
            with np.load(path, allow_pickle=False) as archive:
                assert archive["model_version"].item() == "new"
                assert archive["embeddings"].shape == (1, 1024)

        def _write_meta(self, folder, updated):
            (folder / "meta.json").write_text(json.dumps(updated))

    result = migrate(folder, found, original, FakeRuntime())
    assert result["old_archive_retained"] is True
    assert old_archive.exists() and (images / image_key).exists()
    assert json.loads((folder / "meta.json").read_text())["gallery_file"] == "gallery_0002.npz"
    assert inspect(folder, "new", "new-sha")[2] == "current"


def test_migration_rejects_active_gallery(tmp_path):
    folder = tmp_path / ("c" * 32)
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"gallery_id": folder.name,
                                                  "state": "building", "images": []}))
    with pytest.raises(ValueError, match="not idle"):
        inspect(folder, "new", "new-sha")


def test_empty_collecting_gallery_is_unchanged(tmp_path):
    folder = tmp_path / ("d" * 32)
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"gallery_id": folder.name,
        "state": "collecting", "images": [], "pending": [], "gallery_file": None,
        "job_id": None}))
    assert inspect(folder, "new", "new-sha")[2] == "empty"
