"""The persisted vectors, metadata and model binding survive restart."""

import numpy as np
import pytest

from app.ml.gallery_db import GalleryDatabase


def vectors(n=10):
    values = np.eye(1024, dtype=np.float32)[:n]
    return np.array([f"vehicle-{index}" for index in range(n)]), values


def test_persistent_gallery_and_metadata(tmp_path):
    path = tmp_path / "gallery.sqlite3"
    ids, features = vectors()
    meta = [{"filename": f"car-{i}.png", "bbox": [1, 2, 3, 4]} for i in range(10)]
    GalleryDatabase(path).replace("default", ids, features, "model-v1", "sha-v1", meta)
    restarted = GalleryDatabase(path)
    stored_ids, stored_features = restarted.load("default", "model-v1", "sha-v1")
    assert stored_ids.tolist() == ids.tolist()
    np.testing.assert_array_equal(stored_features, features)
    assert restarted.metadata("default") == meta
    with pytest.raises(ValueError, match="another encoder"):
        restarted.load("default", "model-v2", "sha-v2")


def test_replace_is_atomic_and_rejects_bad_vectors(tmp_path):
    db = GalleryDatabase(tmp_path / "gallery.sqlite3")
    ids, features = vectors()
    db.replace("user:one", ids, features, "v", "sha")
    with pytest.raises(ValueError, match="Invalid gallery"):
        db.replace("user:one", ids, np.zeros_like(features), "v", "sha")
    assert db.load("user:one", "v", "sha")[0].tolist() == ids.tolist()
    replacement_ids = np.array([f"new-{i}" for i in range(10)])
    db.replace("user:one", replacement_ids, features, "v", "sha")
    assert db.load("user:one", "v", "sha")[0].tolist() == replacement_ids.tolist()
