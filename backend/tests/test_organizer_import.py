"""Original JPEG + organizer xywh must be used, never whole-frame fallback."""

import io
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from ml.import_organizer_archive import encode_split, inspect_archive, validate_preprocessing


def make_archive(path, bad_bbox=False):
    image = Image.new("RGB", (100, 80), (255, 0, 0))
    image.paste((0, 255, 0), (20, 10, 60, 50))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    gallery_id, query_id = "a" * 32, "b" * 32
    with ZipFile(path, "w") as archive:
        for name, identity in (("test_gallery.csv", gallery_id), ("test_query.csv", query_id)):
            columns = "image_id,x,y,w,h\n" if not bad_bbox else "image_id\n"
            record = f"{identity},20,10,40,40\n" if not bad_bbox else f"{identity}\n"
            archive.writestr(name, columns + record)
            archive.writestr(f"images/{identity}.jpg", buffer.getvalue())
    return gallery_id


def test_archive_manifest_uses_explicit_bbox_and_model_version(tmp_path):
    path = tmp_path / "archive.zip"
    gallery_id = make_archive(path)
    splits, report = inspect_archive(path)
    assert report["gallery"] == report["query"] == 1
    assert splits["gallery"][1][0][1] == (20, 10, 40, 40)

    class FakeEncoder:
        def embed_crops(self, crops, batch_size):
            assert len(crops) == 1 and crops[0].size == (40, 40)
            vector = np.zeros((1, 1024), dtype=np.float32)
            vector[0, 0] = 1.0
            return vector

    output = tmp_path / "gallery.npz"
    with ZipFile(path) as archive:
        encode_split(archive, splits["gallery"][1], FakeEncoder(),
                     {"model_version": "joint-v1", "model_sha256": "hash"},
                     output, 1, "gallery")
    with np.load(output, allow_pickle=False) as result:
        assert result["gallery_ids"].tolist() == [gallery_id]
        assert result["embeddings"].shape == (1, 1024)
        assert str(result["model_version"].item()) == "joint-v1"


def test_archive_without_bbox_columns_fails_closed(tmp_path):
    path = tmp_path / "archive.zip"
    make_archive(path, bad_bbox=True)
    with pytest.raises(ValueError, match="expected image_id,x,y,w,h"):
        inspect_archive(path)


def test_missing_organizer_csv_has_actionable_error(tmp_path):
    path = tmp_path / "archive.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("README.txt", "not the organizer dataset")
    with pytest.raises(ValueError, match="Missing organizer CSV: test_gallery.csv"):
        inspect_archive(path)


def test_bbox_outside_frame_fails_before_embedding(tmp_path):
    path = tmp_path / "archive.zip"
    make_archive(path)
    splits, _ = inspect_archive(path)
    image_id, _, member = splits["gallery"][1][0]

    class FakeEncoder:
        def embed_crops(self, crops, batch_size):
            raise AssertionError("Invalid BBox must never reach encoder")

    with ZipFile(path) as archive, pytest.raises(ValueError, match="inside the image"):
        encode_split(archive, [(image_id, (90, 10, 40, 40), member)], FakeEncoder(),
                     {}, tmp_path / "gallery.npz", 1, "gallery")


def test_importer_rejects_changed_preprocessing_contract():
    with pytest.raises(ValueError, match="unchanged joint L336 preprocessing"):
        validate_preprocessing({"architecture": "dinov2_vitl14", "input_size": 336})
