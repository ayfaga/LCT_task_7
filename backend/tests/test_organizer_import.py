"""Original JPEG + organizer xywh must be used, never whole-frame fallback."""

import io
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from ml.import_organizer_archive import (ComponentInput, encode_split, inspect_archive,
                                         open_source, validate_preprocessing)


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


def test_same_bbox_contract_from_directory(tmp_path):
    archive_path = tmp_path / "archive.zip"
    gallery_id = make_archive(archive_path)
    folder = tmp_path / "organizer"
    (folder / "images").mkdir(parents=True)
    with ZipFile(archive_path) as archive:
        for name in archive.namelist():
            destination = folder / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
    splits, report = inspect_archive(folder)
    assert report["gallery"] == report["query"] == 1
    assert splits["gallery"][1][0][0] == gallery_id
    with open_source(folder) as source:
        assert source.open(f"images/{gallery_id}.jpg").read(2) == b"\xff\xd8"

        class FakeEncoder:
            def embed_crops(self, crops, batch_size):
                assert len(crops) == 1 and crops[0].size == (40, 40)
                vector = np.zeros((1, 1024), dtype=np.float32)
                vector[0, 0] = 1.0
                return vector

        output = tmp_path / "directory_gallery.npz"
        encode_split(source, splits["gallery"][1], FakeEncoder(),
                     {"model_version": "fixture", "model_sha256": "sha"},
                     output, 1, "gallery")
    with np.load(output, allow_pickle=False) as saved:
        assert saved["gallery_ids"].tolist() == [gallery_id]
        assert saved["embeddings"].shape == (1, 1024)


def test_four_component_fields_mix_zip_folder_and_single_jpeg(tmp_path):
    archive_path = tmp_path / "original.zip"
    gallery_id = make_archive(archive_path)
    second_gallery_id, query_id = "c" * 32, "b" * 32
    gallery_csv = tmp_path / "gallery.csv"
    query_csv = tmp_path / "query.csv"
    gallery_csv.write_text("image_id,x,y,w,h\n" +
                           f"{gallery_id},20,10,40,40\n{second_gallery_id},20,10,40,40\n")
    query_csv.write_text(f"image_id,x,y,w,h\n{query_id},20,10,40,40\n")
    gallery_zip = tmp_path / "gallery.zip"
    query_folder = tmp_path / "query_images"
    query_folder.mkdir()
    with ZipFile(archive_path) as original:
        image = original.read(f"images/{gallery_id}.jpg")
        with ZipFile(gallery_zip, "w") as gallery:
            gallery.writestr(f"nested/{gallery_id}.jpg", image)
        (tmp_path / f"{second_gallery_id}.jpg").write_bytes(image)
        (query_folder / f"{query_id}.jpg").write_bytes(original.read(f"images/{query_id}.jpg"))
    spec = ComponentInput(gallery_csv, (gallery_zip, tmp_path / f"{second_gallery_id}.jpg"),
                          query_csv, (query_folder,))
    splits, report = inspect_archive(spec)
    assert (report["gallery"], report["query"], report["partial"]) == (2, 1, False)
    assert [row[0] for row in splits["gallery"][1]] == [gallery_id, second_gallery_id]
    with open_source(spec) as source:
        assert source.open(splits["gallery"][1][0][2]).read(2) == b"\xff\xd8"
        assert source.open(splits["query"][1][0][2]).read(2) == b"\xff\xd8"


def test_component_partial_requires_explicit_opt_in_and_filters_csv(tmp_path):
    archive_path = tmp_path / "original.zip"
    gallery_id = make_archive(archive_path)
    missing_id, query_id = "c" * 32, "b" * 32
    gallery_csv = tmp_path / "gallery.csv"
    query_csv = tmp_path / "query.csv"
    gallery_csv.write_text("image_id,x,y,w,h\n" +
                           f"{gallery_id},20,10,40,40\n{missing_id},20,10,40,40\n")
    query_csv.write_text(f"image_id,x,y,w,h\n{query_id},20,10,40,40\n")
    gallery_zip = tmp_path / "gallery.zip"
    query_zip = tmp_path / "query.zip"
    with ZipFile(archive_path) as original:
        with ZipFile(gallery_zip, "w") as target:
            target.writestr(f"{gallery_id}.jpg", original.read(f"images/{gallery_id}.jpg"))
        with ZipFile(query_zip, "w") as target:
            target.writestr(f"{query_id}.jpg", original.read(f"images/{query_id}.jpg"))
    spec = ComponentInput(gallery_csv, (gallery_zip,), query_csv, (query_zip,))
    with pytest.raises(ValueError, match="missing JPEG"):
        inspect_archive(spec)
    splits, report = inspect_archive(spec, allow_partial=True)
    assert report["partial"] and report["missing_gallery"] == 1
    assert missing_id not in splits["gallery"][0].decode()


def test_component_duplicate_images_fail_closed(tmp_path):
    archive_path = tmp_path / "original.zip"
    gallery_id = make_archive(archive_path)
    with ZipFile(archive_path) as original:
        gallery_image = tmp_path / f"{gallery_id}.jpg"
        gallery_image.write_bytes(original.read(f"images/{gallery_id}.jpg"))
        gallery_csv = tmp_path / "gallery.csv"
        gallery_csv.write_bytes(original.read("test_gallery.csv"))
        query_csv = tmp_path / "query.csv"
        query_csv.write_bytes(original.read("test_query.csv"))
    spec = ComponentInput(gallery_csv, (gallery_image, gallery_image), query_csv, (archive_path,))
    with pytest.raises(ValueError, match="Duplicate image_id"):
        inspect_archive(spec)
