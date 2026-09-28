"""Exercise uploaded galleries through the backend API and versioned ML index."""

import io
import json
import zipfile
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app.main as backend_module
from app.gallery_store import COMMON_GALLERY_ID, public_gallery
from app.ml.runtime import MLRuntime
from app.ml.gallery_db import GalleryDatabase


def photo(index: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (20, 20), (index, 0, 0)).save(output, format="PNG")
    return output.getvalue()


def archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        for name, content in files.items():
            zipped.writestr(name, content)
    return output.getvalue()


def fake_runtime(folder: Path) -> MLRuntime:
    runtime = MLRuntime.__new__(MLRuntime)
    runtime.user_gallery_dir = folder
    runtime.gallery_db = GalleryDatabase(folder / "gallery.sqlite3")
    runtime.model_version = "test-encoder"
    runtime.model_sha256 = "test-sha"
    runtime.gallery_ids = None
    runtime.gallery_embeddings = None
    from threading import Lock
    runtime._gallery_cache_lock = Lock()
    runtime._gallery_cache = {}
    runtime.policy = SimpleNamespace(
        cosine_threshold=0.394,
        accepted=lambda scores: np.ones((1, 10), dtype=bool),
    )

    def embed(self, image, bbox):
        assert bbox == (0, 0, 20, 20)
        vector = np.zeros(1024, dtype=np.float32)
        vector[image.getpixel((0, 0))[0]] = 1
        return vector

    runtime.embed = MethodType(embed, runtime)
    return runtime


def test_upload_files_zip_incremental_build_and_search(tmp_path, monkeypatch):
    monkeypatch.setenv("LCT_GALLERY_STATE_DIR", str(tmp_path))
    runtime = fake_runtime(tmp_path)

    async def build(gallery_id, job_id):
        return runtime.build_gallery(gallery_id, job_id)

    monkeypatch.setattr(backend_module, "build_ml_gallery", build)
    client = TestClient(backend_module.app)
    response = client.post("/api/galleries", data={"name": "Моя галерея"})
    assert response.status_code == 201
    gallery_id = response.json()["gallery_id"]

    first = client.post(
        f"/api/galleries/{gallery_id}/images",
        files=[("images", (f"car{i}.png", photo(i), "image/png")) for i in range(1, 6)],
    )
    assert first.status_code == 202, first.text
    meta = client.get(f"/api/galleries/{gallery_id}").json()
    assert meta["image_count"] == 5 and not meta["search_ready"]
    assert client.post("/api/identify", files={"image": ("query.png", photo(1))},
                       data={"x": 0, "y": 0, "w": 20, "h": 20,
                             "gallery_id": gallery_id}).status_code == 409

    more = {f"car{i}.png": photo(i) for i in range(6, 11)}
    more["manifest.csv"] = ("filename,gallery_id,x,y,w,h\n" +
                            "".join(f"car{i}.png,id{i},0,0,20,20\n" for i in range(6, 11))).encode()
    second = client.post(f"/api/galleries/{gallery_id}/images",
                         files={"archive": ("cars.zip", archive(more), "application/zip")})
    assert second.status_code == 202, second.text
    meta = client.get(f"/api/galleries/{gallery_id}").json()
    assert meta["image_count"] == 10 and meta["search_ready"]
    assert len(meta["images"]) == 10
    assert client.get(meta["images"][0]["image_url"]).content == photo(1)
    assert client.get("/api/galleries").json()[0]["gallery_id"] == gallery_id

    query = runtime.search(runtime.embed(Image.open(io.BytesIO(photo(1))), (0, 0, 20, 20)),
                           gallery_id=gallery_id)
    assert query["ranked"][0]["gallery_id"] == "car1"
    assert len(query["ranked"]) == 10
    assert len(query["accepted"]) == 10
    assert json.loads((tmp_path / gallery_id / "meta.json").read_text())["generation"] == 2


def test_one_common_gallery_is_shared_and_get_does_not_create_it(tmp_path, monkeypatch):
    monkeypatch.setenv("LCT_GALLERY_STATE_DIR", str(tmp_path))
    runtime = fake_runtime(tmp_path)

    async def build(gallery_id, job_id):
        return runtime.build_gallery(gallery_id, job_id)

    monkeypatch.setattr(backend_module, "build_ml_gallery", build)
    client = TestClient(backend_module.app)
    empty = client.get("/api/common-gallery")
    assert empty.status_code == 200
    assert empty.json()["gallery_id"] == COMMON_GALLERY_ID
    assert empty.json()["image_count"] == 0
    assert not (tmp_path / COMMON_GALLERY_ID).exists()

    upload = client.post(
        "/api/common-gallery/images",
        files=[("images", (f"car{i}.png", photo(i), "image/png")) for i in range(1, 11)],
    )
    assert upload.status_code == 202, upload.text
    assert upload.json()["gallery_id"] == COMMON_GALLERY_ID
    shared = client.get("/api/common-gallery").json()
    assert shared["image_count"] == 10 and shared["search_ready"]
    assert client.get("/api/common-gallery").json()["gallery_id"] == COMMON_GALLERY_ID


def test_common_gallery_accepts_zip_and_separate_bbox_csv_with_nested_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("LCT_GALLERY_STATE_DIR", str(tmp_path))
    runtime = fake_runtime(tmp_path)

    async def build(gallery_id, job_id):
        return runtime.build_gallery(gallery_id, job_id)

    monkeypatch.setattr(backend_module, "build_ml_gallery", build)
    names = [f"images/car{i}.png" for i in range(1, 11)]
    manifest = "filename,gallery_id,x,y,w,h\n" + "".join(
        f"{name},id{i},0,0,20,20\n" for i, name in enumerate(names, 1)
    )
    response = TestClient(backend_module.app).post(
        "/api/common-gallery/images",
        files={
            "archive": ("frames.zip", archive({name: photo(i) for i, name in enumerate(names, 1)}), "application/zip"),
            "manifest": ("gallery_bbox.csv", manifest.encode(), "text/csv"),
        },
    )
    assert response.status_code == 202, response.text
    meta = TestClient(backend_module.app).get("/api/common-gallery").json()
    assert meta["search_ready"] and meta["image_count"] == 10
    assert {row["gallery_id"] for row in meta["images"]} == {f"id{i}" for i in range(1, 11)}


def test_empty_supplied_manifest_never_falls_back_to_whole_frames(tmp_path, monkeypatch):
    monkeypatch.setenv("LCT_GALLERY_STATE_DIR", str(tmp_path))
    response = TestClient(backend_module.app).post(
        "/api/common-gallery/images",
        files={
            "archive": ("frames.zip", archive({"images/car1.png": photo(1)}), "application/zip"),
            "manifest": ("gallery_bbox.csv", b"filename,gallery_id,x,y,w,h\n", "text/csv"),
        },
    )
    assert response.status_code == 422
    assert "missing=" in response.json()["detail"]


def test_large_gallery_status_uses_total_count_and_bounded_preview():
    meta = {
        "gallery_id": COMMON_GALLERY_ID, "name": "shared", "state": "ready",
        "gallery_file": "gallery_0001.npz", "pending": [],
        "images": [{"gallery_id": str(i), "filename": f"car{i}.jpg", "image_key": f"key{i}.jpg"}
                   for i in range(101)],
    }
    info = public_gallery(meta)
    assert info["image_count"] == 101 and info["search_ready"]
    assert info["images_truncated"] and len(info["images"]) == 100


def test_rejects_unsafe_or_incomplete_archives(tmp_path, monkeypatch):
    monkeypatch.setenv("LCT_GALLERY_STATE_DIR", str(tmp_path))
    client = TestClient(backend_module.app)
    gallery_id = client.post("/api/galleries", data={"name": "validation"}).json()["gallery_id"]
    unsafe = archive({"../escape.png": photo(1)})
    response = client.post(f"/api/galleries/{gallery_id}/images",
                           files={"archive": ("cars.zip", unsafe, "application/zip")})
    assert response.status_code == 422
    assert list((tmp_path / gallery_id / "images").iterdir()) == []

    mismatch = archive({"car.png": photo(1), "manifest.csv": b"filename,gallery_id\nother.png,x\n"})
    response = client.post(f"/api/galleries/{gallery_id}/images",
                           files={"archive": ("cars.zip", mismatch, "application/zip")})
    assert response.status_code == 422


def test_replenishment_store_accepts_image_and_text(tmp_path, monkeypatch):
    monkeypatch.setenv("LCT_GALLERY_STATE_DIR", str(tmp_path))
    client = TestClient(backend_module.app)

    response = client.post(
        "/api/replenishment",
        data={"label": "car-01"},
        files={
            "image": ("car-01.png", photo(2), "image/png"),
            "text": ("car-01.txt", b"vehicle plate front side cut", "text/plain"),
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["label"] == "car-01"
    assert payload["image_name"].endswith(".png")
    assert payload["text_name"].endswith(".txt")

    list_response = client.get("/api/replenishment")
    assert list_response.status_code == 200
    assert any(item["label"] == "car-01" for item in list_response.json())

    page = client.get("/replenishment")
    assert page.status_code == 200
    assert "image_id" in page.text and "vehicle_id" in page.text and "camera_id" in page.text
