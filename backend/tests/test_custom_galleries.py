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
from app.ml.runtime import MLRuntime


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
    runtime.model_version = "test-encoder"
    runtime.model_sha256 = "test-sha"
    runtime.gallery_ids = None
    runtime.gallery_embeddings = None
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
