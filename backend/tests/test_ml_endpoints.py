"""Exercise the real HTTP boundary between backend and isolated ML app."""

import io
import concurrent.futures
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.ml.api as ml_api
import app.ml_gateway as gateway
import app.main as backend_module
from app.main import app as backend_app
from app.database import Base


backend = TestClient(backend_app)
ml = TestClient(ml_api.app)


def make_image_bytes():
    image = Image.new("RGB", (640, 360), color=(12, 34, 56))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def request(path, data=None, payload=None, client=backend):
    return client.post(
        path,
        files={"image": ("car.png", make_image_bytes() if payload is None else payload, "image/png")},
        data=data or {"x": 100, "y": 20, "w": 400, "h": 300, "topk": 10},
    )


@pytest.fixture
def connected_services(monkeypatch):
    class FixtureRuntime:
        model_version = "dinov2-l14-l336-fresh16-best13-20260923"
        preprocessing_version = "organizer-bbox-rgb-max448-lanczos-pad-bicubic-imagenet-v1"
        encoder = SimpleNamespace(dimension=1024)
        search_ready = True
        gallery_ids = np.array([f"g-{i}" for i in range(10)])

        def embed(self, image, bbox):
            assert image.size == (640, 360)
            assert bbox == (100, 20, 400, 300)
            return np.r_[1.0, np.zeros(1023, dtype=np.float32)].astype(np.float32)

        def search(self, embedding, topk):
            assert topk == 10
            assert embedding.shape == (1024,)
            item = {"gallery_id": "g-1", "similarity": 0.8, "confidence": 0.8}
            return {"status": "matched", "model_version": self.model_version,
                    "ranked": [item], "accepted": [item], "candidates": [item],
                    "threshold": 0.394,
                    "confidence_semantics": "raw cosine similarity; not a probability"}

    monkeypatch.setenv("LCT_ML_ARTIFACT_DIR", "/fixture")
    monkeypatch.setattr(ml_api, "_load_runtime", lambda *args: FixtureRuntime())
    original = httpx.AsyncClient

    def local_client(*args, **kwargs):
        return original(transport=httpx.ASGITransport(app=ml_api.app), **kwargs)

    monkeypatch.setattr(gateway.httpx, "AsyncClient", local_client)


def test_ml_service_health_ready_and_embedding(connected_services):
    assert ml.get("/health").json() == {"status": "ok"}
    ready = ml.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["gallery_size"] == 10
    response = request("/v1/embeddings", client=ml)
    assert response.status_code == 200, response.text
    assert response.json()["dimension"] == 1024
    assert len(response.json()["embedding"]) == 1024


def test_backend_ready_and_product_inference(connected_services):
    ready = backend.get("/ready")
    assert ready.status_code == 200, ready.text
    assert ready.json()["model_version"] == "dinov2-l14-l336-fresh16-best13-20260923"
    for path in ("/api/identify", "/api/infer", "/v1/search"):
        response = request(path)
        assert response.status_code == 200, (path, response.text)
        body = response.json()
        assert body["status"] == "matched"
        assert body["accepted"] == body["candidates"]
        assert body["threshold"] == 0.394
        assert body["ranked"][0]["confidence"] == body["ranked"][0]["similarity"]
    embedding = request("/v1/embeddings")
    assert embedding.status_code == 200
    assert embedding.json()["dimension"] == 1024


def test_validation_is_preserved_across_boundary(connected_services):
    assert request("/api/identify", data={"x": -1, "y": 20, "w": 400, "h": 300}).status_code == 422
    assert request("/api/identify", data={"x": 100, "y": 20, "w": 400, "h": 300, "topk": 101}).status_code == 422
    assert request("/api/identify", payload=b"not an image").status_code == 422
    assert request("/api/identify", payload=b"").status_code == 422
    assert request("/api/identify", payload=b"x" * (ml_api.MAX_IMAGE_BYTES + 1)).status_code == 413


def test_unready_ml_service_yields_503(connected_services, monkeypatch):
    monkeypatch.delenv("LCT_ML_ARTIFACT_DIR", raising=False)
    assert backend.get("/ready").status_code == 503
    assert request("/api/identify").status_code == 503


def test_refusal_is_successful_empty_answer(connected_services, monkeypatch):
    class RefusalRuntime:
        model_version = "dinov2-l14-l336-fresh16-best13-20260923"
        search_ready = True

        def embed(self, image, bbox):
            return np.r_[1.0, np.zeros(1023, dtype=np.float32)]

        def search(self, embedding, topk):
            ranked = [{"gallery_id": "g-1", "similarity": 0.2, "confidence": 0.2}]
            return {"status": "no_confident_match", "model_version": self.model_version,
                    "ranked": ranked, "accepted": [], "candidates": [],
                    "threshold": 0.394,
                    "confidence_semantics": "raw cosine similarity; not a probability"}

    monkeypatch.setattr(ml_api, "_load_runtime", lambda *args: RefusalRuntime())
    response = request("/api/identify")
    assert response.status_code == 200
    assert response.json()["status"] == "no_confident_match"
    assert response.json()["accepted"] == []


def test_upload_uses_generated_filename_and_rejects_invalid_image(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_module, "UPLOAD_DIR", tmp_path)
    db_engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(db_engine)
    temporary_session = sessionmaker(bind=db_engine)

    def test_db():
        with temporary_session() as session:
            yield session

    backend_module.app.dependency_overrides[backend_module.get_db] = test_db
    payload = make_image_bytes()
    try:
        uploaded = backend.post(
            "/api/upload", params={"title": "test"},
            files={"file": ("../../unsafe.png", payload, "image/png")},
        )
        assert uploaded.status_code == 200, uploaded.text
        filename = uploaded.json()["image_name"]
        assert filename.endswith(".png") and ".." not in filename
        assert (tmp_path / filename).read_bytes() == payload
        assert backend.get(f"/api/images/{filename}").content == payload
        invalid = backend.post(
            "/api/upload", params={"title": "test"},
            files={"file": ("bad.png", b"not an image", "image/png")},
        )
        assert invalid.status_code == 422
    finally:
        backend_module.app.dependency_overrides.pop(backend_module.get_db, None)


def test_concurrent_cold_ready_loads_model_once(monkeypatch):
    count = 0
    model = object()

    def construct(*args):
        nonlocal count
        count += 1
        time.sleep(0.02)
        return model

    monkeypatch.setattr(ml_api, "MLRuntime", construct)
    monkeypatch.setattr(ml_api, "_runtime_instance", None)
    monkeypatch.setattr(ml_api, "_runtime_key", None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        loaded = list(pool.map(lambda _: ml_api._load_runtime("/artifacts", "/gallery", "cpu"), range(8)))
    assert count == 1
    assert all(value is model for value in loaded)
