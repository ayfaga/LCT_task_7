import io
from types import SimpleNamespace

import pytest

from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_image_bytes():
    image = Image.new("RGB", (640, 360), color=(12, 34, 56))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def configured_runtime(monkeypatch):
    class FixtureRuntime:
        model_version = "dinov2-l14-l336-fresh16-best13-20260923"
        preprocessing_version = "organizer-bbox-rgb-max448-lanczos-pad-bicubic-imagenet-v1"
        encoder = SimpleNamespace(dimension=1024)
        search_ready = True

        def embed(self, image, bbox):
            assert image.size == (640, 360)
            assert bbox == (100, 20, 400, 300)
            return SimpleNamespace(tolist=lambda: [1.0] + [0.0] * 1023)

        def search(self, embedding, topk):
            assert topk == 10
            return {"status": "matched", "model_version": self.model_version,
                    "ranked": [{"gallery_id": "g-1", "similarity": 0.8}],
                    "accepted": [{"gallery_id": "g-1", "similarity": 0.8}],
                    "candidates": [{"gallery_id": "g-1", "similarity": 0.8}],
                    "threshold": 0.394,
                    "confidence_semantics": "raw cosine similarity; not a probability"}

    monkeypatch.setenv("LCT_ML_ARTIFACT_DIR", "/fixture")
    monkeypatch.setattr("app.main._load_ml_runtime", lambda *args: FixtureRuntime())


def test_embeddings_endpoint_accepts_file_and_bbox(configured_runtime):
    response = client.post(
        "/v1/embeddings",
        files={"image": ("car.png", make_image_bytes(), "image/png")},
        data={"x": 100, "y": 20, "w": 400, "h": 300},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_version"] == "dinov2-l14-l336-fresh16-best13-20260923"
    assert payload["dimension"] == 1024
    assert payload["preprocessing_version"] == "organizer-bbox-rgb-max448-lanczos-pad-bicubic-imagenet-v1"
    assert len(payload["embedding"]) == 1024
    assert isinstance(payload["embedding"][0], float)


def test_search_endpoint_returns_status_and_candidates(configured_runtime):
    response = client.post(
        "/v1/search",
        files={"image": ("car.png", make_image_bytes(), "image/png")},
        data={"x": 100, "y": 20, "w": 400, "h": 300, "topk": 10},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_version"] == "dinov2-l14-l336-fresh16-best13-20260923"
    assert payload["status"] == "matched"
    assert payload["accepted"] == payload["candidates"]
    assert payload["threshold"] == 0.394


def test_ready_requires_model_and_gallery(monkeypatch):
    monkeypatch.delenv("LCT_ML_ARTIFACT_DIR", raising=False)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_invalid_bbox_returns_422():
    response = client.post(
        "/v1/embeddings",
        files={"image": ("car.png", make_image_bytes(), "image/png")},
        data={"x": -10, "y": 20, "w": 400, "h": 300},
    )

    assert response.status_code == 422
