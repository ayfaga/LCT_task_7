import io

from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_image_bytes():
    image = Image.new("RGB", (640, 360), color=(12, 34, 56))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_embeddings_endpoint_accepts_file_and_bbox():
    response = client.post(
        "/v1/embeddings",
        files={"image": ("car.png", make_image_bytes(), "image/png")},
        data={"x": 100, "y": 220, "w": 640, "h": 360},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_version"] == "reid-base-v1"
    assert payload["dimension"] == 768
    assert len(payload["embedding"]) == 768
    assert isinstance(payload["embedding"][0], float)


def test_search_endpoint_returns_status_and_candidates():
    response = client.post(
        "/v1/search",
        files={"image": ("car.png", make_image_bytes(), "image/png")},
        data={"x": 100, "y": 220, "w": 640, "h": 360, "topk": 10},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_version"] == "reid-base-v1"
    assert payload["status"] in {"matched", "no_confident_match"}
    assert "candidates" in payload


def test_invalid_bbox_returns_422():
    response = client.post(
        "/v1/embeddings",
        files={"image": ("car.png", make_image_bytes(), "image/png")},
        data={"x": -10, "y": 220, "w": 640, "h": 360},
    )

    assert response.status_code == 422
