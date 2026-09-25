"""Real local ML-service HTTP-contract smoke with a versioned weight and gallery."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--gallery", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--expected-gallery-id", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    os.environ["LCT_ML_ARTIFACT_DIR"] = str(args.artifact_dir.resolve())
    os.environ["LCT_ML_GALLERY_PATH"] = str(args.gallery.resolve())
    os.environ["LCT_ML_DEVICE"] = args.device
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.ml.api import app  # noqa: E402

    with Image.open(args.image) as image:
        width, height = image.size
    payload = args.image.read_bytes()
    form = {"x": 0, "y": 0, "w": width, "h": height}
    with TestClient(app) as client:
        ready = client.get("/ready")
        embedding_response = client.post(
            "/v1/embeddings", files={"image": (args.image.name, payload, "image/png")}, data=form,
        )
        search_response = client.post(
            "/v1/search", files={"image": (args.image.name, payload, "image/png")},
            data={**form, "topk": 10},
        )
        invalid = client.post(
            "/v1/embeddings", files={"image": (args.image.name, payload, "image/png")},
            data={**form, "x": -1},
        )
    if (ready.status_code != 200 or embedding_response.status_code != 200
            or search_response.status_code != 200 or invalid.status_code != 422):
        raise AssertionError({"ready": [ready.status_code, ready.text[:300]],
                              "embedding": [embedding_response.status_code, embedding_response.text[:300]],
                              "search": [search_response.status_code, search_response.text[:300]],
                              "invalid": [invalid.status_code, invalid.text[:300]]})
    embedding = embedding_response.json()
    search = search_response.json()
    vector = np.asarray(embedding["embedding"], dtype=np.float32)
    if (vector.shape != (1024,) or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
            or search["model_version"] != embedding["model_version"]
            or search["status"] != "matched"
            or search["ranked"][0]["gallery_id"] != args.expected_gallery_id
            or search["accepted"][0]["gallery_id"] != args.expected_gallery_id):
        raise AssertionError("ML HTTP content contract failed")
    with np.load(args.gallery, allow_pickle=False) as gallery_file:
        gallery_size = len(gallery_file["gallery_ids"])
    observation = {"status": "passed", "gallery_size": gallery_size,
                   "ready_status": ready.status_code, "embedding_status": embedding_response.status_code,
                   "search_status": search_response.status_code, "invalid_bbox_status": invalid.status_code,
                   "top1_gallery_id": search["ranked"][0]["gallery_id"],
                   "model_version": search["model_version"],
                   "note": "One same-image API smoke; not an E2E latency or retrieval-quality benchmark."}
    (args.artifact_dir / "api_smoke.json").write_text(json.dumps(observation, indent=2) + "\n")
    print(json.dumps(observation))


if __name__ == "__main__":
    main()
