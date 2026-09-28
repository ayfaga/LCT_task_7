"""Runs as its own process to avoid FAISS/PyTorch OpenMP conflicts on Mac."""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from fastapi.testclient import TestClient

from ml.ann_service import AnnBundle, create_app, sha256


folder = Path(sys.argv[1])
quantizer = faiss.IndexFlatIP(1024)
index = faiss.IndexIVFFlat(quantizer, 1024, 1, faiss.METRIC_INNER_PRODUCT)
vectors = np.eye(1024, dtype=np.float32)[:10]
index.train(vectors)
index.add(vectors)
index_path = folder / "index.faiss"
faiss.write_index(index, str(index_path))
manifest = {
    "format_version": 1, "dimension": 1024, "model_version": "fixture-joint",
    "total_vectors": 10, "real_gallery": 10, "real_gallery_ids": [f"v{i}" for i in range(10)],
    "synthetic_distractors": 0, "nprobe": 1, "index_sha256": sha256(index_path),
}
(folder / "manifest.json").write_text(json.dumps(manifest))
for _ in range(2):
    client = TestClient(create_app(AnnBundle(folder)))
    assert client.get("/ready").json()["total_vectors"] == 10
    response = client.post("/search", json={"model_version": "fixture-joint",
                                            "embedding": vectors[0].tolist(), "topk": 10})
    assert response.status_code == 200
    assert response.json()["ranked"][0]["item_id"] == "v0"
    assert client.post("/search", json={"model_version": "wrong",
                                        "embedding": vectors[0].tolist()}).status_code == 409
index_path.write_bytes(index_path.read_bytes() + b"tamper")
try:
    AnnBundle(folder)
except ValueError as error:
    assert "SHA256" in str(error)
else:
    raise AssertionError("Tampered index was accepted")
print("ANN_CONTRACT_OK")
