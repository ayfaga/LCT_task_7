"""Persistent, version-bound ANN *demonstration* service for a million-vector load.

Synthetic distractors are deliberately labelled as such. This separate service
does not replace the exact production gallery or the calibrated refusal policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AnnBundle:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported ANN manifest version")
        index_path = self.directory / "index.faiss"
        if sha256(index_path) != self.manifest["index_sha256"]:
            raise ValueError("ANN index SHA256 mismatch")
        self.index = faiss.read_index(str(index_path))
        if self.index.d != self.manifest["dimension"] or self.index.ntotal != self.manifest["total_vectors"]:
            raise ValueError("ANN index/manifest shape mismatch")
        if not self.index.is_trained:
            raise ValueError("ANN index is untrained")
        if self.manifest["dimension"] != 1024:
            raise ValueError("Selected model must return 1024D embeddings")
        if self.manifest["real_gallery"] != len(self.manifest["real_gallery_ids"]):
            raise ValueError("Real gallery ID count mismatch")
        self.index.nprobe = self.manifest["nprobe"]

    def search(self, vector: np.ndarray, topk: int = 10) -> list[dict]:
        vector = np.asarray(vector, dtype=np.float32)
        if (vector.shape != (1024,) or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4)):
            raise ValueError("Expected finite normalized 1024D query")
        if not 1 <= topk <= 100:
            raise ValueError("topk must be 1..100")
        scores, indices = self.index.search(np.ascontiguousarray(vector[None]), topk)
        real_ids = self.manifest["real_gallery_ids"]
        return [{"item_id": real_ids[int(i)] if i < len(real_ids) else f"synthetic:{int(i)-len(real_ids)}",
                 "score_approximate": float(score), "kind": "real" if i < len(real_ids) else "synthetic"}
                for score, i in zip(scores[0], indices[0]) if i >= 0]


class SearchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    embedding: list[float]
    topk: int = 10


def create_app(bundle: AnnBundle) -> FastAPI:
    app = FastAPI(title="LCT ANN load-test service (not production ReID)")

    @app.get("/ready")
    def ready():
        return {"status": "ready", "model_version": bundle.manifest["model_version"],
                "total_vectors": bundle.index.ntotal, "synthetic_distractors": bundle.manifest["synthetic_distractors"]}

    @app.post("/search")
    def search(request: SearchRequest):
        if request.model_version != bundle.manifest["model_version"]:
            raise HTTPException(409, "Encoder version mismatch")
        try:
            ranked = bundle.search(np.asarray(request.embedding, dtype=np.float32), request.topk)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return {"ranked": ranked, "confidence_semantics": "approximate inner product; not a probability",
                "scope": "synthetic-load ANN demonstration; no calibrated refusal"}

    return app


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18022)
    args = parser.parse_args()
    uvicorn.run(create_app(AnnBundle(args.index_dir)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
