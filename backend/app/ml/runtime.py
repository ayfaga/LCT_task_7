"""Versioned E2 encoder, exact cosine gallery, and selected refusal policy."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from .boosting import HybridBoostingPolicy
from .encoder import E2Encoder, file_sha256
from .preprocessing import VERSION


class MLRuntime:
    def __init__(self, artifact_dir: str | Path, gallery_path: str | Path | None, device: str = "auto"):
        artifact_dir = Path(artifact_dir)
        model_manifest = json.loads((artifact_dir / "model_manifest.json").read_text())
        calibration_manifest = json.loads((artifact_dir / "calibration_manifest.json").read_text())
        self.model_version = model_manifest["model_version"]
        self.preprocessing_version = VERSION
        if model_manifest["preprocessing_version"] != VERSION:
            raise ValueError("Preprocessing version differs from model manifest")
        if file_sha256(Path(__file__).with_name("preprocessing.py")) != model_manifest["preprocessing_sha256"]:
            raise ValueError("Preprocessing implementation SHA256 mismatch")
        if model_manifest["embedding_dimension"] != 1024:
            raise ValueError("Unexpected embedding dimension")
        if calibration_manifest["model_version"] != self.model_version:
            raise ValueError("Policy/encoder version mismatch")
        if calibration_manifest["policy_kind"] != "hybrid_hist_gradient_boosting":
            raise ValueError("Unexpected calibration policy")
        policy_file = artifact_dir / calibration_manifest["policy_file"]
        if file_sha256(policy_file) != calibration_manifest["policy_sha256"]:
            raise ValueError("Portable policy SHA256 mismatch")
        self.policy = HybridBoostingPolicy.load(policy_file, self.model_version)
        # A single worker may receive several requests in FastAPI's threadpool.
        # Serializing large ViT forwards bounds activation memory on CPU/GPU.
        self._inference_lock = Lock()
        self.encoder = E2Encoder(
            artifact_dir / model_manifest["model_file"],
            model_manifest["model_sha256"], device=device,
        )
        if (not np.isclose(self.policy.cosine_threshold, calibration_manifest["cosine_threshold"], atol=1e-12, rtol=0)
                or not np.isclose(self.policy.booster_threshold, calibration_manifest["booster_threshold"], atol=1e-12, rtol=0)):
            raise ValueError("Policy thresholds differ from calibration manifest")
        self.gallery_ids = None
        self.gallery_embeddings = None
        if gallery_path is not None:
            with np.load(gallery_path, allow_pickle=False) as gallery:
                version = str(gallery["model_version"].item())
                model_sha = str(gallery["model_sha256"].item())
                embeddings = gallery["embeddings"].astype(np.float32)
                ids = gallery["gallery_ids"].astype(str)
            if version != self.model_version or model_sha != model_manifest["model_sha256"]:
                raise ValueError("Gallery belongs to another encoder")
            if embeddings.ndim != 2 or embeddings.shape[1] != 1024 or len(ids) != len(embeddings):
                raise ValueError("Invalid gallery shape")
            if len(ids) < 10 or len(np.unique(ids)) != len(ids):
                raise ValueError("Boosting policy requires at least 10 uniquely named gallery images")
            if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
                raise ValueError("Gallery embeddings are invalid or not normalized")
            self.gallery_ids = ids
            self.gallery_embeddings = embeddings

    @property
    def search_ready(self) -> bool:
        return self.gallery_embeddings is not None

    def embed(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> np.ndarray:
        with self._inference_lock:
            return self.encoder.embed_image(image, bbox)

    def search(self, query: np.ndarray, topk: int = 10) -> dict:
        if not self.search_ready:
            raise RuntimeError("Gallery is not loaded")
        if not 1 <= topk <= 100:
            raise ValueError("topk must be between 1 and 100")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (1024,) or not np.isfinite(query).all() or not np.isclose(np.linalg.norm(query), 1.0, atol=1e-4):
            raise ValueError("Invalid query embedding")
        # Some macOS Accelerate builds set spurious floating-point flags for
        # finite matmul inputs; validate the output itself rather than flags.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            similarities = self.gallery_embeddings @ query
        if not np.isfinite(similarities).all():
            raise RuntimeError("Cosine search produced non-finite similarities")
        # Match the official offline export contract, including stable ties.
        order = np.argsort(-similarities, kind="stable")
        internal = order[:10]
        accepted = self.policy.accepted(similarities[internal][None, :])[0]
        visible = order[:topk]
        ranked = [{"gallery_id": str(self.gallery_ids[index]),
                   "similarity": float(similarities[index]),
                   "confidence": float(similarities[index])}
                  for index in visible]
        accepted_ids = set(self.gallery_ids[internal[accepted]])
        visible_accepted = [candidate for candidate in ranked if candidate["gallery_id"] in accepted_ids]
        return {
            "status": "matched" if visible_accepted else "no_confident_match",
            "model_version": self.model_version,
            "ranked": ranked,
            "accepted": visible_accepted,
            "candidates": visible_accepted,
            "threshold": self.policy.cosine_threshold,
            "confidence_semantics": "raw cosine similarity; not a probability",
        }
