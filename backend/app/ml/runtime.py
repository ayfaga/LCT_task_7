"""Versioned E2 encoder, exact cosine gallery, and selected refusal policy."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
from PIL import Image

from .boosting import HybridBoostingPolicy
from .encoder import E2Encoder, file_sha256
from .preprocessing import VERSION


class GalleryNotReady(Exception):
    pass


class MLRuntime:
    def __init__(self, artifact_dir: str | Path, gallery_path: str | Path | None, device: str = "auto"):
        artifact_dir = Path(artifact_dir)
        model_manifest = json.loads((artifact_dir / "model_manifest.json").read_text())
        calibration_manifest = json.loads((artifact_dir / "calibration_manifest.json").read_text())
        self.model_version = model_manifest["model_version"]
        self.model_sha256 = model_manifest["model_sha256"]
        self.preprocessing_version = VERSION
        self.user_gallery_dir = Path(os.getenv("LCT_ML_USER_GALLERY_DIR", "/tmp/lct_gallery_state"))
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
            self.gallery_ids, self.gallery_embeddings = self._read_gallery(Path(gallery_path))

    def _read_gallery(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as gallery:
            version = str(gallery["model_version"].item())
            model_sha = str(gallery["model_sha256"].item())
            embeddings = gallery["embeddings"].astype(np.float32)
            ids = gallery["gallery_ids"].astype(str)
        if version != self.model_version or model_sha != self.model_sha256:
            raise ValueError("Gallery belongs to another encoder")
        if embeddings.ndim != 2 or embeddings.shape[1] != 1024 or len(ids) != len(embeddings):
            raise ValueError("Invalid gallery shape")
        if len(ids) == 0 or len(np.unique(ids)) != len(ids):
            raise ValueError("Gallery IDs must be nonempty and unique")
        if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
            raise ValueError("Gallery embeddings are invalid or not normalized")
        return ids, embeddings

    def _custom_folder(self, gallery_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", gallery_id):
            raise FileNotFoundError("Gallery not found")
        folder = self.user_gallery_dir / gallery_id
        if not (folder / "meta.json").is_file():
            raise FileNotFoundError("Gallery not found")
        return folder

    @staticmethod
    def _write_meta(folder: Path, meta: dict) -> None:
        temporary = folder / f".meta.{uuid4().hex}.tmp"
        temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, folder / "meta.json")

    def _custom_gallery(self, gallery_id: str) -> tuple[np.ndarray, np.ndarray]:
        folder = self._custom_folder(gallery_id)
        meta = json.loads((folder / "meta.json").read_text())
        filename = meta.get("gallery_file")
        if not filename or len(meta.get("images", [])) < 10:
            raise GalleryNotReady("Add at least ten gallery images")
        if not re.fullmatch(r"gallery_[0-9]{4,}.npz", filename):
            raise ValueError("Invalid gallery archive name")
        return self._read_gallery(folder / filename)

    def build_gallery(self, gallery_id: str, job_id: str) -> dict:
        folder = self._custom_folder(gallery_id)
        meta = json.loads((folder / "meta.json").read_text())
        if (meta.get("job_id") != job_id or meta.get("state") != "building"
                or not meta.get("pending")):
            raise ValueError("Gallery build job is not active")
        if meta.get("gallery_file"):
            old_filename = meta["gallery_file"]
            if not re.fullmatch(r"gallery_[0-9]{4,}.npz", old_filename):
                raise ValueError("Invalid prior gallery archive")
            old_ids, old_vectors = self._read_gallery(folder / old_filename)
            ids = list(map(str, old_ids))
            vectors = list(old_vectors)
        else:
            ids, vectors = [], []
        for offset, row in enumerate(meta["pending"], 1):
            image_key = row["image_key"]
            if not re.fullmatch(r"[0-9a-f]{32}\.(?:jpg|jpeg|png)", image_key):
                raise ValueError("Invalid stored image key")
            with Image.open(folder / "images" / image_key) as image:
                vector = self.embed(image, tuple(row["bbox"]))
            ids.append(row["gallery_id"])
            vectors.append(vector)
            meta["processed"] = offset
            self._write_meta(folder, meta)
        if len(set(ids)) != len(ids) or len(ids) > 1000:
            raise ValueError("Invalid gallery ID set")
        # A prior worker can finish the archive write and then die before the
        # metadata switch. Preserve that archive and choose a fresh generation.
        generation = int(meta.get("generation", 0)) + 1
        while (folder / f"gallery_{generation:04d}.npz").exists():
            generation += 1
        filename = f"gallery_{generation:04d}.npz"
        path = folder / filename
        np.savez_compressed(path, gallery_ids=np.asarray(ids),
                            embeddings=np.asarray(vectors, dtype=np.float32),
                            model_version=np.asarray(self.model_version),
                            model_sha256=np.asarray(self.model_sha256))
        meta["images"].extend(meta["pending"])
        meta.update(pending=[], job_id=None, generation=generation,
                    gallery_file=filename, state="ready" if len(ids) >= 10 else "collecting",
                    error=None, processed=0)
        self._write_meta(folder, meta)
        return {"gallery_id": gallery_id, "image_count": len(ids),
                "search_ready": len(ids) >= 10, "generation": generation}

    @property
    def search_ready(self) -> bool:
        return self.gallery_embeddings is not None

    def embed(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> np.ndarray:
        with self._inference_lock:
            return self.encoder.embed_image(image, bbox)

    def search(self, query: np.ndarray, topk: int = 10, gallery_id: str | None = None) -> dict:
        if gallery_id:
            gallery_ids, gallery_embeddings = self._custom_gallery(gallery_id)
        else:
            if not self.search_ready:
                raise GalleryNotReady("Default gallery is not loaded")
            gallery_ids, gallery_embeddings = self.gallery_ids, self.gallery_embeddings
        if len(gallery_ids) < 10:
            raise GalleryNotReady("Add at least ten gallery images")
        if not 1 <= topk <= 100:
            raise ValueError("topk must be between 1 and 100")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (1024,) or not np.isfinite(query).all() or not np.isclose(np.linalg.norm(query), 1.0, atol=1e-4):
            raise ValueError("Invalid query embedding")
        # Some macOS Accelerate builds set spurious floating-point flags for
        # finite matmul inputs; validate the output itself rather than flags.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            similarities = gallery_embeddings @ query
        if not np.isfinite(similarities).all():
            raise RuntimeError("Cosine search produced non-finite similarities")
        # Match the official offline export contract, including stable ties.
        order = np.argsort(-similarities, kind="stable")
        internal = order[:10]
        accepted = self.policy.accepted(similarities[internal][None, :])[0]
        visible = order[:topk]
        ranked = [{"gallery_id": str(gallery_ids[index]),
                   "similarity": float(similarities[index]),
                   "confidence": float(similarities[index])}
                  for index in visible]
        accepted_ids = set(gallery_ids[internal[accepted]])
        visible_accepted = [candidate for candidate in ranked if candidate["gallery_id"] in accepted_ids]
        return {
            "status": "matched" if visible_accepted else "no_confident_match",
            "model_version": self.model_version,
            "gallery_id": gallery_id,
            "ranked": ranked,
            "accepted": visible_accepted,
            "candidates": visible_accepted,
            "threshold": self.policy.cosine_threshold,
            "confidence_semantics": "raw cosine similarity; not a probability",
        }
