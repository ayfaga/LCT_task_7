"""Strict offline loading of the final E2 inference weight."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .preprocessing import crop_bbox, tensor_from_crop


os.environ.setdefault("XFORMERS_DISABLED", "1")
VENDOR_DIR = Path(__file__).resolve().parents[2] / "third_party" / "dinov2-main"
sys.path.insert(0, str(VENDOR_DIR))
from dinov2.hub.backbones import dinov2_vitl14  # noqa: E402


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class E2Encoder:
    def __init__(self, model_path: Path, expected_sha256: str, device: str = "auto"):
        if file_sha256(model_path) != expected_sha256:
            raise ValueError("E2 inference weight SHA256 mismatch")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        expected = {
            "format_version": 1,
            "model_kind": "dinov2_normalized_cls",
            "architecture": "large",
            "input_size": 336,
            "embedding_dimension": 1024,
            "embedding_dtype": "float32",
            "embedding_normalization": "l2_after_encoder",
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("Unexpected E2 checkpoint contract")
        self.device = select_device(device)
        self.model = dinov2_vitl14(pretrained=False)
        self.model.load_state_dict(checkpoint["encoder"], strict=True)
        self.model = self.model.eval().to(self.device)
        self.input_size = 336
        self.dimension = 1024

    def embed_crops(self, crops: list[Image.Image], batch_size: int = 1) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(crops), batch_size):
                batch = torch.stack([tensor_from_crop(crop, self.input_size)
                                     for crop in crops[start:start + batch_size]]).to(self.device)
                output = torch.nn.functional.normalize(self.model(batch).float(), dim=1)
                vectors.append(output.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(vectors) if vectors else np.empty((0, self.dimension), dtype=np.float32)

    def embed_image(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> np.ndarray:
        crop = crop_bbox(image, bbox)
        return self.embed_crops([crop])[0]
