"""Diagnostic CLS-to-patch similarity, not causal attention or a ranking feature."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.encoder import E2Encoder  # noqa: E402
from app.ml.preprocessing import crop_bbox, tensor_from_crop  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X", "Y", "W", "H"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = json.loads((args.artifact_dir / "model_manifest.json").read_text())
    encoder = E2Encoder(args.artifact_dir / "model_inference.pt", manifest["model_sha256"], args.device)
    with Image.open(args.image) as image:
        crop = crop_bbox(image, tuple(args.bbox))
    tensor = tensor_from_crop(crop, 336).unsqueeze(0).to(encoder.device)
    with torch.inference_mode():
        features = encoder.model.forward_features(tensor)
        cls = torch.nn.functional.normalize(features["x_norm_clstoken"].float(), dim=-1)
        patches = torch.nn.functional.normalize(features["x_norm_patchtokens"].float(), dim=-1)
        scores = (patches * cls[:, None, :]).sum(-1).cpu().numpy().reshape(24, 24)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Non-finite similarity map")
    lo, hi = np.quantile(scores, [0.05, 0.95])
    normalized = np.clip((scores - lo) / max(hi - lo, 1e-6), 0, 1)
    heat = Image.fromarray(np.uint8(np.round(normalized * 255)), mode="L")
    heat = heat.resize(crop.size, Image.Resampling.BICUBIC)
    rgb = np.asarray(crop.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(heat, dtype=np.float32)[..., None] / 255 * 0.50
    color = np.array([255, 70, 20], dtype=np.float32)
    overlay = Image.fromarray(np.uint8(np.clip(rgb * (1 - alpha) + color * alpha, 0, 255)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.output)
    print(json.dumps({"output": str(args.output), "model_version": manifest["model_version"],
                      "patch_grid": [24, 24], "score_min": float(scores.min()),
                      "score_max": float(scores.max()),
                      "meaning": "CLS-to-patch similarity diagnostic, not causal attention"}))


if __name__ == "__main__":
    main()
