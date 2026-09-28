"""Visualize final-block CLS attention for the frozen joint L336 encoder.

This is the actual softmax attention matrix of the last transformer block,
not CLS/patch cosine and not a causal explanation of retrieval decisions.
"""

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


def cls_attention(norm_tokens: torch.Tensor, attention: torch.nn.Module) -> np.ndarray:
    """Return head-mean last-block CLS→patch probabilities, excluding CLS."""
    batch, tokens, channels = norm_tokens.shape
    heads = attention.num_heads
    qkv = attention.qkv(norm_tokens).reshape(batch, tokens, 3, heads, channels // heads)
    query = qkv[:, :, 0].permute(0, 2, 1, 3)[:, :, :1, :]
    keys = qkv[:, :, 1].permute(0, 2, 3, 1)
    logits = (query @ keys).squeeze(2) * attention.scale
    probabilities = torch.softmax(logits.float(), dim=-1)
    patches = probabilities[:, :, 1:].mean(dim=1)
    side = int((tokens - 1) ** 0.5)
    if batch != 1 or side * side != tokens - 1:
        raise ValueError("Expected one CLS token followed by square patch grid")
    result = patches[0].detach().cpu().numpy().reshape(side, side)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite attention")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    input_tensor = tensor_from_crop(crop, encoder.input_size).unsqueeze(0).to(encoder.device)
    block = encoder.model.blocks[-1]
    captured: list[torch.Tensor] = []
    hook = block.norm1.register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach()))
    try:
        with torch.inference_mode():
            encoder.model.forward_features(input_tensor)
            if len(captured) != 1:
                raise RuntimeError("Expected exactly one final-block activation")
            scores = cls_attention(captured[0], block.attn)
    finally:
        hook.remove()
    lo, hi = np.quantile(scores, [0.05, 0.95])
    normalized = np.clip((scores - lo) / max(hi - lo, 1e-12), 0, 1)
    heat = Image.fromarray(np.uint8(np.round(normalized * 255)), mode="L")
    heat = heat.resize(crop.size, Image.Resampling.BICUBIC)
    rgb = np.asarray(crop.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(heat, dtype=np.float32)[..., None] / 255 * 0.50
    color = np.array([255, 70, 20], dtype=np.float32)
    overlay = Image.fromarray(np.uint8(np.clip(rgb * (1 - alpha) + color * alpha, 0, 255)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.output)
    print(json.dumps({
        "output": str(args.output), "model_version": manifest["model_version"],
        "patch_grid": list(scores.shape), "attention_min": float(scores.min()),
        "attention_max": float(scores.max()), "attention_patch_mass": float(scores.sum()),
        "meaning": "mean-head final-block CLS→patch softmax attention; not causal attribution",
    }))


if __name__ == "__main__":
    main()
