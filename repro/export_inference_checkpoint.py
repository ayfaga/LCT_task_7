"""Strip training-only state and verify an inference-only encoder checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from extract import load_encoder
from train_cuda import file_sha, save_checkpoint


def export_checkpoint(source: Path, weights: Path, output: Path, architecture: str, input_size: int) -> dict:
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError(output)
    training = torch.load(source, map_location="cpu", weights_only=True)
    if "encoder" not in training or not isinstance(training["encoder"], dict):
        raise ValueError("Training checkpoint has no encoder state")
    encoder = load_encoder(weights, architecture)
    encoder.load_state_dict(training["encoder"], strict=True)
    state = {
        "format_version": 1,
        "model_kind": "dinov2_normalized_cls",
        "architecture": architecture,
        "input_size": int(input_size),
        "embedding_dimension": int(encoder.embed_dim),
        "embedding_dtype": "float32",
        "embedding_normalization": "l2_after_encoder",
        "encoder": {key: value.detach().cpu() for key, value in encoder.state_dict().items()},
        "source_checkpoint_sha256": file_sha(source),
        "weights_sha256": file_sha(weights),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, state)
    loaded = torch.load(output, map_location="cpu", weights_only=True)
    verifier = load_encoder(weights, architecture)
    verifier.load_state_dict(loaded["encoder"], strict=True)
    for key, value in state["encoder"].items():
        if not torch.equal(value, loaded["encoder"][key]):
            raise ValueError(f"Inference checkpoint tensor mismatch: {key}")
    manifest = {
        "schema_version": 1,
        "artifact": str(output),
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": state["source_checkpoint_sha256"],
        "weights_sha256": state["weights_sha256"],
        "architecture": architecture,
        "input_size": int(input_size),
        "embedding_dimension": int(encoder.embed_dim),
        "training_only_keys_removed": sorted(set(training) - {"encoder"}),
        "strict_load_verified": True,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", choices=["small", "base", "large"], default="base")
    parser.add_argument("--input-size", type=int, default=336)
    args = parser.parse_args()
    manifest = export_checkpoint(Path(args.source), Path(args.weights), Path(args.output), args.architecture, args.input_size)
    print(json.dumps({"complete": True, **manifest}, indent=2))


if __name__ == "__main__":
    main()
