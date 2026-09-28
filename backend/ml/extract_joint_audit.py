"""Extract frozen joint inference embeddings for organizer dev/calibration only.

Runs locally without training checkpoint or CULAB. Keeps every crop unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.encoder import E2Encoder, file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--crops", required=True, type=Path)
    parser.add_argument("--fold", required=True, choices=["dev", "calibration"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("Batch size and thread count must be positive")
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.artifact_dir / "model_manifest.json").read_text())
    split = json.loads(args.splits.read_text())["identities"]
    frame = pd.read_csv(args.objects)
    frame = frame[(frame.split == "train") & frame.vehicle_id.isin(split[args.fold])].reset_index(drop=True)
    if frame.empty or frame.image_id.duplicated().any():
        raise ValueError("Empty or duplicate-ID fold")
    encoder = E2Encoder(args.artifact_dir / manifest["model_file"], manifest["model_sha256"], args.device)
    vectors = []
    started = time.monotonic()
    for start in range(0, len(frame), args.batch_size):
        crops = []
        for image_id in frame.image_id.iloc[start:start + args.batch_size]:
            with Image.open(args.crops / f"{image_id}.png") as image:
                crops.append(image.convert("RGB"))
        vectors.append(encoder.embed_crops(crops, args.batch_size))
        if (start // args.batch_size + 1) % 20 == 0 or start + len(crops) == len(frame):
            print(json.dumps({"fold": args.fold, "images": start + len(crops),
                              "total": len(frame), "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    embeddings = np.concatenate(vectors)
    if embeddings.shape != (len(frame), 1024) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid embeddings")
    args.output.mkdir(parents=True)
    np.savez(args.output / "features.npz", cls=embeddings,
             image_ids=frame.image_id.to_numpy(dtype=str))
    record = {
        "source": "frozen_inference_weight", "model_version": manifest["model_version"],
        "model_sha256": manifest["model_sha256"], "source_training_checkpoint_sha256":
        manifest["source_training_checkpoint_sha256"],
        "objects_sha256": file_sha256(args.objects), "splits_sha256": file_sha256(args.splits),
        "fold": args.fold, "size": 336, "geometry": "pad", "fp16": False,
        "device": args.device, "threads": args.threads, "batch_size": args.batch_size,
        "rows": len(frame), "elapsed_seconds": time.monotonic() - started,
        "embedding_semantics": "L2-normalized FP32 CLS; no train classifier",
    }
    (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"complete": True, **record}), flush=True)


if __name__ == "__main__":
    main()
