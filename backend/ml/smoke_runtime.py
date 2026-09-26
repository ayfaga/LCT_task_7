"""One real E2 image-to-embedding-to-search smoke check."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.runtime import MLRuntime  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--gallery", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--expected-gallery-id", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    start = time.perf_counter()
    runtime = MLRuntime(args.artifact_dir, args.gallery, args.device)
    load_seconds = time.perf_counter() - start
    with Image.open(args.image) as image:
        embedding = runtime.embed(image, (0, 0, image.width, image.height))
    result = runtime.search(embedding, topk=10)
    top1 = result["ranked"][0]
    if (top1["gallery_id"] != args.expected_gallery_id
            or top1["similarity"] < 0.9999
            or result["status"] != "matched"
            or result["accepted"][0]["gallery_id"] != args.expected_gallery_id
            or embedding.shape != (1024,)
            or not np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)):
        raise AssertionError("Real E2 gallery search failed its smoke contract")
    observation = {
        "status": "passed",
        "model_version": runtime.model_version,
        "policy_kind": "hybrid_hist_gradient_boosting",
        "device": args.device,
        "gallery_size": len(runtime.gallery_ids),
        "embedding_dimension": len(embedding),
        "embedding_norm": float(np.linalg.norm(embedding)),
        "top1_gallery_id": top1["gallery_id"],
        "top1_cosine": top1["similarity"],
        "accepted_count": len(result["accepted"]),
        "model_load_seconds": load_seconds,
        "note": "Single same-image smoke; not retrieval quality or API performance.",
    }
    (args.artifact_dir / "runtime_smoke.json").write_text(json.dumps(observation, indent=2) + "\n")
    print(json.dumps(observation))


if __name__ == "__main__":
    main()
