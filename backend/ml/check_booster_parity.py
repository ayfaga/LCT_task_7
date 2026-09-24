"""Compare the portable inference policy against the original sklearn model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.boosting import HybridBoostingPolicy, _features  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--portable", required=True, type=Path)
    parser.add_argument("--query-features", type=Path)
    parser.add_argument("--gallery-features", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifact = json.loads(args.portable.read_text())
    if sha256(args.model) != artifact["source_model_sha256"]:
        raise ValueError("Source booster model SHA256 mismatch")
    model = joblib.load(args.model)
    portable = HybridBoostingPolicy(artifact, artifact["model_version"])
    if bool(args.query_features) != bool(args.gallery_features):
        raise ValueError("Both query and gallery features must be supplied together")
    if args.query_features:
        with np.load(args.query_features, allow_pickle=False) as query_archive:
            queries = query_archive["embeddings"].astype(np.float32)
            query_version = str(query_archive["model_version"].item())
            query_model_sha = str(query_archive["model_sha256"].item())
        with np.load(args.gallery_features, allow_pickle=False) as gallery_archive:
            gallery = gallery_archive["embeddings"].astype(np.float32)
            gallery_version = str(gallery_archive["model_version"].item())
            gallery_model_sha = str(gallery_archive["model_sha256"].item())
        if (query_version != artifact["model_version"]
                or gallery_version != query_version
                or query_model_sha != gallery_model_sha):
            raise ValueError("Real feature files belong to different encoders")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            similarities = queries @ gallery.T
        if gallery.shape[0] < 10 or not np.isfinite(similarities).all():
            raise ValueError("Invalid real query/gallery cosine matrix")
        top10 = np.argsort(-similarities, axis=1, kind="stable")[:, :10]
        scores = np.take_along_axis(similarities, top10, axis=1)
        source = "unlabeled_test_cosine_top10"
    else:
        rng = np.random.default_rng(20260924)
        # Values span the decision region and include exact threshold edges.
        scores = np.sort(rng.uniform(0.18, 0.82, size=(500, 10)), axis=1)[:, ::-1].copy()
        scores[0] = np.array([0.394, 0.394, 0.394, 0.393, 0.393,
                              0.392, 0.39, 0.38, 0.37, 0.36])
        source = "synthetic_threshold_probe"
    features = _features(scores).reshape(-1, 5)
    reference = model.predict_proba(features)[:, 1].reshape(scores.shape)
    actual = portable.scores(scores)
    max_error = float(np.max(np.abs(reference - actual)))
    reference_accept = (scores >= artifact["cosine_threshold"]) & (
        (np.arange(10)[None, :] < 3) | (reference >= artifact["booster_threshold"])
    )
    different_decisions = int(np.count_nonzero(reference_accept != portable.accepted(scores)))
    if max_error > 1e-12 or different_decisions:
        raise AssertionError(f"Booster parity failed: max_error={max_error}, differences={different_decisions}")
    observation = {"source": source, "queries": len(scores), "candidates": scores.size,
                   "max_abs_score_error": max_error,
                   "different_acceptance_decisions": different_decisions,
                   "portable_sha256": sha256(args.portable)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(observation, indent=2) + "\n")
    print(json.dumps(observation))


if __name__ == "__main__":
    main()
