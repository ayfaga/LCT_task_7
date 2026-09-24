"""Export unlabeled organizer test files with the selected E2 boosting policy.

The query and gallery NPZ files are independently generated with build_gallery.py
from organizer crops. This module loads no identity labels or sealed holdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.boosting import HybridBoostingPolicy  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ids(csv_path: Path) -> np.ndarray:
    with csv_path.open(newline="") as stream:
        ids = [str(row["image_id"]) for row in csv.DictReader(stream)]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate image_id in {csv_path}")
    return np.asarray(ids)


def _load_features(path: Path, expected_ids: np.ndarray, model: dict) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["gallery_ids"].astype(str)
        vectors = archive["embeddings"].astype(np.float32)
        version = str(archive["model_version"].item())
        model_sha = str(archive["model_sha256"].item())
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"Feature ID order differs from organizer CSV: {path}")
    if version != model["model_version"] or model_sha != model["model_sha256"]:
        raise ValueError(f"Feature model version/checksum differs: {path}")
    if vectors.shape != (len(ids), 1024) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid feature shape or values: {path}")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-5):
        raise ValueError(f"Non-normalized feature vectors: {path}")
    return vectors


def export(artifact_dir: Path, query_csv: Path, gallery_csv: Path,
           query_features: Path, gallery_features: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    model = json.loads((artifact_dir / "model_manifest.json").read_text())
    calibration = json.loads((artifact_dir / "calibration_manifest.json").read_text())
    if calibration["model_version"] != model["model_version"]:
        raise ValueError("Model/policy version mismatch")
    policy_path = artifact_dir / calibration["policy_file"]
    if sha256(policy_path) != calibration["policy_sha256"]:
        raise ValueError("Policy checksum mismatch")
    policy = HybridBoostingPolicy.load(policy_path, model["model_version"])
    if (not np.isclose(policy.cosine_threshold, calibration["cosine_threshold"], atol=1e-12, rtol=0)
            or not np.isclose(policy.booster_threshold, calibration["booster_threshold"], atol=1e-12, rtol=0)):
        raise ValueError("Policy threshold mismatch")
    query_ids = _read_ids(query_csv)
    gallery_ids = _read_ids(gallery_csv)
    if len(gallery_ids) < 10 or set(query_ids) & set(gallery_ids):
        raise ValueError("Need at least 10 gallery IDs and disjoint query/gallery IDs")
    queries = _load_features(query_features, query_ids, model)
    gallery = _load_features(gallery_features, gallery_ids, model)
    embeddings = np.concatenate((queries, gallery)).astype(np.float32, copy=False)
    # macOS Accelerate can report spurious FP flags even for finite dot products.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = queries @ gallery.T
    if not np.isfinite(scores).all():
        raise ValueError("Cosine matrix contains non-finite values")
    order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    top_scores = np.take_along_axis(scores, order, axis=1)
    accepted = policy.accepted(top_scores)

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", embeddings, allow_pickle=False)
    with (output / "submission.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id"] + [f"gallery_id_{rank}" for rank in range(1, 11)])
        for query_id, ranking in zip(query_ids, order):
            writer.writerow([query_id, *gallery_ids[ranking]])
    with (output / "candidates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "gallery_id", "confidence"])
        for query_id, ranking, row_scores, row_accepted in zip(query_ids, order, top_scores, accepted):
            if row_accepted.any():
                for gallery_index, cosine in zip(ranking[row_accepted], row_scores[row_accepted]):
                    writer.writerow([query_id, gallery_ids[gallery_index], repr(float(cosine))])
            else:
                writer.writerow([query_id, "", ""])

    # Independent round-trip of the saved embeddings and both CSVs.
    saved = np.load(output / "embeddings.npy", allow_pickle=False)
    if saved.shape != (len(query_ids) + len(gallery_ids), 1024) or saved.dtype != np.float32:
        raise AssertionError("Exported embedding format mismatch")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        recomputed = saved[:len(query_ids)] @ saved[len(query_ids):].T
    if not np.isfinite(recomputed).all():
        raise AssertionError("Exported embeddings yielded non-finite scores")
    expected_order = np.argsort(-recomputed, axis=1, kind="stable")[:, :10]
    if not np.array_equal(expected_order, order):
        raise AssertionError("Exported embedding ranking changed")
    recomputed_top = np.take_along_axis(recomputed, expected_order, axis=1)
    expected_accept = policy.accepted(recomputed_top)
    with (output / "submission.csv").open(newline="") as stream:
        submission = list(csv.reader(stream))
    if len(submission) != len(query_ids) + 1:
        raise AssertionError("Submission query count mismatch")
    for row, query_id, ranking in zip(submission[1:], query_ids, expected_order):
        if row != [query_id, *gallery_ids[ranking]]:
            raise AssertionError("Submission does not match exact cosine top-10")
    expected_rows = []
    for query_id, ranking, row_scores, row_accepted in zip(
            query_ids, expected_order, recomputed_top, expected_accept):
        if row_accepted.any():
            expected_rows.extend((query_id, str(gallery_ids[index]), float(cosine))
                                 for index, cosine in zip(ranking[row_accepted], row_scores[row_accepted]))
        else:
            expected_rows.append((query_id, "", None))
    with (output / "candidates.csv").open(newline="") as stream:
        candidates = list(csv.DictReader(stream))
    if len(candidates) != len(expected_rows):
        raise AssertionError("Candidate row count mismatch")
    for actual, expected in zip(candidates, expected_rows):
        if actual["query_id"] != expected[0] or actual["gallery_id"] != expected[1]:
            raise AssertionError("Candidate IDs or order differ from policy")
        if expected[2] is None:
            if actual["confidence"] != "":
                raise AssertionError("Rejected query must have blank confidence")
        elif abs(float(actual["confidence"]) - expected[2]) > 1e-7:
            raise AssertionError("Candidate confidence differs from raw cosine")
    names = ("submission.csv", "embeddings.npy", "candidates.csv")
    manifest = {
        "schema_version": 1,
        "bundle_version": model["bundle_version"],
        "model_version": model["model_version"],
        "policy_kind": calibration["policy_kind"],
        "policy_sha256": calibration["policy_sha256"],
        "source": {"query_csv_sha256": sha256(query_csv),
                   "gallery_csv_sha256": sha256(gallery_csv),
                   "query_features_sha256": sha256(query_features),
                   "gallery_features_sha256": sha256(gallery_features)},
        "artifacts": {name: {"bytes": (output / name).stat().st_size,
                             "sha256": sha256(output / name)} for name in names},
        "validation": {"queries": len(query_ids), "gallery": len(gallery_ids),
                       "embedding_shape": list(saved.shape),
                       "submission_rows": len(query_ids),
                       "candidates_rows": len(candidates),
                       "roundtrip_top10_exact": True,
                       "boosting_policy_consistent": True,
                       "empty_answers": int((~expected_accept.any(axis=1)).sum())},
        "confidence_semantics": "raw cosine similarity; not a probability",
        "holdout_loaded": False,
        "note": "Unlabeled test export; no hidden score or holdout metric claimed.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--query-csv", required=True, type=Path)
    parser.add_argument("--gallery-csv", required=True, type=Path)
    parser.add_argument("--query-features", required=True, type=Path)
    parser.add_argument("--gallery-features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = export(args.artifact_dir, args.query_csv, args.gallery_csv,
                    args.query_features, args.gallery_features, args.output)
    print(json.dumps({"complete": True, **result["validation"]}))
