"""Verify and complete a versioned joint-L336 inference bundle.

The large inference weight is external to Git; no training checkpoint or data
is copied into the bundle. Existing files are validated, never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.boosting import HybridBoostingPolicy  # noqa: E402
from app.ml.preprocessing import VERSION  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def assemble(directory: Path) -> dict:
    model = json.loads((directory / "model_manifest.json").read_text())
    metadata = json.loads((directory / "model_inference.pt.json").read_text())
    weight = directory / model["model_file"]
    policy_file = directory / "boosting_policy.json"
    policy_payload = json.loads(policy_file.read_text())
    if model["schema_version"] != 1 or model["preprocessing_version"] != VERSION:
        raise ValueError("Model manifest/preprocessing version mismatch")
    preprocessing = Path(__file__).resolve().parents[1] / "app/ml/preprocessing.py"
    if sha256(preprocessing) != model["preprocessing_sha256"]:
        raise ValueError("Preprocessing code SHA mismatch")
    if (weight.stat().st_size != model["model_size_bytes"]
            or sha256(weight) != model["model_sha256"]
            or metadata["sha256"] != model["model_sha256"]
            or metadata["source_checkpoint_sha256"] != model["source_training_checkpoint_sha256"]
            or metadata["weights_sha256"] != model["source_pretrain_sha256"]):
        raise ValueError("Inference weight/lineage mismatch")
    if model["model_size_bytes"] >= 2_000_000_000:
        raise ValueError("Inference weight exceeds 2 GB")
    policy = HybridBoostingPolicy(policy_payload, model["model_version"])
    calibration = {
        "schema_version": 1, "model_version": model["model_version"],
        "policy_kind": "hybrid_hist_gradient_boosting",
        "policy_file": policy_file.name, "policy_sha256": sha256(policy_file),
        "topk": 10, "preserve_first_n": 3,
        "cosine_threshold": policy.cosine_threshold,
        "booster_threshold": policy.booster_threshold,
        "score_semantics": "returned confidence is raw cosine; booster score is internal",
        "selection": "organizer calibration ID-group OOF; dev exploratory; no holdout",
        "official_f1_grain_known": False,
    }
    write_new(directory / "calibration_manifest.json", calibration)
    members = ["model_inference.pt", "model_inference.pt.json", "model_manifest.json",
               "boosting_policy.json", "calibration_manifest.json"]
    manifest = {"schema_version": 1, "bundle_version": model["bundle_version"],
                "members": {name: {"bytes": (directory / name).stat().st_size,
                                   "sha256": sha256(directory / name)} for name in members}}
    write_new(directory / "bundle_manifest.json", manifest)
    return {"bundle_version": model["bundle_version"], "weight_bytes": weight.stat().st_size,
            "weight_sha256": model["model_sha256"], "policy_sha256": calibration["policy_sha256"],
            "bundle_manifest_sha256": sha256(directory / "bundle_manifest.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    print(json.dumps(assemble(parser.parse_args().artifact_dir)))
