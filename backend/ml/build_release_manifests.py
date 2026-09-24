"""Create an honest local manifest for the E2 + portable boosting bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.preprocessing import VERSION  # noqa: E402


MODEL_SHA = "dc910395dda9dc735b1d5571ef7baf75b767964500b6d0a23edeb7cde4d33c52"
JOBLIB_SHA = "c0cd6830e15660217a6cf60b021a9d5581b39657a856c589b36e355ef00757b8"
SOURCE_POLICY_SHA = "a98bdcd4c97cc95e1f9c17049c8215851a6e33b78796b133d45e3a07bf023a3c"
MODEL_VERSION = "dinov2-l14-l336-fresh16-best13-20260923"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def build(directory: Path) -> None:
    expected_files = {
        "model_inference.pt": MODEL_SHA,
        "l336_e2_refusal_boosting_20260924_v1.joblib": JOBLIB_SHA,
        "l336_e2_refusal_boosting_20260924_v1_policy.json": SOURCE_POLICY_SHA,
    }
    for name, expected in expected_files.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Source artifact SHA256 mismatch: {name}")
    checkpoint = torch.load(directory / "model_inference.pt", map_location="cpu", weights_only=True)
    checkpoint_metadata = {key: value for key, value in checkpoint.items() if key != "encoder"}
    if (checkpoint_metadata.get("architecture") != "large"
            or checkpoint_metadata.get("input_size") != 336
            or checkpoint_metadata.get("embedding_dimension") != 1024):
        raise ValueError("Unexpected E2 checkpoint metadata")
    source_policy = json.loads((directory / "l336_e2_refusal_boosting_20260924_v1_policy.json").read_text())
    portable_path = directory / "boosting_policy.json"
    portable = json.loads(portable_path.read_text())
    if (source_policy["model_version"] != MODEL_VERSION
            or portable["model_version"] != MODEL_VERSION
            or portable["source_model_sha256"] != JOBLIB_SHA
            or portable["source_policy_sha256"] != SOURCE_POLICY_SHA):
        raise ValueError("Boosting policy version or provenance mismatch")
    preprocessing_path = Path(__file__).resolve().parents[1] / "app" / "ml" / "preprocessing.py"
    model_manifest = {
        "schema_version": 1,
        "bundle_version": "e2-boosting-hybrid1-20260924",
        "model_version": MODEL_VERSION,
        "model_file": "model_inference.pt",
        "model_sha256": MODEL_SHA,
        "model_size_bytes": (directory / "model_inference.pt").stat().st_size,
        "model_kind": "dinov2_normalized_cls",
        "architecture": "dinov2_vitl14",
        "input_size": 336,
        "embedding_dimension": 1024,
        "embedding_dtype": "float32",
        "embedding_normalization": "l2",
        "preprocessing_version": VERSION,
        "preprocessing_sha256": sha256(preprocessing_path),
        "source_training_checkpoint_sha256": checkpoint_metadata["source_checkpoint_sha256"],
        "source_pretrain_sha256": checkpoint_metadata["weights_sha256"],
        "source_cosine_bundle_manifest_sha256": "fafb9b7b0c5414dc53504613ce925517f5c8bc1657d18c40a65777b55ff7e41d",
    }
    calibration_manifest = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "policy_kind": "hybrid_hist_gradient_boosting",
        "policy_file": "boosting_policy.json",
        "policy_sha256": sha256(portable_path),
        "source_joblib_sha256": JOBLIB_SHA,
        "source_policy_sha256": SOURCE_POLICY_SHA,
        "topk": 10,
        "preserve_first_n": 3,
        "cosine_threshold": 0.394,
        "booster_threshold": 0.112,
        "score_semantics": "raw cosine is returned; booster score is internal",
        "selection": "user selected a balanced tradeoff on 2026-09-24",
        "official_f1_grain_known": False,
        "holdout_evaluated_for_policy_selection": False,
    }
    delivery = {
        "schema_version": 1,
        "bundle_version": "e2-boosting-hybrid1-20260924",
        "status": "local_ml_candidate",
        "source_report": "reports/l336_e2_refusal_boosting_20260924.md",
        "dev_proxy": {
            "encoder_map": 0.7045909961,
            "pair_f1": 0.5325408618,
            "tnr": 0.6605080831,
            "query_correct_id_f1": 0.6633744856,
        },
        "notes": "Proxy dev scores are historical; no official evaluator or hidden holdout score is claimed.",
    }
    test_export_path = directory / "test_export" / "manifest.json"
    if test_export_path.exists():
        test_export = json.loads(test_export_path.read_text())
        checks = test_export["validation"]
        if (test_export["bundle_version"] != model_manifest["bundle_version"]
                or test_export["model_version"] != MODEL_VERSION
                or not checks["roundtrip_top10_exact"]
                or not checks["boosting_policy_consistent"]):
            raise ValueError("Test export is incompatible with the new bundle")
        delivery["unlabeled_test_export"] = checks
    write_json(directory / "model_inference.pt.json", {
        "sha256": MODEL_SHA, "size_bytes": model_manifest["model_size_bytes"], **checkpoint_metadata,
    })
    write_json(directory / "model_manifest.json", model_manifest)
    write_json(directory / "calibration_manifest.json", calibration_manifest)
    write_json(directory / "delivery.json", delivery)
    members = ["model_inference.pt", "boosting_policy.json",
               "l336_e2_refusal_boosting_20260924_v1.joblib",
               "l336_e2_refusal_boosting_20260924_v1_policy.json",
               "model_manifest.json", "calibration_manifest.json",
               "model_inference.pt.json", "delivery.json"]
    if (directory / "runtime_smoke.json").exists():
        members.append("runtime_smoke.json")
    if (directory / "api_smoke.json").exists():
        members.append("api_smoke.json")
    if (directory / "portable_parity.json").exists():
        members.append("portable_parity.json")
    if test_export_path.exists():
        members.extend(f"test_export/{name}" for name in
                       ("manifest.json", "submission.csv", "embeddings.npy", "candidates.csv"))
    bundle = {"schema_version": 1, "bundle_version": model_manifest["bundle_version"],
              "members": {name: {"bytes": (directory / name).stat().st_size,
                                 "sha256": sha256(directory / name)} for name in members}}
    write_json(directory / "bundle_manifest.json", bundle)
    print(json.dumps({"bundle_manifest_sha256": sha256(directory / "bundle_manifest.json"),
                      "files": len(members), "model_size_bytes": model_manifest["model_size_bytes"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    build(parser.parse_args().artifact_dir)
