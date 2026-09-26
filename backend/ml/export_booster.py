"""Export the verified E2 HistGradientBoosting model as portable JSON.

Run this once in a scikit-learn 1.6.1 environment on the downloaded joblib.
The backend serves the resulting JSON without scikit-learn or joblib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier


EXPECTED_MODEL_SHA = "c0cd6830e15660217a6cf60b021a9d5581b39657a856c589b36e355ef00757b8"
EXPECTED_POLICY_SHA = "a98bdcd4c97cc95e1f9c17049c8215851a6e33b78796b133d45e3a07bf023a3c"


def file_sha(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def export(model_path: Path, policy_path: Path, output_path: Path) -> None:
    if sklearn.__version__ != "1.6.1":
        raise ValueError("Export requires scikit-learn 1.6.1")
    if file_sha(model_path) != EXPECTED_MODEL_SHA or file_sha(policy_path) != EXPECTED_POLICY_SHA:
        raise ValueError("Downloaded booster artifact SHA256 mismatch")
    policy = json.loads(policy_path.read_text())
    if (policy.get("kind") != "hybrid_hist_gradient_boosting"
            or policy.get("feature_kind") != "context5"
            or policy.get("model_sha256") != EXPECTED_MODEL_SHA
            or not policy.get("candidate_passed_calibration_gate")):
        raise ValueError("Unexpected booster policy")
    model = joblib.load(model_path)
    if not isinstance(model, HistGradientBoostingClassifier) or model.n_features_in_ != 5:
        raise ValueError("Unexpected booster model")
    trees = []
    for iteration in model._predictors:
        if len(iteration) != 1:
            raise ValueError("Expected one binary classifier tree per iteration")
        predictor = iteration[0]
        nodes = []
        for node in predictor.nodes:
            if node["is_categorical"]:
                raise ValueError("Categorical booster split is unsupported")
            nodes.append([
                int(node["is_leaf"]), int(node["feature_idx"]),
                float(node["num_threshold"]), int(node["left"]),
                int(node["right"]), float(node["value"]),
            ])
        trees.append(nodes)
    artifact = {
        "schema_version": 2,
        "model_version": policy["model_version"],
        "source_model_sha256": EXPECTED_MODEL_SHA,
        "source_policy_sha256": EXPECTED_POLICY_SHA,
        "source_sklearn_version": sklearn.__version__,
        "feature_names": ["candidate_cosine", "top1_cosine", "top10_mean", "top10_std", "candidate_minus_top1"],
        "topk": policy["topk"],
        "preserve_first_n": policy["preserve_first_n"],
        "cosine_threshold": policy["cosine_threshold"],
        "booster_threshold": policy["booster_threshold"],
        "baseline_logit": float(np.asarray(model._baseline_prediction).item()),
        "trees": trees,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, separators=(",", ":")))
    print(json.dumps({"output": str(output_path), "trees": len(trees),
                      "bytes": output_path.stat().st_size, "sha256": file_sha(output_path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.model, args.policy, args.output)
