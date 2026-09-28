"""Fit a version-bound refusal policy for the joint L336 encoder.

Calibration IDs alone select cosine/booster thresholds and fit the booster.
Dev is an exploratory readout. Holdout and test labels are never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from audit_refusal_boosting import (SPECS, best_threshold, hybrid, make_booster,
                                    oof_predictions, predict, xy)
from audit_refusal_policy import attach_ids, bootstrap, metrics, sha256
from protocol import calibrate, open_set_scores


MODEL_VERSION = "dinov2-l14-l336-joint-cityflow-organizer-best19-20260925"
EXPECTED_OBJECTS = "c5eabdd4c17e02265f65b90c0568d51ef7907e099b413ba397f694f6f40c661d"
EXPECTED_SPLITS = "9de62999082a5dd832fc6df53755b94468bf0c072aa8dffadb23c927aa2f1345"
EXPECTED_SPLITS_COPY = "2ea8a4eae4eeea90c2e9068787356d62be16932ecdaa340c84c5839bf22f6d7f"


def fold_pack(frame: pd.DataFrame, identities: list[int], feature_file: Path):
    sub = frame[(frame.split == "train") & frame.vehicle_id.isin(identities)].reset_index(drop=True)
    with np.load(feature_file, allow_pickle=False) as archive:
        image_ids = archive["image_ids"].astype(str)
        if not np.array_equal(image_ids, sub.image_id.to_numpy(dtype=str)):
            raise ValueError(f"Feature/image order mismatch: {feature_file}")
        embeddings = archive["cls"].astype(np.float32)
    if embeddings.shape != (len(sub), 1024) or not np.isfinite(embeddings).all():
        raise ValueError(f"Invalid embeddings: {feature_file}")
    return attach_ids(open_set_scores(embeddings, sub), sub)


def portable_model(model, model_version, cosine_threshold, booster_threshold, model_sha):
    if model.n_features_in_ != 5:
        raise ValueError("Expected five score-context features")
    trees = []
    for iteration in model._predictors:
        if len(iteration) != 1:
            raise ValueError("Expected binary classifier trees")
        nodes = []
        for node in iteration[0].nodes:
            if node["is_categorical"]:
                raise ValueError("Categorical split not portable")
            nodes.append([int(node["is_leaf"]), int(node["feature_idx"]),
                          float(node["num_threshold"]), int(node["left"]),
                          int(node["right"]), float(node["value"])])
        trees.append(nodes)
    return {"schema_version": 2, "model_version": model_version,
            "source_model_sha256": model_sha,
            "source_sklearn_version": sklearn.__version__,
            "feature_names": ["candidate_cosine", "top1_cosine", "top10_mean",
                              "top10_std", "candidate_minus_top1"],
            "topk": 10, "preserve_first_n": 3,
            "cosine_threshold": float(cosine_threshold),
            "booster_threshold": float(booster_threshold),
            "baseline_logit": float(np.asarray(model._baseline_prediction).item()),
            "trees": trees}


def main():
    parser = argparse.ArgumentParser()
    for name in ("objects", "splits", "calibration-features", "dev-features", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.objects) != EXPECTED_OBJECTS or sha256(args.splits) not in (EXPECTED_SPLITS, EXPECTED_SPLITS_COPY):
        raise ValueError("Organizer metadata/split changed")
    frame = pd.read_csv(args.objects)
    identities = json.loads(args.splits.read_text())["identities"]
    if set(identities["calibration"]) & (set(identities["dev"]) | set(identities["holdout"])):
        raise ValueError("Overlapping identity folds")
    cal = fold_pack(frame, identities["calibration"], args.calibration_features)
    dev = fold_pack(frame, identities["dev"], args.dev_features)
    if (cal["queries"], cal["gallery_size"], dev["queries"], dev["gallery_size"]) != (652, 277, 1036, 413):
        raise ValueError("Open-set protocol cardinalities changed")
    cosine_threshold = calibrate(cal)[0]["threshold_cosine"]
    base_cal = cal["scores"] >= cosine_threshold
    base_dev = dev["scores"] >= cosine_threshold
    baseline_cal = metrics(cal, base_cal)
    baseline_dev = metrics(dev, base_dev)
    spec = SPECS["boost_shallow5"]
    oof = oof_predictions(cal, spec)
    chosen = best_threshold(cal, oof, baseline_cal, cosine_threshold, "hybrid")
    passed = chosen["metrics"]["candidate_micro_F1"] >= baseline_cal["candidate_micro_F1"] + .005
    threshold = chosen["threshold"] if passed else 1e-12
    x, y = xy(cal, spec["kind"])
    fitted = make_booster(spec).fit(x, y)
    cal_fitted = predict(fitted, cal, spec["kind"])
    dev_fitted = predict(fitted, dev, spec["kind"])
    selected_dev = hybrid(dev["scores"], dev_fitted, threshold, cosine_threshold)
    args.output.mkdir(parents=True)
    model_path = args.output / "booster.joblib"
    joblib.dump(fitted, model_path)
    artifact = portable_model(fitted, MODEL_VERSION, cosine_threshold, threshold, sha256(model_path))
    policy_path = args.output / "boosting_policy.json"
    policy_path.write_text(json.dumps(artifact, separators=(",", ":")))
    result = {
        "state": "completed", "model_version": MODEL_VERSION,
        "sources_sha256": {"objects": sha256(args.objects), "splits": sha256(args.splits),
                           "calibration_features": sha256(args.calibration_features),
                           "dev_features": sha256(args.dev_features), "script": sha256(Path(__file__))},
        "fit_fold": "organizer calibration only", "readout_fold": "organizer dev (reused exploratory)",
        "holdout_opened": False, "cosine_threshold": float(cosine_threshold),
        "booster_threshold": float(threshold), "passed_calibration_gain_gate": bool(passed),
        "calibration": {"cosine": baseline_cal, "oof_hybrid": chosen["metrics"],
                        "fit_hybrid": metrics(cal, hybrid(cal["scores"], cal_fitted, threshold, cosine_threshold))},
        "dev": {"cosine": baseline_dev, "hybrid": metrics(dev, selected_dev),
                "hybrid_minus_cosine_bootstrap": bootstrap(dev, base_dev, selected_dev)},
        "joblib_sha256": sha256(model_path), "portable_sha256": sha256(policy_path),
        "limits": "Proxy open-set splits; repeated past dev/calibration use; official F1 grain unknown."
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"state": "completed", "cosine_threshold": cosine_threshold,
                      "booster_threshold": threshold, "passed_gate": bool(passed),
                      "calibration": result["calibration"], "dev": result["dev"],
                      "portable_sha256": result["portable_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
