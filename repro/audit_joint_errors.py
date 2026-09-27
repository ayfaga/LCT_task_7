"""Frozen joint L336 error audit on organizer dev/calibration; never holdout.

Requires already-extracted selected-model features. No tuning is performed.
All query slices share the same deterministic open-set gallery within a fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_joint_refusal import (EXPECTED_OBJECTS, EXPECTED_SPLITS, EXPECTED_SPLITS_COPY,
                                 MODEL_VERSION, fold_pack)
from audit_refusal_policy import sha256
from protocol import retrieval_metrics

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.ml.boosting import HybridBoostingPolicy  # noqa: E402


def summarize_queries(pack: dict, frame: pd.DataFrame, decisions: np.ndarray) -> tuple[pd.DataFrame, list[dict]]:
    queries = frame.iloc[pack["query_indices"]].reset_index(drop=True)
    gallery = frame.iloc[pack["gallery_indices"]].reset_index(drop=True)
    ranked_indices = pack["ranking"]
    ranked_vehicle = gallery.vehicle_id.to_numpy()[ranked_indices]
    ranked_image = gallery.image_id.astype(str).to_numpy()[ranked_indices]
    any_correct = (decisions & pack["truth"]).any(axis=1)
    any_accepted = decisions.any(axis=1)
    known = pack["known"]
    errors = np.select(
        [~known & any_accepted, known & ~any_accepted, known & any_accepted & ~any_correct],
        ["false_accept_unknown", "false_reject_known", "wrong_accept_known"],
        default="correct_or_unresolved",
    )
    output = pd.DataFrame({
        "image_id": queries.image_id.astype(str).to_numpy(),
        "vehicle_id": queries.vehicle_id.astype(int).to_numpy(),
        "camera_id": queries.camera_id.astype(int).to_numpy(),
        "min_side": queries.min_side.astype(float).to_numpy(),
        "brightness": queries.brightness.astype(float).to_numpy(),
        "sharpness_128": queries.sharpness_128.astype(float).to_numpy(),
        "known": known.astype(bool), "top1_correct": pack["truth"][:, 0].astype(bool),
        "accepted": any_accepted, "correct_id_accepted": any_correct,
        "error_type": errors, "top1_vehicle_id": ranked_vehicle[:, 0].astype(int),
        "top1_image_id": ranked_image[:, 0],
        "top1_cosine": pack["scores"][:, 0],
        "accepted_count": decisions.sum(axis=1),
    })
    output["resolution_slice"] = pd.cut(
        output.min_side, bins=[0, 256, 384, np.inf], right=False,
        labels=["<256", "256–383", "≥384"], include_lowest=True,
    ).astype(str)
    output["brightness_slice"] = pd.cut(
        output.brightness, bins=[-np.inf, 70, 180, np.inf], right=False,
        labels=["dark<70", "70–179", "bright≥180"],
    ).astype(str)
    output["sharpness_slice"] = pd.cut(
        output.sharpness_128, bins=[-np.inf, 100, 1000, np.inf], right=False,
        labels=["blur_proxy<100", "100–999", "sharp_proxy≥1000"],
    ).astype(str)
    slices = []
    for column in ("resolution_slice", "brightness_slice", "sharpness_slice"):
        for value, group in output.groupby(column, observed=True):
            slices.append({"slice": column, "value": str(value), "queries": len(group),
                           "known": int(group.known.sum()),
                           "unknown": int((~group.known).sum()),
                           "top1_accuracy_known": float(group.loc[group.known, "top1_correct"].mean())
                           if group.known.any() else None,
                           "false_accept_unknown": int((group.error_type == "false_accept_unknown").sum()),
                           "false_accept_rate_unknown": float((group.loc[~group.known, "error_type"] ==
                                                                 "false_accept_unknown").mean()) if (~group.known).any() else None,
                           "false_reject_known": int((group.error_type == "false_reject_known").sum()),
                           "wrong_accept_known": int((group.error_type == "wrong_accept_known").sum()),
                           "known_without_correct_accept_rate": float((~group.loc[group.known,
                                                                                 "correct_id_accepted"]).mean())
                           if group.known.any() else None})
    return output, slices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--fold", required=True, choices=["dev", "calibration"])
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.objects) != EXPECTED_OBJECTS or sha256(args.splits) not in (EXPECTED_SPLITS, EXPECTED_SPLITS_COPY):
        raise ValueError("Organizer metadata/split hash mismatch")
    extraction = json.loads((args.features.parent / "run.json").read_text())
    expected_checkpoint = "e1e3c03476d3b1af8eb9d9323b896d4e1e0859064597aaa66c79ee1b3e6e8d83"
    from_training_checkpoint = (extraction.get("checkpoint_sha256") == expected_checkpoint
                                and extraction.get("identity_split") == args.fold)
    from_frozen_inference = (extraction.get("source") == "frozen_inference_weight"
                             and extraction.get("source_training_checkpoint_sha256") == expected_checkpoint
                             and extraction.get("fold") == args.fold
                             and extraction.get("model_version") == MODEL_VERSION
                             and extraction.get("model_sha256") ==
                             "507b4f2e34384e2366ab5364e147331831664493b35b40041507d88634da05f5"
                             and extraction.get("objects_sha256") == sha256(args.objects)
                             and extraction.get("splits_sha256") == sha256(args.splits))
    if (not (from_training_checkpoint or from_frozen_inference) or extraction.get("size") != 336
            or extraction.get("geometry") != "pad" or extraction.get("fp16") is not False):
        raise ValueError("Features were not extracted from the frozen joint checkpoint in FP32")
    identities = json.loads(args.splits.read_text())["identities"]
    frame = pd.read_csv(args.objects)
    fold = frame[(frame.split == "train") & frame.vehicle_id.isin(identities[args.fold])].reset_index(drop=True)
    pack = fold_pack(frame, identities[args.fold], args.features)
    with np.load(args.features, allow_pickle=False) as archive:
        embeddings = archive["cls"].astype(np.float32)
    retrieval, details = retrieval_metrics(embeddings, fold, exclude_all_same_camera=True)
    policy_artifact = json.loads(args.policy.read_text())
    policy = HybridBoostingPolicy(policy_artifact, MODEL_VERSION)
    decisions = policy.accepted(pack["scores"])
    rows, slices = summarize_queries(pack, fold, decisions)
    args.output.mkdir(parents=True)
    rows.to_csv(args.output / "open_set_queries.csv", index=False)
    identity_rows = rows.groupby(["vehicle_id", "known"], observed=True).agg(
        queries=("image_id", "size"),
        false_accept_unknown=("error_type", lambda col: int((col == "false_accept_unknown").sum())),
        false_reject_known=("error_type", lambda col: int((col == "false_reject_known").sum())),
        wrong_accept_known=("error_type", lambda col: int((col == "wrong_accept_known").sum())),
        correct_id_accepted=("correct_id_accepted", "sum"),
    ).reset_index()
    identity_rows.to_csv(args.output / "open_set_identities.csv", index=False)
    examples = rows[rows.error_type != "correct_or_unresolved"].sort_values(
        ["error_type", "top1_cosine"], ascending=[True, False]
    ).groupby("error_type", sort=True).head(10)
    examples.to_csv(args.output / "error_examples.csv", index=False)
    pd.DataFrame(details).to_csv(args.output / "cross_camera_queries.csv", index=False)
    (args.output / "slices.json").write_text(json.dumps(slices, ensure_ascii=False, indent=2) + "\n")
    candidate_tp = int((decisions & pack["truth"]).sum())
    candidate_fp = int((decisions & ~pack["truth"]).sum())
    candidate_fn = int(pack["total_positives"] - candidate_tp)
    unknown = ~pack["known"]
    summary = {
        "fold": args.fold, "model_version": policy.model_version,
        "objects_sha256": sha256(args.objects), "splits_sha256": sha256(args.splits),
        "features_sha256": sha256(args.features), "policy_sha256": sha256(args.policy),
        "retrieval": retrieval, "open_set_gallery_size": pack["gallery_size"],
        "open_set_queries": pack["queries"], "known_queries": int(pack["known"].sum()),
        "error_counts": rows.error_type.value_counts().to_dict(),
        "candidate_micro_f1": float(2 * candidate_tp / max(2 * candidate_tp + candidate_fp + candidate_fn, 1)),
        "unknown_tnr": float((~decisions.any(axis=1))[unknown].mean()) if unknown.any() else None,
        "unknown_identity_macro_tnr": float((1 - identity_rows.loc[~identity_rows.known,
            "false_accept_unknown"] / identity_rows.loc[~identity_rows.known, "queries"]).mean())
            if (~identity_rows.known).any() else None,
        "false_accept_unknown_identities": int((identity_rows.loc[~identity_rows.known,
            "false_accept_unknown"] > 0).sum()),
        "candidate_tp_fp_fn": [candidate_tp, candidate_fp, candidate_fn],
        "missing_viewpoint_and_occlusion_labels": True,
        "refusal_readout": "fitted calibration; not OOF" if args.fold == "calibration"
        else "reused exploratory dev; not sealed test",
        "scope": "same deterministic gallery for every slice; no holdout or threshold fitting",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
