"""Compare two frozen finalists and cheap embedding fusions without opening holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import (
    calibrate,
    make_splits,
    normalize,
    open_set_scores,
    refusal_metrics,
    retrieval_metrics,
)


def load_cls(path: str, expected_ids: np.ndarray) -> np.ndarray:
    payload = np.load(Path(path) / "features.npz", allow_pickle=False)
    if not np.array_equal(payload["image_ids"].astype(str), expected_ids.astype(str)):
        raise ValueError(f"Feature order mismatch: {path}")
    return normalize(payload["cls"])


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, draws: int, seed: int) -> dict:
    paired = a.merge(
        b,
        on=["image_id", "vehicle_id", "camera_id"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    paired["delta_ap"] = paired.ap_a - paired.ap_b
    paired["delta_rank1"] = paired.rank1_a - paired.rank1_b
    paired["delta_rank5"] = (
        (paired.first_positive_rank_a <= 5).astype(int)
        - (paired.first_positive_rank_b <= 5).astype(int)
    )
    grouped = paired.groupby("vehicle_id", sort=True).agg(
        delta_ap_sum=("delta_ap", "sum"),
        delta_rank1_sum=("delta_rank1", "sum"),
        delta_rank5_sum=("delta_rank5", "sum"),
        queries=("image_id", "size"),
    )
    values = grouped.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(draws, len(values)))
    picked = values[sampled]
    denominator = picked[:, :, 3].sum(axis=1)
    boot = {
        "mAP": picked[:, :, 0].sum(axis=1) / denominator,
        "Rank1": picked[:, :, 1].sum(axis=1) / denominator,
        "Rank5": picked[:, :, 2].sum(axis=1) / denominator,
    }
    result = {
        "queries": int(len(paired)),
        "identities": int(len(grouped)),
        "draws": int(draws),
        "delta_a_minus_b": {},
    }
    for name, samples in boot.items():
        observed = float(
            paired["delta_ap" if name == "mAP" else f"delta_{name.lower()}"].mean()
        )
        result["delta_a_minus_b"][name] = {
            "observed_pp": 100 * observed,
            "ci95_pp": (100 * np.quantile(samples, [0.025, 0.975])).tolist(),
            "bootstrap_probability_positive": float((samples > 0).mean()),
        }
    return result, paired


def refusal_report(emb: np.ndarray, frame: pd.DataFrame, splits: dict) -> dict:
    calibration_mask = (frame.split == "train") & frame.vehicle_id.isin(splits["calibration"])
    dev_mask = (frame.split == "train") & frame.vehicle_id.isin(splits["dev"])
    calibration_pack = open_set_scores(
        emb[calibration_mask], frame.loc[calibration_mask].reset_index(drop=True)
    )
    best, _ = calibrate(calibration_pack)
    dev_pack = open_set_scores(emb[dev_mask], frame.loc[dev_mask].reset_index(drop=True))
    return {
        "calibration": best,
        "dev": refusal_metrics(dev_pack, best["threshold_cosine"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", default="outputs/eda/objects.csv")
    parser.add_argument("--splits", default="repro/configs/identity_splits.json")
    parser.add_argument("--features-a", required=True)
    parser.add_argument("--features-b", required=True)
    parser.add_argument("--label-a", default="finalist_a")
    parser.add_argument("--label-b", default="finalist_b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    frame = pd.read_csv(args.objects)
    image_ids = frame.image_id.to_numpy(dtype=str)
    embeddings_a = load_cls(args.features_a, image_ids)
    embeddings_b = load_cls(args.features_b, image_ids)
    if embeddings_a.shape != embeddings_b.shape:
        raise ValueError("Finalist embeddings must have the same shape for this comparison")

    splits = make_splits(frame, args.splits)
    dev_mask = (frame.split == "train") & frame.vehicle_id.isin(splits["dev"])
    dev_frame = frame.loc[dev_mask].reset_index(drop=True)

    metrics_a, details_a = retrieval_metrics(embeddings_a[dev_mask], dev_frame, True)
    metrics_b, details_b = retrieval_metrics(embeddings_b[dev_mask], dev_frame, True)
    bootstrap, paired = paired_bootstrap(
        pd.DataFrame(details_a), pd.DataFrame(details_b), args.bootstrap_draws, args.seed
    )

    candidates = {args.label_a: embeddings_a, args.label_b: embeddings_b}
    alphas = np.linspace(0.0, 1.0, 11)
    for alpha in alphas:
        tag = f"{alpha:.1f}"
        candidates[f"blend_a{tag}"] = normalize(alpha * embeddings_a + (1.0 - alpha) * embeddings_b)
        candidates[f"concat_a{tag}"] = normalize(
            np.concatenate(
                [np.sqrt(alpha) * embeddings_a, np.sqrt(1.0 - alpha) * embeddings_b], axis=1
            )
        )

    leaderboard = []
    candidate_metrics = {}
    for name, embeddings in candidates.items():
        retrieval, _ = retrieval_metrics(embeddings[dev_mask], dev_frame, True)
        standard, _ = retrieval_metrics(embeddings[dev_mask], dev_frame, False)
        refusal = refusal_report(embeddings, frame, splits)
        candidate_metrics[name] = {
            "retrieval": retrieval,
            "standard_reid": standard,
            "refusal": refusal,
            "dimension": int(embeddings.shape[1]),
        }
        leaderboard.append(
            {
                "candidate": name,
                "dimension": int(embeddings.shape[1]),
                "dev_mAP": retrieval["mAP"],
                "dev_Rank1": retrieval["Rank1"],
                "dev_Rank5": retrieval["Rank5"],
                "standard_mAP": standard["mAP"],
                "calibration_threshold": refusal["calibration"]["threshold_cosine"],
                "dev_candidate_F1": refusal["dev"]["candidate_micro_F1"],
                "dev_TNR": refusal["dev"]["TNR"],
                "dev_precision": refusal["dev"]["precision"],
                "dev_recall": refusal["dev"]["recall"],
            }
        )

    leaderboard_frame = pd.DataFrame(leaderboard).sort_values(
        ["dev_mAP", "dev_candidate_F1"], ascending=False, kind="stable"
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    leaderboard_frame.to_csv(output / "leaderboard.csv", index=False)
    paired.to_csv(output / "paired_query_deltas.csv", index=False)
    best_name = str(leaderboard_frame.iloc[0].candidate)
    summary = {
        "labels": {"a": args.label_a, "b": args.label_b},
        "base_metrics": {args.label_a: metrics_a, args.label_b: metrics_b},
        "paired_ID_bootstrap": bootstrap,
        "best_dev_candidate": best_name,
        "best_dev_metrics": candidate_metrics[best_name],
        "candidates": candidate_metrics,
        "selection_scope": "dev selects candidate; calibration selects each threshold; holdout not loaded",
        "limitations": [
            "One trained seed per finalist; ID bootstrap is not training-seed variance.",
            "Fusion requires both encoders at inference unless later distilled.",
            "Proxy evaluator only; official evaluator remains unavailable.",
        ],
        "holdout_opened": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "complete": True,
        "best_dev_candidate": best_name,
        "best_dev_metrics": candidate_metrics[best_name],
        "paired_ID_bootstrap": bootstrap,
        "holdout_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
