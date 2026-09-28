"""Small, dependency-light refusal policy over an already sorted cosine top-10.

This module never reads images, identity labels, cameras or licence plates. It
only decides which retrieval candidates pass a versioned acceptance threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FEATURES = {
    "score_only": ("candidate_cosine",),
    "context4": ("candidate_cosine", "top1_cosine", "top10_mean", "top10_std"),
    "context5": ("candidate_cosine", "top1_cosine", "top10_mean", "top10_std", "candidate_minus_top1"),
}
QUERY_FEATURES = ("top1_cosine", "top1_minus_top2", "top10_mean", "top10_std")


def _check_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 10 or not np.isfinite(scores).all():
        raise ValueError("Refusal policy requires a finite [query, 10] cosine matrix")
    if (scores[:, :-1] < scores[:, 1:] - 1e-7).any():
        raise ValueError("Cosine candidates must be sorted in descending order")
    if ((scores < -1.0001) | (scores > 1.0001)).any():
        raise ValueError("Cosine scores must lie in [-1, 1]")
    return scores


def pair_features(scores: np.ndarray, kind: str) -> np.ndarray:
    scores = _check_scores(scores)
    if kind not in FEATURES:
        raise ValueError(f"Unknown pair feature set: {kind}")
    top = scores[:, :1]
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    values = {
        "candidate_cosine": scores,
        "top1_cosine": np.broadcast_to(top, scores.shape),
        "top10_mean": np.broadcast_to(mean, scores.shape),
        "top10_std": np.broadcast_to(std, scores.shape),
        "candidate_minus_top1": scores - top,
    }
    return np.stack([values[name] for name in FEATURES[kind]], axis=2)


def query_features(scores: np.ndarray) -> np.ndarray:
    scores = _check_scores(scores)
    return np.stack((scores[:, 0], scores[:, 0] - scores[:, 1],
                     scores.mean(axis=1), scores.std(axis=1)), axis=1)


def _predict_logit(model: dict, matrix: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    if matrix.shape[-1] != len(mean) or len(scale) != len(mean) or len(coef) != len(mean):
        raise ValueError("Model feature dimension mismatch")
    if not np.isfinite(matrix).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(coef).all() or (scale <= 0).any():
        raise ValueError("Invalid refusal model coefficients/features")
    z = ((matrix - mean) / scale) @ coef + float(model["intercept"])
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def policy_scores(scores: np.ndarray, policy: dict) -> np.ndarray:
    scores = _check_scores(scores)
    kind = policy["kind"]
    if kind == "cosine":
        return scores
    feature_kind = "context4" if kind == "hybrid_context4" else kind
    if feature_kind not in FEATURES:
        raise ValueError(f"Unsupported refusal policy kind: {kind}")
    pair = _predict_logit(policy["pair_model"], pair_features(scores, feature_kind).reshape(-1, len(FEATURES[feature_kind]))).reshape(scores.shape)
    if policy.get("query_model") is not None:
        query = _predict_logit(policy["query_model"], query_features(scores))
        pair *= query[:, None]
    if (pair[:, :-1] < pair[:, 1:] - 1e-12).any():
        raise ValueError("Refusal policy changed cosine candidate ranking")
    return pair


def accepted(scores: np.ndarray, policy: dict) -> np.ndarray:
    scores = _check_scores(scores)
    learned = policy_scores(scores, policy)
    result = learned >= float(policy["threshold"])
    if policy["kind"] == "hybrid_context4":
        preserve = int(policy["preserve_first_n"])
        cosine_threshold = float(policy["baseline_cosine_threshold"])
        if not 1 <= preserve <= 10 or not np.isfinite(cosine_threshold):
            raise ValueError("Invalid hybrid refusal policy")
        result = (scores >= cosine_threshold) & ((np.arange(10)[None, :] < preserve) | result)
    # A monotone score makes the accepted candidates a top-ranked prefix.
    if (result[:, 1:] & ~result[:, :-1]).any():
        raise ValueError("Accepted candidates must be a ranked prefix")
    return result


def load_policy(path: str | Path, expected_model_version: str | None = None) -> dict:
    policy = json.loads(Path(path).read_text())
    if policy.get("schema_version") != 1 or policy.get("topk") != 10:
        raise ValueError("Unsupported refusal policy schema or top-k")
    if expected_model_version is not None and policy.get("model_version") != expected_model_version:
        raise ValueError("Refusal policy belongs to another encoder version")
    threshold = float(policy["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Invalid refusal threshold")
    return policy
