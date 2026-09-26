"""Portable scorer for a versioned histogram-boosting refusal policy.

The scorer consumes only sorted cosine similarities. It never reads images,
metadata, camera IDs, licence plates, or identity labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "candidate_cosine",
    "top1_cosine",
    "top10_mean",
    "top10_std",
    "candidate_minus_top1",
)


def _features(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 10:
        raise ValueError("Boosting policy requires a [queries, 10] cosine matrix")
    if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.0001):
        raise ValueError("Cosine matrix contains invalid values")
    if np.any(scores[:, 1:] > scores[:, :-1] + 1e-7):
        raise ValueError("Cosine candidates must be sorted in descending order")
    top1 = np.broadcast_to(scores[:, :1], scores.shape)
    mean = np.broadcast_to(scores.mean(axis=1, keepdims=True), scores.shape)
    std = np.broadcast_to(scores.std(axis=1, keepdims=True), scores.shape)
    return np.stack((scores, top1, mean, std, scores - top1), axis=-1)


def _tree_values(tree: list[list[float | int]], features: np.ndarray) -> np.ndarray:
    """Traverse an sklearn HistGradientBoosting tree on finite numeric inputs.

    Each node is [is_leaf, feature_index, threshold, left, right, value].
    Leaf values in the trained predictor already include learning_rate.
    """
    indices = np.zeros(features.shape[0], dtype=np.int32)
    rows = np.arange(features.shape[0])
    while True:
        nodes = [tree[int(index)] for index in indices]
        active = np.fromiter((not node[0] for node in nodes), dtype=bool, count=len(nodes))
        if not active.any():
            return np.fromiter((node[5] for node in nodes), dtype=np.float64, count=len(nodes))
        active_rows = rows[active]
        active_nodes = [nodes[int(index)] for index in active_rows]
        feature_indices = np.fromiter((node[1] for node in active_nodes), dtype=np.int32)
        thresholds = np.fromiter((node[2] for node in active_nodes), dtype=np.float64)
        go_left = features[active_rows, feature_indices] <= thresholds
        indices[active_rows] = np.where(
            go_left,
            np.fromiter((node[3] for node in active_nodes), dtype=np.int32),
            np.fromiter((node[4] for node in active_nodes), dtype=np.int32),
        )


class HybridBoostingPolicy:
    def __init__(self, artifact: dict, expected_model_version: str):
        if artifact.get("schema_version") != 2:
            raise ValueError("Unsupported portable boosting schema")
        if artifact.get("model_version") != expected_model_version:
            raise ValueError("Boosting policy belongs to another encoder")
        if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Boosting feature order mismatch")
        if artifact.get("topk") != 10 or artifact.get("preserve_first_n") != 3:
            raise ValueError("Unsupported boosting top-k or preserved prefix")
        if not artifact.get("trees") or not np.isfinite(float(artifact["baseline_logit"])):
            raise ValueError("Incomplete boosting ensemble")
        self.model_version = expected_model_version
        self.cosine_threshold = float(artifact["cosine_threshold"])
        self.booster_threshold = float(artifact["booster_threshold"])
        self.baseline_logit = float(artifact["baseline_logit"])
        self.trees = artifact["trees"]
        if not (0 < self.booster_threshold < 1) or not (-1 <= self.cosine_threshold <= 1):
            raise ValueError("Invalid boosting thresholds")
        for tree in self.trees:
            if not tree:
                raise ValueError("Empty boosting tree")
            for node in tree:
                if len(node) != 6 or node[0] not in (0, 1):
                    raise ValueError("Invalid boosting node")
                if node[0] == 0 and (not 0 <= node[1] < len(FEATURE_NAMES)
                                     or not 0 <= node[3] < len(tree)
                                     or not 0 <= node[4] < len(tree)):
                    raise ValueError("Invalid boosting split")
                if not np.isfinite(node[5]) or (node[0] == 0 and not np.isfinite(node[2])):
                    raise ValueError("Non-finite boosting node")

    @classmethod
    def load(cls, path: str | Path, expected_model_version: str):
        return cls(json.loads(Path(path).read_text()), expected_model_version)

    def scores(self, cosine_scores: np.ndarray) -> np.ndarray:
        features = _features(cosine_scores)
        flat = features.reshape(-1, len(FEATURE_NAMES))
        logits = np.full(len(flat), self.baseline_logit, dtype=np.float64)
        for tree in self.trees:
            logits += _tree_values(tree, flat)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        result = probability.reshape(features.shape[:2])
        if np.any(result[:, 1:] > result[:, :-1] + 1e-10):
            raise ValueError("Boosting policy changed cosine candidate order")
        return result

    def accepted(self, cosine_scores: np.ndarray) -> np.ndarray:
        cosine_scores = np.asarray(cosine_scores, dtype=np.float64)
        booster_scores = self.scores(cosine_scores)
        ranks = np.arange(10)[None, :]
        accepted = (cosine_scores >= self.cosine_threshold) & (
            (ranks < 3) | (booster_scores >= self.booster_threshold)
        )
        if np.any(accepted[:, 1:] & ~accepted[:, :-1]):
            raise ValueError("Accepted candidates must be a cosine prefix")
        return accepted
