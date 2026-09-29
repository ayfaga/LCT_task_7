"""Small, weight-free contract test for the hybrid submission exporter."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ml.export_test_hybrid import export
from app.ml.boosting import HybridBoostingPolicy
from app.ml.runtime import MLRuntime


def _csv(path: Path, ids: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id"])
        writer.writerows((image_id,) for image_id in ids)


@pytest.mark.parametrize("batch_ranking", ["exact_cosine", "transductive_aqe_k5_alpha025"])
def test_exact_top10_and_hybrid_rejection_roundtrip(tmp_path, batch_ranking):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    policy = {
        "schema_version": 2,
        "model_version": "fixture-e2",
        "feature_names": ["candidate_cosine", "top1_cosine", "top10_mean",
                          "top10_std", "candidate_minus_top1"],
        "topk": 10,
        "preserve_first_n": 3,
        "cosine_threshold": .394,
        "booster_threshold": .112,
        "baseline_logit": -3.0,
        "trees": [[[1, 0, 0.0, 0, 0, 0.0]]],
    }
    policy_path = artifacts / "boosting_policy.json"
    policy_path.write_text(json.dumps(policy))
    (artifacts / "model_manifest.json").write_text(json.dumps({
        "bundle_version": "fixture-bundle", "model_version": "fixture-e2", "model_sha256": "fixture-sha",
        "batch_ranking": batch_ranking,
    }))
    (artifacts / "calibration_manifest.json").write_text(json.dumps({
        "model_version": "fixture-e2", "policy_kind": "hybrid_hist_gradient_boosting",
        "policy_file": "boosting_policy.json",
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "cosine_threshold": .394, "booster_threshold": .112,
    }))
    query_ids = ["q0", "q1", "q-absent"]
    gallery_ids = [f"g{index}" for index in range(10)]
    query_csv, gallery_csv = tmp_path / "query.csv", tmp_path / "gallery.csv"
    _csv(query_csv, query_ids)
    _csv(gallery_csv, gallery_ids)
    query_vectors = np.eye(1024, dtype=np.float32)[[0, 1, 10]]
    gallery_vectors = np.eye(1024, dtype=np.float32)[:10]
    query_features, gallery_features = tmp_path / "query.npz", tmp_path / "gallery.npz"
    for path, ids, vectors in (
            (query_features, query_ids, query_vectors),
            (gallery_features, gallery_ids, gallery_vectors)):
        np.savez_compressed(path, gallery_ids=np.asarray(ids), embeddings=vectors,
                            model_version=np.asarray("fixture-e2"),
                            model_sha256=np.asarray("fixture-sha"))
    result = export(artifacts, query_csv, gallery_csv,
                    query_features, gallery_features, tmp_path / "output")
    assert result["validation"]["embedding_shape"] == [13, 1024]
    assert result["validation"]["candidates_rows"] == 3
    assert result["validation"]["empty_answers"] == 1
    assert result["ranking_algorithm"] == batch_ranking
    saved = np.load(tmp_path / "output" / "embeddings.npy", allow_pickle=False)
    if batch_ranking == "transductive_aqe_k5_alpha025":
        assert result["embeddings_semantics"] == "transductive_aqe_expanded"
        assert not np.array_equal(saved[:3], query_vectors)
    with (tmp_path / "output" / "submission.csv").open(newline="") as stream:
        submission = list(csv.reader(stream))[1:]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = saved[:3] @ saved[3:].T
    for index, row in enumerate(submission):
        expected = np.argsort(-scores[index], kind="stable")[:10]
        assert row == [query_ids[index], *np.asarray(gallery_ids)[expected]]
    runtime = object.__new__(MLRuntime)
    runtime.model_version = "fixture-e2"
    runtime.batch_ranking = batch_ranking
    runtime.gallery_ids = np.asarray(gallery_ids)
    runtime.gallery_embeddings = gallery_vectors
    runtime.policy = HybridBoostingPolicy.load(policy_path, "fixture-e2")
    api_rows = runtime.search_batch(query_vectors, topk=10)
    with (tmp_path / "output" / "candidates.csv").open(newline="") as stream:
        candidate_rows = list(csv.DictReader(stream))
    for index, api_row in enumerate(api_rows):
        assert [item["gallery_id"] for item in api_row["ranked"]] == submission[index][1:]
        expected_accepted = [row["gallery_id"] for row in candidate_rows
                             if row["query_id"] == query_ids[index] and row["gallery_id"]]
        assert [item["gallery_id"] for item in api_row["accepted"]] == expected_accepted
    with (tmp_path / "output" / "candidates.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["gallery_id"] == "g0"
    assert rows[1]["gallery_id"] == "g1"
    assert rows[2] == {"query_id": "q-absent", "gallery_id": "", "confidence": ""}
