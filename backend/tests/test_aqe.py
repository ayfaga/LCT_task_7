"""Production AQE must reproduce the frozen research formula."""

import numpy as np
import pytest
from types import SimpleNamespace

from app.ml.aqe import expand_transductive
from app.ml.runtime import MLRuntime


def _reference(query, gallery, k=5, alpha=.25):
    cohort = np.concatenate((query, gallery)).astype(np.float32)
    cohort /= np.linalg.norm(cohort, axis=1, keepdims=True)
    cosine = cohort @ cohort.T
    np.fill_diagonal(cosine, -np.inf)
    order = np.argsort(-cosine, axis=1, kind="stable")[:, :k]
    expanded = (1 - alpha) * cohort + alpha * cohort[order].mean(axis=1)
    expanded /= np.linalg.norm(expanded, axis=1, keepdims=True)
    return expanded[:len(query)], expanded[len(query):]


def test_aqe_matches_research_formula_across_blocks():
    rng = np.random.default_rng(20260929)
    query = rng.normal(size=(8, 16)).astype(np.float32)
    gallery = rng.normal(size=(11, 16)).astype(np.float32)
    expected_query, expected_gallery = _reference(query, gallery)
    actual_query, actual_gallery = expand_transductive(query, gallery, working_memory_mib=1)
    np.testing.assert_allclose(actual_query, expected_query, atol=2e-7)
    np.testing.assert_allclose(actual_gallery, expected_gallery, atol=2e-7)
    np.testing.assert_allclose(actual_query @ actual_gallery.T,
                               expected_query @ expected_gallery.T, atol=2e-7)


def test_aqe_rejects_bad_vectors_and_undersized_cohort():
    with pytest.raises(ValueError, match="nonempty"):
        expand_transductive(np.empty((0, 16)), np.ones((10, 16)))
    with pytest.raises(ValueError, match="more cohort"):
        expand_transductive(np.ones((1, 16)), np.ones((4, 16)))
    with pytest.raises(ValueError, match="finite"):
        expand_transductive(np.full((1, 16), np.nan), np.ones((10, 16)))


def test_batch_runtime_ranks_by_aqe_but_applies_refusal_to_raw_cosine():
    rng = np.random.default_rng(29)
    gallery = rng.normal(size=(10, 1024)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    queries = np.stack((gallery[0] + gallery[1], gallery[1] + gallery[2]))
    queries = queries.astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    runtime = object.__new__(MLRuntime)
    runtime.model_version = "fixture-strong"
    runtime.batch_ranking = "transductive_aqe_k5_alpha025"
    runtime.gallery_ids = np.array([f"g{i}" for i in range(10)])
    runtime.gallery_embeddings = gallery
    runtime.policy = SimpleNamespace(
        cosine_threshold=.0,
        accepted=lambda scores: scores >= 0,
    )

    expanded_query, expanded_gallery = expand_transductive(queries, gallery)
    response = runtime.search_batch(queries)
    assert len(response) == 2
    assert all(row["query_cohort_size"] == 2 for row in response)
    for index, row in enumerate(response):
        expected = np.argsort(-(expanded_gallery @ expanded_query[index]), kind="stable")
        assert [item["gallery_id"] for item in row["ranked"]] == [f"g{i}" for i in expected]
        assert row["ranking_algorithm"] == "transductive_aqe_k5_alpha025"
        np.testing.assert_allclose(
            [item["similarity"] for item in row["ranked"]],
            (gallery @ queries[index])[expected], atol=1e-6,
        )
        assert [item["gallery_id"] for item in row["accepted"]] == [
            item["gallery_id"] for item in row["ranked"] if item["similarity"] >= 0
        ]


def test_one_query_in_batch_keeps_online_cosine_semantics():
    rng = np.random.default_rng(29)
    gallery = rng.normal(size=(10, 1024)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    runtime = object.__new__(MLRuntime)
    runtime.model_version = "fixture-strong"
    runtime.batch_ranking = "transductive_aqe_k5_alpha025"
    runtime.gallery_ids = np.array([f"g{i}" for i in range(10)])
    runtime.gallery_embeddings = gallery
    runtime.policy = SimpleNamespace(cosine_threshold=.0,
                                     accepted=lambda scores: scores >= 0)
    response = runtime.search_batch(gallery[:1])[0]
    assert response["ranking_algorithm"] == "exact_cosine"
    assert response["ranked"][0]["gallery_id"] == "g0"
