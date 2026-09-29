"""Exact one-hop transductive AQE for a complete query cohort and gallery.

The k-nearest-neighbor step follows the frozen research implementation. Its
blockwise implementation bounds intermediate memory; CPU work is still
quadratic in the number of gallery and query embeddings.
"""

from __future__ import annotations

import numpy as np


def _unit_vectors(values: np.ndarray, dimension: int) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != dimension or not np.isfinite(vectors).all():
        raise ValueError("AQE requires finite [images, embedding_dimension] vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("AQE cannot use zero embeddings")
    return vectors / norms


def expand_transductive(
    queries: np.ndarray,
    gallery: np.ndarray,
    *,
    k: int = 5,
    alpha: float = 0.25,
    working_memory_mib: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand all queries and gallery images against their shared cohort.

    Unlike online/gallery-only expansion, adding or removing a query can
    change the entire gallery representation. No identity/camera labels are
    read here. Inputs and results are L2-normalized float32 embeddings.
    """
    gallery = np.asarray(gallery, dtype=np.float32)
    if gallery.ndim != 2:
        raise ValueError("Gallery embeddings must be a matrix")
    queries = _unit_vectors(queries, gallery.shape[1])
    gallery = _unit_vectors(gallery, gallery.shape[1])
    if len(queries) == 0 or len(gallery) == 0:
        raise ValueError("AQE requires nonempty query and gallery sets")
    if k < 1 or len(queries) + len(gallery) <= k:
        raise ValueError("AQE requires more cohort images than neighbors")
    if not 0 <= alpha <= 1 or working_memory_mib < 1:
        raise ValueError("Invalid AQE alpha or working memory budget")

    cohort = np.concatenate((queries, gallery), axis=0)
    expanded = np.empty_like(cohort)
    # Similarities (float32), stable argsort indices (int64), and temporary
    # ordering buffers share this bounded working space. This bounds memory,
    # not the O(N^2) compute cost of exact neighbor discovery.
    block_size = max(1, min(256, working_memory_mib * 1024**2 // (16 * len(cohort))))
    for start in range(0, len(cohort), block_size):
        stop = min(start + block_size, len(cohort))
        similarities = cohort[start:stop] @ cohort.T
        similarities[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        neighbors = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        mixed = (1 - alpha) * cohort[start:stop] + alpha * cohort[neighbors].mean(axis=1)
        expanded[start:stop] = _unit_vectors(mixed, cohort.shape[1])
    return expanded[:len(queries)], expanded[len(queries):]
