"""Bounded million-vector FAISS demonstration, not the serving index.

Real organizer embeddings seed the evaluation; deterministic synthetic
unit vectors are load-only distractors. This script never changes model
inference, refusal policy, or the user gallery.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import faiss
import numpy as np


def unit_random(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    values = rng.standard_normal((count, dimension), dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all():
        raise FloatingPointError("Synthetic load contains non-finite vectors")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery-features", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--synthetic-count", type=int, default=1_000_000)
    parser.add_argument("--batch", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--nlist", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.synthetic_count >= 1_000_000 and args.batch > 0 and 1 <= args.nprobe <= args.nlist):
        raise ValueError("Expected at least one million vectors and valid IVF parameters")
    with np.load(args.gallery_features, allow_pickle=False) as payload:
        gallery = payload["embeddings"].astype(np.float32)
    with np.load(args.query_features, allow_pickle=False) as payload:
        queries = payload["embeddings"][:args.queries].astype(np.float32)
    if gallery.shape[1] != 1024 or queries.shape[1] != 1024 or len(queries) != args.queries:
        raise ValueError("Need real 1024D selected-model embeddings")
    faiss.omp_set_num_threads(4)
    rng = np.random.default_rng(20260927)
    train = unit_random(rng, 10_000, 1024)
    quantizer = faiss.IndexFlatIP(1024)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, 1024, args.nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    started = time.perf_counter()
    index.train(np.ascontiguousarray(np.concatenate([train, gallery])))
    train_seconds = time.perf_counter() - started
    index.nprobe = args.nprobe

    # Exact streaming oracle: retain only top-10 scores/indices for each query.
    best_scores = np.full((len(queries), 10), -np.inf, dtype=np.float32)
    best_ids = np.full((len(queries), 10), -1, dtype=np.int64)

    def update_exact(block: np.ndarray, offset: int) -> float:
        start = time.perf_counter()
        # Accelerate on macOS can raise FP warnings for finite dot products.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = queries @ block.T
        if not np.isfinite(scores).all():
            raise FloatingPointError("Exact-search oracle has non-finite scores")
        for row in range(len(queries)):
            local = np.argpartition(scores[row], -10)[-10:]
            merged_scores = np.concatenate([best_scores[row], scores[row, local]])
            merged_ids = np.concatenate([best_ids[row], local.astype(np.int64) + offset])
            keep = np.argsort(-merged_scores, kind="stable")[:10]
            best_scores[row] = merged_scores[keep]
            best_ids[row] = merged_ids[keep]
        return time.perf_counter() - start

    exact_dot_seconds = update_exact(gallery, 0)
    started = time.perf_counter()
    index.add(np.ascontiguousarray(gallery))
    add_seconds = time.perf_counter() - started
    for offset in range(0, args.synthetic_count, args.batch):
        block = unit_random(rng, min(args.batch, args.synthetic_count - offset), 1024)
        exact_dot_seconds += update_exact(block, len(gallery) + offset)
        started = time.perf_counter()
        index.add(block)
        add_seconds += time.perf_counter() - started
    if index.ntotal != len(gallery) + args.synthetic_count:
        raise AssertionError("Index cardinality mismatch")
    search_times = []
    for _ in range(5):
        start = time.perf_counter()
        _, approximate_ids = index.search(np.ascontiguousarray(queries), 10)
        search_times.append(time.perf_counter() - start)
    recall = float(np.mean([len(set(approximate_ids[i]) & set(best_ids[i])) / 10
                            for i in range(len(queries))]))
    result = {
        "status": "completed", "index": "FAISS IVF-SQ8 inner product", "dimension": 1024,
        "real_gallery": len(gallery), "synthetic_distractors": args.synthetic_count,
        "total_vectors": index.ntotal, "query_count": len(queries),
        "nlist": args.nlist, "nprobe": args.nprobe, "training_seconds": train_seconds,
        "addition_seconds": add_seconds, "exact_streamed_dot_seconds": exact_dot_seconds,
        "ann_search_batch_seconds_median": float(np.median(search_times)),
        "ann_search_per_query_ms_median": float(np.median(search_times)) * 1000 / len(queries),
        "recall_at_10_vs_exact": recall,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
        "scope": "Mac synthetic-load tie-break demo; not production index or organizer speed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
