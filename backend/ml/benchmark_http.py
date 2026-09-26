"""Repeatable local HTTP check; reports observed latency, not organizer speed."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path

import httpx


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--bbox", type=int, nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--expected-top1", required=True)
    parser.add_argument("--sequential", type=int, default=10)
    parser.add_argument("--concurrent", type=int, default=12)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.image.read_bytes()
    x, y, w, h = args.bbox
    form = {"x": x, "y": y, "w": w, "h": h, "topk": 10}
    files = {"image": (args.image.name, payload, "image/png")}
    url = args.url.rstrip("/")
    times = {"warmup": [], "sequential": [], "concurrent": []}
    states = []

    with httpx.Client(timeout=120.0) as client:
        ready = client.get(f"{url}/ready")
        ready.raise_for_status()
        openapi = client.get(f"{url}/openapi.json")
        openapi.raise_for_status()
        paths = openapi.json()["paths"]
        if not all(route in paths for route in ("/api/identify", "/api/infer", "/v1/search", "/v1/embeddings")):
            raise AssertionError("Required API paths are missing from OpenAPI")

        def once() -> float:
            started = time.perf_counter()
            response = client.post(f"{url}/api/identify", data=form, files=files)
            duration = time.perf_counter() - started
            response.raise_for_status()
            body = response.json()
            ranked = body["ranked"]
            accepted = body["accepted"]
            if (len(ranked) != 10 or ranked[0]["gallery_id"] != args.expected_top1
                    or body["status"] != "matched" or body["candidates"] != accepted
                    or any(a not in ranked for a in accepted)
                    or any(abs(item["confidence"] - item["similarity"]) > 1e-7 for item in ranked)
                    or any(ranked[i]["similarity"] < ranked[i + 1]["similarity"] - 1e-6
                           for i in range(9))):
                raise AssertionError("Inconsistent model response")
            states.append((body["model_version"], tuple(item["gallery_id"] for item in ranked),
                           tuple(item["gallery_id"] for item in accepted)))
            return duration

        for _ in range(2):
            times["warmup"].append(once())
        for _ in range(args.sequential):
            times["sequential"].append(once())
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            times["concurrent"] = list(pool.map(lambda _: once(), range(args.concurrent)))
        concurrent_wall = time.perf_counter() - started
        embedding = client.post(f"{url}/v1/embeddings", data={k: v for k, v in form.items() if k != "topk"}, files=files)
        embedding.raise_for_status()
        if embedding.json()["dimension"] != 1024 or len(embedding.json()["embedding"]) != 1024:
            raise AssertionError("Embedding response is invalid")
        invalid = client.post(f"{url}/api/identify", data={**form, "w": 0}, files=files)
        if invalid.status_code != 422:
            raise AssertionError(f"Invalid BBox returned {invalid.status_code}, not 422")

    if len(set(states)) != 1:
        raise AssertionError("Identical image yielded inconsistent responses")
    summary = {
        "status": "passed", "model_version": ready.json()["model_version"],
        "gallery_size": ready.json()["gallery_size"], "host_test_scope": "local Docker Desktop, CPU, 750 gallery",
        "requests": {"warmup": 2, "sequential": args.sequential, "concurrent": args.concurrent,
                     "workers": args.workers, "failures": 0},
        "sequential_seconds": {"median": statistics.median(times["sequential"]),
                               "p95": percentile(times["sequential"], .95),
                               "max": max(times["sequential"])},
        "concurrent_seconds": {"median": statistics.median(times["concurrent"]),
                               "p95": percentile(times["concurrent"], .95),
                               "max": max(times["concurrent"]),
                               "wall": concurrent_wall,
                               "completed_requests_per_second": args.concurrent / concurrent_wall},
        "openapi_paths_checked": ["/api/identify", "/api/infer", "/v1/search", "/v1/embeddings"],
        "invalid_bbox_http": 422,
        "note": "Repeated same-image API load check. Not organizer hardware, batch FPS, or hidden quality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
