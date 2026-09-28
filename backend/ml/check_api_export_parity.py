"""Compare live API predictions against the frozen unlabeled test export."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from zipfile import ZipFile

import httpx
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    parser.add_argument("--export", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--crop-dir", type=Path)
    source.add_argument("--organizer-archive", type=Path,
                        help="Use original test JPEG+xywh, not cached PNG crops")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with (args.export / "submission.csv").open(newline="") as stream:
        submission = list(csv.DictReader(stream))
    accepted = defaultdict(list)
    refused = []
    with (args.export / "candidates.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["gallery_id"]:
                accepted[row["query_id"]].append(row["gallery_id"])
            else:
                refused.append(row["query_id"])
    by_query = {row["query_id"]: row for row in submission}
    selected = [submission[i]["query_id"] for i in (0, 100, 500, 1000)] + refused[:2]
    observations = []
    archive_context = ZipFile(args.organizer_archive) if args.organizer_archive else nullcontext(None)
    with httpx.Client(timeout=120.0) as client, archive_context as archive:
        bboxes = {}
        if archive is not None:
            table = csv.DictReader(io.StringIO(archive.read("test_query.csv").decode("utf-8-sig")))
            if not table.fieldnames or not {"image_id", "x", "y", "w", "h"}.issubset(table.fieldnames):
                raise ValueError("Organizer query CSV must provide image_id,x,y,w,h")
            bboxes = {row["image_id"]: {key: int(row[key]) for key in ("x", "y", "w", "h")}
                      for row in table}
        for query_id in selected:
            if archive is not None:
                filename = f"{query_id}.jpg"
                payload = archive.read(f"images/{filename}")
                bbox = bboxes[query_id]
                mime = "image/jpeg"
            else:
                path = args.crop_dir / f"{query_id}.png"
                with Image.open(path) as image:
                    width, height = image.size
                filename, payload, mime = path.name, path.read_bytes(), "image/png"
                bbox = {"x": 0, "y": 0, "w": width, "h": height}
            response = client.post(
                f"{args.url.rstrip('/')}/api/identify",
                data={**bbox, "topk": 10},
                files={"image": (filename, payload, mime)},
            )
            response.raise_for_status()
            result = response.json()
            expected_rank = [by_query[query_id][f"gallery_id_{i}"] for i in range(1, 11)]
            actual_rank = [item["gallery_id"] for item in result["ranked"]]
            expected_accepted = accepted[query_id]
            actual_accepted = [item["gallery_id"] for item in result["accepted"]]
            if actual_rank != expected_rank or actual_accepted != expected_accepted:
                raise AssertionError({"query_id": query_id, "rank_match": actual_rank == expected_rank,
                                      "accepted_match": actual_accepted == expected_accepted,
                                      "actual_rank": actual_rank, "expected_rank": expected_rank,
                                      "actual_accepted": actual_accepted,
                                      "expected_accepted": expected_accepted})
            if (result["status"] == "no_confident_match") != (query_id in refused):
                raise AssertionError("Refusal status differs from offline export")
            observations.append({"query_id": query_id, "top10_equal": True,
                                 "accepted_equal": True, "refused": query_id in refused})
    output = {"status": "passed", "compared_queries": len(selected),
              "input": "original organizer JPEG+xywh" if args.organizer_archive else "cached PNG crop",
              "ordinary_queries": sum(not row["refused"] for row in observations),
              "refusal_queries": sum(row["refused"] for row in observations),
              "observations": observations,
              "note": "API-vs-existing-unlabeled-export parity; not a labeled quality metric."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
