"""Exercise user-facing error responses against the running two-service API."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import httpx
from PIL import Image


def image_bytes(format: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color="red").save(buffer, format=format)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    form = {"x": 0, "y": 0, "w": 32, "h": 24, "topk": 10}
    png = image_bytes("PNG")
    cases = (
        ("jpeg", {**form}, image_bytes("JPEG"), 200),
        ("negative_bbox", {**form, "x": -1}, png, 422),
        ("bbox_outside_image", {**form, "w": 33}, png, 422),
        ("zero_width", {**form, "w": 0}, png, 422),
        ("topk_zero", {**form, "topk": 0}, png, 422),
        ("topk_101", {**form, "topk": 101}, png, 422),
        ("corrupt_image", {**form}, b"garbage", 422),
        ("empty_image", {**form}, b"", 422),
        ("unsupported_bmp", {**form}, image_bytes("BMP"), 422),
        ("too_large", {**form}, b"x" * (20 * 1024 * 1024 + 1), 413),
    )
    observations = []
    with httpx.Client(timeout=120.0) as client:
        for name, data, payload, expected in cases:
            response = client.post(
                f"{args.url.rstrip('/')}/api/identify", data=data,
                files={"image": ("query.img", payload, "application/octet-stream")},
            )
            observations.append({"case": name, "status": response.status_code, "expected": expected})
            if response.status_code != expected:
                raise AssertionError({"case": name, "expected": expected,
                                      "actual": response.status_code, "body": response.text[:500]})
    result = {"status": "passed", "cases": len(cases), "observations": observations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
