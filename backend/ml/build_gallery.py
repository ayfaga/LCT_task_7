"""Create a version-bound exact-cosine gallery from image records.

Input JSONL: {"gallery_id":"...","image_path":"...","bbox":[x,y,w,h]}
For an existing organizer crop, omit bbox to embed the full crop unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.encoder import E2Encoder  # noqa: E402
from app.ml.preprocessing import crop_bbox  # noqa: E402


def _records(records_file: Path | None, organizer_gallery_csv: Path | None, crop_dir: Path | None):
    if organizer_gallery_csv is not None:
        if crop_dir is None:
            raise ValueError("--crop-dir is required with --organizer-gallery-csv")
        with organizer_gallery_csv.open(newline="") as stream:
            for row in csv.DictReader(stream):
                image_id = row["image_id"]
                yield {"gallery_id": image_id, "image_path": str(crop_dir / f"{image_id}.png")}
    else:
        if records_file is None:
            raise ValueError("Provide --records or --organizer-gallery-csv")
        with records_file.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def build(artifact_dir: Path, records_file: Path | None, organizer_gallery_csv: Path | None,
          crop_dir: Path | None, output_file: Path, device: str, batch_size: int) -> None:
    manifest = json.loads((artifact_dir / "model_manifest.json").read_text())
    encoder = E2Encoder(artifact_dir / manifest["model_file"], manifest["model_sha256"], device)
    ids: list[str] = []
    seen_ids: set[str] = set()
    embeddings: list[np.ndarray] = []
    pending: list[Image.Image] = []
    for line_number, record in enumerate(_records(records_file, organizer_gallery_csv, crop_dir), 1):
        gallery_id = str(record["gallery_id"])
        if not gallery_id or gallery_id in seen_ids:
            raise ValueError(f"Duplicate/empty gallery_id on line {line_number}")
        with Image.open(record["image_path"]) as image:
            crop = image.convert("RGB")
            if record.get("bbox") is not None:
                crop = crop_bbox(crop, tuple(record["bbox"]))
            else:
                crop.thumbnail((448, 448), Image.Resampling.LANCZOS)
        pending.append(crop)
        ids.append(gallery_id)
        seen_ids.add(gallery_id)
        if len(pending) >= batch_size:
            embeddings.extend(encoder.embed_crops(pending, batch_size=batch_size))
            pending.clear()
            if len(ids) % 100 < batch_size:
                print(json.dumps({"embedded": len(ids)}), flush=True)
    if pending:
        embeddings.extend(encoder.embed_crops(pending, batch_size=batch_size))
    if len(ids) < 10:
        raise ValueError("Boosting policy requires at least 10 gallery images")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, gallery_ids=np.asarray(ids),
                        embeddings=np.stack(embeddings).astype(np.float32),
                        model_version=np.asarray(manifest["model_version"]),
                        model_sha256=np.asarray(manifest["model_sha256"]))
    print(json.dumps({"gallery_images": len(ids), "dimension": 1024, "output": str(output_file)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--records", type=Path)
    sources.add_argument("--organizer-gallery-csv", type=Path)
    parser.add_argument("--crop-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    build(args.artifact_dir, args.records, args.organizer_gallery_csv,
          args.crop_dir, args.output, args.device, args.batch_size)
