"""Stream organizer JPEG+BBox archive into versioned ReID gallery and export.

Never treats a whole original frame as a vehicle. Reads only test query/gallery
members from the ZIP; train labels and holdout are not used.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.encoder import E2Encoder, file_sha256  # noqa: E402
from app.ml.preprocessing import VERSION, crop_bbox  # noqa: E402

from ml.export_test_hybrid import export  # noqa: E402


IMAGE_ID = re.compile(r"[0-9a-f]{32}\Z")
EXPECTED_COLUMNS = {"image_id", "x", "y", "w", "h"}
MAX_JPEG_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 50_000_000
MAX_CSV_BYTES = 8 * 1024 * 1024


def validate_preprocessing(manifest: dict) -> None:
    implementation = Path(__file__).resolve().parents[1] / "app/ml/preprocessing.py"
    if (manifest.get("preprocessing_version") != VERSION
            or manifest.get("preprocessing_sha256") != file_sha256(implementation)
            or manifest.get("architecture") != "dinov2_vitl14"
            or manifest.get("input_size") != 336
            or manifest.get("embedding_dimension") != 1024):
        raise ValueError("Organizer importer requires the unchanged joint L336 preprocessing contract")


def read_records(archive: ZipFile, csv_name: str) -> tuple[bytes, list[tuple[str, tuple[int, int, int, int], str]]]:
    try:
        csv_info = archive.getinfo(csv_name)
    except KeyError as error:
        raise ValueError(f"Missing organizer CSV: {csv_name}") from error
    if csv_info.flag_bits & 1 or csv_info.file_size > MAX_CSV_BYTES:
        raise ValueError(f"{csv_name}: encrypted or oversized CSV")
    raw = archive.read(csv_info)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if not reader.fieldnames or not EXPECTED_COLUMNS.issubset(reader.fieldnames):
        raise ValueError(f"{csv_name}: expected image_id,x,y,w,h")
    records = []
    ids = set()
    for line, row in enumerate(reader, 2):
        image_id = (row["image_id"] or "").strip()
        if not IMAGE_ID.fullmatch(image_id) or image_id in ids:
            raise ValueError(f"{csv_name}:{line}: invalid or duplicate image_id")
        try:
            bbox = tuple(int(row[key]) for key in ("x", "y", "w", "h"))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{csv_name}:{line}: invalid integer BBox") from error
        x, y, w, h = bbox
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError(f"{csv_name}:{line}: empty or negative BBox")
        member = f"images/{image_id}.jpg"
        try:
            info = archive.getinfo(member)
        except KeyError as error:
            raise ValueError(f"{csv_name}:{line}: missing JPEG {member}") from error
        if info.flag_bits & 1 or info.file_size > MAX_JPEG_BYTES:
            raise ValueError(f"{csv_name}:{line}: encrypted or oversized JPEG")
        records.append((image_id, bbox, member))
        ids.add(image_id)
    if not records:
        raise ValueError(f"{csv_name}: no records")
    return raw, records


def inspect_archive(path: Path) -> tuple[dict, dict]:
    with ZipFile(path) as archive:
        names = [info.filename for info in archive.infolist()]
        if len(names) != len(set(names)):
            raise ValueError("Archive has duplicate member names")
        gallery_csv, gallery = read_records(archive, "test_gallery.csv")
        query_csv, query = read_records(archive, "test_query.csv")
    if set(row[0] for row in gallery) & set(row[0] for row in query):
        raise ValueError("Query and gallery image IDs overlap")
    return {"gallery": (gallery_csv, gallery), "query": (query_csv, query)}, {
        "gallery": len(gallery), "query": len(query), "bbox_source": "organizer test CSV",
        "uses_train_or_holdout": False,
    }


def encode_split(archive: ZipFile, records: list[tuple[str, tuple[int, int, int, int], str]],
                 encoder: E2Encoder, manifest: dict, output: Path, batch_size: int, split: str) -> None:
    vectors = []
    for start in range(0, len(records), batch_size):
        crops = []
        for image_id, bbox, member in records[start:start + batch_size]:
            with archive.open(member) as stream:
                data = stream.read(MAX_JPEG_BYTES + 1)
            if len(data) > MAX_JPEG_BYTES:
                raise ValueError(f"{member}: JPEG too large")
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG" or image.width * image.height > MAX_PIXELS:
                    raise ValueError(f"{member}: invalid JPEG format/dimensions")
                crops.append(crop_bbox(image, bbox))
        vectors.append(encoder.embed_crops(crops, batch_size))
        if (start // batch_size + 1) % 20 == 0 or start + len(crops) == len(records):
            print(json.dumps({"split": split, "embedded": start + len(crops), "total": len(records)}), flush=True)
    embeddings = np.concatenate(vectors)
    if embeddings.shape != (len(records), 1024) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid extracted embeddings")
    np.savez_compressed(output, gallery_ids=np.asarray([row[0] for row in records]),
                        embeddings=embeddings.astype(np.float32),
                        model_version=np.asarray(manifest["model_version"]),
                        model_sha256=np.asarray(manifest["model_sha256"]))


def build(archive_path: Path, artifact_dir: Path, output: Path, device: str = "cpu",
          batch_size: int = 4, limit_per_split: int = 0) -> dict:
    if batch_size < 1 or limit_per_split < 0:
        raise ValueError("Invalid batch size or limit")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    splits, inspection = inspect_archive(archive_path)
    if limit_per_split:
        splits = {name: (raw, records[:limit_per_split]) for name, (raw, records) in splits.items()}
    if len(splits["gallery"][1]) < 10 or len(splits["query"][1]) < 1:
        raise ValueError("Need at least 10 gallery and one query record")
    manifest = json.loads((artifact_dir / "model_manifest.json").read_text())
    validate_preprocessing(manifest)
    encoder = E2Encoder(artifact_dir / manifest["model_file"], manifest["model_sha256"], device)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ZipFile(archive_path) as archive:
        for name in ("gallery", "query"):
            raw, records = splits[name]
            # A smoke run writes only the selected records, never the original full CSV.
            if limit_per_split:
                lines = ["image_id,x,y,w,h"] + [f"{image_id},{','.join(map(str, bbox))}"
                                                   for image_id, bbox, _ in records]
                (output / f"test_{name}.csv").write_text("\n".join(lines) + "\n")
            else:
                (output / f"test_{name}.csv").write_bytes(raw)
            encode_split(archive, records, encoder, manifest, output / f"test_{name}.npz", batch_size, name)
    exported = export(artifact_dir, output / "test_query.csv", output / "test_gallery.csv",
                      output / "test_query.npz", output / "test_gallery.npz", output / "submission")
    result = {
        "status": "completed", "scope": "SMOKE ONLY; not a full submission" if limit_per_split else "full organizer test archive",
        "archive_sha256": file_sha256(archive_path), "model_version": manifest["model_version"],
        "model_sha256": manifest["model_sha256"], "archive_records": inspection,
        "processed": {name: len(records) for name, (_, records) in splits.items()},
        "gallery_npz": str(output / "test_gallery.npz"),
        "query_npz": str(output / "test_query.npz"),
        "submission_validation": exported["validation"],
        "elapsed_seconds": time.monotonic() - started,
        "default_gallery_not_changed": True,
    }
    (output / "import_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit-per-split", type=int, default=0, help="smoke test only")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.inspect_only:
        _, report = inspect_archive(args.archive)
        print(json.dumps(report))
        return
    import torch
    torch.set_num_threads(args.threads)
    build(args.archive, args.artifact_dir, args.output, args.device,
          args.batch_size, args.limit_per_split)


if __name__ == "__main__":
    main()
