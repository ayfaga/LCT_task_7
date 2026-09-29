"""Stream organizer JPEG+BBox archive into versioned ReID gallery and export.

Never treats a whole original frame as a vehicle. Reads only test query/gallery
members from the ZIP; train labels and holdout are not used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import stat
import sys
import time
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Callable
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


@dataclass(frozen=True)
class ComponentInput:
    """Four independent fields, with repeatable image sources per split."""

    gallery_csv: Path
    gallery_images: tuple[Path, ...]
    query_csv: Path
    query_images: tuple[Path, ...]


class ComponentSource:
    """Expose mixed ZIPs, folders and JPEGs through the organizer reader API."""

    def __init__(self, spec: ComponentInput):
        self.spec = spec
        self._stack = ExitStack()
        self._members: dict[str, tuple[Path | ZipFile, object]] = {}

    def __enter__(self):
        try:
            for split in ("gallery", "query"):
                csv_path = getattr(self.spec, f"{split}_csv")
                if csv_path.is_symlink() or not csv_path.is_file():
                    raise ValueError(f"{split} CSV does not exist or is a symlink: {csv_path}")
                for source in getattr(self.spec, f"{split}_images"):
                    self._add_source(split, source)
            return self
        except Exception:
            self._stack.close()
            raise

    def __exit__(self, *_):
        return self._stack.__exit__(*_)

    def _add_image(self, split: str, name: str, owner: Path | ZipFile, entry: object) -> None:
        basename = PurePosixPath(name).name
        if basename.startswith(".") or Path(basename).suffix.lower() not in {".jpg", ".jpeg"}:
            return
        image_id = Path(basename).stem
        if not IMAGE_ID.fullmatch(image_id):
            return
        key = f"{split}/images/{image_id}.jpg"
        if key in self._members:
            raise ValueError(f"Duplicate image_id in selected sources: {image_id}")
        self._members[key] = (owner, entry)

    def _add_source(self, split: str, source: Path) -> None:
        if source.is_symlink() or not source.exists():
            raise ValueError(f"Image source does not exist or is a symlink: {source}")
        if source.is_dir():
            for image in source.rglob("*"):
                if image.is_symlink():
                    raise ValueError(f"Symlink in image folder: {image}")
                if image.is_file():
                    self._add_image(split, image.name, image, image)
            return
        if not source.is_file():
            raise ValueError(f"Unsupported image source: {source}")
        if source.suffix.lower() in {".jpg", ".jpeg"}:
            if not IMAGE_ID.fullmatch(source.stem):
                raise ValueError(f"JPEG filename must be its 32-hex image_id: {source.name}")
            self._add_image(split, source.name, source, source)
            return
        if source.suffix.lower() != ".zip":
            raise ValueError(f"Use a ZIP, image folder or JPEG: {source}")
        archive = self._stack.enter_context(ZipFile(source))
        names = [info.filename for info in archive.infolist()]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate ZIP member names: {source}")
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if (member.is_absolute() or ".." in member.parts or "\\" in info.filename
                    or mode == stat.S_IFLNK):
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            if not info.is_dir():
                self._add_image(split, info.filename, archive, info)

    def getinfo(self, name: str):
        if name in {"test_gallery.csv", "test_query.csv"}:
            split = "gallery" if name == "test_gallery.csv" else "query"
            path = getattr(self.spec, f"{split}_csv")
            return SimpleNamespace(filename=name, file_size=path.stat().st_size, flag_bits=0)
        try:
            owner, entry = self._members[name]
        except KeyError as error:
            raise KeyError(name) from error
        if isinstance(owner, ZipFile):
            return entry
        return SimpleNamespace(filename=name, file_size=owner.stat().st_size, flag_bits=0)

    def read(self, item) -> bytes:
        name = item.filename if hasattr(item, "filename") else item
        if name not in {"test_gallery.csv", "test_query.csv"}:
            raise KeyError(name)
        split = "gallery" if name == "test_gallery.csv" else "query"
        return getattr(self.spec, f"{split}_csv").read_bytes()

    def open(self, name: str):
        owner, entry = self._members[name]
        return owner.open(entry) if isinstance(owner, ZipFile) else owner.open("rb")


class DirectorySource:
    """Read the same organizer layout from a folder without copying it to ZIP."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def _path(self, name: str) -> Path:
        if name not in {"test_gallery.csv", "test_query.csv"} and not re.fullmatch(
                r"images/[0-9a-f]{32}\.jpg", name):
            raise KeyError(name)
        path = self.root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.root):
            raise KeyError(name)
        return path

    def getinfo(self, name: str):
        path = self._path(name)
        return SimpleNamespace(filename=name, file_size=path.stat().st_size, flag_bits=0)

    def read(self, item) -> bytes:
        return self._path(item.filename if hasattr(item, "filename") else item).read_bytes()

    def open(self, name: str):
        return self._path(name).open("rb")


def open_source(path: Path | ComponentInput):
    if isinstance(path, ComponentInput):
        return ComponentSource(path)
    if path.is_dir():
        return DirectorySource(path)
    if path.is_file():
        return ZipFile(path)
    raise FileNotFoundError(f"Organizer input not found: {path}")


def input_sha256(path: Path | ComponentInput, splits: dict) -> str:
    if isinstance(path, Path) and path.is_file():
        return file_sha256(path)
    digest = hashlib.sha256()
    with open_source(path) as source:
        for name in ("gallery", "query"):
            csv_name = f"test_{name}.csv"
            digest.update(csv_name.encode() + b"\0")
            digest.update(source.read(csv_name) if isinstance(path, ComponentInput)
                          else splits[name][0])
            for _, _, member in splits[name][1]:
                digest.update(member.encode() + b"\0")
                with source.open(member) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
    return digest.hexdigest()


def validate_preprocessing(manifest: dict) -> None:
    implementation = Path(__file__).resolve().parents[1] / "app/ml/preprocessing.py"
    if (manifest.get("preprocessing_version") != VERSION
            or manifest.get("preprocessing_sha256") != file_sha256(implementation)
            or manifest.get("architecture") != "dinov2_vitl14"
            or manifest.get("input_size") != 336
            or manifest.get("embedding_dimension") != 1024):
        raise ValueError("Organizer importer requires the unchanged joint L336 preprocessing contract")


def read_records(archive: ZipFile | DirectorySource | ComponentSource, csv_name: str,
                 allow_missing: bool = False, missing: list[str] | None = None
                 ) -> tuple[bytes, list[tuple[str, tuple[int, int, int, int], str]]]:
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
    matched_rows = []
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
        split = "gallery" if csv_name == "test_gallery.csv" else "query"
        member = (f"{split}/images/{image_id}.jpg" if isinstance(archive, ComponentSource)
                  else f"images/{image_id}.jpg")
        try:
            info = archive.getinfo(member)
        except KeyError as error:
            if not allow_missing:
                raise ValueError(f"{csv_name}:{line}: missing JPEG {member}") from error
            if missing is not None:
                missing.append(image_id)
            ids.add(image_id)
            continue
        if info.flag_bits & 1 or info.file_size > MAX_JPEG_BYTES:
            raise ValueError(f"{csv_name}:{line}: encrypted or oversized JPEG")
        records.append((image_id, bbox, member))
        matched_rows.append(row)
        ids.add(image_id)
    if not records:
        raise ValueError(f"{csv_name}: no records")
    if allow_missing and missing:
        filtered = io.StringIO()
        writer = csv.DictWriter(filtered, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(matched_rows)
        raw = filtered.getvalue().encode("utf-8")
    return raw, records


def inspect_archive(path: Path | ComponentInput, allow_partial: bool = False) -> tuple[dict, dict]:
    if allow_partial and not isinstance(path, ComponentInput):
        raise ValueError("Partial mode is only available for four-field component input")
    missing_gallery: list[str] = []
    missing_query: list[str] = []
    with open_source(path) as archive:
        if isinstance(archive, ZipFile):
            names = [info.filename for info in archive.infolist()]
            if len(names) != len(set(names)):
                raise ValueError("Archive has duplicate member names")
        gallery_csv, gallery = read_records(archive, "test_gallery.csv", allow_partial, missing_gallery)
        query_csv, query = read_records(archive, "test_query.csv", allow_partial, missing_query)
    if set(row[0] for row in gallery) & set(row[0] for row in query):
        raise ValueError("Query and gallery image IDs overlap")
    return {"gallery": (gallery_csv, gallery), "query": (query_csv, query)}, {
        "gallery": len(gallery), "query": len(query), "bbox_source": "organizer test CSV",
        "uses_train_or_holdout": False,
        "missing_gallery": len(missing_gallery), "missing_query": len(missing_query),
        "partial": bool(missing_gallery or missing_query),
    }


def encode_split(archive: ZipFile | DirectorySource, records: list[tuple[str, tuple[int, int, int, int], str]],
                 encoder: E2Encoder, manifest: dict, output: Path, batch_size: int, split: str,
                 progress: Callable[[str, int, int], None] | None = None,
                 inference_lock=None) -> None:
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
        with inference_lock if inference_lock is not None else nullcontext():
            vectors.append(encoder.embed_crops(crops, batch_size))
        if progress is not None:
            progress(split, start + len(crops), len(records))
        if (start // batch_size + 1) % 20 == 0 or start + len(crops) == len(records):
            print(json.dumps({"split": split, "embedded": start + len(crops), "total": len(records)}), flush=True)
    embeddings = np.concatenate(vectors)
    if embeddings.shape != (len(records), 1024) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid extracted embeddings")
    np.savez_compressed(output, gallery_ids=np.asarray([row[0] for row in records]),
                        embeddings=embeddings.astype(np.float32),
                        model_version=np.asarray(manifest["model_version"]),
                        model_sha256=np.asarray(manifest["model_sha256"]))


def build(archive_path: Path | ComponentInput, artifact_dir: Path, output: Path, device: str = "cpu",
          batch_size: int = 4, limit_per_split: int = 0,
          encoder: E2Encoder | None = None,
          progress: Callable[[str, int, int], None] | None = None,
          inference_lock=None, allow_partial: bool = False) -> dict:
    if batch_size < 1 or limit_per_split < 0:
        raise ValueError("Invalid batch size or limit")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    splits, inspection = inspect_archive(archive_path, allow_partial=allow_partial)
    if limit_per_split:
        splits = {name: (raw, records[:limit_per_split]) for name, (raw, records) in splits.items()}
    if len(splits["gallery"][1]) < 10 or len(splits["query"][1]) < 1:
        raise ValueError("Need at least 10 gallery and one query record")
    manifest = json.loads((artifact_dir / "model_manifest.json").read_text())
    validate_preprocessing(manifest)
    encoder = encoder or E2Encoder(artifact_dir / manifest["model_file"], manifest["model_sha256"], device)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with open_source(archive_path) as archive:
        for name in ("gallery", "query"):
            raw, records = splits[name]
            # A smoke run writes only the selected records, never the original full CSV.
            if limit_per_split:
                lines = ["image_id,x,y,w,h"] + [f"{image_id},{','.join(map(str, bbox))}"
                                                   for image_id, bbox, _ in records]
                (output / f"test_{name}.csv").write_text("\n".join(lines) + "\n")
            else:
                (output / f"test_{name}.csv").write_bytes(raw)
            encode_split(archive, records, encoder, manifest, output / f"test_{name}.npz",
                         batch_size, name, progress, inference_lock)
    exported = export(artifact_dir, output / "test_query.csv", output / "test_gallery.csv",
                      output / "test_query.npz", output / "test_gallery.npz", output / "submission")
    source_hash = input_sha256(archive_path, splits)
    scope = ("SMOKE ONLY; not a full submission" if limit_per_split
             else "PARTIAL TEST ONLY; not a full submission" if inspection["partial"]
             else "full organizer test archive")
    input_kind = ("components" if isinstance(archive_path, ComponentInput)
                  else "directory" if archive_path.is_dir() else "zip")
    exported["input_scope"] = scope
    exported["input_kind"] = input_kind
    exported["input_sha256"] = source_hash
    (output / "submission" / "manifest.json").write_text(json.dumps(exported, indent=2) + "\n")
    result = {
        "status": "completed", "scope": scope,
        "archive_sha256": source_hash if isinstance(archive_path, Path) and archive_path.is_file() else None,
        "input_sha256": source_hash,
        "input_kind": input_kind,
        "model_version": manifest["model_version"],
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
    parser.add_argument("--archive", "--input", dest="archive", type=Path)
    parser.add_argument("--gallery-csv", type=Path)
    parser.add_argument("--query-csv", type=Path)
    parser.add_argument("--gallery-images", type=Path, action="append", default=[],
                        help="Repeat for each gallery ZIP, folder or JPEG")
    parser.add_argument("--query-images", type=Path, action="append", default=[],
                        help="Repeat for each query ZIP, folder or JPEG")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Explicit subset test only; never a full submission")
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
    component_fields = (args.gallery_csv, args.query_csv, args.gallery_images, args.query_images)
    if args.archive and (any(component_fields) or args.allow_partial):
        parser.error("Choose --input OR the four separate component fields")
    if not args.archive and not all(component_fields):
        parser.error("Provide --input OR all four fields: --gallery-csv, --gallery-images, "
                     "--query-csv, --query-images")
    source = (args.archive if args.archive else ComponentInput(
        args.gallery_csv, tuple(args.gallery_images), args.query_csv, tuple(args.query_images)))
    if args.inspect_only:
        _, report = inspect_archive(source, allow_partial=args.allow_partial)
        print(json.dumps(report))
        return
    import torch
    torch.set_num_threads(args.threads)
    build(source, args.artifact_dir, args.output, args.device,
          args.batch_size, args.limit_per_split, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
