"""Re-embed persisted user galleries after an encoder switch, retaining old archives.

Run with both API services stopped. Dry-run is the default; --apply writes a
new versioned NPZ and atomically updates meta.json only after verification.
Source images, prior NPZ generations and unrelated galleries are not removed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.runtime import MLRuntime  # noqa: E402


def inspect(folder: Path, model_version: str, model_sha: str) -> tuple[dict, bytes, str]:
    if not re.fullmatch(r"[0-9a-f]{32}", folder.name):
        raise ValueError(f"Invalid gallery folder name: {folder.name}")
    meta_path = folder / "meta.json"
    original = meta_path.read_bytes()
    meta = json.loads(original)
    if meta.get("gallery_id") != folder.name:
        raise ValueError(f"Gallery ID mismatch: {folder.name}")
    if meta.get("job_id") or meta.get("pending") or meta.get("state") not in {"ready", "collecting"}:
        raise ValueError(f"Gallery is not idle: {folder.name}")
    rows = meta.get("images", [])
    if not isinstance(rows, list) or len(rows) > 1000:
        raise ValueError(f"Invalid image count: {folder.name}")
    if not rows and not meta.get("gallery_file"):
        return meta, original, "empty"
    if not rows:
        raise ValueError(f"Empty gallery has an archive: {folder.name}")
    ids = [str(row.get("gallery_id", "")) for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate/empty image IDs: {folder.name}")
    for row in rows:
        if not re.fullmatch(r"[0-9a-f]{32}\.(?:jpg|jpeg|png)", str(row.get("image_key", ""))):
            raise ValueError(f"Invalid image key: {folder.name}")
        if not (folder / "images" / row["image_key"]).is_file():
            raise FileNotFoundError(f"Stored image missing: {folder.name}/{row['image_key']}")
        if not isinstance(row.get("bbox"), list) or len(row["bbox"]) != 4:
            raise ValueError(f"Invalid BBox: {folder.name}")
    old_file = str(meta.get("gallery_file", ""))
    if not re.fullmatch(r"gallery_[0-9]{4,}.npz", old_file):
        raise ValueError(f"Invalid gallery archive: {folder.name}")
    with np.load(folder / old_file, allow_pickle=False) as archive:
        old_version = str(archive["model_version"].item())
        old_sha = str(archive["model_sha256"].item())
        old_ids = archive["gallery_ids"].astype(str).tolist()
    if old_ids != ids:
        raise ValueError(f"Image metadata/archive order mismatch: {folder.name}")
    status = "current" if (old_version, old_sha) == (model_version, model_sha) else "migration_needed"
    return meta, original, status


def migrate(folder: Path, meta: dict, original: bytes, runtime: MLRuntime) -> dict:
    vectors = []
    rows = meta["images"]
    for row in rows:
        with Image.open(folder / "images" / row["image_key"]) as image:
            vectors.append(runtime.embed(image, tuple(row["bbox"])))
    embeddings = np.stack(vectors).astype(np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Nonfinite embedding: {folder.name}")
    generation = int(meta.get("generation", 0)) + 1
    while (folder / f"gallery_{generation:04d}.npz").exists():
        generation += 1
    filename = f"gallery_{generation:04d}.npz"
    target = folder / filename
    temporary = folder / f".gallery.{uuid4().hex}.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, gallery_ids=np.asarray([row["gallery_id"] for row in rows]),
            embeddings=embeddings, model_version=np.asarray(runtime.model_version),
            model_sha256=np.asarray(runtime.model_sha256),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    runtime._read_gallery(target)  # Verify the complete archive before switching metadata.
    meta_path = folder / "meta.json"
    if meta_path.read_bytes() != original:
        raise RuntimeError(f"Metadata changed during migration: {folder.name}")
    meta.update(gallery_file=filename, generation=generation)
    runtime._write_meta(folder, meta)
    return {"gallery_id": folder.name, "images": len(rows), "new_archive": filename,
            "old_archive_retained": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--gallery-state-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.artifact_dir / "model_manifest.json").read_text())
    folders = sorted(path for path in args.gallery_state_dir.iterdir()
                     if path.is_dir() and (path / "meta.json").is_file())
    inspected = []
    for folder in folders:
        meta, original, status = inspect(folder, manifest["model_version"],
                                         manifest["model_sha256"])
        inspected.append((folder, meta, original, status))
    print(json.dumps({"mode": "apply" if args.apply else "dry_run", "galleries": [
        {"gallery_id": folder.name, "images": len(meta["images"]), "status": status}
        for folder, meta, _, status in inspected]}, ensure_ascii=False), flush=True)
    if not args.apply:
        return
    runtime = MLRuntime(args.artifact_dir, None, args.device)
    changed = [migrate(folder, meta, original, runtime)
               for folder, meta, original, status in inspected if status == "migration_needed"]
    print(json.dumps({"status": "complete", "migrated": changed,
                      "already_current": sum(status == "current" for _, _, _, status in inspected)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
