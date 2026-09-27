"""Fail fast on an incomplete local ML bundle, without importing torch."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class ArtifactError(RuntimeError):
    """The configured inference files cannot be used."""


def verify_artifacts(artifact_dir: Path, gallery_path: Path | None = None) -> dict:
    manifest_path = artifact_dir / "model_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactError(f"Model manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        filename = manifest["model_file"]
        expected_size = int(manifest["model_size_bytes"])
        expected_sha = manifest["model_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArtifactError(f"Invalid model manifest: {manifest_path}") from exc
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ArtifactError("Model manifest has an invalid model_file")
    weight = artifact_dir / filename
    if not weight.is_file():
        raise ArtifactError(f"Model weight is missing: {weight}. Run `git lfs pull`.")
    with weight.open("rb") as source:
        first_bytes = source.read(128)
    if first_bytes.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ArtifactError(
            f"{weight} is a Git LFS pointer, not the model weight. "
            "Install Git LFS and run `git lfs pull`, then retry."
        )
    actual_size = weight.stat().st_size
    if actual_size != expected_size:
        raise ArtifactError(
            f"Model weight size mismatch: {weight} has {actual_size} bytes; "
            f"expected {expected_size}. Re-download with `git lfs pull`."
        )
    digest = hashlib.sha256()
    with weight.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        raise ArtifactError(f"Model weight SHA256 mismatch: {weight}. Re-download the weight.")
    if gallery_path is not None and not gallery_path.is_file():
        raise ArtifactError(
            f"Gallery file is missing: {gallery_path}. Set LCT_ML_GALLERY_PATH "
            "to an existing gallery; see README.md."
        )
    return manifest


if __name__ == "__main__":
    try:
        verify_artifacts(
            Path(os.environ["LCT_ML_ARTIFACT_DIR"]),
            Path(os.environ["LCT_ML_GALLERY_PATH"]) if os.getenv("LCT_ML_GALLERY_PATH") else None,
        )
    except (ArtifactError, KeyError) as exc:
        raise SystemExit(f"ML artifact preflight failed: {exc}") from exc
    print("ML artifact preflight passed", flush=True)
