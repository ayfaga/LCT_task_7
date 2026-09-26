"""Verify all SHA256/size entries in the local E2 hybrid bundle manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory: Path) -> dict:
    manifest_path = directory / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("members"):
        raise ValueError("Unsupported or empty bundle manifest")
    for name, expected in manifest["members"].items():
        member = Path(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe bundle member path: {name}")
        path = directory / member
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing file or size mismatch: {name}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"SHA256 mismatch: {name}")
    result = {"status": "verified", "bundle_version": manifest["bundle_version"],
              "members": len(manifest["members"]),
              "bundle_manifest_sha256": sha256(manifest_path)}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    verify(parser.parse_args().artifact_dir)
