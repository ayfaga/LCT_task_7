"""Startup must explain bad artifacts before installing dependencies or loading torch."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from artifact_preflight import ArtifactError, verify_artifacts

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import quick_start


def bundle(tmp_path: Path, weight: bytes = b"valid test model") -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_inference.pt").write_bytes(weight)
    (model_dir / "model_manifest.json").write_text(json.dumps({
        "model_file": "model_inference.pt",
        "model_size_bytes": len(weight),
        "model_sha256": hashlib.sha256(weight).hexdigest(),
    }), encoding="utf-8")
    gallery = tmp_path / "gallery.npz"
    gallery.write_bytes(b"placeholder gallery")
    return model_dir, gallery


def test_valid_artifacts_pass(tmp_path):
    model_dir, gallery = bundle(tmp_path)
    assert verify_artifacts(model_dir, gallery)["model_size_bytes"] == len(b"valid test model")


def test_lfs_pointer_gives_actionable_error(tmp_path):
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64
    model_dir, gallery = bundle(tmp_path, pointer)
    with pytest.raises(ArtifactError, match="Git LFS pointer.*git lfs pull"):
        verify_artifacts(model_dir, gallery)


def test_missing_gallery_does_not_get_silently_replaced(tmp_path):
    model_dir, gallery = bundle(tmp_path)
    gallery.unlink()
    with pytest.raises(ArtifactError, match="LCT_ML_GALLERY_PATH"):
        verify_artifacts(model_dir, gallery)


def test_corrupted_weight_rejected(tmp_path):
    model_dir, gallery = bundle(tmp_path)
    (model_dir / "model_inference.pt").write_bytes(b"x" * len(b"valid test model"))
    with pytest.raises(ArtifactError, match="SHA256 mismatch"):
        verify_artifacts(model_dir, gallery)


def test_size_mismatch_rejected(tmp_path):
    model_dir, gallery = bundle(tmp_path)
    (model_dir / "model_inference.pt").write_bytes(b"short")
    with pytest.raises(ArtifactError, match="size mismatch"):
        verify_artifacts(model_dir, gallery)


def test_quick_start_respects_gallery_override_and_fails_before_install(
    tmp_path, monkeypatch, capsys,
):
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64
    model_dir, gallery = bundle(tmp_path, pointer)
    monkeypatch.setenv("LCT_ML_ARTIFACT_DIR", str(model_dir))
    monkeypatch.setenv("LCT_ML_GALLERY_PATH", str(gallery))
    monkeypatch.setattr(quick_start, "install_dependencies", lambda: pytest.fail("pip ran"))
    monkeypatch.setattr("sys.argv", ["quick_start.py"])
    assert quick_start.configured_paths() == (model_dir.resolve(), gallery.resolve())
    assert quick_start.main() == 1
    assert "Git LFS pointer" in capsys.readouterr().err
