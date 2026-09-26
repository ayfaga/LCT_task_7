from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
BACKEND_REQUIREMENTS = BACKEND_DIR / "requirements.txt"
ML_REQUIREMENTS = BACKEND_DIR / "requirements-ml.txt"
MODEL_ARTIFACT_DIR = ROOT / "model_artifacts" / "joint_l336"
DEFAULT_GALLERY = MODEL_ARTIFACT_DIR / "test_gallery_joint.npz"


def run_command(command: list[str], description: str, cwd: Path | None = None) -> None:
    print(f"\n=== {description} ===")
    print("Command:", " ".join(command))
    result = subprocess.run(command, cwd=str(cwd or ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def ensure_default_gallery() -> Path:
    if DEFAULT_GALLERY.exists():
        return DEFAULT_GALLERY

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "NumPy is not available yet. Re-run after install_dependencies() has completed."
        ) from exc

    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((MODEL_ARTIFACT_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(42)
    ids = [f"gallery_{index:04d}" for index in range(10)]
    vectors = rng.normal(size=(10, 1024)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    np.savez_compressed(
        DEFAULT_GALLERY,
        gallery_ids=np.asarray(ids),
        embeddings=vectors,
        model_version=np.asarray(manifest["model_version"]),
        model_sha256=np.asarray(manifest["model_sha256"]),
    )
    return DEFAULT_GALLERY


def install_dependencies() -> None:
    if not BACKEND_REQUIREMENTS.exists():
        raise FileNotFoundError(f"Requirements file not found: {BACKEND_REQUIREMENTS}")
    if not ML_REQUIREMENTS.exists():
        raise FileNotFoundError(f"ML requirements file not found: {ML_REQUIREMENTS}")

    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip")
    run_command([sys.executable, "-m", "pip", "install", "-r", str(BACKEND_REQUIREMENTS)], "Installing backend dependencies")
    run_command([sys.executable, "-m", "pip", "install", "-r", str(ML_REQUIREMENTS)], "Installing ML dependencies")


def start_local_stack() -> None:
    gallery_path = ensure_default_gallery()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["LCT_ML_ARTIFACT_DIR"] = str(MODEL_ARTIFACT_DIR)
    env["LCT_ML_GALLERY_PATH"] = str(gallery_path)
    env["LCT_ML_USER_GALLERY_DIR"] = str(ROOT / "gallery_state")
    env["LCT_UPLOAD_DIR"] = str(ROOT / "uploads")
    env["LCT_GALLERY_STATE_DIR"] = str(ROOT / "gallery_state")
    env["DATABASE_URL"] = f"sqlite:///{ROOT / 'local.db'}"

    print("\n=== Starting LCT stack ===")
    print("Backend: http://127.0.0.1:8000")
    print("ML: http://127.0.0.1:8001")
    print("Press Ctrl+C to stop all services.\n")

    ml_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.ml.api:app", "--host", "127.0.0.1", "--port", "8001"],
        cwd=str(BACKEND_DIR),
        env=env,
    )
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(BACKEND_DIR),
        env=env,
    )

    try:
        ml_process.wait()
    finally:
        backend_process.terminate()
        ml_process.terminate()


if __name__ == "__main__":
    install_dependencies()
    start_local_stack()
