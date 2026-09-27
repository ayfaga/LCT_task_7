"""Start the local backend and ML service with an explicit artifact check."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
BACKEND_REQUIREMENTS = BACKEND_DIR / "requirements.txt"
ML_REQUIREMENTS = BACKEND_DIR / "requirements-ml.txt"
DEFAULT_ARTIFACT_DIR = ROOT / "model_artifacts" / "joint_l336"

sys.path.insert(0, str(BACKEND_DIR))
from artifact_preflight import ArtifactError, verify_artifacts  # noqa: E402


def run_command(command: list[str], description: str) -> None:
    print(f"\n=== {description} ===", flush=True)
    result = subprocess.run(command, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {result.returncode}")


def install_dependencies() -> None:
    for path in (BACKEND_REQUIREMENTS, ML_REQUIREMENTS):
        if not path.is_file():
            raise FileNotFoundError(f"Requirements file not found: {path}")
    run_command([sys.executable, "-m", "pip", "install", "-r", str(BACKEND_REQUIREMENTS)],
                "Installing backend dependencies")
    run_command([sys.executable, "-m", "pip", "install", "-r", str(ML_REQUIREMENTS)],
                "Installing ML dependencies")


def configured_paths() -> tuple[Path, Path | None]:
    artifact_dir = Path(os.environ.get("LCT_ML_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))).expanduser().resolve()
    gallery_setting = os.environ.get("LCT_ML_GALLERY_PATH", "")
    gallery_path = Path(gallery_setting).expanduser().resolve() if gallery_setting else None
    return artifact_dir, gallery_path


def wait_ready(name: str, port: int, process: subprocess.Popen, timeout: float = 120) -> None:
    url = f"http://127.0.0.1:{port}/ready"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited with code {process.returncode}; check its log above")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                if response.status == 200:
                    return
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                raise RuntimeError(f"{name} is not ready (HTTP 503); check its log above") from exc
            raise RuntimeError(f"{name} /ready returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError(f"{name} did not become ready at {url} within {timeout}s")


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def check_port_available(port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
    except OSError as exc:
        raise RuntimeError(f"Port {port} is unavailable; choose another --backend-port/--ml-port") from exc


def start_local_stack(artifact_dir: Path, gallery_path: Path | None,
                      backend_port: int, ml_port: int) -> None:
    if backend_port == ml_port or not all(1 <= port <= 65535 for port in (backend_port, ml_port)):
        raise ValueError("Backend and ML ports must be different and in 1..65535")
    check_port_available(backend_port)
    check_port_available(ml_port)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["LCT_ML_ARTIFACT_DIR"] = str(artifact_dir)
    env["LCT_ML_GALLERY_PATH"] = str(gallery_path) if gallery_path else ""
    env["LCT_ML_SERVICE_URL"] = f"http://127.0.0.1:{ml_port}"
    env.setdefault("LCT_ML_USER_GALLERY_DIR", str(ROOT / "gallery_state"))
    env.setdefault("LCT_ML_GALLERY_DB_PATH", str(ROOT / "gallery_state" / "gallery.sqlite3"))
    env.setdefault("LCT_UPLOAD_DIR", str(ROOT / "uploads"))
    env.setdefault("LCT_GALLERY_STATE_DIR", str(ROOT / "gallery_state"))
    env.setdefault("DATABASE_URL", f"sqlite:///{ROOT / 'local.db'}")

    print(f"\nBackend: http://127.0.0.1:{backend_port}/docs", flush=True)
    print(f"ML: http://127.0.0.1:{ml_port}/ready", flush=True)
    if gallery_path is None:
        print("No default gallery configured. Upload a gallery through the API/browser.",
              flush=True)
    print("Press Ctrl+C to stop both services.\n", flush=True)
    ml_process = None
    backend_process = None
    try:
        ml_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.ml.api:app", "--host", "127.0.0.1",
             "--port", str(ml_port)], cwd=str(BACKEND_DIR), env=env,
        )
        wait_ready("ML service", ml_port, ml_process)
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
             "--port", str(backend_port)], cwd=str(BACKEND_DIR), env=env,
        )
        wait_ready("Backend", backend_port, backend_process)
        print("Both services are ready.", flush=True)
        while ml_process.poll() is None and backend_process.poll() is None:
            time.sleep(0.5)
        raise RuntimeError("A service stopped unexpectedly; check its log above")
    finally:
        stop_process(backend_process)
        stop_process(ml_process)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Use dependencies already installed")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--ml-port", type=int, default=8001)
    args = parser.parse_args()
    artifact_dir, gallery_path = configured_paths()
    try:
        verify_artifacts(artifact_dir, gallery_path)
        if not args.skip_install:
            install_dependencies()
        start_local_stack(artifact_dir, gallery_path, args.backend_port, args.ml_port)
    except (ArtifactError, RuntimeError, ValueError, KeyboardInterrupt) as exc:
        if not isinstance(exc, KeyboardInterrupt):
            print(f"Startup failed: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
