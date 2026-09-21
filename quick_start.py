from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_REQUIREMENTS = ROOT / "backend" / "requirements.txt"


def run_command(command: list[str], description: str) -> None:
    print(f"\n=== {description} ===")
    print("Command:", " ".join(command))
    result = subprocess.run(command, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def install_dependencies() -> None:
    if not BACKEND_REQUIREMENTS.exists():
        raise FileNotFoundError(f"Requirements file not found: {BACKEND_REQUIREMENTS}")

    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip")
    run_command([sys.executable, "-m", "pip", "install", "-r", str(BACKEND_REQUIREMENTS)], "Installing backend dependencies")


def start_server() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    print("\n=== Starting server ===")
    print("Open: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server.\n")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(ROOT),
        env=env,
        check=False,
    )


if __name__ == "__main__":
    install_dependencies()
    start_server()
