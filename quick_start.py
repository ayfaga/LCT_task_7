"""Запуск локального backend и ML-сервиса с проверкой артефактов."""

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


# --------------------------------------------------------------------------- #
# Утилиты запуска
# --------------------------------------------------------------------------- #

def run_command(command: list[str], description: str) -> None:
    print(f"\n=== {description} ===", flush=True)
    result = subprocess.run(command, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{description} завершилось с кодом {result.returncode}")


def install_dependencies() -> None:
    for path in (BACKEND_REQUIREMENTS, ML_REQUIREMENTS):
        if not path.is_file():
            raise FileNotFoundError(f"Не найден файл зависимостей: {path}")
    run_command(
        [sys.executable, "-m", "pip", "install", "-r", str(BACKEND_REQUIREMENTS)],
        "Установка backend-зависимостей",
    )
    run_command(
        [sys.executable, "-m", "pip", "install", "-r", str(ML_REQUIREMENTS)],
        "Установка ML-зависимостей",
    )


def configured_paths() -> tuple[Path, Path | None]:
    artifact_dir = Path(
        os.environ.get("LCT_ML_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))
    ).expanduser().resolve()
    gallery_setting = os.environ.get("LCT_ML_GALLERY_PATH", "")
    gallery_path = (
        Path(gallery_setting).expanduser().resolve() if gallery_setting else None
    )
    return artifact_dir, gallery_path


# --------------------------------------------------------------------------- #
# Проверка ML-runtime без изменения версии артефактов
# --------------------------------------------------------------------------- #

def verify_ml_runtime_loadable(artifact_dir: Path, gallery_path: Path | None) -> None:
    """Проверяет MLRuntime; несовпадение SHA — ошибка, а не повод менять bundle."""
    if os.environ.get("LCT_SKIP_ML_CHECK") == "1":
        print("[warn] LCT_SKIP_ML_CHECK=1 — preflight ML пропущен.", flush=True)
        return

    try:
        from app.ml.runtime import MLRuntime  # type: ignore
    except Exception as exc:
        raise ArtifactError(f"Не удалось импортировать app.ml.runtime: {exc}") from exc

    device = os.environ.get("LCT_ML_DEVICE", "cpu")
    gallery_arg = str(gallery_path) if gallery_path else None

    try:
        MLRuntime(str(artifact_dir), gallery_arg, device)
    except ValueError as exc:
        raise ArtifactError(f"ML runtime не загрузился: {exc}. Проверьте версию кода и артефактов; манифест не менялся.") from exc
    except Exception as exc:
        raise ArtifactError(f"ML runtime не загрузился: {type(exc).__name__}: {exc}") from exc
    print("[ok] ML preflight прошёл.", flush=True)


# --------------------------------------------------------------------------- #
# Ожидание готовности и запуск
# --------------------------------------------------------------------------- #

def wait_ready(name: str, port: int, process: subprocess.Popen, timeout: float = 120) -> None:
    url = f"http://127.0.0.1:{port}/ready"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{name} завершился с кодом {process.returncode}; см. лог выше"
            )
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                if response.status == 200:
                    return
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                raise RuntimeError(
                    f"{name} не готов (HTTP 503); см. лог выше"
                ) from exc
            raise RuntimeError(f"{name} /ready вернул HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError(f"{name} не поднялся по адресу {url} за {timeout}с")


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
        raise RuntimeError(
            f"Порт {port} занят; выберите другой --backend-port/--ml-port"
        ) from exc


def start_local_stack(
    artifact_dir: Path,
    gallery_path: Path | None,
    backend_port: int,
    ml_port: int,
) -> None:
    if backend_port == ml_port or not all(1 <= port <= 65535 for port in (backend_port, ml_port)):
        raise ValueError("Порты backend и ML должны различаться и быть в диапазоне 1..65535")
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
    print(f"ML:      http://127.0.0.1:{ml_port}/ready", flush=True)
    if gallery_path is None:
        print(
            "Галерея по умолчанию не задана. Загрузите её через API/браузер.",
            flush=True,
        )
    print("Ctrl+C — остановить оба сервиса.\n", flush=True)

    ml_process: subprocess.Popen | None = None
    backend_process: subprocess.Popen | None = None
    try:
        ml_process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "app.ml.api:app",
                "--host", "127.0.0.1", "--port", str(ml_port),
            ],
            cwd=str(BACKEND_DIR),
            env=env,
        )
        wait_ready("ML-сервис", ml_port, ml_process)

        backend_process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "app.main:app",
                "--host", "127.0.0.1", "--port", str(backend_port),
            ],
            cwd=str(BACKEND_DIR),
            env=env,
        )
        wait_ready("Backend", backend_port, backend_process)

        print("Оба сервиса готовы.", flush=True)
        while ml_process.poll() is None and backend_process.poll() is None:
            time.sleep(0.5)
        raise RuntimeError("Один из сервисов неожиданно остановился; см. лог выше")
    finally:
        stop_process(backend_process)
        stop_process(ml_process)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Не устанавливать зависимости (они уже стоят)",
    )
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--ml-port", type=int, default=8001)
    args = parser.parse_args()

    artifact_dir, gallery_path = configured_paths()

    try:
        verify_artifacts(artifact_dir, gallery_path)
        if not args.skip_install:
            install_dependencies()
        verify_ml_runtime_loadable(artifact_dir, gallery_path)
        start_local_stack(artifact_dir, gallery_path, args.backend_port, args.ml_port)
    except (ArtifactError, RuntimeError, ValueError, KeyboardInterrupt) as exc:
        if not isinstance(exc, KeyboardInterrupt):
            print(f"Запуск не удался: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
