"""Isolate backend tests from local application data and disk capacity."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_test_state = Path(tempfile.mkdtemp(prefix="lct_backend_tests_"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_test_state / 'local.db'}")
os.environ.setdefault("LCT_ML_USER_GALLERY_DIR", str(_test_state / "galleries"))
os.environ.setdefault("LCT_ML_GALLERY_DB_PATH", str(_test_state / "gallery.sqlite3"))

import app.gallery_store as gallery_store


@pytest.fixture(autouse=True)
def test_disk_reserve(monkeypatch):
    # Production still reserves 256 MiB; tiny fixture uploads do not depend on
    # the current Mac/CI disk capacity. Resource-guard behavior is tested apart.
    monkeypatch.setattr(gallery_store, "DISK_RESERVE_BYTES", 0)
