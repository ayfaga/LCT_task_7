"""Keep FAISS out of the PyTorch pytest process on macOS."""

import os
import subprocess
import sys
from pathlib import Path


def test_ann_contract_in_child_process(tmp_path):
    root = Path(__file__).resolve().parents[3]
    environment = {**os.environ, "PYTHONPATH": str(root / "backend")}
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("ann_contract_check.py")), str(tmp_path)],
        env=environment, capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ANN_CONTRACT_OK" in completed.stdout
