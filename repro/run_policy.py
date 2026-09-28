"""Explicit research permission is not a submission/compliance approval."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_run_policy(research_only=False, release=False, gate_path=None):
    path = Path(gate_path) if gate_path else ROOT / 'repro/configs/privacy_gate.json'
    gate = json.loads(path.read_text())
    approved = gate.get('status') == 'approved'
    if not approved and not (research_only and not release and gate.get('research_allowed') is True):
        raise RuntimeError('Release is not approved; research needs explicit --research-only: ' + gate['reason'])
    return {
        'purpose': 'research_only' if research_only else 'approved',
        'release_approved': approved and not research_only,
        'gate_status': gate['status'],
        'gate_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'plate_features': 'No OCR, text, plate embedding or plate-derived ranking input. Residual pixels not certified absent.',
    }
