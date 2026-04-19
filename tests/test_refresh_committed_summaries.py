from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "refresh_committed_summaries.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_refresh_script_check_mode_is_read_only() -> None:
    tracked = [
        ROOT / "results" / "diffusion1d_main_summary.json",
        ROOT / "results" / "diffusion1d_baseline_field.npz",
        ROOT / "results" / "diffusion1d_stl_field.npz",
    ]
    before = {path: _sha256(path) for path in tracked}

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    after = {path: _sha256(path) for path in tracked}
    assert before == after
    assert "Verified" in proc.stdout


def test_refresh_script_default_preserves_committed_main_field_sidecars() -> None:
    tracked = [
        ROOT / "results" / "diffusion1d_main_summary.json",
        ROOT / "results" / "diffusion1d_baseline_field.npz",
        ROOT / "results" / "diffusion1d_stl_field.npz",
    ]
    before = {path: _sha256(path) for path in tracked}

    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    after = {path: _sha256(path) for path in tracked}
    assert before == after
    assert "Committed summaries refreshed" in proc.stdout
