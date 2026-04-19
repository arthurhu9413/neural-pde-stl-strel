from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_example_status_script_runs() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "example_status.py"), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["summary"]["reported_results"] >= 1
    assert payload["missing_evidence"] == []
