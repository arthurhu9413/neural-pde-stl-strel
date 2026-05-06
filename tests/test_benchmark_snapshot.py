from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_benchmark_snapshot_json() -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "show_benchmark_snapshot.py"), "--json"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout)
    assert data["source"] == "results/benchmark_training.csv"
    assert isinstance(data["rows"], list) and data["rows"]
    row = data["rows"][0]
    assert row["experiment"]
    assert row["config"]
    assert row["device"]
