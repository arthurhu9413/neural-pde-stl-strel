from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_all_figures.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generate_all_figures_supports_scratch_output(tmp_path: Path) -> None:
    tracked_fig = ROOT / "figs" / "architecture_diagram.png"
    before = _sha256(tracked_fig)

    manifest = tmp_path / "figure_manifest.json"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--dpi",
        "110",
        "--output-root",
        str(tmp_path),
        "--manifest",
        str(manifest),
        "--check",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["generated_count"] == payload["expected_count"]
    assert (tmp_path / "figs" / "architecture_diagram.png").is_file()
    assert (tmp_path / "assets" / "diffusion1d_baseline_field.png").is_file()
    assert _sha256(tracked_fig) == before
