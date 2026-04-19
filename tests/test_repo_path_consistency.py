from __future__ import annotations

from pathlib import Path


def test_heat2d_moonlight_default_json_matches_committed_artifact_name() -> None:
    text = Path("scripts/eval_heat2d_moonlight.py").read_text(encoding="utf-8")
    assert "heat2d_strel_monitoring.json" in text
    assert "heat2d_moonlight.json" not in text


def test_framework_survey_examples_use_real_docs_filename() -> None:
    text = Path("scripts/framework_survey.py").read_text(encoding="utf-8")
    assert "docs/FRAMEWORK_SURVEY.md" in text
    assert "docs/framework_survey.md" not in text


def test_heat2d_metadata_uses_posix_repo_relative_outdir() -> None:
    meta_text = Path("assets/heat2d_scalar/meta.json").read_text(encoding="utf-8")
    assert '"outdir": "assets/heat2d_scalar"' in meta_text
    assert "assets\\heat2d_scalar" not in meta_text

    script_text = Path("scripts/gen_heat2d_frames.py").read_text(encoding="utf-8")
    assert "v.as_posix()" in script_text
