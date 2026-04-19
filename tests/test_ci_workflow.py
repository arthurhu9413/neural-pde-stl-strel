from __future__ import annotations

from pathlib import Path


def test_ci_figure_job_uses_scratch_validation() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "--output-root .figure-check" in text
    assert "--manifest .figure-check/figure_manifest.json" in text
    assert "git diff --exit-code -- figs/ assets/" not in text


def test_ci_summary_job_uses_semantic_validation() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python scripts/refresh_committed_summaries.py --check" in text
    assert "git diff --exit-code -- results/" not in text



def test_ci_uses_minimal_permissions_and_cpu_torch() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "permissions:" in text
    assert "contents: read" in text
    assert "cache-dependency-path:" in text
    assert "https://download.pytorch.org/whl/cpu" in text
    assert 'pip install -e ".[dev,plot]"' in text
    assert 'pip install -e ".[torch,dev,plot]"' not in text


def test_ci_lint_job_uses_repo_ruff_config_and_honest_label() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "name: Lint" in text
    assert "Lint & type check" not in text
    assert "ruff check src/ tests/ scripts/" in text
    assert "--select E9,F63,F7,F82" not in text
