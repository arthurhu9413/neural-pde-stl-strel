from __future__ import annotations

from pathlib import Path

import yaml


def test_makefile_diffusion_targets_use_yaml_runner() -> None:
    text = Path("Makefile").read_text(encoding="utf-8")

    assert "scripts/run_experiment.py -c configs/diffusion1d_baseline.yaml" in text
    assert "scripts/run_experiment.py -c configs/diffusion1d_stl.yaml" in text
    assert "scripts/train_diffusion_stl.py --config" not in text
    assert "--override optim.epochs=200" not in text


def test_main_yaml_configs_have_explicit_tags() -> None:
    expected = {
        "configs/diffusion1d_baseline.yaml": ("diffusion1d", "baseline"),
        "configs/diffusion1d_stl.yaml": ("diffusion1d", "stl"),
        "configs/heat2d_baseline.yaml": ("heat2d", "heat2d_baseline"),
        "configs/heat2d_stl_safe.yaml": ("heat2d", "heat2d_stl_safe"),
        "configs/heat2d_stl_eventually.yaml": ("heat2d", "heat2d_stl_eventually"),
        "configs/neuromancer_sine_bound.yaml": ("neuromancer_sine_bound", "neuromancer_sine_bound"),
    }

    for rel, (exp, tag) in expected.items():
        cfg = yaml.safe_load(Path(rel).read_text(encoding="utf-8"))
        assert cfg["experiment"] == exp
        assert cfg["tag"] == tag


def test_makefile_has_non_destructive_figure_check() -> None:
    text = Path("Makefile").read_text(encoding="utf-8")

    assert "figures-check:" in text
    assert "--output-root $(FIGURE_CHECK_DIR)" in text
    assert "--manifest $(FIGURE_CHECK_DIR)/figure_manifest.json" in text


def test_makefile_has_non_destructive_refresh_check() -> None:
    text = Path("Makefile").read_text(encoding="utf-8")

    assert "refresh-check:" in text
    assert "scripts/refresh_committed_summaries.py --check" in text



def test_makefile_has_linux_cpu_install_target() -> None:
    text = Path("Makefile").read_text(encoding="utf-8")

    assert "install-cpu-linux:" in text
    assert "https://download.pytorch.org/whl/cpu" in text
    assert 'pip install -e ".[plot,dev]"' in text or '$(PIP) install -e ".[plot,dev]"' in text


def test_makefile_install_extra_wording_matches_pyproject_all_extra() -> None:
    text = Path("Makefile").read_text(encoding="utf-8")

    assert 'Install all extras declared in pyproject.toml' in text
    assert 'Install all optional dependencies' not in text


def test_makefile_lint_uses_repo_ruff_config() -> None:
    text = Path("Makefile").read_text(encoding="utf-8")

    assert "lint:" in text
    assert "python -m ruff check src/ tests/ scripts/" in text or "$(PYTHON) -m ruff check src/ tests/ scripts/" in text
    assert "--select E9,F63,F7,F82" not in text
