from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path

import setuptools.build_meta as build_meta


REPO_ROOT = Path(__file__).resolve().parents[1]
_SKIP_NAMES = {
    ".cache",
    ".figure-check",
    ".git",
    ".mplconfig",
    ".pycache",
    ".pytest_cache",
    ".ruff_cache",
    ".tmp",
    ".venv",
    "__pycache__",
    "build",
    "data",
    "dist",
    "logs",
    "runs",
}


def _ignore_for_copy(src_dir: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in _SKIP_NAMES:
            ignored.add(name)
            continue
        if name.endswith(".egg-info"):
            ignored.add(name)
            continue
        if name.endswith(".pyc"):
            ignored.add(name)
            continue
    return ignored


def test_sdist_includes_reproducibility_artifacts(tmp_path) -> None:
    work = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, work, ignore=_ignore_for_copy)

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    old_cwd = Path.cwd()
    os.chdir(work)
    try:
        sdist_name = build_meta.build_sdist(str(dist_dir))
    finally:
        os.chdir(old_cwd)

    sdist_path = dist_dir / sdist_name
    assert sdist_path.exists()

    with tarfile.open(sdist_path, "r:gz") as tf:
        names = sorted(tf.getnames())

    root = names[0].split("/", 1)[0]
    must_have = (
        ".env.example",
        ".github/workflows/ci.yml",
        ".gitignore",
        "CHANGELOG.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "LICENSE",
        "Makefile",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-extra.txt",
        "MANIFEST.in",
        "docs/PUBLICATION_NOTES.md",
        "configs/diffusion1d_baseline.yaml",
        "scripts/generate_all_figures.py",
        "results/diffusion1d_main_summary.json",
        "assets/diffusion1d_baseline_field.png",
        "figs/architecture_diagram.png",
        "tests/test_docs_paper_support.py",
        "src/neural_pde_stl_strel/__init__.py",
        "src/neural_pde_stl_strel/py.typed",
    )
    name_set = set(names)
    for rel in must_have:
        assert f"{root}/{rel}" in name_set, rel

    forbidden_fragments = (
        ".pytest_cache/",
        ".ruff_cache/",
        ".figure-check/",
        "/build/",
        "/dist/",
        "__pycache__/",
    )
    assert not any(any(fragment in name for fragment in forbidden_fragments) for name in names)
    assert not any(name.endswith(".pyc") for name in names)
