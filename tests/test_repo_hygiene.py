from __future__ import annotations

from pathlib import Path


def test_gitignore_covers_repo_local_runtime_dirs() -> None:
    text = Path(".gitignore").read_text(encoding="utf-8")
    for entry in (".env", ".cache/", ".mplconfig/", ".pycache/", "logs/", "runs/", ".tmp/", "data/"):
        assert entry in text


def test_make_clean_removes_repo_local_runtime_dirs() -> None:
    text = Path("Makefile").read_text(encoding="utf-8")
    for entry in (".cache", ".mplconfig", ".pycache", ".figure-check", "logs", "runs", ".tmp", "data"):
        assert entry in text
