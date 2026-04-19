from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
STALE_IDENTIFIERS = ("physical" + "_ai_stl", "physical" + "-ai-stl", "PHYSICAL" + "_AI_STL")
SKIP_PARTS = {
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


def _token_from_points(*points: int) -> str:
    return "".join(chr(point) for point in points)


FORBIDDEN_TOKENS = (
    _token_from_points(98, 101, 110, 32, 119, 111, 111, 100, 105, 110, 103),
    _token_from_points(115, 97, 109, 117, 101, 108, 32, 115, 97, 115, 97, 107, 105),
    _token_from_points(97, 110, 110, 101, 32, 116, 117, 109, 108, 105, 110),
    _token_from_points(116, 97, 121, 108, 111, 114, 32, 116, 46, 32, 106, 111, 104, 110, 115, 111, 110),
    _token_from_points(
        118, 97, 110, 100, 101, 114, 98, 105, 108, 116, 32, 117, 110, 105, 118, 101, 114, 115, 105, 116, 121
    ),
    _token_from_points(115, 97, 105, 118, 32, 50, 48, 50, 54),
    _token_from_points(110, 102, 109, 32, 50, 48, 50, 54),
)


def _should_skip_path(path: Path) -> bool:
    return any(part in SKIP_PARTS or part.endswith(".egg-info") for part in path.parts)


def _iter_utf8_text_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip_path(path):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pt", ".npz", ".npy", ".pyc"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        files.append((path, text))
    return files


def test_readme_title_uses_repo_name() -> None:
    first_line = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "# neural-pde-stl-strel"


def test_no_stale_project_identifiers_remain() -> None:
    offenders: list[str] = []
    for path, text in _iter_utf8_text_files():
        for token in STALE_IDENTIFIERS:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    assert offenders == []


def test_no_removed_collaborator_or_venue_tokens_remain() -> None:
    offenders: list[str] = []
    for path, text in _iter_utf8_text_files():
        lowered = text.lower()
        for token in FORBIDDEN_TOKENS:
            if token in lowered:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    assert offenders == []


def test_pyproject_all_extra_is_explicit_union() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "neural-pde-stl-strel[" not in text

    try:
        import tomllib
    except ModuleNotFoundError:
        for needle in (
            '"torch>=2.0"',
            r'"rtamt>=0.3; python_version < \"3.12\""',
            '"moonlight>=0.3"',
            '"neuromancer"',
            '"torchphysics"',
            '"matplotlib>=3.7"',
            '"pandas>=2.0"',
            '"pytest>=7.0"',
            '"ruff>=0.4"',
        ):
            assert needle in text
        return

    data = tomllib.loads(text)
    all_deps = set(data["project"]["optional-dependencies"]["all"])
    expected = {
        'torch>=2.0',
        'rtamt>=0.3; python_version < "3.12"',
        'moonlight>=0.3',
        'neuromancer',
        'torchphysics',
        'matplotlib>=3.7',
        'pandas>=2.0',
        'pytest>=7.0',
        'ruff>=0.4',
    }
    assert expected.issubset(all_deps)
    assert not any(dep.startswith("neural-pde-stl-strel[") for dep in all_deps)


def test_pyproject_excludes_bytecode_from_wheels() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for needle in (
        "[tool.setuptools]",
        "include-package-data = false",
        "[tool.setuptools.package-data]",
        'neural_pde_stl_strel = ["py.typed"]',
        "[tool.setuptools.exclude-package-data]",
        '"*" = ["*.pyc", "__pycache__/*"]',
    ):
        assert needle in text


def test_citation_metadata_matches_project_version() -> None:
    citation = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_line = next(line for line in pyproject.splitlines() if line.strip().startswith('version = '))
    version = version_line.split('"')[1]

    assert citation["title"] == "neural-pde-stl-strel"
    assert citation["version"] == version
    assert citation["repository-code"].endswith("/neural-pde-stl-strel")
    assert citation["authors"] == [{"family-names": "Hu", "given-names": "Arthur"}]


def test_pyproject_uses_spdx_license_metadata() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "MIT"' in text
    assert 'license-files = ["LICENSE"]' in text
    assert "License :: OSI Approved :: MIT License" not in text


def test_manifest_in_includes_repo_artifacts() -> None:
    text = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for needle in (
        "graft .github",
        "graft assets",
        "graft configs",
        "graft docs",
        "graft figs",
        "graft results",
        "graft scripts",
        "graft src",
        "graft tests",
        "include .env.example",
        "include .gitignore",
        "include CITATION.cff",
        "include CHANGELOG.md",
        "include CONTRIBUTING.md",
        "include Makefile",
        "include pyproject.toml",
        "include README.md",
        "include LICENSE",
        "include requirements*.txt",
    ):
        assert needle in text


def test_pyproject_ruff_default_matches_repo_lint_gate() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    try:
        import tomllib
    except ModuleNotFoundError:
        assert 'select = ["E9", "F63", "F7", "F82"]' in text
        assert "Lint & type check" not in (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        return

    data = tomllib.loads(text)
    assert data["tool"]["ruff"]["lint"]["select"] == ["E9", "F63", "F7", "F82"]
