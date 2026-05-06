from __future__ import annotations

import json

import pytest

from neural_pde_stl_strel import __version__
from neural_pde_stl_strel.__main__ import main


def test_cli_version_flag(capsys) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_about_flag(capsys) -> None:
    assert main(["--about"]) == 0
    out = capsys.readouterr().out
    assert "neural_pde_stl_strel" in out
    assert "Dependency probes:" in out


def test_cli_about_flag_json(capsys) -> None:
    assert main(["--about", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "about"
    assert payload["version"] == __version__


def test_cli_about_brief_and_json_conflict(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["about", "--brief", "--json"])
    assert excinfo.value.code == 2
    assert "--brief and --json are mutually exclusive" in capsys.readouterr().err


def test_cli_version_flag_rejects_json_and_brief(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version", "--json"])
    assert excinfo.value.code == 2
    assert "--version cannot be combined with --brief or --json" in capsys.readouterr().err

    with pytest.raises(SystemExit) as excinfo:
        main(["--version", "--brief"])
    assert excinfo.value.code == 2
    assert "--version cannot be combined with --brief or --json" in capsys.readouterr().err
