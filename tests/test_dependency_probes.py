from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from neural_pde_stl_strel import about, optional_dependencies, require_optional
from neural_pde_stl_strel.__main__ import _evaluate_requirements, _requirement_groups, main
from neural_pde_stl_strel._versioning import version_satisfies_minimum
from neural_pde_stl_strel.monitors import BackendInfo, require_backend


def test_dependency_report_includes_pyyaml() -> None:
    deps = optional_dependencies()
    assert "pyyaml" in deps
    assert deps["pyyaml"]["available"] is True


def test_about_report_mentions_pyyaml() -> None:
    assert "pyyaml" in about()


def test_core_requirement_group_tracks_pyyaml() -> None:
    groups = _requirement_groups(optional_dependencies().keys())
    assert "numpy" in groups["core"]
    assert "pyyaml" in groups["core"]


def test_core_requirement_group_matches_base_runtime_deps() -> None:
    groups = _requirement_groups(optional_dependencies().keys())
    assert groups["core"] == {"numpy", "pyyaml"}
    assert "torch" not in groups["core"]


def test_core_requirement_defaults_to_all_policy() -> None:
    report = {
        "numpy": {"available": True},
        "pyyaml": {"available": False},
        "torch": {"available": True},
    }
    rc, evaluations, warnings = _evaluate_requirements(["core"], report, default_policy="any")

    assert rc == 1
    assert warnings == []
    assert evaluations == [
        {
            "spec": "core",
            "group": "core",
            "policy": "all",
            "want": ["numpy", "pyyaml"],
            "have": ["numpy"],
            "missing": ["pyyaml"],
            "satisfied": False,
        }
    ]


def test_core_requirement_rejects_too_old_versions() -> None:
    report = {
        "numpy": {"available": True, "version": "1.0.0"},
        "pyyaml": {"available": True, "version": "5.0.0"},
        "torch": {"available": True, "version": "2.0.0"},
    }
    rc, evaluations, warnings = _evaluate_requirements(["core"], report, default_policy="any")

    assert rc == 1
    assert any("numpy" in warning and "1.24" in warning for warning in warnings)
    assert any("pyyaml" in warning and "6.0" in warning for warning in warnings)
    assert evaluations == [
        {
            "spec": "core",
            "group": "core",
            "policy": "all",
            "want": ["numpy", "pyyaml"],
            "have": [],
            "missing": ["numpy", "pyyaml"],
            "satisfied": False,
        }
    ]


def test_version_guard_fallback_works_without_packaging(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "packaging", None)
    monkeypatch.setitem(sys.modules, "packaging.version", None)

    assert version_satisfies_minimum("2.10.0+cpu", "2.0") is True
    assert version_satisfies_minimum("6.0.3", "6.0") is True
    assert version_satisfies_minimum("1.0.0", "1.24") is False
    assert version_satisfies_minimum("5.0.0", "6.0") is False
    assert version_satisfies_minimum("1.24.0rc1", "1.24") is False


def test_core_requirement_rejects_too_old_versions_without_packaging(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "packaging", None)
    monkeypatch.setitem(sys.modules, "packaging.version", None)

    report = {
        "numpy": {"available": True, "version": "1.0.0"},
        "pyyaml": {"available": True, "version": "5.0.0"},
        "torch": {"available": True, "version": "2.10.0+cpu"},
    }
    rc, evaluations, warnings = _evaluate_requirements(["core"], report, default_policy="any")

    assert rc == 1
    assert any("numpy" in warning and "1.24" in warning for warning in warnings)
    assert any("pyyaml" in warning and "6.0" in warning for warning in warnings)
    assert evaluations == [
        {
            "spec": "core",
            "group": "core",
            "policy": "all",
            "want": ["numpy", "pyyaml"],
            "have": [],
            "missing": ["numpy", "pyyaml"],
            "satisfied": False,
        }
    ]


def test_require_optional_rejects_too_old_version_without_packaging(monkeypatch) -> None:
    import neural_pde_stl_strel as pkg

    monkeypatch.setitem(sys.modules, "packaging", None)
    monkeypatch.setitem(sys.modules, "packaging.version", None)
    monkeypatch.setattr(pkg, "_probe_module", lambda name: (True, "1.0.0"))

    with pytest.raises(ImportError, match="version >= 1.24 required; found 1.0.0"):
        require_optional("numpy", min_version="1.24")


def test_require_backend_rejects_too_old_version_without_packaging(monkeypatch) -> None:
    import neural_pde_stl_strel.monitors as monitors

    monkeypatch.setitem(sys.modules, "packaging", None)
    monkeypatch.setitem(sys.modules, "packaging.version", None)
    monkeypatch.setattr(
        monitors,
        "probe_backend",
        lambda name: BackendInfo(name=name, available=True, version="0.1", dist=name, error=None),
    )

    with pytest.raises(RuntimeError, match="older than required 0.3"):
        require_backend("moonlight", min_version="0.3")


def test_all_requirement_defaults_to_all_policy() -> None:
    report = {
        "numpy": {"available": True},
        "pyyaml": {"available": True},
        "moonlight": {"available": False},
    }
    rc, evaluations, warnings = _evaluate_requirements(["all"], report, default_policy="any")

    assert rc == 1
    assert warnings == []
    assert evaluations == [
        {
            "spec": "all",
            "group": "all",
            "policy": "all",
            "want": ["moonlight", "numpy", "pyyaml"],
            "have": ["numpy", "pyyaml"],
            "missing": ["moonlight"],
            "satisfied": False,
        }
    ]


def test_physics_requirement_still_defaults_to_any_policy() -> None:
    report = {
        "neuromancer": {"available": False},
        "physicsnemo": {"available": False},
        "modulus": {"available": False},
        "torchphysics": {"available": True},
    }
    rc, evaluations, warnings = _evaluate_requirements(["physics"], report, default_policy="any")

    assert rc == 0
    assert warnings == []
    assert evaluations == [
        {
            "spec": "physics",
            "group": "physics",
            "policy": "any",
            "want": ["modulus", "neuromancer", "physicsnemo", "torchphysics"],
            "have": ["torchphysics"],
            "missing": [],
            "satisfied": True,
        }
    ]


def test_cli_doctor_json_includes_pyyaml(capsys) -> None:
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "pyyaml" in payload["optional_deps"]




def test_cli_doctor_defaults_to_core_when_require_omitted(monkeypatch, capsys) -> None:
    import neural_pde_stl_strel.__main__ as cli

    report = {
        name: {"available": True, "version": "9.9.9"}
        for name in optional_dependencies().keys()
    }
    report["pyyaml"] = {"available": False, "version": None, "pip": "pip install pyyaml"}

    monkeypatch.setattr(cli, "optional_dependencies", lambda **kwargs: report)

    assert cli.main(["doctor", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)

    assert payload["exit_code"] == 1
    assert payload["requirements"]["requested_specs"] == []
    assert payload["requirements"]["specs"] == ["core"]
    assert payload["requirements"]["evaluations"] == [
        {
            "spec": "core",
            "group": "core",
            "policy": "all",
            "want": ["numpy", "pyyaml"],
            "have": ["numpy"],
            "missing": ["pyyaml"],
            "satisfied": False,
        }
    ]


def test_cli_doctor_rejects_too_old_core_versions(monkeypatch, capsys) -> None:
    import neural_pde_stl_strel.__main__ as cli

    report = {
        name: {"available": True, "version": "9.9.9"}
        for name in optional_dependencies().keys()
    }
    report["numpy"] = {"available": True, "version": "1.0.0", "pip": "pip install numpy"}
    report["pyyaml"] = {"available": True, "version": "5.0.0", "pip": "pip install pyyaml"}

    monkeypatch.setattr(cli, "optional_dependencies", lambda **kwargs: report)

    assert cli.main(["doctor", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)

    assert payload["exit_code"] == 1
    assert any("numpy" in warning and "1.24" in warning for warning in payload["warnings"])
    assert any("pyyaml" in warning and "6.0" in warning for warning in payload["warnings"])
    assert payload["requirements"]["evaluations"] == [
        {
            "spec": "core",
            "group": "core",
            "policy": "all",
            "want": ["numpy", "pyyaml"],
            "have": [],
            "missing": ["numpy", "pyyaml"],
            "satisfied": False,
        }
    ]


def test_install_extras_docs_cover_spatial_helpers() -> None:
    text = Path("docs/INSTALL_EXTRAS.md").read_text(encoding="utf-8")
    assert "spatial-spec" in text
    assert "frameworks/spatial_spec_hello.py" in text
    assert 'git+https://github.com/KTH-RPL-Planiacs/SpaTiaL.git#subdirectory=spatial-lib' in text
    assert "monitors/spatial_demo.py" in text


def test_make_doctor_script_core_includes_pyyaml() -> None:
    text = Path("scripts/check_env.py").read_text(encoding="utf-8")
    assert 'display="PyYAML"' in text
    assert 'import_names=("yaml",)' in text
    assert 'pip_names=("pyyaml",)' in text
    assert 'required=True' in text
    assert "all core requirements are satisfied (Python >= 3.10, NumPy >= 1.24, PyYAML >= 6.0)." in text
