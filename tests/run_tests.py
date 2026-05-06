#!/usr/bin/env python
"""Standalone test runner for neural-pde-stl-strel core functionality.

This script validates the core library without requiring pytest or PyTorch.
It tests all numpy-based modules and the CLI/utility infrastructure.

Usage:
    python tests/run_tests.py           # from repo root
    PYTHONPATH=src python tests/run_tests.py  # if not installed
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import traceback
import warnings
from pathlib import Path

# Ensure src/ is on the path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np


# ── Test infrastructure ──────────────────────────────────────────────────────

_PASS = 0
_FAIL = 0
_SKIP = 0
_ERRORS: list[str] = []


def _run(name: str, fn):
    global _PASS, _FAIL
    try:
        fn()
        _PASS += 1
        print(f"  \033[32m[PASS]\033[0m {name}")
    except Exception as e:
        _FAIL += 1
        _ERRORS.append(f"{name}: {e}")
        print(f"  \033[31m[FAIL]\033[0m {name}: {e}")
        traceback.print_exc(limit=2)


def _skip(name: str, reason: str):
    global _SKIP
    _SKIP += 1
    print(f"  \033[33m[SKIP]\033[0m {name}: {reason}")


_SKIP_SCAN_PARTS = {
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
_FORBIDDEN_TOKENS = (
    "".join(chr(point) for point in (98, 101, 110, 32, 119, 111, 111, 100, 105, 110, 103)),
    "".join(chr(point) for point in (115, 97, 109, 117, 101, 108, 32, 115, 97, 115, 97, 107, 105)),
    "".join(chr(point) for point in (97, 110, 110, 101, 32, 116, 117, 109, 108, 105, 110)),
    "".join(chr(point) for point in (116, 97, 121, 108, 111, 114, 32, 116, 46, 32, 106, 111, 104, 110, 115, 111, 110)),
    "".join(
        chr(point)
        for point in (
            118,
            97,
            110,
            100,
            101,
            114,
            98,
            105,
            108,
            116,
            32,
            117,
            110,
            105,
            118,
            101,
            114,
            115,
            105,
            116,
            121,
        )
    ),
    "".join(chr(point) for point in (115, 97, 105, 118, 32, 50, 48, 50, 54)),
    "".join(chr(point) for point in (110, 102, 109, 32, 50, 48, 50, 54)),
)


def _should_skip_scan_path(path: Path) -> bool:
    return any(part in _SKIP_SCAN_PARTS or part.endswith(".egg-info") for part in path.parts)


def _iter_repo_text_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path in _REPO.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip_scan_path(path):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pt", ".npz", ".npy", ".pyc"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        files.append((path, text))
    return files


# ── Test: Package imports ────────────────────────────────────────────────────

def test_package_import():
    import neural_pde_stl_strel
    assert isinstance(neural_pde_stl_strel.__version__, str) and neural_pde_stl_strel.__version__


def test_about():
    import neural_pde_stl_strel
    info = neural_pde_stl_strel.about()
    assert "neural_pde_stl_strel" in info
    assert "Dependency probes:" in info
    assert "pyyaml" in info


def test_optional_dependencies():
    import neural_pde_stl_strel
    deps = neural_pde_stl_strel.optional_dependencies()
    assert isinstance(deps, dict)
    assert "numpy" in deps
    assert deps["numpy"]["available"] is True
    assert "pyyaml" in deps
    assert deps["pyyaml"]["available"] is True


def test_require_optional_missing():
    import neural_pde_stl_strel
    try:
        neural_pde_stl_strel.require_optional("nonexistent_package_xyz")
        raise AssertionError("Should have raised ImportError")
    except ImportError:
        pass


def test_cli_version():
    from neural_pde_stl_strel.__main__ import main
    rc = main(["version"])
    assert rc == 0


def test_cli_version_flag():
    from neural_pde_stl_strel.__main__ import main
    rc = main(["--version"])
    assert rc == 0


def test_cli_about():
    from neural_pde_stl_strel.__main__ import main
    rc = main(["about"])
    assert rc == 0


def test_cli_about_flag():
    from neural_pde_stl_strel.__main__ import main
    rc = main(["--about"])
    assert rc == 0


def test_cli_about_brief():
    from neural_pde_stl_strel.__main__ import main
    rc = main(["about", "--brief"])
    assert rc == 0


def test_cli_about_json():
    from neural_pde_stl_strel.__main__ import main
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["about", "--json"])
    assert rc == 0
    data = json.loads(buf.getvalue())
    import neural_pde_stl_strel
    assert data["version"] == neural_pde_stl_strel.__version__


def test_cli_about_flag_json():
    from neural_pde_stl_strel.__main__ import main
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["--about", "--json"])
    assert rc == 0
    data = json.loads(buf.getvalue())
    import neural_pde_stl_strel
    assert data["command"] == "about"
    assert data["version"] == neural_pde_stl_strel.__version__


def test_cli_doctor():
    from neural_pde_stl_strel.__main__ import main
    rc = main(["doctor"])
    assert rc == 0




def test_cli_doctor_defaults_to_core_when_require_omitted():
    import io
    import contextlib
    import neural_pde_stl_strel
    import neural_pde_stl_strel.__main__ as cli

    report = {
        name: {"available": True, "version": "9.9.9"}
        for name in neural_pde_stl_strel.optional_dependencies().keys()
    }
    report["numpy"] = {"available": False, "version": None, "pip": "pip install numpy"}

    original = cli.optional_dependencies
    buf = io.StringIO()
    err = io.StringIO()
    try:
        cli.optional_dependencies = lambda **kwargs: report
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            rc = cli.main(["doctor", "--json"])
    finally:
        cli.optional_dependencies = original

    data = json.loads(buf.getvalue())
    assert rc == 1
    assert data["exit_code"] == 1
    assert data["requirements"]["requested_specs"] == []
    assert data["requirements"]["specs"] == ["core"]
    assert data["requirements"]["evaluations"] == [
        {
            "spec": "core",
            "group": "core",
            "policy": "all",
            "want": ["numpy", "pyyaml"],
            "have": ["pyyaml"],
            "missing": ["numpy"],
            "satisfied": False,
        }
    ]


def test_cli_doctor_rejects_too_old_core_versions():
    import io
    import contextlib
    import neural_pde_stl_strel
    import neural_pde_stl_strel.__main__ as cli

    report = {
        name: {"available": True, "version": "9.9.9"}
        for name in neural_pde_stl_strel.optional_dependencies().keys()
    }
    report["numpy"] = {"available": True, "version": "1.0.0", "pip": "pip install numpy"}
    report["pyyaml"] = {"available": True, "version": "5.0.0", "pip": "pip install pyyaml"}

    original = cli.optional_dependencies
    buf = io.StringIO()
    err = io.StringIO()
    try:
        cli.optional_dependencies = lambda **kwargs: report
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            rc = cli.main(["doctor", "--json"])
    finally:
        cli.optional_dependencies = original

    data = json.loads(buf.getvalue())
    assert rc == 1
    assert data["exit_code"] == 1
    assert any("numpy" in warning and "1.24" in warning for warning in data["warnings"])
    assert any("pyyaml" in warning and "6.0" in warning for warning in data["warnings"])
    assert data["requirements"]["evaluations"] == [
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


def test_version_guard_fallback_without_packaging():
    import neural_pde_stl_strel._versioning as versioning

    original_packaging = sys.modules.get("packaging")
    original_packaging_version = sys.modules.get("packaging.version")
    sys.modules["packaging"] = None
    sys.modules["packaging.version"] = None
    try:
        assert versioning.version_satisfies_minimum("2.10.0+cpu", "2.0") is True
        assert versioning.version_satisfies_minimum("1.0.0", "1.24") is False
        assert versioning.version_satisfies_minimum("5.0.0", "6.0") is False
        assert versioning.version_satisfies_minimum("1.24.0rc1", "1.24") is False
    finally:
        if original_packaging is None:
            sys.modules.pop("packaging", None)
        else:
            sys.modules["packaging"] = original_packaging
        if original_packaging_version is None:
            sys.modules.pop("packaging.version", None)
        else:
            sys.modules["packaging.version"] = original_packaging_version


def test_cli_doctor_rejects_too_old_core_versions_without_packaging():
    import io
    import contextlib
    import neural_pde_stl_strel
    import neural_pde_stl_strel.__main__ as cli

    report = {
        name: {"available": True, "version": "9.9.9"}
        for name in neural_pde_stl_strel.optional_dependencies().keys()
    }
    report["numpy"] = {"available": True, "version": "1.0.0", "pip": "pip install numpy"}
    report["pyyaml"] = {"available": True, "version": "5.0.0", "pip": "pip install pyyaml"}

    original_reporter = cli.optional_dependencies
    original_packaging = sys.modules.get("packaging")
    original_packaging_version = sys.modules.get("packaging.version")
    buf = io.StringIO()
    err = io.StringIO()
    try:
        cli.optional_dependencies = lambda **kwargs: report
        sys.modules["packaging"] = None
        sys.modules["packaging.version"] = None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            rc = cli.main(["doctor", "--json"])
    finally:
        cli.optional_dependencies = original_reporter
        if original_packaging is None:
            sys.modules.pop("packaging", None)
        else:
            sys.modules["packaging"] = original_packaging
        if original_packaging_version is None:
            sys.modules.pop("packaging.version", None)
        else:
            sys.modules["packaging.version"] = original_packaging_version

    data = json.loads(buf.getvalue())
    assert rc == 1
    assert data["exit_code"] == 1
    assert any("numpy" in warning and "1.24" in warning for warning in data["warnings"])
    assert any("pyyaml" in warning and "6.0" in warning for warning in data["warnings"])
    assert data["requirements"]["evaluations"] == [
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


def test_cli_pip():
    from neural_pde_stl_strel.__main__ import main
    rc = main(["pip"])
    assert rc == 0


def test_cli_about_brief_and_json_conflict():
    from neural_pde_stl_strel.__main__ import main
    import contextlib
    import io

    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            main(["about", "--brief", "--json"])
        raise AssertionError("Expected parser conflict for about --brief --json")
    except SystemExit as exc:
        assert exc.code == 2
    assert "--brief and --json are mutually exclusive" in buf.getvalue()


def test_cli_version_flag_rejects_json_and_brief():
    from neural_pde_stl_strel.__main__ import main
    import contextlib
    import io

    for argv in (["--version", "--json"], ["--version", "--brief"]):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                main(argv)
            raise AssertionError(f"Expected parser conflict for {argv}")
        except SystemExit as exc:
            assert exc.code == 2
        assert "--version cannot be combined with --brief or --json" in buf.getvalue()


def test_benchmark_snapshot_json():
    import subprocess

    cmd = [sys.executable, str(_REPO / "scripts" / "show_benchmark_snapshot.py"), "--json"]
    out = subprocess.check_output(cmd, text=True, cwd=_REPO)
    data = json.loads(out)
    assert data["source"] == "results/benchmark_training.csv"
    assert isinstance(data["rows"], list) and len(data["rows"]) >= 1
    assert {"experiment", "config", "epochs", "wall_time_s", "peak_memory_mb", "device"} <= set(data["rows"][0])


# ── Test: Seed utility ───────────────────────────────────────────────────────

def test_seed_everything():
    from neural_pde_stl_strel.utils.seed import seed_everything
    seed_everything(42)
    a = np.random.rand(5)
    seed_everything(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


# ── Test: PDE simulation ────────────────────────────────────────────────────

def test_cfl_number():
    from neural_pde_stl_strel.pde_example import cfl_number
    r = cfl_number(dt=0.1, alpha=0.1, dx=1.0)
    assert abs(r - 0.01) < 1e-10


def test_cfl_validation():
    from neural_pde_stl_strel.pde_example import cfl_number
    try:
        cfl_number(dt=0.1, alpha=0.1, dx=0.0)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass
    try:
        cfl_number(dt=0.1, alpha=-0.1, dx=1.0)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def test_simulate_diffusion_basic():
    from neural_pde_stl_strel.pde_example import simulate_diffusion
    u = simulate_diffusion(10, 5, dt=0.1, alpha=0.1)
    assert u.shape == (6, 10)
    assert u[0, 0] == 1.0  # hotspot at x=0


def test_simulate_diffusion_custom_ic():
    from neural_pde_stl_strel.pde_example import simulate_diffusion
    ic = np.sin(np.linspace(0, np.pi, 20))
    u = simulate_diffusion(20, 10, dt=0.01, alpha=0.1, initial=ic, dx=0.05)
    assert u.shape == (11, 20)
    np.testing.assert_allclose(u[0], ic)


def test_simulate_diffusion_zero_steps():
    from neural_pde_stl_strel.pde_example import simulate_diffusion
    u = simulate_diffusion(10, 0, dt=0.1, alpha=0.1)
    assert u.shape == (1, 10)


def test_simulate_diffusion_length1():
    from neural_pde_stl_strel.pde_example import simulate_diffusion
    u = simulate_diffusion(1, 5, dt=0.1, alpha=0.1)
    assert u.shape == (6, 1)


def test_simulate_diffusion_length2():
    from neural_pde_stl_strel.pde_example import simulate_diffusion
    u = simulate_diffusion(2, 5, dt=0.1, alpha=0.1)
    assert u.shape == (6, 2)


def test_simulate_diffusion_stability_warning():
    from neural_pde_stl_strel.pde_example import simulate_diffusion
    # r = alpha*dt/dx^2 = 1.0*1.0/1.0 = 1.0 > 0.5
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simulate_diffusion(10, 1, dt=1.0, alpha=1.0, dx=1.0)
        assert any(issubclass(x.category, RuntimeWarning) for x in w)


def test_simulate_diffusion_validation():
    from neural_pde_stl_strel.pde_example import simulate_diffusion
    try:
        simulate_diffusion(0, 5)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass
    try:
        simulate_diffusion(10, -1)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def test_simulate_diffusion_with_clipping():
    from neural_pde_stl_strel.pde_example import simulate_diffusion_with_clipping
    u = simulate_diffusion_with_clipping(
        10, 5, dt=0.1, alpha=0.1, lower=-0.5, upper=0.5
    )
    assert np.all(u >= -0.5)
    assert np.all(u <= 0.5)


def test_clipping_bounds_validation():
    from neural_pde_stl_strel.pde_example import simulate_diffusion_with_clipping
    try:
        simulate_diffusion_with_clipping(10, 5, lower=1.0, upper=0.0)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def test_diffusion_conservation():
    """Mass should be approximately conserved with Neumann BCs."""
    from neural_pde_stl_strel.pde_example import simulate_diffusion
    ic = np.zeros(50)
    ic[20:30] = 1.0
    u = simulate_diffusion(50, 100, dt=0.001, alpha=0.1, initial=ic, dx=0.02)
    mass_initial = u[0].sum()
    mass_final = u[-1].sum()
    assert abs(mass_final - mass_initial) / abs(mass_initial + 1e-12) < 0.05


# ── Test: Robustness computation ─────────────────────────────────────────────

def test_compute_robustness():
    from neural_pde_stl_strel.pde_example import compute_robustness
    r = compute_robustness(np.array([0.2, 0.5, 0.8]), 0.0, 1.0)
    assert abs(r - 0.2) < 1e-10


def test_compute_robustness_violation():
    from neural_pde_stl_strel.pde_example import compute_robustness
    r = compute_robustness(np.array([0.2, 1.5, 0.8]), 0.0, 1.0)
    assert r < 0  # violation


def test_compute_robustness_empty():
    from neural_pde_stl_strel.pde_example import compute_robustness
    try:
        compute_robustness(np.array([]), 0.0, 1.0)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def test_spatiotemporal_robustness():
    from neural_pde_stl_strel.pde_example import compute_spatiotemporal_robustness
    field = np.random.rand(5, 10) * 0.5 + 0.25  # in [0.25, 0.75]
    r = compute_spatiotemporal_robustness(field, 0.0, 1.0)
    assert r >= 0.25


def test_spatiotemporal_robustness_2d_only():
    from neural_pde_stl_strel.pde_example import compute_spatiotemporal_robustness
    try:
        compute_spatiotemporal_robustness(np.array([1.0, 2.0]), 0.0, 1.0)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def test_pointwise_bounds_margin():
    from neural_pde_stl_strel.pde_example import pointwise_bounds_margin
    m = pointwise_bounds_margin(np.array([0.3, 0.7, 1.2]), 0.0, 1.0)
    np.testing.assert_allclose(m, [0.3, 0.3, -0.2])


# ── Test: STL temporal operators ─────────────────────────────────────────────

def test_stl_globally():
    from neural_pde_stl_strel.pde_example import stl_globally_robustness
    rho = np.array([0.5, 0.3, 0.1, 0.4, 0.2])
    g = stl_globally_robustness(rho, window=3)
    assert len(g) == len(rho)
    # At index 2: min(0.5, 0.3, 0.1) = 0.1
    assert abs(g[2] - 0.1) < 1e-10


def test_stl_eventually():
    from neural_pde_stl_strel.pde_example import stl_eventually_robustness
    rho = np.array([0.5, 0.3, 0.1, 0.4, 0.2])
    e = stl_eventually_robustness(rho, window=3)
    assert len(e) == len(rho)
    # At index 2: max(0.5, 0.3, 0.1) = 0.5
    assert abs(e[2] - 0.5) < 1e-10


def test_stl_window_validation():
    from neural_pde_stl_strel.pde_example import stl_globally_robustness
    try:
        stl_globally_robustness(np.array([1.0]), window=0)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass


def test_stl_nan_propagation():
    from neural_pde_stl_strel.pde_example import stl_globally_robustness
    rho = np.array([0.5, np.nan, 0.3])
    g = stl_globally_robustness(rho, window=2)
    assert np.isnan(g[1])


# ── Test: Spatiotemporal operators ───────────────────────────────────────────

def test_rect_globally_bounds():
    from neural_pde_stl_strel.pde_example import stl_rect_globally_bounds
    field = np.random.rand(5, 10) * 0.5 + 0.25
    r = stl_rect_globally_bounds(field, 0.0, 1.0, t_window=2, x_window=3)
    assert r.shape == (5, 10)


def test_rect_eventually_bounds():
    from neural_pde_stl_strel.pde_example import stl_rect_eventually_bounds
    field = np.random.rand(5, 10) * 0.5 + 0.25
    r = stl_rect_eventually_bounds(field, 0.0, 1.0, t_window=2, x_window=3)
    assert r.shape == (5, 10)


# ── Test: YAML config parsing ────────────────────────────────────────────────

def test_yaml_configs():
    import yaml
    config_dir = _REPO / "configs"
    for f in sorted(config_dir.glob("*.yaml")):
        with open(f) as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict), f"{f.name} did not parse as dict"


# ── Test: Results files integrity ────────────────────────────────────────────

def test_results_files():
    results_dir = _REPO / "results"
    for f in results_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        assert isinstance(data, (dict, list)), f"{f.name} invalid JSON"
    for f in results_dir.glob("*.csv"):
        text = f.read_text()
        assert len(text) > 0, f"{f.name} is empty"


# ── Test: Heat2D data ────────────────────────────────────────────────────────

def test_heat2d_data():
    data_path = _REPO / "assets" / "heat2d_scalar" / "field_xy_t.npy"
    if not data_path.exists():
        _skip("test_heat2d_data", "field_xy_t.npy not found")
        return
    field = np.load(data_path)
    assert field.ndim == 3
    assert field.dtype == np.float32


def test_heat2d_metadata():
    meta_path = _REPO / "assets" / "heat2d_scalar" / "meta.json"
    if not meta_path.exists():
        _skip("test_heat2d_metadata", "meta.json not found")
        return
    with open(meta_path) as f:
        meta = json.load(f)
    assert "alpha" in meta
    assert "method" in meta


# ── Test: Figure files exist ─────────────────────────────────────────────────

def test_figure_files_exist():
    figs_dir = _REPO / "figs"
    expected = [
        "architecture_diagram.png",
        "training_pipeline.png",
        "stl_semantics_overview.png",
        "method_landscape.png",
        "summary_results.png",
        "comprehensive_results.png",
        "diffusion1d_training_dynamics.png",
        "diffusion1d_lambda_ablation.png",
        "diffusion1d_comparison.png",
        "heat2d_field_evolution.png",
        "heat2d_stl_traces.png",
        "pde_fields_overview.png",
        "quality_dashboard.png",
        "diffusion1d_ablations.png",
        "benchmark_cost.png",
    ]
    for fname in expected:
        path = figs_dir / fname
        assert path.exists(), f"Missing figure: {fname}"
        assert path.stat().st_size > 1000, f"Figure too small: {fname}"


def test_diffusion_ablation_summary_sync():
    """Figure-side ablation CSV should match the committed result summary."""

    results_path = _REPO / "results" / "diffusion1d_ablation_summary.csv"
    figs_path = _REPO / "figs" / "diffusion1d_ablations_summary.csv"

    with results_path.open(newline="", encoding="utf-8") as fh:
        results_rows = list(csv.DictReader(fh))
    with figs_path.open(newline="", encoding="utf-8") as fh:
        figure_rows = list(csv.DictReader(fh))

    assert len(results_rows) == len(figure_rows), "Ablation row count mismatch"
    assert results_rows, "Expected non-empty ablation summary"

    for idx, (res_row, fig_row) in enumerate(zip(results_rows, figure_rows, strict=True)):
        assert res_row.keys() == fig_row.keys(), f"Column mismatch at row {idx}"
        for key in res_row:
            try:
                res_val = float(res_row[key])
                fig_val = float(fig_row[key])
            except ValueError:
                assert res_row[key] == fig_row[key], f"Value mismatch at row {idx}, column {key}"
            else:
                assert math.isclose(res_val, fig_val, rel_tol=1e-9, abs_tol=1e-12), (
                    f"Numeric mismatch at row {idx}, column {key}: "
                    f"{res_val!r} != {fig_val!r}"
                )


# ── Test: Python file compilation ────────────────────────────────────────────

def test_all_python_compile():
    errors = []
    for py_file in sorted(_REPO.rglob("*.py")):
        if _should_skip_scan_path(py_file):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            compile(source, str(py_file), "exec")
        except Exception as e:
            errors.append(f"{py_file.relative_to(_REPO)}: {e}")
    assert not errors, f"Compilation errors:\n" + "\n".join(errors)


# ── Test: No trailing whitespace in source ───────────────────────────────────

def test_no_trailing_whitespace():
    violations = []
    for py_file in sorted((_REPO / "src").rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        for i, line in enumerate(py_file.read_text().splitlines(), 1):
            if line != line.rstrip():
                violations.append(f"{py_file.relative_to(_REPO)}:{i}")
                break  # one per file is enough
    assert not violations, f"Trailing whitespace in: {violations[:5]}"


# ── Test: Logger utility ─────────────────────────────────────────────────────

def test_csv_logger():
    import tempfile
    from neural_pde_stl_strel.utils.logger import CSVLogger
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        logger = CSVLogger(path, header=["epoch", "loss"], overwrite=True)
        logger.append({"epoch": 1, "loss": 0.5})
        logger.append({"epoch": 2, "loss": 0.3})
        del logger  # flush on garbage collection
        text = Path(path).read_text()
        assert "epoch" in text
        assert "0.5" in text
    finally:
        os.unlink(path)


# ── Test: Documentation & Structure ──────────────────────────────────────────

def test_readme_exists():
    readme = _REPO / "README.md"
    assert readme.exists(), "README.md not found"
    content = readme.read_text()
    assert len(content) > 1000, "README.md too short (< 1000 chars)"
    assert "## " in content, "README.md missing section headers"
    assert "STL" in content, "README.md should mention STL"


def test_changelog_exists():
    changelog = _REPO / "CHANGELOG.md"
    assert changelog.exists(), "CHANGELOG.md not found"
    content = changelog.read_text()
    assert "0.1.0" in content, "CHANGELOG.md should document version 0.1.0"


def test_docs_complete():
    docs_dir = _REPO / "docs"
    assert docs_dir.is_dir(), "docs/ directory not found"
    expected = [
        "DATASET_RECOMMENDATIONS.md",
        "FRAMEWORK_SURVEY.md",
        "INSTALL_EXTRAS.md",
        "READING_LIST.md",
        "REPRODUCIBILITY.md",
        "MANUSCRIPT_OUTLINE.md",
        "PUBLICATION_NOTES.md",
        "SPECIFICATIONS.md",
    ]
    for name in expected:
        path = docs_dir / name
        assert path.exists(), f"docs/{name} not found"
        assert path.stat().st_size > 100, f"docs/{name} too small (< 100 bytes)"


def test_license_exists():
    lic = _REPO / "LICENSE"
    assert lic.exists(), "LICENSE file not found"
    content = lic.read_text()
    assert "MIT" in content or "License" in content, "LICENSE should contain license text"


def test_citation_metadata():
    import yaml

    path = _REPO / "CITATION.cff"
    assert path.exists(), "CITATION.cff not found"
    data = yaml.safe_load(path.read_text())
    assert data["title"] == "neural-pde-stl-strel"
    assert data["repository-code"].endswith("/neural-pde-stl-strel")
    assert data.get("authors") == [{"family-names": "Hu", "given-names": "Arthur"}]


def test_no_stale_project_identifiers():
    stale = ("physical" + "_ai_stl", "physical" + "-ai-stl", "PHYSICAL" + "_AI_STL")
    offenders = []
    for path, file_text in _iter_repo_text_files():
        for token in stale:
            if token in file_text:
                offenders.append(f"{path.relative_to(_REPO)}: {token}")
    assert not offenders, f"Stale project identifiers found: {offenders[:5]}"


def test_no_removed_collaborator_or_venue_tokens():
    offenders = []
    for path, file_text in _iter_repo_text_files():
        lowered = file_text.lower()
        for token in _FORBIDDEN_TOKENS:
            if token in lowered:
                offenders.append(f"{path.relative_to(_REPO)}: {token}")
    assert not offenders, f"Removed collaborator/venue tokens found: {offenders[:5]}"


def test_pyproject_valid():
    pp = _REPO / "pyproject.toml"
    assert pp.exists(), "pyproject.toml not found"
    content = pp.read_text()
    assert "[build-system]" in content, "pyproject.toml missing [build-system]"
    assert "[project]" in content, "pyproject.toml missing [project]"
    assert 'name = "neural-pde-stl-strel"' in content, "pyproject.toml missing project name"
    assert "requires-python" in content, "pyproject.toml missing requires-python"
    # Try parsing with tomllib if available (Python 3.11+)
    try:
        import tomllib
        with open(pp, "rb") as f:
            data = tomllib.load(f)
        assert "project" in data
        assert data["project"]["name"] == "neural-pde-stl-strel"
    except ImportError:
        pass  # tomllib not available on Python 3.10


def test_pyproject_all_extra_is_explicit_union():
    pp = _REPO / "pyproject.toml"
    content = pp.read_text(encoding="utf-8")
    assert "neural-pde-stl-strel[" not in content, "all extra must not self-reference the package"

    expected = (
        '"torch>=2.0"',
        r'"rtamt>=0.3; python_version < \"3.12\""',
        '"moonlight>=0.3"',
        '"neuromancer"',
        '"torchphysics"',
        '"matplotlib>=3.7"',
        '"pandas>=2.0"',
        '"pytest>=7.0"',
        '"ruff>=0.4"',
    )
    for needle in expected:
        assert needle in content, f"Missing expected dependency in explicit all extra: {needle}"


def test_pyproject_excludes_bytecode_from_wheels():
    pp = _REPO / "pyproject.toml"
    content = pp.read_text(encoding="utf-8")
    for needle in (
        "[tool.setuptools]",
        "include-package-data = false",
        "[tool.setuptools.package-data]",
        'neural_pde_stl_strel = ["py.typed"]',
        "[tool.setuptools.exclude-package-data]",
        '"*" = ["*.pyc", "__pycache__/*"]',
    ):
        assert needle in content, f"Missing wheel-hygiene setting: {needle}"


def test_pyproject_uses_spdx_license_metadata():
    pp = _REPO / "pyproject.toml"
    content = pp.read_text(encoding="utf-8")
    assert 'license = "MIT"' in content
    assert 'license-files = ["LICENSE"]' in content
    assert 'License :: OSI Approved :: MIT License' not in content


def test_manifest_in_captures_repo_artifacts():
    manifest = (_REPO / "MANIFEST.in").read_text(encoding="utf-8")
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
        assert needle in manifest, f"Missing MANIFEST rule: {needle}"


def test_all_modules_have_docstrings():
    src_dir = _REPO / "src" / "neural_pde_stl_strel"
    missing = []
    for py_file in sorted(src_dir.rglob("*.py")):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        content = py_file.read_text()
        # Check that file has a docstring (triple-quoted string near the top)
        if '"""' not in content[:2000] and "'''" not in content[:2000]:
            rel = py_file.relative_to(_REPO)
            missing.append(str(rel))
    assert not missing, f"Files missing module docstrings: {missing[:5]}"


def test_no_bare_except():
    violations = []
    for py_file in sorted((_REPO / "src").rglob("*.py")):
        lines = py_file.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == "except:" or stripped.startswith("except: "):
                violations.append(f"{py_file.relative_to(_REPO)}:{i}")
    assert not violations, f"Bare except clauses found: {violations[:5]}"


def test_version_consistency():
    """Version in __init__.py must match pyproject.toml."""
    import neural_pde_stl_strel
    init_ver = neural_pde_stl_strel.__version__
    pyproject = _REPO / "pyproject.toml"
    for line in pyproject.read_text().splitlines():
        if line.strip().startswith('version = '):
            toml_ver = line.split('"')[1]
            assert init_ver == toml_ver, (
                f"__init__.__version__={init_ver!r} != "
                f"pyproject.toml version={toml_ver!r}"
            )
            return
    raise AssertionError("Could not find version in pyproject.toml")


def test_all_configs_have_required_keys():
    """All YAML configs should be valid dicts with at least 3 keys."""
    import yaml
    for yf in sorted((_REPO / "configs").glob("*.yaml")):
        cfg = yaml.safe_load(yf.read_text())
        assert isinstance(cfg, dict), f"{yf.name}: not a dict"
        assert len(cfg) >= 3, f"{yf.name}: only {len(cfg)} keys"


def test_method_landscape_figure():
    """Method landscape figure should exist and be non-trivial."""
    fig = _REPO / "figs" / "method_landscape.png"
    assert fig.exists(), "figs/method_landscape.png not found"
    assert fig.stat().st_size > 10000, "method_landscape.png too small"


def test_no_circular_imports():
    """Core subpackages should import without errors."""
    import importlib
    for sub in ["utils", "monitoring", "physics", "models", "datasets"]:
        mod = importlib.import_module(f"neural_pde_stl_strel.{sub}")
        assert mod is not None, f"Failed to import neural_pde_stl_strel.{sub}"


def test_no_long_lines():
    """All Python files should have lines <= 120 characters."""
    violations = []
    src = _REPO / "src"
    for py in sorted(src.rglob("*.py")):
        with open(py) as fh:
            for i, line in enumerate(fh, 1):
                if len(line.rstrip()) > 120:
                    violations.append(f"{py.relative_to(_REPO)}:{i}")
    assert not violations, (
        f"{len(violations)} lines > 120 chars in src/: "
        + ", ".join(violations[:5])
    )


def test_py_typed_marker():
    """Package should include py.typed marker for PEP 561."""
    marker = _REPO / "src" / "neural_pde_stl_strel" / "py.typed"
    assert marker.exists(), "py.typed marker missing"


def test_gitignore_exists():
    """.gitignore should be present."""
    assert (_REPO / ".gitignore").exists()


def test_ci_workflow_exists():
    """GitHub Actions CI workflow should be present."""
    ci = _REPO / ".github" / "workflows" / "ci.yml"
    assert ci.exists(), "CI workflow not found"
    content = ci.read_text()
    assert "pytest" in content or "run_tests" in content, (
        "CI workflow should run tests"
    )


def test_all_subpackages_have_init():
    """Every subdirectory under src/neural_pde_stl_strel/ with .py files
    should have an __init__.py."""
    pkg_root = _REPO / "src" / "neural_pde_stl_strel"
    missing = []
    for dirpath in sorted(pkg_root.rglob("*")):
        if not dirpath.is_dir():
            continue
        if dirpath.name == "__pycache__":
            continue
        py_files = list(dirpath.glob("*.py"))
        if py_files and not (dirpath / "__init__.py").exists():
            missing.append(str(dirpath.relative_to(_REPO)))
    assert not missing, f"Missing __init__.py: {missing}"


def test_contributing_exists():
    """CONTRIBUTING.md should exist."""
    assert (_REPO / "CONTRIBUTING.md").exists()


def test_readme_has_key_sections():
    """README.md should have essential sections for a research repo."""
    text = (_REPO / "README.md").read_text()
    required = [
        "Quickstart", "Case Study", "References",
        "Reproducibility", "Framework Integration",
    ]
    missing = [s for s in required if s not in text]
    assert not missing, f"README missing sections: {missing}"


def test_no_debug_prints_in_library():
    """Library code should not have bare print() calls (use logging)."""
    pkg_root = _REPO / "src" / "neural_pde_stl_strel"
    violations = []
    whitelist = {
        "__main__.py", "pde_example.py",
    }
    # Experiment runners and framework stubs are CLI-like; prints are OK
    whitelist_dirs = {"experiments", "frameworks", "monitors"}
    for pyfile in sorted(pkg_root.rglob("*.py")):
        if pyfile.name in whitelist:
            continue
        if any(d in pyfile.parts for d in whitelist_dirs):
            continue
        for i, line in enumerate(pyfile.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("print(") and not stripped.startswith("# "):
                violations.append(
                    f"{pyfile.relative_to(_REPO)}:{i}"
                )
    assert not violations, (
        f"print() in library code ({len(violations)} found): "
        + ", ".join(violations[:5])
    )


def test_makefile_exists():
    """Makefile should exist and have key targets."""
    mf = _REPO / "Makefile"
    assert mf.exists()
    text = mf.read_text()
    targets = ["install", "test", "clean", "lint"]
    missing = [t for t in targets if f"{t}:" not in text]
    assert not missing, f"Makefile missing targets: {missing}"


def test_no_mixed_line_endings():
    """All Python files should use Unix line endings (LF only)."""
    violations = []
    for root, _, files in os.walk(str(_REPO)):
        if ".git" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            with open(path, "rb") as fh:
                content = fh.read()
            if b"\r\n" in content:
                rel = os.path.relpath(path, str(_REPO))
                violations.append(rel)
    assert not violations, f"CRLF line endings: {violations}"


def test_yaml_configs_parseable():
    """All YAML configs should parse without errors."""
    import yaml
    configs_dir = _REPO / "configs"
    for yf in sorted(configs_dir.glob("*.yaml")):
        with open(yf) as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict), f"{yf.name}: not a dict"


def test_readme_image_refs_valid():
    """All image references in README.md should point to existing files."""
    readme = (_REPO / "README.md").read_text()
    pattern = re.compile(r"!\[.*?\]\((.*?)\)")
    missing = []
    for match in pattern.finditer(readme):
        path = match.group(1)
        if path.startswith("http"):
            continue
        full = _REPO / path
        if not full.exists():
            missing.append(path)
    assert not missing, f"Broken image refs: {missing}"


def test_env_example_exists():
    """.env.example should exist for documenting env vars."""
    assert (_REPO / ".env.example").exists()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("neural-pde-stl-strel  --  Standalone Test Suite")
    print("=" * 60)

    sections = [
        ("Package Imports & CLI", [
            ("package_import", test_package_import),
            ("about", test_about),
            ("optional_dependencies", test_optional_dependencies),
            ("require_optional_missing", test_require_optional_missing),
            ("cli_version", test_cli_version),
            ("cli_version_flag", test_cli_version_flag),
            ("cli_about", test_cli_about),
            ("cli_about_flag", test_cli_about_flag),
            ("cli_about_brief", test_cli_about_brief),
            ("cli_about_json", test_cli_about_json),
            ("cli_about_flag_json", test_cli_about_flag_json),
            ("cli_doctor", test_cli_doctor),
            ("cli_doctor_rejects_too_old_core_versions", test_cli_doctor_rejects_too_old_core_versions),
            ("version_guard_fallback_without_packaging", test_version_guard_fallback_without_packaging),
            ("cli_doctor_rejects_too_old_core_versions_without_packaging", test_cli_doctor_rejects_too_old_core_versions_without_packaging),
            ("cli_pip", test_cli_pip),
            ("cli_about_brief_and_json_conflict", test_cli_about_brief_and_json_conflict),
            ("cli_version_flag_rejects_json_and_brief", test_cli_version_flag_rejects_json_and_brief),
            ("benchmark_snapshot_json", test_benchmark_snapshot_json),
        ]),
        ("Utilities", [
            ("seed_everything", test_seed_everything),
            ("csv_logger", test_csv_logger),
        ]),
        ("PDE Simulation", [
            ("cfl_number", test_cfl_number),
            ("cfl_validation", test_cfl_validation),
            ("diffusion_basic", test_simulate_diffusion_basic),
            ("diffusion_custom_ic", test_simulate_diffusion_custom_ic),
            ("diffusion_zero_steps", test_simulate_diffusion_zero_steps),
            ("diffusion_length1", test_simulate_diffusion_length1),
            ("diffusion_length2", test_simulate_diffusion_length2),
            ("diffusion_stability_warning", test_simulate_diffusion_stability_warning),
            ("diffusion_validation", test_simulate_diffusion_validation),
            ("diffusion_with_clipping", test_simulate_diffusion_with_clipping),
            ("clipping_bounds_validation", test_clipping_bounds_validation),
            ("diffusion_conservation", test_diffusion_conservation),
        ]),
        ("Robustness Computation", [
            ("compute_robustness", test_compute_robustness),
            ("robustness_violation", test_compute_robustness_violation),
            ("robustness_empty", test_compute_robustness_empty),
            ("spatiotemporal_robustness", test_spatiotemporal_robustness),
            ("spatiotemporal_2d_only", test_spatiotemporal_robustness_2d_only),
            ("pointwise_bounds_margin", test_pointwise_bounds_margin),
        ]),
        ("STL Temporal Operators", [
            ("stl_globally", test_stl_globally),
            ("stl_eventually", test_stl_eventually),
            ("stl_window_validation", test_stl_window_validation),
            ("stl_nan_propagation", test_stl_nan_propagation),
        ]),
        ("Spatiotemporal Operators", [
            ("rect_globally_bounds", test_rect_globally_bounds),
            ("rect_eventually_bounds", test_rect_eventually_bounds),
        ]),
        ("Data & Configuration", [
            ("yaml_configs", test_yaml_configs),
            ("results_files", test_results_files),
            ("heat2d_data", test_heat2d_data),
            ("heat2d_metadata", test_heat2d_metadata),
            ("figure_files_exist", test_figure_files_exist),
            ("diffusion_ablation_summary_sync", test_diffusion_ablation_summary_sync),
        ]),
        ("Documentation & Structure", [
            ("readme_exists", test_readme_exists),
            ("changelog_exists", test_changelog_exists),
            ("docs_complete", test_docs_complete),
            ("license_exists", test_license_exists),
            ("citation_metadata", test_citation_metadata),
            ("pyproject_valid", test_pyproject_valid),
            ("pyproject_all_extra_explicit", test_pyproject_all_extra_is_explicit_union),
            ("pyproject_excludes_bytecode_from_wheels", test_pyproject_excludes_bytecode_from_wheels),
            ("pyproject_spdx_license_metadata", test_pyproject_uses_spdx_license_metadata),
            ("manifest_in_captures_repo_artifacts", test_manifest_in_captures_repo_artifacts),
            ("all_modules_have_docstrings", test_all_modules_have_docstrings),
            ("contributing_exists", test_contributing_exists),
            ("readme_has_key_sections", test_readme_has_key_sections),
            ("readme_image_refs_valid", test_readme_image_refs_valid),
            ("makefile_exists", test_makefile_exists),
            ("env_example_exists", test_env_example_exists),
        ]),
        ("Code Quality", [
            ("all_python_compile", test_all_python_compile),
            ("no_trailing_whitespace", test_no_trailing_whitespace),
            ("no_bare_except", test_no_bare_except),
            ("no_long_lines", test_no_long_lines),
            ("no_debug_prints_in_library", test_no_debug_prints_in_library),
            ("no_mixed_line_endings", test_no_mixed_line_endings),
            ("yaml_configs_parseable", test_yaml_configs_parseable),
            ("version_consistency", test_version_consistency),
            ("no_stale_project_identifiers", test_no_stale_project_identifiers),
            ("no_removed_collaborator_or_venue_tokens", test_no_removed_collaborator_or_venue_tokens),
            ("config_required_keys", test_all_configs_have_required_keys),
            ("method_landscape_figure", test_method_landscape_figure),
            ("no_circular_imports", test_no_circular_imports),
            ("py_typed_marker", test_py_typed_marker),
            ("gitignore_exists", test_gitignore_exists),
            ("ci_workflow_exists", test_ci_workflow_exists),
            ("all_subpackages_have_init", test_all_subpackages_have_init),
        ]),
    ]

    for section_name, tests in sections:
        print(f"\n── {section_name} ──")
        for name, fn in tests:
            _run(name, fn)

    print("\n" + "=" * 60)
    total = _PASS + _FAIL + _SKIP
    print(f"Results: {_PASS} passed, {_FAIL} failed, {_SKIP} skipped / {total} total")
    if _ERRORS:
        print("\nFailures:")
        for e in _ERRORS:
            print(f"  ✗ {e}")
    print("=" * 60)

    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
