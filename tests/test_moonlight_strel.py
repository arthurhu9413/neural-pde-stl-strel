"""Integration tests / executable examples for MoonLight STREL monitoring.

Repository-facing intent
These tests are deliberately written as *toy, actually-run examples* that:

* load a concrete STREL script (``scripts/specs/contain_hotspot.mls``),
* build a tiny spatial graph (3x3 grid),
* generate spatio-temporal traces that clearly **satisfy** and **falsify**
  containment specifications, and
* save at least one figure so results are immediately portable to a report/deck.

The goal is for this file to double as:

1) regression tests for the MoonLight glue code, and
2) a self-contained, easy-to-explain demo of *spatial STL/STREL monitoring*.

Specs under test
The repository's demo STREL script defines (among others):

* ``contain``: ``eventually(globally(nowhere_hot))``
  (i.e., the hotspot eventually disappears forever), and
* ``contain_within_5``: a bounded variant requiring a 5-time-unit window of
  ``nowhere_hot`` within a deadline.

We use 3 minimal traces:

* *Transient hotspot*  -> contain **holds**.
* *Persistent hotspot* -> contain **fails**.
* *Re-ignite* (quenches, then returns) -> contain **fails**, but contain_within_5
  **holds**.

MoonLight is a JVM-backed optional dependency. If it's not available (or the JVM
can't be started), we *skip* the runtime tests rather than failing CI.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Keep matplotlib imports headless (this file saves a figure when MoonLight works).
os.environ.setdefault("MPLBACKEND", "Agg")


def _import_strel_module():
    """Import the MoonLight STREL demo module.

    Tests are sometimes run from a fresh clone without an editable install.
    If that happens, fall back to adding ``repo_root/src`` to ``sys.path``.
    """

    try:
        return importlib.import_module("neural_pde_stl_strel.monitors.moonlight_strel_hello")
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"
        if src_dir.is_dir():
            sys.path.insert(0, str(src_dir))
        return importlib.import_module("neural_pde_stl_strel.monitors.moonlight_strel_hello")


def _skip_if_moonlight_runtime_issue(exc: BaseException) -> None:
    """Skip tests when MoonLight is unavailable due to optional/JVM issues.

    We intentionally avoid matching the generic substring "java" because MoonLight
    exceptions are often Java exceptions even when the *inputs* are malformed.
    Those should fail loudly.
    """

    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        pytest.skip(f"MoonLight not available: {exc!r}")

    msg = str(exc).lower()
    runtime_markers = (
        "jnius",
        "pyjnius",
        "jvm",
        "libjvm",
        "java_home",
        "could not find jvm",
        "no jvm",
        "failed to create jvm",
        "unsatisfiedlinkerror",
        "unsupported class file major version",
        "noclassdeffounderror",
        "classnotfound",
    )
    if any(m in msg for m in runtime_markers):
        pytest.skip(f"MoonLight runtime unavailable: {exc!r}")


def _extract_values(arr: np.ndarray) -> np.ndarray:
    """Extract the 'value' column from common MoonLight outputs.

    MoonLight's Python examples often return a 2-column matrix ``[index, value]``
    (e.g., ``[time, value]``). We treat the second column as the monitored signal.

    Returns
    -------
    np.ndarray
        1D float array of values (may be empty if conversion fails).
    """

    a = np.asarray(arr)

    def to_float_1d(x: Any) -> np.ndarray:
        try:
            return np.asarray(x, dtype=float).reshape(-1)
        except Exception:
            flat: list[float] = []
            try:
                for v in np.asarray(x).reshape(-1):
                    try:
                        flat.append(float(v))
                    except Exception:
                        continue
            except Exception:
                pass
            return np.asarray(flat, dtype=float)

    if a.ndim == 2 and a.shape[1] == 2:
        return to_float_1d(a[:, 1])
    if a.ndim == 1 and a.size == 2:
        return to_float_1d([a[1]])
    return to_float_1d(a)


def _any_positive(arr: np.ndarray) -> bool:
    """Heuristic satisfaction check that ignores an index/time column.

    For boolean semantics we expect values in {0,1}. For quantitative semantics,
    positive typically means satisfied and negative means violated.
    """

    values = _extract_values(arr)
    if values.size == 0:
        return False
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(np.any(finite > 0.0))


def _make_hotspot_field(nt: int, *, hot_times: set[int], amp: float = 2.0) -> np.ndarray:
    """Return a (3, 3, nt) field with a center-cell hotspot at selected times."""

    if nt <= 0:
        raise ValueError("nt must be positive")

    u = np.zeros((3, 3, nt), dtype=float)
    cx = cy = 1
    for t in hot_times:
        if 0 <= t < nt:
            u[cx, cy, t] = amp
    return u


def test_moonlight_strel_public_api_contract() -> None:
    mod = _import_strel_module()

    assert getattr(mod, "__all__", None) == ["strel_hello"]
    assert hasattr(mod, "strel_hello")

    sig = inspect.signature(mod.strel_hello)
    assert list(sig.parameters) == []


def test_contain_hotspot_spec_file_is_present_and_has_explicit_specs() -> None:
    """The STREL script should live in-repo and include explicit formulas."""

    mod = _import_strel_module()

    spec_path = mod._resolve_spec_file()  # noqa: SLF001
    assert spec_path is not None, "Unable to locate scripts/specs/contain_hotspot.mls"
    assert spec_path.is_file(), f"Missing STREL spec file: {spec_path}"

    text = spec_path.read_text(encoding="utf-8")

    # Keep these checks simple and robust (no exact formatting assumptions).
    assert "domain boolean" in text
    assert "signal" in text and "hot" in text

    # Specs that our demos/tests rely on.
    assert "formula contain" in text
    assert "formula contain_hotspot" in text
    assert "formula contain_within_5" in text


def test_strel_hello_smoke_runs_or_skips_cleanly() -> None:
    """`strel_hello()` should either run or fail with a clear optional-dep error."""

    mod = _import_strel_module()

    try:
        out = mod.strel_hello()
    except BaseException as exc:
        _skip_if_moonlight_runtime_issue(exc)
        raise

    assert isinstance(out, np.ndarray)
    assert out.size > 0

    vals = _extract_values(out)
    assert vals.size > 0
    assert np.all(np.isfinite(vals))

    # The default trace in `strel_hello` quenches the hotspot, so containment holds.
    assert _any_positive(out)


def test_toy_traces_satisfy_and_falsify_containment_and_save_a_figure(tmp_path: Path):
    """Run multiple traces and show satisfied and falsified cases + a figure."""

    mod = _import_strel_module()

    if importlib.util.find_spec("moonlight") is None:
        pytest.skip("moonlight not installed")

    # Load the .mls script once.
    try:
        spec_path = mod._resolve_spec_file()  # noqa: SLF001
        if spec_path is None or not spec_path.exists():
            pytest.skip("contain_hotspot.mls not found")

        script = mod.load_script_from_file(str(spec_path))
        if getattr(mod, "_helper_set_domain", None) is not None:  # noqa: SLF001
            mod._helper_set_domain(script, "boolean")  # noqa: SLF001
    except BaseException as exc:  # pragma: no cover
        _skip_if_moonlight_runtime_issue(exc)
        raise

    # Build a small grid graph.
    if getattr(mod, "_helper_build_grid_graph", None) is not None:  # noqa: SLF001
        graph = mod._helper_build_grid_graph(3, 3, return_format="triples")  # noqa: SLF001
    else:
        graph = mod._build_grid_graph_local(3, 3)  # noqa: SLF001

    to_signal = (
        mod._helper_field_to_signal  # noqa: SLF001
        if getattr(mod, "_helper_field_to_signal", None) is not None  # noqa: SLF001
        else mod._field_to_signal_local  # noqa: SLF001
    )

    def monitor(formula_name: str, field: np.ndarray) -> np.ndarray:
        try:
            sig = to_signal(field, threshold=1.0)
            mon = mod.get_monitor(script, formula_name)
            raw = mod._monitor_graph_time_series(mon, graph, sig)  # noqa: SLF001
            return mod._to_ndarray(raw)  # noqa: SLF001
        except BaseException as exc:  # pragma: no cover
            _skip_if_moonlight_runtime_issue(exc)
            raise

    # We choose nt=8 so that contain_within_5 (tau=5) has room to be satisfied.
    nt = 8

    # 1) Transient hotspot: should satisfy contain.
    u_transient = _make_hotspot_field(nt, hot_times={0})
    out_transient = monitor("contain", u_transient)
    assert _any_positive(out_transient)

    # 2) Persistent hotspot: should falsify contain.
    u_persistent = _make_hotspot_field(nt, hot_times=set(range(nt)))
    out_persistent = monitor("contain", u_persistent)
    assert not _any_positive(out_persistent)

    # 3) Re-ignite: falsifies contain, but satisfies contain_within_5.
    u_reignite = _make_hotspot_field(nt, hot_times={0, nt - 1})
    out_reignite_contain = monitor("contain", u_reignite)
    out_reignite_within = monitor("contain_within_5", u_reignite)

    assert not _any_positive(out_reignite_contain)
    assert _any_positive(out_reignite_within)

    # Save a figure that is useful for reports/slides: trace + monitoring outputs.
    import matplotlib.pyplot as plt

    t = np.arange(nt, dtype=float)
    max_hot = (u_reignite > 1.0).max(axis=(0, 1)).astype(float)

    vals_contain = _extract_values(out_reignite_contain)
    vals_within = _extract_values(out_reignite_within)

    fig_path = tmp_path / "moonlight_strel_reignite_trace.png"

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, max_hot, marker="o", label="max(hot)")

    if vals_contain.size:
        x = t if vals_contain.size == nt else np.arange(vals_contain.size, dtype=float)
        ax.plot(x, vals_contain, label="contain")
    if vals_within.size:
        x = t if vals_within.size == nt else np.arange(vals_within.size, dtype=float)
        ax.plot(x, vals_within, label="contain_within_5")

    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.set_title("Toy STREL trace + monitored specs")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    assert fig_path.exists() and fig_path.stat().st_size > 0


def test_strel_hello_graceful_unavailable(monkeypatch: pytest.MonkeyPatch):
    """If our glue layer is missing, error should be explicit (not cryptic)."""

    mod = _import_strel_module()

    # Force the internal guard.
    monkeypatch.setattr(mod, "load_script_from_file", None)

    with pytest.raises(
        RuntimeError,
        match=r"MoonLight is not installed|MoonLight helper is missing|MoonLight is not available",
    ):
        mod.strel_hello()


def test_strel_hello_missing_moonlight_package_raises_importerror(monkeypatch: pytest.MonkeyPatch):
    """When `moonlight` isn't installed, we should raise a clear ImportError.

    This test only runs when MoonLight is actually absent, so we don't fight the
    real import system in environments where MoonLight is installed.
    """

    if importlib.util.find_spec("moonlight") is not None:
        pytest.skip("moonlight is installed; skip negative-path import test")

    mod = _import_strel_module()

    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ):
        if name == "moonlight":
            raise ModuleNotFoundError("No module named 'moonlight'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        mod.strel_hello()
