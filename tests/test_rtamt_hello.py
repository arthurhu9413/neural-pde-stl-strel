# tests/test_rtamt_hello.py
"""RTAMT smoke tests.

This file validates:

* RTAMT remains an *optional* dependency (our wrappers are importable without it).
* A tiny, deterministic STL example produces the expected robustness value.
* Our compatibility helpers in :mod:`neural_pde_stl_strel.monitoring.rtamt_monitor`
  reproduce the same result.

The core example uses the STL safety property:

    G (u <= 1.0)

On the sampled trace ``u = [0.2, 0.4, 1.1]`` (unit sampling), the quantitative
STL robustness is:

    min_t (1.0 - u(t)) = -0.1

RTAMT is optional in this repository. Tests that require RTAMT will skip if it is
not installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

# Make the package importable whether or not it has been installed yet.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_RTAMT_MODULE = "rtamt"

# Deterministic toy traces for a single variable `u`.
_U_MAX = 1.0
_TRACE_VIOLATES = (0.2, 0.4, 1.1)
_TRACE_SATISFIES = (0.2, 0.4, 0.9)
_EXPECTED_RHO_VIOLATES = -0.1
_EXPECTED_RHO_SATISFIES = 0.1
_ABS_TOL = 1e-9


def _rtamt_is_available() -> bool:
    """Return True if ``rtamt`` can be found on the import path."""

    return importlib.util.find_spec(_RTAMT_MODULE) is not None


def _import_rtamt_or_skip() -> ModuleType:
    """Return the imported ``rtamt`` module or skip the calling test.

    We intentionally avoid importing RTAMT at *module* import time so the test
    suite remains runnable in a minimal install.
    """

    if not _rtamt_is_available():
        pytest.skip("RTAMT not installed; install `rtamt` to run this test.")

    try:
        return importlib.import_module(_RTAMT_MODULE)
    except Exception as exc:  # pragma: no cover
        # RTAMT is optional; if it is present but broken (e.g. incompatible
        # Python version / missing native backend), skip with a helpful reason.
        pytest.skip(f"RTAMT found but could not be imported: {exc!r}")


def _build_always_upper_bound_spec():
    """Build the spec for ``G (u <= U)``.

    We prefer RTAMT's dense-time interpretation (the default in our wrapper)
    because it is what we use for continuous-time traces. If a particular RTAMT
    install lacks dense-time classes, we fall back to the discrete-time
    specification so the test still provides value.
    """

    from neural_pde_stl_strel.monitoring.rtamt_monitor import stl_always_upper_bound

    try:
        return stl_always_upper_bound("u", u_max=_U_MAX)
    except RuntimeError:
        return stl_always_upper_bound("u", u_max=_U_MAX, time_semantics="discrete")


def test_wrappers_do_not_import_rtamt_eagerly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing our RTAMT wrappers must *not* import RTAMT.

    This ensures RTAMT stays an optional dependency and that users without RTAMT
    can still import and run the rest of the project.
    """

    # Ensure a clean import path for this check.
    for module_name in list(sys.modules):
        if module_name == _RTAMT_MODULE or module_name.startswith(f"{_RTAMT_MODULE}."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    for module_name in (
        "neural_pde_stl_strel.monitoring.rtamt_monitor",
        "neural_pde_stl_strel.monitors.rtamt_hello",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    wrappers = importlib.import_module("neural_pde_stl_strel.monitoring.rtamt_monitor")
    hello = importlib.import_module("neural_pde_stl_strel.monitors.rtamt_hello")

    assert hasattr(wrappers, "stl_always_upper_bound")
    assert hasattr(wrappers, "evaluate_series")
    assert hasattr(hello, "stl_hello_offline")

    assert not any(
        name == _RTAMT_MODULE or name.startswith(f"{_RTAMT_MODULE}.")
        for name in sys.modules
    )


def test_rtamt_offline_matches_expected() -> None:
    """The canned hello-world offline monitor returns robustness -0.1."""

    _import_rtamt_or_skip()

    from neural_pde_stl_strel.monitors.rtamt_hello import stl_hello_offline

    rho = stl_hello_offline()
    assert isinstance(rho, (int, float))
    assert rho == pytest.approx(_EXPECTED_RHO_VIOLATES, abs=_ABS_TOL)
    assert rho < 0  # violated


def test_robustness_is_negative_when_series_exceeds_upper_bound() -> None:
    """G(u <= U) should be violated when any sample exceeds U (negative rho)."""

    _import_rtamt_or_skip()

    from neural_pde_stl_strel.monitoring.rtamt_monitor import (
        evaluate_series,
        satisfied,
    )

    spec = _build_always_upper_bound_spec()
    rho = evaluate_series(spec, "u", _TRACE_VIOLATES, dt=1.0)

    assert isinstance(rho, (int, float))
    assert rho == pytest.approx(_EXPECTED_RHO_VIOLATES, abs=_ABS_TOL)
    assert not satisfied(rho)


def test_evaluate_series_accepts_timestamped_samples() -> None:
    """Helpers should accept both raw samples and (time, value) pairs."""

    _import_rtamt_or_skip()

    from neural_pde_stl_strel.monitoring.rtamt_monitor import evaluate_series

    spec = _build_always_upper_bound_spec()

    # RTAMT examples often use a list-of-lists style for traces.
    timestamped = [[0.0, 0.2], [1.0, 0.4], [2.0, 1.1]]
    rho = evaluate_series(spec, "u", timestamped)

    assert isinstance(rho, (int, float))
    assert rho == pytest.approx(_EXPECTED_RHO_VIOLATES, abs=_ABS_TOL)


def test_robustness_is_positive_when_series_stays_below_upper_bound() -> None:
    """G(u <= U) should be satisfied when all samples stay below U (positive rho)."""

    _import_rtamt_or_skip()

    from neural_pde_stl_strel.monitoring.rtamt_monitor import (
        evaluate_series,
        satisfied,
    )

    spec = _build_always_upper_bound_spec()
    rho = evaluate_series(spec, "u", _TRACE_SATISFIES, dt=1.0)

    assert isinstance(rho, (int, float))
    assert rho == pytest.approx(_EXPECTED_RHO_SATISFIES, abs=_ABS_TOL)
    assert satisfied(rho)
