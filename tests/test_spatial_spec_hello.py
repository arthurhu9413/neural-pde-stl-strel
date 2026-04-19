from __future__ import annotations

"""Tests for :mod:`neural_pde_stl_strel.frameworks.spatial_spec_hello`.

This repository treats **SpaTiaL Specifications** as an *optional* dependency:

- PyPI distribution: ``spatial-spec``
- Import name: ``spatial_spec``

These tests focus on *our* small helper module (not the full semantics of the
external library). They are designed to be:

- **Hermetic / CI-friendly**: do *not* require ``spatial-spec`` to be installed.
- **Behavior-focused**: validate lazy-import behavior, version resolution
  policy, and user-facing error messages.
- **Optionally integrative**: if ``spatial-spec`` *is* installed, run a tiny
  end-to-end example that (a) both satisfies and falsifies a spec and (b)
  produces a simple plot (useful for demos / docs).

Project motivation: keep the repo auditable by showing concrete, actually-run
examples with clearly written specs and figures, even when optional
dependencies are involved.
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

HELPER_MOD = "neural_pde_stl_strel.frameworks.spatial_spec_hello"
EXPECTED_DIST_NAME = "spatial-spec"
EXPECTED_MODULE_NAME = "spatial_spec"


def _import_helper() -> ModuleType:
    """Import and return the helper module under test."""

    return importlib.import_module(HELPER_MOD)


def _import_helper_fresh(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the helper module after removing any cached copy."""

    monkeypatch.delitem(sys.modules, HELPER_MOD, raising=False)
    return importlib.import_module(HELPER_MOD)


def _dummy_spatial_spec(version: str | None = None) -> ModuleType:
    """Create a tiny stand-in module object for ``spatial_spec``."""

    mod = ModuleType(EXPECTED_MODULE_NAME)
    if version is not None:
        # The helper prefers the module's __version__ if present.
        mod.__version__ = version  # type: ignore[attr-defined]
    return mod


def test_helper_import_is_lazy_wrt_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the helper must *not* import ``spatial_spec``.

    We force ``spatial_spec`` to be un-importable even if it is installed by
    inserting ``None`` in ``sys.modules``.

    Per Python's import system, a ``None`` entry indicates an import in progress
    that failed, and subsequent import attempts should raise
    :class:`ModuleNotFoundError`.

    If the helper imported ``spatial_spec`` at module import time, this test
    would fail immediately.
    """

    monkeypatch.setitem(sys.modules, EXPECTED_MODULE_NAME, None)
    helper = _import_helper_fresh(monkeypatch)

    assert helper.SPATIAL_SPEC_DIST_NAME == EXPECTED_DIST_NAME
    assert helper.SPATIAL_SPEC_MODULE_NAME == EXPECTED_MODULE_NAME


def test_public_api_surface_is_stable() -> None:
    """The helper should export a small, stable API for downstream use."""

    helper = _import_helper()

    assert helper.__all__ == [
        "SPATIAL_SPEC_DIST_NAME",
        "SPATIAL_SPEC_MODULE_NAME",
        "spatial_spec_version",
        "spatial_spec_available",
    ]

    # Core call sites used throughout the repository.
    assert callable(helper.spatial_spec_version)
    assert callable(helper.spatial_spec_available)


def test_spatial_spec_version_prefers_dunder_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``spatial_spec.__version__`` is present (and non-blank), return it."""

    helper = _import_helper()

    monkeypatch.setitem(sys.modules, EXPECTED_MODULE_NAME, _dummy_spatial_spec("9.9.9-test"))
    assert helper.spatial_spec_version() == "9.9.9-test"


def test_spatial_spec_version_ignores_blank_dunder_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank ``__version__`` should be treated as missing and fall back."""

    helper = _import_helper()

    monkeypatch.setitem(sys.modules, EXPECTED_MODULE_NAME, _dummy_spatial_spec("  "))

    def fake_metadata_version(dist_name: str) -> str:
        assert dist_name == EXPECTED_DIST_NAME
        return "0.0.0-meta"

    monkeypatch.setattr(helper._metadata, "version", fake_metadata_version)

    assert helper.spatial_spec_version() == "0.0.0-meta"


def test_spatial_spec_version_falls_back_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``__version__`` is absent, fall back to :func:`importlib.metadata.version`."""

    helper = _import_helper()

    monkeypatch.setitem(sys.modules, EXPECTED_MODULE_NAME, _dummy_spatial_spec(None))

    def fake_metadata_version(dist_name: str) -> str:
        assert dist_name == EXPECTED_DIST_NAME
        return "1.2.3"

    monkeypatch.setattr(helper._metadata, "version", fake_metadata_version)

    assert helper.spatial_spec_version() == "1.2.3"


def test_spatial_spec_version_returns_unknown_when_metadata_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If metadata lookup fails, return the literal ``"unknown"``."""

    helper = _import_helper()

    monkeypatch.setitem(sys.modules, EXPECTED_MODULE_NAME, _dummy_spatial_spec(None))

    def raise_anything(_dist_name: str) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(helper._metadata, "version", raise_anything)

    assert helper.spatial_spec_version() == "unknown"


def test_spatial_spec_version_raises_clean_importerror_when_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``spatial_spec`` cannot be imported, raise a helpful :class:`ImportError`."""

    helper = _import_helper()

    # Force the import to fail even if the optional dependency is installed.
    monkeypatch.setitem(sys.modules, EXPECTED_MODULE_NAME, None)

    with pytest.raises(ImportError) as excinfo:
        helper.spatial_spec_version()

    msg = str(excinfo.value)
    assert EXPECTED_DIST_NAME in msg
    assert EXPECTED_MODULE_NAME in msg
    assert "pip install" in msg


def test_spatial_spec_available_truth_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """``spatial_spec_available`` should never raise and should match reality."""

    helper = _import_helper()

    monkeypatch.setitem(sys.modules, EXPECTED_MODULE_NAME, _dummy_spatial_spec("0.0"))
    assert helper.spatial_spec_available() is True

    monkeypatch.setitem(sys.modules, EXPECTED_MODULE_NAME, None)
    assert helper.spatial_spec_available() is False


def test_spatial_spec_toy_example_runs_if_installed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Optional integration smoke test (skips if ``spatial-spec`` is not installed).

    A minimal, actually-run toy example:

    - Create a static object ``a``.
    - Create a dynamic object ``b`` that starts left of ``a`` and ends right of ``a``.
    - Monitor:
        * a satisfied bounded-eventually spec: ``(F [0,1] (a leftof b))``
        * a falsified bounded-always spec:     ``(G [0,1] (a leftof b))``

    We also render a simple two-panel plot (t=0 and t=1) into ``tmp_path`` to
    ensure the dependency is usable in a headless environment.
    """

    helper = _import_helper()
    if not helper.spatial_spec_available():
        pytest.skip("optional dependency `spatial-spec` is not installed")

    # Version strings are useful for demos/bug reports; the helper should be able
    # to resolve it when the package is installed.
    version = helper.spatial_spec_version()
    assert isinstance(version, str)
    assert version.strip() and version != "unknown"

    # Ensure a headless-safe Matplotlib backend *before* importing spatial_spec.geometry.
    monkeypatch.setenv("MPLBACKEND", "Agg")

    import numpy as np

    # Imports below are intentionally inside the test.
    from spatial_spec.geometry import DynamicObject, Polygon, PolygonCollection, StaticObject
    from spatial_spec.logic import Spatial

    def square(*, x0: float, y0: float, side: float = 1.0) -> PolygonCollection:
        verts = np.array(
            [
                [x0, y0],
                [x0, y0 + side],
                [x0 + side, y0 + side],
                [x0 + side, y0],
            ],
            dtype=float,
        )
        return PolygonCollection({Polygon(verts)})

    # Trace construction -------------------------------------------------
    a = StaticObject(square(x0=0.0, y0=0.0))

    # b starts left of a at t=0 and moves to the right of a at t=1.
    b_dyn = DynamicObject()
    b_dyn.addObject(square(x0=-3.0, y0=0.0), 0)
    b_dyn.addObject(square(x0=3.0, y0=0.0), 1)

    # Monitoring ---------------------------------------------------------
    stl = Spatial(quantitative=True)

    # spatial-spec exposes `assign_variable` (not `assign_var`) on the `Spatial` object.
    stl.assign_variable("a", a)
    stl.assign_variable("b", b_dyn)

    def parse_or_fail(formula: str) -> object:
        """Parse *formula* and fail with a clear message if parsing fails."""

        tree = stl.parse(formula)
        assert tree is not None, f"Failed to parse formula: {formula!r}"
        return tree

    atom_formula = "(a leftof b)"
    eventually_formula = "(F [0,1] (a leftof b))"
    always_formula = "(G [0,1] (a leftof b))"

    tree_atom = parse_or_fail(atom_formula)
    tree_eventually = parse_or_fail(eventually_formula)
    tree_always = parse_or_fail(always_formula)

    # Sanity-check the atomic predicate at each time.
    rho_atom_t0 = stl.interpret(tree_atom, lower=0, upper=0)
    rho_atom_t1 = stl.interpret(tree_atom, lower=1, upper=1)

    assert isinstance(rho_atom_t0, (float, np.floating))
    assert isinstance(rho_atom_t1, (float, np.floating))
    assert rho_atom_t0 < 0, "At t=0, b is left of a, so (a leftof b) should be violated"
    assert rho_atom_t1 > 0, "At t=1, b is right of a, so (a leftof b) should be satisfied"

    rho_eventually = stl.interpret(tree_eventually, lower=0, upper=1)
    rho_always = stl.interpret(tree_always, lower=0, upper=1)

    assert isinstance(rho_eventually, (float, np.floating))
    assert isinstance(rho_always, (float, np.floating))

    assert rho_eventually > 0, "The bounded eventually spec should be satisfied for the constructed trace"
    assert rho_always < 0, "The bounded always spec should be falsified for the constructed trace"

    # Plotting -----------------------------------------------------------
    # This isn't meant to be beautiful; it simply proves we can generate a
    # headless-safe figure for demos/docs.
    import matplotlib.pyplot as plt

    rho_by_t = {0: float(rho_atom_t0), 1: float(rho_atom_t1)}

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
    for t, ax in enumerate(axes):
        plt.sca(ax)  # spatial_spec's plot helpers call pyplot-level autoscale/axis.
        a.getObject(t).plot(ax=ax, label=False)
        b_dyn.getObject(t).plot(ax=ax, label=False)
        ax.set_title(f"t={t} (rho={rho_by_t[t]:+.3g})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    out = tmp_path / "spatial_spec_toy_trace.png"
    fig.savefig(out)
    plt.close(fig)

    assert out.is_file()
    assert out.stat().st_size > 0
