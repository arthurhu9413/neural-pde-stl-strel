# ruff: noqa: I001
from __future__ import annotations

"""neural_pde_stl_strel.frameworks.spatial_spec_hello

Import-safe helpers for the **SpaTiaL Specifications** Python package.

The upstream SpaTiaL project has multiple Python entry points. This helper is
specifically for the lightweight *PyPI* distribution:

- **PyPI distribution**: ``spatial-spec``
- **Import name**: ``spatial_spec``

This module intentionally keeps a tiny public API (see :data:`__all__`) because
it is used by CI smoke tests and environment checks.

If executed as a module (``python -m neural_pde_stl_strel.frameworks.spatial_spec_hello``),
it runs a minimal *actually-run* demo that:

1) constructs two rectangles over two time steps,
2) monitors one **eventually** and one **always** specification, and
3) saves a compact plot showing the trace.

This provides a convenient, reproducible figure for the repository/report.

Notes
-----
- We do *not* import ``spatial_spec.geometry`` at module import time. That
  submodule imports Matplotlib, which is undesirable for import-time side
  effects in CI and headless environments.
- The full SpaTiaL repository also ships a separate library (often installed
  from Git) that is imported as ``spatial`` (not ``spatial_spec``). This helper
  targets only the PyPI ``spatial-spec`` distribution.

References
- PyPI (``spatial-spec``): https://pypi.org/project/spatial-spec/
- Upstream repository: https://github.com/KTH-RPL-Planiacs/SpaTiaL

"""

from dataclasses import dataclass
from importlib import import_module, metadata as _metadata
from pathlib import Path
from typing import Any


SPATIAL_SPEC_DIST_NAME: str = "spatial-spec"
SPATIAL_SPEC_MODULE_NAME: str = "spatial_spec"

__all__ = [
    "SPATIAL_SPEC_DIST_NAME",
    "SPATIAL_SPEC_MODULE_NAME",
    "spatial_spec_version",
    "spatial_spec_available",
]


def _install_hint() -> str:
    # Keep "pip install" in the hint because tests look for that substring.
    return f"python -m pip install {SPATIAL_SPEC_DIST_NAME}"


def _missing_spatial_spec_error(*, missing_dep: str | None = None) -> ImportError:
    """Create a consistent ImportError with actionable installation help."""

    extra = ""
    if missing_dep:
        extra = (
            "\n\n"
            f"Note: `{SPATIAL_SPEC_MODULE_NAME}` was found, but a dependency failed to import: {missing_dep!r}. "
            "You may need to (re)install that dependency in your environment."
        )

    return ImportError(
        "SpaTiaL is not installed (or failed to import). "
        f"Install the PyPI distribution `{SPATIAL_SPEC_DIST_NAME}` "
        f"(module `{SPATIAL_SPEC_MODULE_NAME}`).\n"
        f"Example:  {_install_hint()}"
        f"{extra}"
    )


def _require_spatial_spec() -> Any:
    """Import and return :mod:`spatial_spec` with a friendly error on failure."""

    try:
        return import_module(SPATIAL_SPEC_MODULE_NAME)
    except ModuleNotFoundError as e:
        # If *our* module is missing, provide the standard install guidance.
        if getattr(e, "name", None) == SPATIAL_SPEC_MODULE_NAME:
            raise _missing_spatial_spec_error() from e
        # Otherwise, the distribution may be present but a dependency is missing.
        raise _missing_spatial_spec_error(missing_dep=getattr(e, "name", None) or str(e)) from e


def spatial_spec_version() -> str:
    """Return the installed SpaTiaL (``spatial-spec``) version string.

    Preference order:

    1) ``spatial_spec.__version__`` if present;
    2) ``importlib.metadata.version('spatial-spec')``;
    3) ``'unknown'``.

    Raises:
        ImportError: if the module cannot be imported.
    """

    mod = _require_spatial_spec()

    ver = getattr(mod, "__version__", None)
    if isinstance(ver, str) and ver.strip():
        return ver.strip()

    # Some tooling normalizes distribution names to underscores; try both.
    candidates = (SPATIAL_SPEC_DIST_NAME, SPATIAL_SPEC_DIST_NAME.replace("-", "_"))
    for dist_name in candidates:
        try:
            return _metadata.version(dist_name)
        except _metadata.PackageNotFoundError:
            continue
        except Exception:
            # Metadata backends can vary; treat any other failure as "unknown".
            continue

    return "unknown"


def spatial_spec_available() -> bool:
    """Return ``True`` if ``spatial_spec`` imports, else ``False``."""

    try:
        _require_spatial_spec()
    except ImportError:
        return False
    return True


# Optional: runnable, plot-producing "hello" demo


@dataclass(frozen=True)
class SpatialSpecHelloResult:
    """Outputs of :func:`run_spatial_spec_hello`.

    Attributes:
        phi_eventually: An STL-style eventually specification (finite horizon).
        phi_always: An STL-style always specification (finite horizon).
        robustness_eventually: Quantitative robustness of ``phi_eventually``.
        robustness_always: Quantitative robustness of ``phi_always``.
        plot_path: Where the demo plot was saved (or ``None`` if plotting disabled).
    """

    phi_eventually: str
    phi_always: str
    robustness_eventually: float
    robustness_always: float
    plot_path: Path | None


def run_spatial_spec_hello(
    *,
    out_path: str | Path | None = "spatial_spec_hello.png",
    show: bool = False,
    dpi: int = 200,
) -> SpatialSpecHelloResult:
    """Run a tiny end-to-end demo (monitor + plot) using ``spatial-spec``.

    The example is intentionally *toy-sized* for explanation:

    - ``plate`` is static.
    - ``fork`` moves from right-of to left-of the plate.

    We monitor two finite-horizon formulas over the 2-step trace:

    - Eventually: ``(F [0,1] (fork leftof plate))``  (satisfied)
    - Always:     ``(G [0,1] (fork leftof plate))``  (violated)

    Args:
        out_path: File path to save the plot. Set to ``None`` to disable plotting.
        show: If ``True``, attempt to display the plot (requires a GUI backend).
        dpi: DPI for the saved figure.

    Returns:
        A :class:`SpatialSpecHelloResult` with formulas, robustness, and plot path.

    Raises:
        ImportError: if ``spatial-spec`` (or its runtime deps) are not available.
    """

    # Import optional deps lazily (and in a headless-safe way).
    import os

    if not show:
        # Must be set *before* importing Matplotlib (spatial_spec.geometry imports pyplot).
        os.environ.setdefault("MPLBACKEND", "Agg")

    _require_spatial_spec()

    import numpy as np

    # These imports pull in Matplotlib and Shapely (via spatial_spec.geometry).
    from spatial_spec.geometry import DynamicObject, Polygon, PolygonCollection, StaticObject
    from spatial_spec.logic import Spatial

    # Rectangle helper (clockwise). Units are arbitrary.
    def rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
        verts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
        return Polygon(verts)

    plate = StaticObject(PolygonCollection([rect(-0.1, -0.05, 0.1, 0.05)]))

    fork = DynamicObject()
    fork.addObject(PolygonCollection([rect(0.25, -0.04, 0.45, 0.04)]), time=0)  # right of plate
    fork.addObject(PolygonCollection([rect(-0.45, -0.04, -0.25, 0.04)]), time=1)  # left of plate

    phi_eventually = "(F [0,1] (fork leftof plate))"
    phi_always = "(G [0,1] (fork leftof plate))"

    sp = Spatial(quantitative=True)
    sp.assign_variable("plate", plate)
    sp.assign_variable("fork", fork)

    rob_eventually = float(sp.interpret(sp.parse(phi_eventually)))
    rob_always = float(sp.interpret(sp.parse(phi_always)))

    plot_file: Path | None = None

    if out_path is not None:
        import matplotlib.pyplot as plt

        plot_file = Path(out_path)
        plot_file.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(ncols=2, figsize=(7.5, 3.25), constrained_layout=True)
        times = [0, 1]
        for ax, t in zip(axes, times, strict=True):
            ax.set_title(f"t = {t}")

            # Make sure SpaTiaL's plotting helpers target the intended axes.
            plt.sca(ax)

            plate.getObject(t).plot(ax=ax, color="C0", label=False)
            fork.getObject(t).plot(ax=ax, color="C1", label=False)

            # Annotate object labels at polygon centroids for readability.
            for name, obj in ("plate", plate), ("fork", fork):
                polys = list(obj.getObject(t).polygons)
                if not polys:
                    continue
                centers = np.vstack([p.center for p in polys])
                cx, cy = centers.mean(axis=0)
                ax.text(cx, cy, name, ha="center", va="center", fontsize=10, fontweight="bold")

            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.5, alpha=0.4)

        fig.suptitle(
            "SpaTiaL spatial-spec hello: eventually satisfied, always violated\n"
            f"rob(F)= {rob_eventually:.3g},  rob(G)= {rob_always:.3g}",
            fontsize=10,
        )

        fig.savefig(plot_file, dpi=int(dpi), bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return SpatialSpecHelloResult(
        phi_eventually=phi_eventually,
        phi_always=phi_always,
        robustness_eventually=rob_eventually,
        robustness_always=rob_always,
        plot_path=plot_file,
    )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="SpaTiaL (spatial-spec) import check + tiny runnable demo (monitor + plot)."
    )
    parser.add_argument(
        "--out",
        default="spatial_spec_hello.png",
        help="Where to save the demo plot (set to empty to disable).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window (requires a GUI backend).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for the saved plot (default: 200).",
    )

    args = parser.parse_args()

    try:
        _require_spatial_spec()
    except ImportError:
        print(f"{SPATIAL_SPEC_MODULE_NAME} available: False")
        raise

    print(f"{SPATIAL_SPEC_MODULE_NAME} available: True")
    print(f"{SPATIAL_SPEC_MODULE_NAME} version:   {spatial_spec_version()}")

    out: str | None = args.out
    if out.strip() == "":
        out = None

    res = run_spatial_spec_hello(out_path=out, show=bool(args.show), dpi=int(args.dpi))

    print("\nSpecifications monitored:")
    print(f"  eventually: {res.phi_eventually}  -> robustness = {res.robustness_eventually:.6g}")
    print(f"  always:     {res.phi_always}  -> robustness = {res.robustness_always:.6g}")

    if res.plot_path is not None:
        print(f"\nSaved plot: {res.plot_path}")


if __name__ == "__main__":
    _main()
