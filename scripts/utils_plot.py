# -*- coding: utf-8 -*-
from __future__ import annotations

"""scripts.utils_plot

Lightweight, dependency-minimal plotting helpers used across the
neural-pde-stl-strel repository.

Design goals
- **Robust I/O**: Accept NumPy arrays, PyTorch tensors, and Python sequences.
- **CPS-friendly**: Handle common shapes that arise in 1-D/2-D PDE demos and
  time-series traces (e.g., STL robustness signals).
- **Headless-safe**: Never call ``plt.show()``; always save to disk and close.
- **Fast**: Optionally downsample very large heatmaps to avoid rendering
  bottlenecks while preserving qualitative structure.
- **Pretty by default**: Sensible labels, titles, grids, and colorbars.

Notes
-----
- If the user has not explicitly selected a Matplotlib backend (via the
  ``MPLBACKEND`` environment variable), we default to the non-interactive
  ``Agg`` backend to avoid crashes on headless machines.

All public functions return the :class:`pathlib.Path` to the saved figure.
"""

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeAlias

import matplotlib

# Use a non-interactive backend by default (safe for CI/headless servers).
# Respect an explicitly configured backend (e.g., in notebooks).
if os.environ.get("MPLBACKEND") is None:
    try:  # pragma: no cover - backend selection depends on import order/environment
        matplotlib.use("Agg", force=True)
    except Exception:
        # Backend was likely chosen already (e.g., pyplot imported elsewhere).
        pass

import matplotlib.pyplot as plt
import numpy as np

# Optional torch dependency (commonly available in this repo)
try:  # pragma: no cover - don't require torch in minimal envs
    import torch  # type: ignore[import-not-found]

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# A permissive "array like" union used throughout this module.
if _HAS_TORCH:
    ArrayLike: TypeAlias = torch.Tensor | np.ndarray | Sequence[float] | Sequence[Sequence[float]]
else:
    ArrayLike: TypeAlias = np.ndarray | Sequence[float] | Sequence[Sequence[float]]


# Default output paths (kept as module constants so we can detect "default"
# when supporting legacy keyword aliases like out_path=...).
_DEFAULT_OUT_U_XT = "figs/diffusion_heatmap.png"
_DEFAULT_OUT_U_XY_FRAME = "figs/heat2d_t0.png"
_DEFAULT_OUT_TIME_SLICES = "figs/diffusion_slices.png"
_DEFAULT_OUT_MEAN_OVER_TIME = "figs/mean_over_time.png"


# Internals

def _pathify(path: str | Path) -> Path:
    """Return a normalized :class:`pathlib.Path`.

    We expand environment variables and ``~`` to reduce surprises when the
    caller supplies user paths.
    """
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Best-effort conversion to a NumPy array on CPU without gradients."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _as_1d_coord(x: ArrayLike, *, name: str) -> np.ndarray:
    """Convert a coordinate input into a 1-D NumPy array.

    Accepts:
    - 1-D arrays/vectors
    - row/column vectors with shape (n, 1) or (1, n)
    - meshgrids where one axis is repeated (common from ``np.meshgrid``)

    Raises a ValueError if a 1-D axis cannot be inferred.
    """
    a = _to_numpy(x)

    if a.ndim == 1:
        return a.reshape(-1)

    if a.ndim == 2:
        # Treat (n,1)/(1,n) as a vector.
        if 1 in a.shape and a.size == int(max(a.shape)):
            return a.reshape(-1)

        # Meshgrid heuristic: one dimension is constant along the other.
        # Example: X(x,t) is constant across columns -> use first column.
        # Example: T(x,t) is constant across rows    -> use first row.
        if np.allclose(a, a[:, [0]], equal_nan=True):
            return a[:, 0].reshape(-1)
        if np.allclose(a, a[[0], :], equal_nan=True):
            return a[0, :].reshape(-1)

        raise ValueError(
            f"{name} appears 2-D but is neither a vector nor an axis-repeated meshgrid; "
            f"shape={a.shape}"
        )

    raise ValueError(f"{name} must be 1-D or 2-D (got ndim={a.ndim}, shape={a.shape}).")


def _ensure_parent_dir(path: str | Path) -> None:
    """Create parent directory for *path* if needed (no-op otherwise)."""
    p = _pathify(path)
    if p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)


def _resolve_out(
    *,
    out: str | Path,
    out_path: str | Path | None,
    default_out: str | Path,
) -> Path:
    """Resolve legacy ``out_path=`` alias.

    Many scripts use the keyword ``out=...``. Some older/Makefile snippets use
    ``out_path=...``. This helper makes both work without editing other files.

    Rules:
    - If only ``out`` is provided: use it.
    - If ``out_path`` is provided and ``out`` is still the default: use
      ``out_path``.
    - If both are provided and resolve to different paths: raise.
    """
    out_p = _pathify(out)
    if out_path is None:
        return out_p

    out_path_p = _pathify(out_path)
    default_p = _pathify(default_out)

    if out_p != default_p and out_p != out_path_p:
        raise ValueError(
            "Specify only one of 'out' or 'out_path' (they resolve to different paths): "
            f"out={out_p!s}, out_path={out_path_p!s}"
        )

    return out_path_p


def _savefig(fig: plt.Figure, out: Path, *, dpi: int) -> None:
    """Save *fig* to *out* and always close the figure."""
    _ensure_parent_dir(out)
    try:
        # Use the OO API to avoid pyplot state leaks when generating many plots.
        fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    finally:
        plt.close(fig)


def _maybe_downsample(img: np.ndarray, *, max_elems: int) -> tuple[np.ndarray, int, int]:
    """Downsample a 2-D image if it has more than *max_elems* pixels.

    Returns the (possibly downsampled) image and the strides used along (row, col).
    """
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {img.shape}")

    rows, cols = img.shape
    elems = int(rows) * int(cols)
    if elems <= int(max_elems):
        return img, 1, 1

    # Choose an integer stride to get close to max_elems while preserving aspect.
    stride = max(1, int(np.ceil(np.sqrt(elems / float(max_elems)))))
    return img[::stride, ::stride], stride, stride


def _extent_from_coords(*, y: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(x_min, x_max, y_min, y_max)`` extent tuple for ``imshow``."""
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    return (x_min, x_max, y_min, y_max)


def _reshape_to_grid(u: np.ndarray, *, n_rows: int, n_cols: int) -> np.ndarray:
    """Arrange *u* into a ``(n_rows, n_cols)`` grid, accepting common alternatives.

    Acceptable inputs:
    - ``u.shape == (n_rows, n_cols)``
    - ``u.shape == (n_cols, n_rows)`` -> will be transposed
    - ``u`` is 1-D with size ``n_rows * n_cols`` -> reshaped row-major
    """
    if u.ndim == 2 and u.shape == (n_rows, n_cols):
        return u
    if u.ndim == 2 and u.shape == (n_cols, n_rows):
        return u.T
    if u.ndim == 1 and u.size == n_rows * n_cols:
        return u.reshape(n_rows, n_cols)
    raise ValueError(f"u with shape {u.shape} cannot be arranged to ({n_rows}, {n_cols}).")


def _sorted_with_index(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted array and the permutation indices that sort it ascending."""
    idx = np.argsort(a, kind="stable")
    return a[idx], idx


def _nearest_indices(sorted_vals: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Return indices of *sorted_vals* nearest to each query in *queries*.

    Assumes *sorted_vals* is 1-D and sorted ascending.
    """
    if sorted_vals.ndim != 1:
        raise ValueError("sorted_vals must be 1-D")
    if sorted_vals.size == 0:
        raise ValueError("sorted_vals must be non-empty")

    q = np.asarray(queries, dtype=float).reshape(-1)

    # Insertion points in [0, n]
    right = np.searchsorted(sorted_vals, q, side="left")

    # Candidate neighbors
    right = np.clip(right, 0, sorted_vals.size - 1)
    left = np.clip(right - 1, 0, sorted_vals.size - 1)

    dist_r = np.abs(sorted_vals[right] - q)
    dist_l = np.abs(sorted_vals[left] - q)

    use_left = dist_l <= dist_r
    out = np.where(use_left, left, right)
    return out.astype(int)


# Public API

def plot_u_xt(
    u: ArrayLike,
    x: ArrayLike,
    t: ArrayLike,
    *,
    out: str | Path = _DEFAULT_OUT_U_XT,
    out_path: str | Path | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    add_colorbar: bool = True,
    title: str = "Diffusion1D u(x,t)",
    max_elems: int = 2_000_000,
    interpolation: str = "nearest",
    dpi: int = 150,
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
    cbar_label: str = "u(x, t)",
    aspect: str = "auto",
    x_label: str = "x",
    t_label: str = "t",
) -> Path:
    """Heatmap for a scalar field :math:`u(x,t)`.

    Orientation convention
    This function renders **space on the horizontal axis** and **time on the
    vertical axis** (i.e., x is plotted left-to-right, t bottom-to-top). This
    matches the convention used in the repository's report/slide figures.

    Parameters
    u:
        Field values on a Cartesian product of space *x* and time *t*. Accepted
        shapes: ``(n_x, n_t)``, ``(n_t, n_x)``, or a flat vector of size
        ``n_x*n_t``.
    x:
        Spatial coordinates. May be 1-D (length ``n_x``) or a meshgrid with a
        repeated axis (e.g., ``X`` from ``np.meshgrid``).
    t:
        Time coordinates. May be 1-D (length ``n_t``) or a meshgrid with a
        repeated axis.
    out / out_path:
        Output image path. ``out_path`` is a legacy alias (kept so Makefile
        snippets continue to work).
    max_elems:
        If the rendered image exceeds this many pixels, we downsample with a
        uniform stride for responsiveness.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    out_p = _resolve_out(out=out, out_path=out_path, default_out=_DEFAULT_OUT_U_XT)

    u_np = _to_numpy(u)
    x_np = _as_1d_coord(x, name="x")
    t_np = _as_1d_coord(t, name="t")

    # Sort coordinates (robust to user passing non-monotonic arrays)
    x_sorted, ix = _sorted_with_index(x_np)
    t_sorted, it = _sorted_with_index(t_np)

    # Arrange u to (n_x, n_t) then permute to the sorted order.
    # After that, transpose for plotting so x is horizontal and t is vertical.
    grid_x_t = _reshape_to_grid(u_np, n_rows=x_np.size, n_cols=t_np.size)
    grid_x_t = grid_x_t[ix, :][:, it]
    grid_t_x = np.asarray(grid_x_t, dtype=float).T
    grid_t_x = np.ma.masked_invalid(grid_t_x)

    # Optional downsampling (preserve axes by also slicing coordinates).
    grid_ds, sy, sx = _maybe_downsample(grid_t_x, max_elems=int(max_elems))
    extent = _extent_from_coords(y=t_sorted[::sy], x=x_sorted[::sx])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        grid_ds,
        origin="lower",
        aspect=str(aspect),
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation=str(interpolation),
        cmap=cmap,
    )
    ax.set_xlabel(str(x_label))
    ax.set_ylabel(str(t_label))
    ax.set_title(str(title))

    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if cbar_label:
            cbar.set_label(str(cbar_label))

    fig.tight_layout()
    _savefig(fig, out_p, dpi=int(dpi))
    return out_p


def plot_u_xy_frame(
    u_xy: ArrayLike,
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    out: str | Path = _DEFAULT_OUT_U_XY_FRAME,
    out_path: str | Path | None = None,
    add_colorbar: bool = True,
    title: str = "2D field u(x,y)",
    dpi: int = 150,
    interpolation: str = "nearest",
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
    cbar_label: str = "u(x, y)",
    aspect: str | None = None,
    max_elems: int | None = None,
) -> Path:
    """Render a single 2-D scalar frame :math:`u(x,y)`.

    If *x* and *y* are provided, we use their sizes to infer the ``(n_y, n_x)``
    shape and build an ``extent`` so axes reflect physical units. If they are
    omitted, the input is interpreted as an image grid. A 1-D input is reshaped
    to a square grid when possible.

    Parameters
    max_elems:
        Optional pixel budget; if provided and the grid exceeds this size, we
        downsample for faster rendering.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    out_p = _resolve_out(out=out, out_path=out_path, default_out=_DEFAULT_OUT_U_XY_FRAME)

    u_np = _to_numpy(u_xy)

    extent = None
    if x is not None and y is not None:
        x_np = _as_1d_coord(x, name="x")
        y_np = _as_1d_coord(y, name="y")

        # Sort coordinates for robustness and align grid accordingly.
        x_sorted, ix = _sorted_with_index(x_np)
        y_sorted, iy = _sorted_with_index(y_np)

        grid = _reshape_to_grid(u_np, n_rows=y_np.size, n_cols=x_np.size)
        grid = grid[iy, :][:, ix]
        extent = _extent_from_coords(y=y_sorted, x=x_sorted)
    else:
        # Keep as-is if already 2-D, otherwise attempt a square reshape.
        if u_np.ndim == 2:
            grid = u_np
        else:
            side = int(round(np.sqrt(u_np.size)))
            if side * side != int(u_np.size):
                raise ValueError(
                    "u_xy must be 2-D or have perfect-square size when x/y are omitted. "
                    f"Got size={u_np.size}."
                )
            grid = u_np.reshape(side, side)

    grid = np.ma.masked_invalid(np.asarray(grid, dtype=float))

    if max_elems is not None:
        grid, _, _ = _maybe_downsample(grid, max_elems=int(max_elems))

    aspect_used = aspect
    if aspect_used is None:
        # For physical coordinate plots, preserve geometry; for pixel images, fill the Axes.
        aspect_used = "equal" if extent is not None else "auto"

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        grid,
        origin="lower",
        aspect=str(aspect_used),
        extent=extent,  # type: ignore[arg-type]
        interpolation=str(interpolation),
        cmap=cmap,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(str(title))

    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if cbar_label:
            cbar.set_label(str(cbar_label))

    fig.tight_layout()
    _savefig(fig, out_p, dpi=int(dpi))
    return out_p


def plot_time_slices(
    u: ArrayLike,
    x: ArrayLike,
    t: ArrayLike,
    *,
    times: Iterable[float] | None = None,
    num_slices: int = 4,
    out: str | Path = _DEFAULT_OUT_TIME_SLICES,
    out_path: str | Path | None = None,
    u_bounds: tuple[float | None, float | None] | None = None,
    title: str = "u(x,t) at selected times",
    dpi: int = 150,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Plot spatial slices :math:`u(x, t_k)` at chosen time instants.

    If *times* is ``None`` we pick *num_slices* evenly spaced indices. Otherwise
    we select the nearest index for each requested time (after sorting *t*).

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    out_p = _resolve_out(out=out, out_path=out_path, default_out=_DEFAULT_OUT_TIME_SLICES)

    u_np = _to_numpy(u)
    x_np = _as_1d_coord(x, name="x")
    t_np = _as_1d_coord(t, name="t")

    x_sorted, ix = _sorted_with_index(x_np)
    t_sorted, it = _sorted_with_index(t_np)

    U = _reshape_to_grid(u_np, n_rows=x_np.size, n_cols=t_np.size)
    U = U[ix, :][:, it]  # align with sorted coordinates

    if t_sorted.size == 0:
        raise ValueError("t must be non-empty")

    if times is None:
        idxs = np.linspace(0, t_sorted.size - 1, num=int(num_slices), dtype=int)
    else:
        times_arr = np.asarray(list(times), dtype=float)
        idxs = _nearest_indices(t_sorted, times_arr)

    # Preserve requested order but avoid duplicate legend entries.
    ordered_unique: list[int] = []
    seen: set[int] = set()
    for i in idxs.tolist():
        ii = int(i)
        if ii not in seen:
            ordered_unique.append(ii)
            seen.add(ii)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for k in ordered_unique:
        ax.plot(x_sorted, U[:, k], label=f"t={t_sorted[k]:.3g}")

    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(str(title))

    if u_bounds is not None:
        umin, umax = u_bounds
        if umin is not None:
            ax.axhline(float(umin), linestyle="--", linewidth=1.0, alpha=0.6, label="u_min")
        if umax is not None:
            ax.axhline(float(umax), linestyle="--", linewidth=1.0, alpha=0.6, label="u_max")

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", framealpha=0.8, fontsize="small")

    fig.tight_layout()
    _savefig(fig, out_p, dpi=int(dpi))
    return out_p


def plot_spatial_mean_over_time(
    u: ArrayLike,
    t: ArrayLike,
    *,
    mean_dims: tuple[int, ...] | None = None,
    time_dim: int = -1,
    out: str | Path = _DEFAULT_OUT_MEAN_OVER_TIME,
    out_path: str | Path | None = None,
    u_max: float | None = None,
    var_name: str = "u",
    title: str | None = None,
    dpi: int = 150,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Plot the spatial mean of *u* against time.

    By default we average over all axes **except** ``time_dim`` (which defaults
    to ``-1``). You can override this by specifying ``mean_dims`` directly.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    out_p = _resolve_out(out=out, out_path=out_path, default_out=_DEFAULT_OUT_MEAN_OVER_TIME)

    u_np = _to_numpy(u)
    t_np = _as_1d_coord(t, name="t")

    if u_np.ndim == 0:
        raise ValueError("u must not be a scalar")

    time_dim_norm = int(time_dim) % int(u_np.ndim)

    if mean_dims is None:
        mean_axes = tuple(i for i in range(u_np.ndim) if i != time_dim_norm)
    else:
        mean_axes = tuple(int(i) for i in mean_dims)

    series = u_np.mean(axis=mean_axes)
    series = np.asarray(series, dtype=float).reshape(-1)

    if series.size != t_np.size:
        raise ValueError(
            "Cannot align mean series with provided time vector: "
            f"mean_series.size={series.size}, t.size={t_np.size}"
        )

    # Sort time for nicer plots, and reorder the series accordingly.
    t_sorted, it = _sorted_with_index(t_np)
    series = series[it]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(t_sorted, series)
    ax.set_xlabel("t")
    ax.set_ylabel(f"mean {var_name}")

    if title is None:
        title = f"Temporal evolution of mean {var_name}"
    ax.set_title(str(title))

    if u_max is not None:
        ax.axhline(float(u_max), linestyle="--", linewidth=1.0, alpha=0.6, label=f"{var_name}_max")
        ax.legend(loc="best", framealpha=0.8, fontsize="small")

    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_p, dpi=int(dpi))
    return out_p


# Backwards-compatible aliases (if other scripts import the old names)

def plot_u_1d(
    u: ArrayLike,
    X: ArrayLike,
    T: ArrayLike,
    out: str | Path = _DEFAULT_OUT_U_XT,
    out_path: str | Path | None = None,
) -> Path:
    """Alias kept for backward compatibility: calls :func:`plot_u_xt`."""
    return plot_u_xt(u=u, x=X, t=T, out=out, out_path=out_path)


def plot_u_2d_frame(
    u_frame: ArrayLike,
    out: str | Path = _DEFAULT_OUT_U_XY_FRAME,
    out_path: str | Path | None = None,
) -> Path:
    """Alias kept for backward compatibility: calls :func:`plot_u_xy_frame`."""
    return plot_u_xy_frame(u_xy=u_frame, out=out, out_path=out_path)
