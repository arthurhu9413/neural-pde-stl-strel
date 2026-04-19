"""Grid and sampling utilities.

This module standardizes *where* we evaluate models over continuous space(-time)
boxes. It is used for both:

* **Training**: sampling interior / boundary points for PINN-style residual and
  constraint losses.
* **Monitoring**: building regular tensor-product grids used to approximate
  continuous spatial/temporal quantifiers in STL / STREL monitors.

Ordering invariants
The coordinate tables returned by :func:`grid1d`, :func:`grid2d`, and
:func:`grid3d` are ordered so that the *last* axis varies fastest (time is last).
This matches :func:`torch.cartesian_prod` / :func:`itertools.product` ordering
and makes reshapes like

    u = model(XT).reshape(X.shape)

correct when ``XT`` and ``X`` come from the same grid call.

Note
----
Continuous spatial quantifiers (e.g., ``max_x u(x, t)``) are approximated on a
finite grid. Choose monitoring grid resolutions accordingly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

Tensor = torch.Tensor
DeviceLike = torch.device | str


# Utilities


def _resolve_dtype(dtype: torch.dtype | None) -> torch.dtype:
    """Resolve ``None`` to the current default dtype."""
    return torch.get_default_dtype() if dtype is None else dtype


def _validate_floating_dtype(dtype: torch.dtype) -> None:
    """Ensure ``dtype`` is a real floating-point dtype."""
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be a floating point type (got {dtype}).")


def _validate_nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative (got {value}).")


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive (got {value}).")


def _validate_interval(name: str, lo: float, hi: float) -> None:
    """Validate a numeric interval ``[lo, hi]`` (finite and not inverted)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{name}_min and {name}_max must be finite (got {lo}, {hi}).")
    if hi < lo:
        raise ValueError(f"{name}_max must be >= {name}_min (got {lo}..{hi}).")


def _as_tensor(x: float | Tensor, *, device: DeviceLike, dtype: torch.dtype) -> Tensor:
    """Return ``x`` as a 0-D tensor on ``device``/``dtype``.

    If ``x`` is already a tensor, it will be moved/cast as needed.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _linspace(
    min_v: float,
    max_v: float,
    n: int,
    *,
    device: DeviceLike,
    dtype: torch.dtype,
) -> Tensor:
    """A small validated wrapper around :func:`torch.linspace`.

    Notes:
        ``torch.linspace`` is inclusive by default; we rely on this to include both
        endpoints in monitoring grids.
    """
    _validate_positive_int("n", n)
    return torch.linspace(min_v, max_v, n, device=device, dtype=dtype)


def _stack_flat(*meshes: Tensor) -> Tensor:
    """Flatten meshgrid tensors to an ``(N, d)`` coordinate table."""
    if not meshes:
        raise ValueError("No meshes provided.")
    shape = meshes[0].shape
    if any(m.shape != shape for m in meshes[1:]):
        raise ValueError("All meshes must have the same shape.")
    flat_cols = [m.reshape(-1) for m in meshes]
    return torch.stack(flat_cols, dim=-1)


# Public API - Regular grids


def grid1d(
    n_x: int = 128,
    n_t: int = 100,
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    return_cartesian: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Construct a 1-D space x time grid.

    Returns:
        (X, T, XT):
            * X, T have shape ``(n_x, n_t)`` (created with ``indexing='ij'``).
            * XT is an ``(n_x * n_t, 2)`` table of coordinates ``[x, t]``.

        When ``return_cartesian`` is True, ``XT`` is built with
        :func:`torch.cartesian_prod`. Otherwise, it is a flattened stack of the
        mesh tensors (identical values and ordering, different memory layout).
    """
    _validate_positive_int("n_x", n_x)
    _validate_positive_int("n_t", n_t)
    _validate_interval("x", x_min, x_max)
    _validate_interval("t", t_min, t_max)

    dtype = _resolve_dtype(dtype)
    _validate_floating_dtype(dtype)
    x = _linspace(x_min, x_max, n_x, device=device, dtype=dtype)
    t = _linspace(t_min, t_max, n_t, device=device, dtype=dtype)

    # NOTE: With indexing='ij', X.shape == T.shape == (n_x, n_t) and the last
    # axis (time) varies fastest in the flattened representation. This matches
    # torch.cartesian_prod(x, t) ordering.
    X, T = torch.meshgrid(x, t, indexing="ij")
    XT = torch.cartesian_prod(x, t) if return_cartesian else _stack_flat(X, T)
    return X, T, XT


def grid2d(
    n_x: int = 64,
    n_y: int = 64,
    n_t: int = 50,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    return_cartesian: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Construct a 2-D (x, y) x time grid.

    Returns:
        (X, Y, T, XYT):
            * X, Y, T have shape ``(n_x, n_y, n_t)``.
            * XYT is an ``(n_x * n_y * n_t, 3)`` table of coordinates
              ``[x, y, t]``.

        When ``return_cartesian`` is True, ``XYT`` is built with
        :func:`torch.cartesian_prod`. Otherwise, it is the flattened stack of the
        mesh tensors (identical values and ordering, different memory layout).
    """
    _validate_positive_int("n_x", n_x)
    _validate_positive_int("n_y", n_y)
    _validate_positive_int("n_t", n_t)
    _validate_interval("x", x_min, x_max)
    _validate_interval("y", y_min, y_max)
    _validate_interval("t", t_min, t_max)

    dtype = _resolve_dtype(dtype)
    _validate_floating_dtype(dtype)
    x = _linspace(x_min, x_max, n_x, device=device, dtype=dtype)
    y = _linspace(y_min, y_max, n_y, device=device, dtype=dtype)
    t = _linspace(t_min, t_max, n_t, device=device, dtype=dtype)

    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    XYT = torch.cartesian_prod(x, y, t) if return_cartesian else _stack_flat(X, Y, T)
    return X, Y, T, XYT


def grid3d(
    n_x: int = 32,
    n_y: int = 32,
    n_z: int = 32,
    n_t: int = 20,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    z_min: float = 0.0,
    z_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    return_cartesian: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Construct a 3-D (x, y, z) x time grid.

    Returns:
        (X, Y, Z, T, XYZT):
            * X, Y, Z, T have shape ``(n_x, n_y, n_z, n_t)``.
            * XYZT is an ``(n_x * n_y * n_z * n_t, 4)`` table of coordinates
              ``[x, y, z, t]``.

        When ``return_cartesian`` is True, ``XYZT`` is built with
        :func:`torch.cartesian_prod`. Otherwise, it is the flattened stack of the
        mesh tensors (identical values and ordering, different memory layout).
    """
    _validate_positive_int("n_x", n_x)
    _validate_positive_int("n_y", n_y)
    _validate_positive_int("n_z", n_z)
    _validate_positive_int("n_t", n_t)
    _validate_interval("x", x_min, x_max)
    _validate_interval("y", y_min, y_max)
    _validate_interval("z", z_min, z_max)
    _validate_interval("t", t_min, t_max)

    dtype = _resolve_dtype(dtype)
    _validate_floating_dtype(dtype)
    x = _linspace(x_min, x_max, n_x, device=device, dtype=dtype)
    y = _linspace(y_min, y_max, n_y, device=device, dtype=dtype)
    z = _linspace(z_min, z_max, n_z, device=device, dtype=dtype)
    t = _linspace(t_min, t_max, n_t, device=device, dtype=dtype)

    X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing="ij")
    XYZT = torch.cartesian_prod(x, y, z, t) if return_cartesian else _stack_flat(X, Y, Z, T)
    return X, Y, Z, T, XYZT


# Spacing


def spacing1d(
    n_x: int,
    n_t: int,
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    *,
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Return grid spacings ``(dx, dt)`` for a 1-D (x, t) tensor grid.

    Notes:
        The standard definition is ``dx = (x_max - x_min) / (n_x - 1)`` for
        ``n_x >= 2`` (same for ``dt``). When ``n_x == 1`` (degenerate grid), we
        return ``x_max - x_min`` to avoid division-by-zero.
    """
    _validate_positive_int("n_x", n_x)
    _validate_positive_int("n_t", n_t)
    _validate_interval("x", x_min, x_max)
    _validate_interval("t", t_min, t_max)

    dtype = _resolve_dtype(dtype)
    _validate_floating_dtype(dtype)
    dx = _as_tensor((x_max - x_min) / max(n_x - 1, 1), device=device, dtype=dtype)
    dt = _as_tensor((t_max - t_min) / max(n_t - 1, 1), device=device, dtype=dtype)
    return dx, dt


def spacing2d(
    n_x: int,
    n_y: int,
    n_t: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    t_min: float,
    t_max: float,
    *,
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return grid spacings ``(dx, dy, dt)`` for a 2-D (x, y, t) tensor grid."""
    _validate_positive_int("n_x", n_x)
    _validate_positive_int("n_y", n_y)
    _validate_positive_int("n_t", n_t)
    _validate_interval("x", x_min, x_max)
    _validate_interval("y", y_min, y_max)
    _validate_interval("t", t_min, t_max)

    dtype = _resolve_dtype(dtype)
    _validate_floating_dtype(dtype)
    dx = _as_tensor((x_max - x_min) / max(n_x - 1, 1), device=device, dtype=dtype)
    dy = _as_tensor((y_max - y_min) / max(n_y - 1, 1), device=device, dtype=dtype)
    dt = _as_tensor((t_max - t_min) / max(n_t - 1, 1), device=device, dtype=dtype)
    return dx, dy, dt


def spacing3d(
    n_x: int,
    n_y: int,
    n_z: int,
    n_t: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    t_min: float,
    t_max: float,
    *,
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return grid spacings ``(dx, dy, dz, dt)`` for a 3-D (x, y, z, t) grid."""
    _validate_positive_int("n_x", n_x)
    _validate_positive_int("n_y", n_y)
    _validate_positive_int("n_z", n_z)
    _validate_positive_int("n_t", n_t)
    _validate_interval("x", x_min, x_max)
    _validate_interval("y", y_min, y_max)
    _validate_interval("z", z_min, z_max)
    _validate_interval("t", t_min, t_max)

    dtype = _resolve_dtype(dtype)
    _validate_floating_dtype(dtype)
    dx = _as_tensor((x_max - x_min) / max(n_x - 1, 1), device=device, dtype=dtype)
    dy = _as_tensor((y_max - y_min) / max(n_y - 1, 1), device=device, dtype=dtype)
    dz = _as_tensor((z_max - z_min) / max(n_z - 1, 1), device=device, dtype=dtype)
    dt = _as_tensor((t_max - t_min) / max(n_t - 1, 1), device=device, dtype=dtype)
    return dx, dy, dz, dt


# Random / quasi-random samplers


_SampleMethod = Literal["sobol", "uniform", "rand", "random", "lhs"]


def _lhs_unit_samples(num: int, dim: int, *, seed: int | None, dtype: torch.dtype) -> Tensor:
    """Vanilla Latin hypercube samples in ``[0, 1) ** dim`` as a CPU tensor.

    For each dimension independently, draws one sample in each of ``num``
    equiprobable strata and randomly permutes their order. This is a simple
    implementation intended for lightweight PINN sampling.

    Reproducibility:
        * If ``seed`` is provided, we use a dedicated CPU generator seeded with
          that value.
        * If ``seed`` is ``None``, we use PyTorch's global RNG state, so calls to
          :func:`torch.manual_seed` control determinism.
    """
    _validate_nonnegative_int("num", num)
    _validate_nonnegative_int("dim", dim)

    _validate_floating_dtype(dtype)

    if num == 0:
        return torch.empty(0, dim, dtype=dtype)
    if dim == 0:
        return torch.empty(num, 0, dtype=dtype)

    if seed is None:
        eps = torch.rand(num, dim, dtype=dtype)
        cols = []
        for j in range(dim):
            perm = torch.randperm(num).to(dtype=dtype)
            cols.append((perm + eps[:, j]) / float(num))
        return torch.stack(cols, dim=1)

    g = torch.Generator()
    g.manual_seed(int(seed))
    eps = torch.rand(num, dim, generator=g, dtype=dtype)
    cols = []
    for j in range(dim):
        perm = torch.randperm(num, generator=g).to(dtype=dtype)
        cols.append((perm + eps[:, j]) / float(num))
    return torch.stack(cols, dim=1)


def _unit_samples(
    num: int,
    dim: int,
    *,
    method: _SampleMethod,
    device: DeviceLike,
    dtype: torch.dtype,
    seed: int | None,
) -> Tensor:
    """Draw samples in the unit hypercube ``[0, 1) ** dim``.

    Supported methods:
        * ``"sobol"``: scrambled Sobol low-discrepancy sequence.
        * ``"uniform"`` / ``"rand"`` / ``"random"``: i.i.d. uniform samples.
        * ``"lhs"``: Latin hypercube sampling.

    Notes:
        Samples are generated on CPU for consistent behavior across devices and
        then moved to ``device``/``dtype``.
    """
    _validate_nonnegative_int("num", num)
    _validate_nonnegative_int("dim", dim)

    _validate_floating_dtype(dtype)
    work_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32

    if dim == 0:
        return torch.empty(num, 0, device=device, dtype=dtype)
    if num == 0:
        return torch.empty(0, dim, device=device, dtype=dtype)

    m = method.strip().lower()
    if m not in {"sobol", "uniform", "rand", "random", "lhs"}:
        raise ValueError("method must be one of {'sobol','uniform','rand','random','lhs'}")

    if m == "sobol":
        engine = torch.quasirandom.SobolEngine(
            dimension=dim,
            scramble=True,
            seed=None if seed is None else int(seed),
        )
        # SobolEngine.draw supports a dtype kwarg in newer PyTorch versions.
        try:
            u_cpu = engine.draw(num, dtype=work_dtype)
        except TypeError:
            u_cpu = engine.draw(num)
    elif m in {"uniform", "rand", "random"}:
        if seed is None:
            u_cpu = torch.rand(num, dim, dtype=work_dtype)
        else:
            g = torch.Generator()
            g.manual_seed(int(seed))
            u_cpu = torch.rand(num, dim, generator=g, dtype=work_dtype)
    else:  # LHS
        u_cpu = _lhs_unit_samples(num, dim, seed=seed, dtype=work_dtype)

    return u_cpu.to(device=device, dtype=dtype)

def sample_interior_1d(
    n: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    method: _SampleMethod = "sobol",
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    seed: int | None = None,
) -> Tensor:
    """Sample ``n`` points in a 1-D (x, t) box.

    Returns:
        Tensor of shape ``(n, 2)`` with rows ``[x, t]``.
    """
    _validate_nonnegative_int("n", n)
    _validate_interval("x", x_min, x_max)
    _validate_interval("t", t_min, t_max)

    dtype = _resolve_dtype(dtype)
    _validate_floating_dtype(dtype)
    if n == 0:
        return torch.empty(0, 2, device=device, dtype=dtype)

    u = _unit_samples(n, 2, method=method, device=device, dtype=dtype, seed=seed)
    mins = torch.tensor([x_min, t_min], device=device, dtype=dtype)
    maxs = torch.tensor([x_max, t_max], device=device, dtype=dtype)
    return mins + u * (maxs - mins)


def sample_interior_2d(
    n: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    method: _SampleMethod = "sobol",
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    seed: int | None = None,
) -> Tensor:
    """Sample ``n`` points in a 2-D (x, y, t) box.

    Returns:
        Tensor of shape ``(n, 3)`` with rows ``[x, y, t]``.
    """
    _validate_nonnegative_int("n", n)
    _validate_interval("x", x_min, x_max)
    _validate_interval("y", y_min, y_max)
    _validate_interval("t", t_min, t_max)

    dtype = _resolve_dtype(dtype)
    _validate_floating_dtype(dtype)
    if n == 0:
        return torch.empty(0, 3, device=device, dtype=dtype)

    u = _unit_samples(n, 3, method=method, device=device, dtype=dtype, seed=seed)
    mins = torch.tensor([x_min, y_min, t_min], device=device, dtype=dtype)
    maxs = torch.tensor([x_max, y_max, t_max], device=device, dtype=dtype)
    return mins + u * (maxs - mins)


def sample_interior_3d(
    n: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    z_min: float = 0.0,
    z_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    method: _SampleMethod = "sobol",
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    seed: int | None = None,
) -> Tensor:
    """Sample ``n`` points in a 3-D (x, y, z, t) box.

    Returns:
        Tensor of shape ``(n, 4)`` with rows ``[x, y, z, t]``.
    """
    _validate_nonnegative_int("n", n)
    _validate_interval("x", x_min, x_max)
    _validate_interval("y", y_min, y_max)
    _validate_interval("z", z_min, z_max)
    _validate_interval("t", t_min, t_max)

    dtype = _resolve_dtype(dtype)
    _validate_floating_dtype(dtype)
    if n == 0:
        return torch.empty(0, 4, device=device, dtype=dtype)

    u = _unit_samples(n, 4, method=method, device=device, dtype=dtype, seed=seed)
    mins = torch.tensor([x_min, y_min, z_min, t_min], device=device, dtype=dtype)
    maxs = torch.tensor([x_max, y_max, z_max, t_max], device=device, dtype=dtype)
    return mins + u * (maxs - mins)

def _largest_remainder_counts(n_total: int, weights: Sequence[float]) -> list[int]:
    """Allocate integer counts that sum to ``n_total`` via largest remainders."""
    _validate_nonnegative_int("n_total", n_total)
    if len(weights) == 0:
        return []

    w = torch.tensor(weights, dtype=torch.float64)
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be non-negative finite numbers")

    s = float(w.sum().item())
    if s <= 0.0:
        raise ValueError("sum of weights must be positive")

    w = w / s
    exact = w * float(n_total)
    base = exact.floor().to(dtype=torch.int64)

    counts = base.tolist()
    remaining = int(n_total - int(base.sum().item()))

    if remaining > 0:
        # Distribute the remaining points to the largest fractional remainders.
        frac = (exact - base).tolist()
        order = sorted(range(len(frac)), key=lambda i: frac[i], reverse=True)
        for i in range(remaining):
            counts[order[i % len(order)]] += 1

    return counts


def sample_boundary_1d(
    n_total: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    method: _SampleMethod = "sobol",
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    seed: int | None = None,
) -> Tensor:
    """Sample boundary points on the 1-D spatial boundary x time.

    We split ``n_total`` between the left (``x = x_min``) and right
    (``x = x_max``) faces.

    Returns:
        Tensor of shape ``(n_total, 2)`` with rows ``[x, t]``.
    """
    _validate_nonnegative_int("n_total", n_total)
    _validate_interval("x", x_min, x_max)
    _validate_interval("t", t_min, t_max)

    dtype = _resolve_dtype(dtype)
    _validate_floating_dtype(dtype)
    if n_total == 0:
        return torch.empty(0, 2, device=device, dtype=dtype)

    n_left = n_total // 2
    n_right = n_total - n_left

    u_left = _unit_samples(n_left, 1, method=method, device=device, dtype=dtype, seed=seed)
    u_right = _unit_samples(
        n_right,
        1,
        method=method,
        device=device,
        dtype=dtype,
        seed=None if seed is None else seed + 1,
    )

    t_left = t_min + u_left * (t_max - t_min)
    t_right = t_min + u_right * (t_max - t_min)

    x_left = torch.full((n_left, 1), x_min, device=device, dtype=dtype)
    x_right = torch.full((n_right, 1), x_max, device=device, dtype=dtype)

    left = torch.cat([x_left, t_left], dim=1)
    right = torch.cat([x_right, t_right], dim=1)
    return torch.cat([left, right], dim=0)


def sample_boundary_2d(
    n_total: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    method: _SampleMethod = "sobol",
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    seed: int | None = None,
    split: Sequence[float] | None = None,
) -> Tensor:
    """Sample boundary points on the 2-D rectangular boundary x time.

    Faces (in order):
        1) left   (x = x_min)
        2) right  (x = x_max)
        3) bottom (y = y_min)
        4) top    (y = y_max)

    Args:
        n_total: Total number of samples over all faces.
        split: Optional length-4 sequence of non-negative weights controlling how
            many samples are allocated to each face. If omitted, uses equal
            weights.

    Returns:
        Tensor of shape ``(n_total, 3)`` with rows ``[x, y, t]``.
    """
    _validate_nonnegative_int("n_total", n_total)
    _validate_interval("x", x_min, x_max)
    _validate_interval("y", y_min, y_max)
    _validate_interval("t", t_min, t_max)

    dtype = _resolve_dtype(dtype)
    _validate_floating_dtype(dtype)
    if n_total == 0:
        return torch.empty(0, 3, device=device, dtype=dtype)

    if split is None:
        split = (0.25, 0.25, 0.25, 0.25)
    if len(split) != 4:
        raise ValueError("split must have length 4 for the 4 boundary faces")

    counts = _largest_remainder_counts(n_total, split)

    def _face_samples(n: int, seed_shift: int) -> Tensor:
        # Each face samples (space_along_face, t) from [0,1)^2.
        return _unit_samples(
            n,
            2,
            method=method,
            device=device,
            dtype=dtype,
            seed=None if seed is None else seed + seed_shift,
        )

    # left: x fixed; sample y and t
    u_left = _face_samples(counts[0], 0)
    y_left = y_min + u_left[:, 0:1] * (y_max - y_min)
    t_left = t_min + u_left[:, 1:2] * (t_max - t_min)
    left = torch.cat(
        [
            torch.full((counts[0], 1), x_min, device=device, dtype=dtype),
            y_left,
            t_left,
        ],
        dim=1,
    )

    # right: x fixed; sample y and t
    u_right = _face_samples(counts[1], 1)
    y_right = y_min + u_right[:, 0:1] * (y_max - y_min)
    t_right = t_min + u_right[:, 1:2] * (t_max - t_min)
    right = torch.cat(
        [
            torch.full((counts[1], 1), x_max, device=device, dtype=dtype),
            y_right,
            t_right,
        ],
        dim=1,
    )

    # bottom: y fixed; sample x and t
    u_bottom = _face_samples(counts[2], 2)
    x_bottom = x_min + u_bottom[:, 0:1] * (x_max - x_min)
    t_bottom = t_min + u_bottom[:, 1:2] * (t_max - t_min)
    bottom = torch.cat(
        [
            x_bottom,
            torch.full((counts[2], 1), y_min, device=device, dtype=dtype),
            t_bottom,
        ],
        dim=1,
    )

    # top: y fixed; sample x and t
    u_top = _face_samples(counts[3], 3)
    x_top = x_min + u_top[:, 0:1] * (x_max - x_min)
    t_top = t_min + u_top[:, 1:2] * (t_max - t_min)
    top = torch.cat(
        [
            x_top,
            torch.full((counts[3], 1), y_max, device=device, dtype=dtype),
            t_top,
        ],
        dim=1,
    )

    return torch.cat([left, right, bottom, top], dim=0)


def sample_boundary_3d(
    n_total: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    z_min: float = 0.0,
    z_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    method: _SampleMethod = "sobol",
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    seed: int | None = None,
    split: Sequence[float] | None = None,
) -> Tensor:
    """Sample boundary points on a 3-D box boundary x time (6 faces).

    Faces (order): x-min, x-max, y-min, y-max, z-min, z-max.

    Returns:
        Tensor of shape ``(n_total, 4)`` with rows ``[x, y, z, t]``.
    """
    _validate_nonnegative_int("n_total", n_total)
    _validate_interval("x", x_min, x_max)
    _validate_interval("y", y_min, y_max)
    _validate_interval("z", z_min, z_max)
    _validate_interval("t", t_min, t_max)

    dtype = _resolve_dtype(dtype)
    _validate_floating_dtype(dtype)
    if n_total == 0:
        return torch.empty(0, 4, device=device, dtype=dtype)

    if split is None:
        split = (1.0 / 6.0,) * 6
    if len(split) != 6:
        raise ValueError("split must have length 6 for the 6 boundary faces")

    counts = _largest_remainder_counts(n_total, split)

    def _face_samples(n: int, seed_shift: int) -> Tensor:
        # Each face samples (space1, space2, t) from [0,1)^3.
        return _unit_samples(
            n,
            3,
            method=method,
            device=device,
            dtype=dtype,
            seed=None if seed is None else seed + seed_shift,
        )

    # x-min face: x fixed; sample y, z, t
    u = _face_samples(counts[0], 0)
    y = y_min + u[:, 0:1] * (y_max - y_min)
    z = z_min + u[:, 1:2] * (z_max - z_min)
    t = t_min + u[:, 2:3] * (t_max - t_min)
    xmin = torch.cat(
        [
            torch.full((counts[0], 1), x_min, device=device, dtype=dtype),
            y,
            z,
            t,
        ],
        dim=1,
    )

    # x-max face
    u = _face_samples(counts[1], 1)
    y = y_min + u[:, 0:1] * (y_max - y_min)
    z = z_min + u[:, 1:2] * (z_max - z_min)
    t = t_min + u[:, 2:3] * (t_max - t_min)
    xmax = torch.cat(
        [
            torch.full((counts[1], 1), x_max, device=device, dtype=dtype),
            y,
            z,
            t,
        ],
        dim=1,
    )

    # y-min face: y fixed; sample x, z, t
    u = _face_samples(counts[2], 2)
    x = x_min + u[:, 0:1] * (x_max - x_min)
    z = z_min + u[:, 1:2] * (z_max - z_min)
    t = t_min + u[:, 2:3] * (t_max - t_min)
    ymin = torch.cat(
        [
            x,
            torch.full((counts[2], 1), y_min, device=device, dtype=dtype),
            z,
            t,
        ],
        dim=1,
    )

    # y-max face
    u = _face_samples(counts[3], 3)
    x = x_min + u[:, 0:1] * (x_max - x_min)
    z = z_min + u[:, 1:2] * (z_max - z_min)
    t = t_min + u[:, 2:3] * (t_max - t_min)
    ymax = torch.cat(
        [
            x,
            torch.full((counts[3], 1), y_max, device=device, dtype=dtype),
            z,
            t,
        ],
        dim=1,
    )

    # z-min face: z fixed; sample x, y, t
    u = _face_samples(counts[4], 4)
    x = x_min + u[:, 0:1] * (x_max - x_min)
    y = y_min + u[:, 1:2] * (y_max - y_min)
    t = t_min + u[:, 2:3] * (t_max - t_min)
    zmin = torch.cat(
        [
            x,
            y,
            torch.full((counts[4], 1), z_min, device=device, dtype=dtype),
            t,
        ],
        dim=1,
    )

    # z-max face
    u = _face_samples(counts[5], 5)
    x = x_min + u[:, 0:1] * (x_max - x_min)
    y = y_min + u[:, 1:2] * (y_max - y_min)
    t = t_min + u[:, 2:3] * (t_max - t_min)
    zmax = torch.cat(
        [
            x,
            y,
            torch.full((counts[5], 1), z_max, device=device, dtype=dtype),
            t,
        ],
        dim=1,
    )

    return torch.cat([xmin, xmax, ymin, ymax, zmin, zmax], dim=0)


# Axis-aligned box domains


@dataclass(frozen=True)
class Box1D:
    """Axis-aligned (x, t) rectangle."""

    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def __post_init__(self) -> None:
        _validate_interval("x", self.x_min, self.x_max)
        _validate_interval("t", self.t_min, self.t_max)

    def grid(
        self,
        n_x: int,
        n_t: int,
        *,
        device: DeviceLike = "cpu",
        dtype: torch.dtype | None = None,
        return_cartesian: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return grid1d(
            n_x=n_x,
            n_t=n_t,
            x_min=self.x_min,
            x_max=self.x_max,
            t_min=self.t_min,
            t_max=self.t_max,
            device=device,
            dtype=dtype,
            return_cartesian=return_cartesian,
        )

    def sample_interior(
        self,
        n: int,
        *,
        method: _SampleMethod = "sobol",
        device: DeviceLike = "cpu",
        dtype: torch.dtype | None = None,
        seed: int | None = None,
    ) -> Tensor:
        return sample_interior_1d(
            n=n,
            x_min=self.x_min,
            x_max=self.x_max,
            t_min=self.t_min,
            t_max=self.t_max,
            method=method,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    def sample_boundary(
        self,
        n_total: int,
        *,
        method: _SampleMethod = "sobol",
        device: DeviceLike = "cpu",
        dtype: torch.dtype | None = None,
        seed: int | None = None,
    ) -> Tensor:
        return sample_boundary_1d(
            n_total=n_total,
            x_min=self.x_min,
            x_max=self.x_max,
            t_min=self.t_min,
            t_max=self.t_max,
            method=method,
            device=device,
            dtype=dtype,
            seed=seed,
        )


@dataclass(frozen=True)
class Box2D:
    """Axis-aligned (x, y, t) rectangular slab."""

    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def __post_init__(self) -> None:
        _validate_interval("x", self.x_min, self.x_max)
        _validate_interval("y", self.y_min, self.y_max)
        _validate_interval("t", self.t_min, self.t_max)

    def grid(
        self,
        n_x: int,
        n_y: int,
        n_t: int,
        *,
        device: DeviceLike = "cpu",
        dtype: torch.dtype | None = None,
        return_cartesian: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return grid2d(
            n_x=n_x,
            n_y=n_y,
            n_t=n_t,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            t_min=self.t_min,
            t_max=self.t_max,
            device=device,
            dtype=dtype,
            return_cartesian=return_cartesian,
        )

    def sample_interior(
        self,
        n: int,
        *,
        method: _SampleMethod = "sobol",
        device: DeviceLike = "cpu",
        dtype: torch.dtype | None = None,
        seed: int | None = None,
    ) -> Tensor:
        return sample_interior_2d(
            n=n,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            t_min=self.t_min,
            t_max=self.t_max,
            method=method,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    def sample_boundary(
        self,
        n_total: int,
        *,
        method: _SampleMethod = "sobol",
        device: DeviceLike = "cpu",
        dtype: torch.dtype | None = None,
        seed: int | None = None,
        split: Sequence[float] | None = None,
    ) -> Tensor:
        return sample_boundary_2d(
            n_total=n_total,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            t_min=self.t_min,
            t_max=self.t_max,
            method=method,
            device=device,
            dtype=dtype,
            seed=seed,
            split=split,
        )


@dataclass(frozen=True)
class Box3D:
    """Axis-aligned (x, y, z, t) rectangular slab."""

    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    z_min: float = 0.0
    z_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def __post_init__(self) -> None:
        _validate_interval("x", self.x_min, self.x_max)
        _validate_interval("y", self.y_min, self.y_max)
        _validate_interval("z", self.z_min, self.z_max)
        _validate_interval("t", self.t_min, self.t_max)

    def grid(
        self,
        n_x: int,
        n_y: int,
        n_z: int,
        n_t: int,
        *,
        device: DeviceLike = "cpu",
        dtype: torch.dtype | None = None,
        return_cartesian: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return grid3d(
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            n_t=n_t,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            z_min=self.z_min,
            z_max=self.z_max,
            t_min=self.t_min,
            t_max=self.t_max,
            device=device,
            dtype=dtype,
            return_cartesian=return_cartesian,
        )

    def sample_interior(
        self,
        n: int,
        *,
        method: _SampleMethod = "sobol",
        device: DeviceLike = "cpu",
        dtype: torch.dtype | None = None,
        seed: int | None = None,
    ) -> Tensor:
        return sample_interior_3d(
            n=n,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            z_min=self.z_min,
            z_max=self.z_max,
            t_min=self.t_min,
            t_max=self.t_max,
            method=method,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    def sample_boundary(
        self,
        n_total: int,
        *,
        method: _SampleMethod = "sobol",
        device: DeviceLike = "cpu",
        dtype: torch.dtype | None = None,
        seed: int | None = None,
        split: Sequence[float] | None = None,
    ) -> Tensor:
        return sample_boundary_3d(
            n_total=n_total,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            z_min=self.z_min,
            z_max=self.z_max,
            t_min=self.t_min,
            t_max=self.t_max,
            method=method,
            device=device,
            dtype=dtype,
            seed=seed,
            split=split,
        )


__all__ = [
    # grids
    "grid1d",
    "grid2d",
    "grid3d",
    # spacing
    "spacing1d",
    "spacing2d",
    "spacing3d",
    # samplers
    "sample_interior_1d",
    "sample_interior_2d",
    "sample_interior_3d",
    "sample_boundary_1d",
    "sample_boundary_2d",
    "sample_boundary_3d",
    # domains
    "Box1D",
    "Box2D",
    "Box3D",
]
