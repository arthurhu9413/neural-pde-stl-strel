from __future__ import annotations

"""neural_pde_stl_strel.physics.heat2d

Utilities for the 2-D heat/diffusion equation used by the project.

We model a scalar field ``u(x, y, t)`` (e.g., temperature) on a rectangular
spatial domain ``[x_min, x_max] × [y_min, y_max]`` over a time interval
``[t_min, t_max]``.

PDE (conservative form with diagonal diffusivity)

The default strong-form residual corresponds to

    u_t - ∂_x(α_x u_x) - ∂_y(α_y u_y) = s(x, y, t),

where ``α_x`` and ``α_y`` may be constant or depend on ``(x, y, t)``.

For constant, isotropic diffusivity (``α_x = α_y = α``) and ``s = 0`` this reduces
to the classic heat equation

    u_t = α (u_xx + u_yy).

This module provides:

* ``residual_heat2d`` - autograd-based PDE residual.
* ``bc_ic_heat2d`` - boundary/initial condition losses on a space-time slab.
* ``make_dirichlet_mask`` + ``MaskedModel`` - a "trial solution" style output
  transform to hard-enforce homogeneous Dirichlet boundary conditions.

Design goals
1) **Correctness first.** Derivatives are computed via PyTorch autograd with
   ``create_graph=True`` so gradients can flow through PDE residuals.
2) **Plug-and-play.** Signatures match experiment usage in this repository.
3) **CPU-friendly.** No heavyweight dependencies; sampling supports Sobol.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

# Optional import (kept local to avoid import cycles at module load time).
# Provides Sobol boundary sampling for rectangles, if available.
try:  # pragma: no cover - convenience only
    from ..training.grids import sample_boundary_2d
except Exception:  # pragma: no cover
    sample_boundary_2d = None  # type: ignore


# Small helpers


def _check_coords(coords: Tensor) -> Tensor:
    """Validate a coordinate tensor.

    Parameters
    coords:
        Tensor shaped ``(N, 3)`` with columns ``[x, y, t]``.

    Returns
    -------
    Tensor
        The same tensor (for chaining).
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3) with columns [x, y, t]")
    return coords


def _split_coords(coords: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Split ``coords`` (N,3) into ``(x, y, t)`` each shaped (N,1)."""
    coords = _check_coords(coords)
    return coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]


def _grad(y: Tensor, x: Tensor, *, retain_graph: bool = True) -> Tensor:
    """d(y)/d(x) with autograd (vector-Jacobian product).

    Always uses ``create_graph=True`` so the result can participate in higher-order
    derivatives and backpropagation.
    """
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=retain_graph,
    )[0]


def _unit_samples(
    n: int,
    d: int,
    *,
    method: str = "sobol",
    seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Sample ``n`` points from ``[0, 1]^d``.

    Notes
    -----
    * ``method='sobol'`` draws on CPU then moves to ``device``.
    * If ``seed`` is provided, the returned samples are deterministic.
    """
    if n <= 0:
        return torch.empty((0, d), device=device, dtype=dtype or torch.get_default_dtype())

    method_l = method.lower()
    dt = dtype or torch.get_default_dtype()

    if method_l == "sobol":
        engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        u = engine.draw(n).to(dtype=dt, device="cpu")
        return u.to(device=device)

    if method_l == "uniform":
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
        return torch.rand((n, d), generator=gen, device=device, dtype=dt)

    raise ValueError(f"Unknown sampling method: {method!r} (expected 'sobol' or 'uniform')")


def _as_tensor(x: float | Tensor, *, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(float(x), device=device, dtype=dtype)


def _as_column(x: float | Tensor, n: int, *, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    """Convert/broadcast a scalar or vector to shape (n,1)."""
    t = _as_tensor(x, device=device, dtype=dtype)
    if t.ndim == 0:
        return t.view(1, 1).expand(n, 1)
    if t.ndim == 1:
        if t.shape[0] != n:
            raise ValueError(f"Expected a length-{n} vector, got shape {tuple(t.shape)}")
        return t.view(n, 1)
    if t.ndim == 2:
        if t.shape != (n, 1):
            raise ValueError(f"Expected shape (n,1) with n={n}, got shape {tuple(t.shape)}")
        return t
    raise ValueError("Expected scalar/1D/2D tensor")


def _field_1col(
    f: float | Tensor | Callable[[Tensor], Tensor] | None,
    coords: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Evaluate and normalize a scalar field to shape (N,1)."""
    coords = _check_coords(coords)
    n = coords.shape[0]

    if f is None:
        return torch.zeros((n, 1), device=coords.device, dtype=dtype)

    val = f(coords) if callable(f) else f

    if torch.is_tensor(val):
        v = val.to(device=coords.device, dtype=dtype)
        if v.ndim == 0:
            return v.view(1, 1).expand(n, 1)
        if v.ndim == 1:
            if v.shape[0] != n:
                raise ValueError(f"Field returned length-{v.shape[0]} vector for N={n} coords")
            return v.view(n, 1)
        if v.ndim == 2:
            if v.shape != (n, 1):
                raise ValueError(f"Expected field shape (N,1) for N={n}, got {tuple(v.shape)}")
            return v
        raise ValueError("Field tensor must be scalar, 1D, or 2D")

    # Python scalar
    return torch.full((n, 1), float(val), device=coords.device, dtype=dtype)


def _alpha_field(
    alpha: float | tuple[float, float] | Tensor | Callable[[Tensor], Tensor],
    coords: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Normalize diffusivity specification to a tensor shaped (N,2).

    Returned tensor columns are (α_x, α_y).
    """
    coords = _check_coords(coords)
    n = coords.shape[0]

    # Tuple -> diagonal coefficients.
    if isinstance(alpha, tuple):
        if len(alpha) != 2:
            raise ValueError("alpha tuple must have length 2: (alpha_x, alpha_y)")
        ax = _as_column(alpha[0], n, device=coords.device, dtype=dtype)
        ay = _as_column(alpha[1], n, device=coords.device, dtype=dtype)
        return torch.cat([ax, ay], dim=1)

    # Callable or scalar/tensor.
    a = alpha(coords) if callable(alpha) else alpha

    if not torch.is_tensor(a):
        a = torch.tensor(float(a), device=coords.device, dtype=dtype)

    a = a.to(device=coords.device, dtype=dtype)

    if a.ndim == 0:
        a = a.view(1, 1).expand(n, 1)
    elif a.ndim == 1:
        if a.shape[0] != n:
            raise ValueError(f"alpha length {a.shape[0]} does not match N={n}")
        a = a.view(n, 1)
    elif a.ndim == 2:
        if a.shape[0] != n or a.shape[1] not in (1, 2):
            raise ValueError(
                f"alpha tensor must have shape (N,1) or (N,2) with N={n}, got {tuple(a.shape)}"
            )
    else:
        raise ValueError("alpha tensor must be scalar, (N,), (N,1), or (N,2)")

    if a.shape[1] == 1:
        a = a.expand(n, 2)

    return a


# Core PDE residual


def residual_heat2d(
    model: torch.nn.Module,
    coords: Tensor,
    alpha: float | tuple[float, float] | Tensor | Callable[[Tensor], Tensor] = 0.1,
    *,
    source: float | Tensor | Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    r"""Return the strong-form residual for the 2-D heat equation.

    The PDE is taken as

        u_t - ∂_x(α_x u_x) - ∂_y(α_y u_y) = s(x, y, t).

    For constant, isotropic diffusion (``α_x = α_y = α``) this reduces to

        u_t - α (u_xx + u_yy) = s.

    Parameters
    model:
        Neural field mapping ``(x, y, t) -> u``.
    coords:
        Tensor of shape ``(N, 3)`` with columns ``[x, y, t]``.
    alpha:
        Diffusivity. Supported forms:

        * float (isotropic)
        * tuple(float, float) for diagonal anisotropy (α_x, α_y)
        * Tensor broadcastable to (N,1) or (N,2)
        * Callable(coords)->Tensor returning (N,1) or (N,2)

    source:
        Optional source term ``s(coords)``.

    Returns
    -------
    Tensor
        Residual tensor of shape ``(N, 1)``.
    """
    coords = _check_coords(coords).requires_grad_(True)

    u = model(coords)
    if u.ndim == 1:
        u = u.view(-1, 1)
    if u.ndim != 2 or u.shape != (coords.shape[0], 1):
        raise ValueError(f"model(coords) must have shape (N,1); got {tuple(u.shape)}")

    du = _grad(u, coords, retain_graph=True)
    u_x = du[:, 0:1]
    u_y = du[:, 1:2]
    u_t = du[:, 2:3]

    a = _alpha_field(alpha, coords, dtype=u.dtype)
    ax = a[:, 0:1]
    ay = a[:, 1:2]

    # Divergence of diagonal flux: ∂_x(ax * u_x) + ∂_y(ay * u_y)
    flux_x = ax * u_x
    flux_y = ay * u_y

    dflux_x = _grad(flux_x, coords, retain_graph=True)
    dflux_y = _grad(flux_y, coords, retain_graph=True)

    div = dflux_x[:, 0:1] + dflux_y[:, 1:2]

    s = _field_1col(source, coords, dtype=u.dtype)

    return u_t - div - s


# Boundary / initial condition losses


@dataclass(frozen=True)
class SquareDomain2D:
    """Axis-aligned 2-D spatial rectangle with a time interval."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    t_min: float
    t_max: float

    @property
    def x_mid(self) -> float:
        return 0.5 * (self.x_min + self.x_max)

    @property
    def y_mid(self) -> float:
        return 0.5 * (self.y_min + self.y_max)


def gaussian_ic(
    x: Tensor,
    y: Tensor,
    *,
    center: tuple[float, float] = (0.5, 0.5),
    sharpness: float = 50.0,
    amplitude: float | Tensor = 1.0,
) -> Tensor:
    """Gaussian initial condition on the spatial plane.

    Returns
    -------
    Tensor
        Shape matches ``x``/``y`` (typically (N,1)).
    """
    cx, cy = center
    amp = _as_tensor(amplitude, device=x.device, dtype=x.dtype)
    r2 = (x - cx).square() + (y - cy).square()
    return amp * torch.exp(-sharpness * r2)


def bc_ic_heat2d(
    model: torch.nn.Module,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    t_min: float,
    t_max: float,
    *,
    n_boundary: int = 512,
    n_initial: int = 512,
    boundary: str = "dirichlet",
    boundary_value: float | Callable[[Tensor, Tensor, Tensor], Tensor] | None = 0.0,
    neumann_flux: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor] | None = None,
    ic: Callable[[Tensor, Tensor], Tensor] | None = None,
    sampler: str = "sobol",
    boundary_split: Sequence[float] | None = None,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Compute boundary and initial condition losses for a 2-D heat problem.

    Parameters
    boundary:
        Either ``'dirichlet'`` or ``'neumann'``.

        * Dirichlet: enforce ``u(x,y,t) = g(x,y,t)`` on the spatial boundary.
        * Neumann: enforce outward normal derivative ``∂u/∂n = q``.

    boundary_value:
        For Dirichlet BCs, either a scalar or a callable ``g(x,y,t)``.

    neumann_flux:
        For Neumann BCs, callable returning the target normal derivative
        ``q(x,y,t,u_x,u_y)``. If ``None``, the target is 0 everywhere.

    Returns
    -------
    (loss_bc, loss_ic)
        Two scalars (0-dim tensors).
    """
    dom = SquareDomain2D(
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        t_min=float(t_min),
        t_max=float(t_max),
    )

    if boundary.lower() not in {"dirichlet", "neumann"}:
        raise ValueError("boundary must be either 'dirichlet' or 'neumann'")

    # Boundary samples (space boundary over time)
    if n_boundary <= 0:
        bc_coords = torch.empty((0, 3), device=device, dtype=dtype)
    elif sample_boundary_2d is not None:
        bc_coords = sample_boundary_2d(
            x_min=dom.x_min,
            x_max=dom.x_max,
            y_min=dom.y_min,
            y_max=dom.y_max,
            t_min=dom.t_min,
            t_max=dom.t_max,
            n_total=n_boundary,
            seed=seed,
            method=sampler,
            split=boundary_split,
            device=device,
            dtype=dtype,
        )
    else:  # pragma: no cover (should not happen in-repo)
        # Fallback: sample in the box, then snap to a random face.
        u = _unit_samples(n_boundary, 3, method=sampler, seed=seed, device=device, dtype=dtype)
        xb = dom.x_min + (dom.x_max - dom.x_min) * u[:, 0:1]
        yb = dom.y_min + (dom.y_max - dom.y_min) * u[:, 1:2]
        tb = dom.t_min + (dom.t_max - dom.t_min) * u[:, 2:3]

        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + 17)
        sides = torch.randint(0, 4, (n_boundary, 1), generator=gen, device=device)

        xb[sides == 0] = dom.x_min
        xb[sides == 1] = dom.x_max
        yb[sides == 2] = dom.y_min
        yb[sides == 3] = dom.y_max

        bc_coords = torch.cat([xb, yb, tb], dim=1)

    # Boundary loss
    if bc_coords.numel() == 0:
        loss_bc = torch.zeros((), device=device, dtype=dtype)

    elif boundary.lower() == "dirichlet":
        u_b = model(bc_coords)
        if u_b.ndim == 1:
            u_b = u_b.view(-1, 1)
        if u_b.ndim != 2 or u_b.shape != (bc_coords.shape[0], 1):
            raise ValueError(f"model(bc_coords) must have shape (N,1); got {tuple(u_b.shape)}")

        x_b, y_b, t_b = _split_coords(bc_coords)

        if callable(boundary_value):
            target_val = boundary_value(x_b, y_b, t_b)
        else:
            target_val = 0.0 if boundary_value is None else float(boundary_value)

        target = _field_1col(target_val, bc_coords, dtype=u_b.dtype)
        loss_bc = (u_b - target).square().mean()

    else:  # Neumann
        bc_coords = bc_coords.requires_grad_(True)
        u_b = model(bc_coords)
        if u_b.ndim == 1:
            u_b = u_b.view(-1, 1)
        if u_b.ndim != 2 or u_b.shape != (bc_coords.shape[0], 1):
            raise ValueError(f"model(bc_coords) must have shape (N,1); got {tuple(u_b.shape)}")

        x_b, y_b, t_b = _split_coords(bc_coords)
        du = _grad(u_b, bc_coords, retain_graph=True)
        u_x = du[:, 0:1]
        u_y = du[:, 1:2]

        # Use 1D boolean masks to avoid PyTorch's element-wise boolean indexing
        # flattening surprises when mask has shape (N,1).
        x_min_t = x_b.new_tensor(dom.x_min)
        x_max_t = x_b.new_tensor(dom.x_max)
        y_min_t = y_b.new_tensor(dom.y_min)
        y_max_t = y_b.new_tensor(dom.y_max)

        left = torch.isclose(x_b, x_min_t, atol=eps, rtol=0.0).squeeze(1)
        right = torch.isclose(x_b, x_max_t, atol=eps, rtol=0.0).squeeze(1)
        bottom = torch.isclose(y_b, y_min_t, atol=eps, rtol=0.0).squeeze(1)
        top = torch.isclose(y_b, y_max_t, atol=eps, rtol=0.0).squeeze(1)

        errs: list[Tensor] = []

        def _accum(mask: Tensor, n_x: float, n_y: float) -> None:
            if not bool(mask.any()):
                return

            g = n_x * u_x[mask] + n_y * u_y[mask]

            if neumann_flux is None:
                target = torch.zeros_like(g)
            else:
                target_val = neumann_flux(
                    x_b[mask],
                    y_b[mask],
                    t_b[mask],
                    u_x[mask],
                    u_y[mask],
                )

                if torch.is_tensor(target_val):
                    target = target_val.to(device=g.device, dtype=g.dtype)
                    if target.ndim == 0:
                        target = target.view(1, 1).expand_as(g)
                    elif target.ndim == 1:
                        target = target.view(-1, 1)
                else:
                    target = torch.full_like(g, float(target_val))

                if target.shape != g.shape:
                    raise ValueError(
                        "neumann_flux must return a scalar or a tensor matching the face batch; "
                        f"expected {tuple(g.shape)}, got {tuple(target.shape)}"
                    )

            errs.append((g - target).square())

        # Outward normals for each face
        _accum(left, -1.0, 0.0)
        _accum(right, 1.0, 0.0)
        _accum(bottom, 0.0, -1.0)
        _accum(top, 0.0, 1.0)

        loss_bc = torch.cat(errs, dim=0).mean() if errs else torch.zeros((), device=device, dtype=dtype)

    # Initial condition samples (t = t_min plane)
    if n_initial <= 0:
        loss_ic = torch.zeros((), device=device, dtype=dtype)
    else:
        ic_seed = None if seed is None else seed + 1
        u0 = _unit_samples(n_initial, 2, method=sampler, seed=ic_seed, device=device, dtype=dtype)
        x0 = dom.x_min + (dom.x_max - dom.x_min) * u0[:, 0:1]
        y0 = dom.y_min + (dom.y_max - dom.y_min) * u0[:, 1:2]
        t0 = torch.full_like(x0, dom.t_min)
        ic_coords = torch.cat([x0, y0, t0], dim=1)

        u_pred0 = model(ic_coords)
        if u_pred0.ndim == 1:
            u_pred0 = u_pred0.view(-1, 1)
        if u_pred0.ndim != 2 or u_pred0.shape != (n_initial, 1):
            raise ValueError(f"model(ic_coords) must have shape (N,1); got {tuple(u_pred0.shape)}")

        if ic is None:
            target0_val = gaussian_ic(
                x0,
                y0,
                center=(dom.x_mid, dom.y_mid),
                sharpness=50.0,
                amplitude=1.0,
            )
        else:
            target0_val = ic(x0, y0)

        if torch.is_tensor(target0_val):
            target0 = target0_val.to(device=device, dtype=dtype)
            if target0.ndim == 1:
                target0 = target0.view(-1, 1)
            if target0.ndim == 0:
                target0 = target0.view(1, 1).expand_as(u_pred0)
        else:
            target0 = torch.full_like(u_pred0, float(target0_val))

        if target0.shape != u_pred0.shape:
            raise ValueError(
                f"Initial condition must return shape {tuple(u_pred0.shape)}; got {tuple(target0.shape)}"
            )

        loss_ic = (u_pred0 - target0).square().mean()

    return loss_bc, loss_ic


# Exact Dirichlet enforcement via output transform


def make_dirichlet_mask(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    pow: int = 1,
    normalize: bool = True,
) -> Callable[[Tensor], Tensor]:
    """Return a multiplicative mask that is zero on the spatial boundary.

    The mask is based on the classic "trial solution" construction:

        m(x, y) = (x - x_min)(x_max - x) (y - y_min)(y_max - y).

    If ``normalize=True``, we scale each 1-D factor by its maximum value so the
    mask is in ``[0, 1]`` (achieves 1 at the spatial center). The mask ignores
    time; it depends only on (x,y).

    Parameters
    pow:
        Optional exponent applied to the mask. Larger values make the model output
        decay more sharply near the boundary.

    Returns
    -------
    Callable
        A function ``mask(coords)->Tensor`` returning shape ``(N,1)``.
    """
    if not isinstance(pow, int) or pow < 1:
        raise ValueError("pow must be a positive integer")

    x_min_f = float(x_min)
    x_max_f = float(x_max)
    y_min_f = float(y_min)
    y_max_f = float(y_max)

    width = x_max_f - x_min_f
    height = y_max_f - y_min_f
    if width <= 0 or height <= 0:
        raise ValueError("Expected x_max > x_min and y_max > y_min")

    # Max of (x-x_min)(x_max-x) over the interval is width^2 / 4.
    mx_max = (width * width) / 4.0
    my_max = (height * height) / 4.0

    def _mask(coords: Tensor) -> Tensor:
        coords = _check_coords(coords)
        x = coords[:, 0:1]
        y = coords[:, 1:2]

        mx = (x - x_min_f) * (x_max_f - x)
        my = (y - y_min_f) * (y_max_f - y)

        # Guard against tiny negative values from floating point noise.
        mx = torch.clamp(mx, min=0.0)
        my = torch.clamp(my, min=0.0)

        if normalize:
            mx = mx / mx_max
            my = my / my_max

        m = mx * my
        return m.pow(pow) if pow != 1 else m

    return _mask


class MaskedModel(torch.nn.Module):
    """Wrap a base model with a multiplicative spatial mask.

    If the mask is zero on the boundary, the wrapped model satisfies homogeneous
    Dirichlet boundary conditions exactly:

        u(x,y,t) = mask(x,y) * base(x,y,t)  =>  u|∂Ω = 0.
    """

    def __init__(self, base: torch.nn.Module, mask_fn: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.base = base
        self.mask_fn = mask_fn

    def forward(self, coords: Tensor) -> Tensor:
        return self.mask_fn(coords) * self.base(coords)


__all__ = [
    "residual_heat2d",
    "bc_ic_heat2d",
    "SquareDomain2D",
    "gaussian_ic",
    "make_dirichlet_mask",
    "MaskedModel",
]
