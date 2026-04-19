from __future__ import annotations

"""neural_pde_stl_strel.physics.diffusion1d

Physics utilities for the 1-D diffusion (heat) equation used by the experiments.

We model a *scalar spatio-temporal field* ``u(x, t)`` on a space-time slab:

    x ∈ [x_left, x_right],   t ∈ [t_min, t_max].

The default PDE is the classic 1-D heat equation with constant diffusivity ``α``:

    u_t = α u_xx.

For convenience, the residual helper also accepts an ``alpha(coords)`` callable or
per-sample tensor. **Important:** in that case this module implements the
*non-divergence form* ``u_t = α(x,t) · u_xx``. If you need the physically common
divergence form ``u_t = (α u_x)_x``, you must include the extra ``α_x u_x`` term
yourself.

This module is intentionally "physics-only": it provides PDE residuals, boundary
and initial-condition penalties, and a simple mask wrapper for enforcing
homogeneous Dirichlet boundaries by construction. STL/STREL monitoring and
differentiable robustness penalties live in :mod:`neural_pde_stl_strel.monitoring` and
are wired up in :mod:`neural_pde_stl_strel.experiments`.

In the diffusion experiments, these pieces are combined in a standard PINN-style
objective (schematically)

    loss = loss_pde + loss_bcic + (λ · loss_stl),

where ``loss_bcic`` is produced by :func:`boundary_loss` and ``λ`` corresponds to
the experiment/config parameter often named ``stl_weight``.

For temporal-logic monitoring, remember that ``u(x,t)`` is a *field*. To obtain a
1-D time-series suitable for STL, you typically (a) pick one or more probe
locations (e.g., ``x=0.5``) or (b) apply a spatial reduction such as
``max_x u(x,t)``, ``mean_x u(x,t)``, or a smooth approximation thereof. The
diffusion experiment scripts implement these reductions on a sampled spatial
grid.

Conventions
* Coordinate tensors are always shaped ``(N, 2)`` with columns ``[x, t]``.
* Models are expected to map ``coords -> u`` with output shape ``(N, 1)``.
* Neumann/Robin options in :func:`boundary_loss` interpret derivatives as
  partial derivatives with respect to **+x** (i.e., ``u_x``). If you want
  outward-normal derivatives, remember: ``du/dn = -u_x`` at ``x=x_left`` and
  ``du/dn = +u_x`` at ``x=x_right``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

Alpha = float | Tensor | Callable[[Tensor], Tensor]


# Autograd helpers

def _check_coords(coords: Tensor) -> Tensor:
    """Validate the standard coordinate layout ``(N,2) == [x, t]``."""
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError("coords must have shape (N, 2) with columns [x, t]")
    if not coords.is_floating_point():
        raise TypeError("coords must be a floating-point tensor for autograd derivatives")
    return coords


def _check_scalar_field(u: Tensor, n: int) -> Tensor:
    """Ensure model outputs are shaped ``(N,1)`` (PINN scalar field)."""
    if u.ndim != 2 or u.shape != (n, 1):
        raise ValueError(f"model(coords) must return shape (N, 1); got {tuple(u.shape)}")
    return u


def _grad(y: Tensor, x: Tensor) -> Tensor:
    """Element-wise gradient ``∂y/∂x`` for batched coordinate networks.

    Assumes the batch dimension corresponds to independent samples (true for
    standard coordinate-MLPs without cross-sample coupling such as batch-norm).
    Returns a tensor of shape ``(N, D)``.
    """
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def _as_tensor(x: float | Tensor, *, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    """Convert a float or tensor to the requested ``(device, dtype)``."""
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(float(x), device=device, dtype=dtype)


def _as_col(x: Tensor, n: int, *, name: str) -> Tensor:
    """Normalize a tensor to shape ``(N,1)`` with helpful errors."""
    if x.ndim == 0:
        return x.reshape(1, 1).expand(n, 1)
    if x.ndim == 1:
        if x.shape[0] != n:
            raise ValueError(f"{name} has shape {(x.shape[0],)} but expected (N,) with N={n}")
        return x.reshape(n, 1)
    if x.ndim == 2 and x.shape[1] == 1:
        if x.shape[0] != n:
            raise ValueError(
                f"{name} has shape {tuple(x.shape)} but expected (N,1) with N={n}"
            )
        return x
    raise ValueError(f"{name} must be scalar, (N,), or (N,1); got shape {tuple(x.shape)}")


def _alpha_field(alpha: Alpha, coords: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Normalize diffusivity ``α`` to a per-sample column tensor of shape ``(N,1)``."""
    n = coords.shape[0]
    device = coords.device
    if callable(alpha):
        a = alpha(coords)
        if not isinstance(a, Tensor):
            raise TypeError("alpha(coords) must return a torch.Tensor")
        a = a.to(device=device, dtype=dtype)
        return _as_col(a, n, name="alpha(coords)")

    if isinstance(alpha, Tensor):
        a = alpha.to(device=device, dtype=dtype)
        return _as_col(a, n, name="alpha")

    # float
    return torch.full((n, 1), float(alpha), device=device, dtype=dtype)


# Core PDE residual: u_t - α u_xx

def pde_residual(
    model: torch.nn.Module,
    coords: Tensor,
    alpha: Alpha = 0.1,
) -> Tensor:
    r"""Return the strong-form residual ``r(x,t) = u_t − α(x,t) u_xx``.

    Parameters
    model
        Neural field mapping coordinates to the scalar solution value:
        ``model([x,t]) = u(x,t)``.
    coords
        Tensor of shape ``(N,2)`` with columns ``[x, t]``.
    alpha
        Diffusivity specification: scalar (float/0-D tensor), per-sample tensor
        ``(N,)`` or ``(N,1)``, or a callable ``alpha(coords)`` returning one of
        those.

    Returns
    -------
    Tensor
        Residual values with shape ``(N,1)``.
    """
    coords = _check_coords(coords)
    # Avoid mutating the caller's tensor while ensuring a leaf for autograd.
    coords_req = coords.detach().requires_grad_(True)

    u = _check_scalar_field(model(coords_req), coords_req.shape[0])

    du = _grad(u, coords_req)  # (N,2) -> [u_x, u_t]
    u_x = du[:, 0:1]
    u_t = du[:, 1:2]

    u_xx = _grad(u_x, coords_req)[:, 0:1]
    a = _alpha_field(alpha, coords_req, dtype=u.dtype)

    return u_t - a * u_xx


def residual_loss(
    model: torch.nn.Module,
    coords: Tensor,
    alpha: Alpha = 0.1,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Mean-squared PDE residual loss over collocation points."""
    r = pde_residual(model, coords, alpha)
    sq = r.square()

    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    if reduction == "none":
        return sq
    raise ValueError("reduction must be one of {'mean','sum','none'}")


# Boundary and initial conditions (soft penalties)

def _unit_samples(
    n: int,
    d: int,
    *,
    method: Literal["sobol", "uniform"],
    device: torch.device | str,
    dtype: torch.dtype,
    seed: int | None,
) -> Tensor:
    """Sample ``n`` points in the unit hypercube ``[0,1]^d``.

    Notes
    -----
    * ``sobol`` uses :class:`torch.quasirandom.SobolEngine`.
    * ``uniform`` uses a local :class:`torch.Generator` when ``seed`` is provided
      for per-call reproducibility.
    """
    if n <= 0:
        return torch.empty((0, d), device=device, dtype=dtype)

    if method == "sobol":
        # If seed is None, use a deterministic (but scrambled) sequence.
        sobol_seed = 0 if seed is None else int(seed)
        engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=sobol_seed)
        # SobolEngine draws on CPU; move to the requested device afterwards.
        u = engine.draw(n).to(dtype=dtype)
        return u.to(device=device)

    if method == "uniform":
        if seed is None:
            return torch.rand((n, d), device=device, dtype=dtype)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        u = torch.rand((n, d), generator=gen, device="cpu", dtype=dtype)
        return u.to(device=device)

    raise ValueError("method must be 'sobol' or 'uniform'")


def sine_ic(
    x: Tensor,
    *,
    x_left: float = 0.0,
    x_right: float = 1.0,
    amplitude: float | Tensor = 1.0,
) -> Tensor:
    r"""Sine initial condition respecting homogeneous Dirichlet boundaries.

    .. math::

        u(x, 0) = A\,\sin\big(\pi (x - x_\mathrm{left}) / L\big),\quad L=x_\mathrm{right}-x_\mathrm{left}.
    """
    if not x.is_floating_point():
        raise TypeError("x must be floating-point")
    if x_right <= x_left:
        raise ValueError("Expected x_right > x_left")

    l = _as_tensor(x_right - x_left, device=x.device, dtype=x.dtype)
    k = torch.pi / l
    a = _as_tensor(amplitude, device=x.device, dtype=x.dtype)
    return a * torch.sin(k * (x - x_left))


def bc_ic_targets(
    x: Tensor,
    t: Tensor,
    *,
    x_left: float,
    x_right: float,
    bc_left: float | Callable[[Tensor], Tensor] = 0.0,
    bc_right: float | Callable[[Tensor], Tensor] = 0.0,
    ic: Callable[[Tensor], Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute target values for boundary/initial conditions.

    Returns ``(u_L(t), u_R(t), u_0(x))``.
    """
    if ic is None:
        u0 = sine_ic(x, x_left=x_left, x_right=x_right)
    else:
        u0 = ic(x)

    if callable(bc_left):
        u_l = bc_left(t)
    else:
        u_l = torch.full_like(t, fill_value=float(bc_left))

    if callable(bc_right):
        u_r = bc_right(t)
    else:
        u_r = torch.full_like(t, fill_value=float(bc_right))

    return u_l, u_r, u0


def _eval_target(
    target: float | Callable[[Tensor], Tensor],
    t: Tensor,
    *,
    name: str,
) -> Tensor:
    """Evaluate a boundary target specified as a float or callable of time."""
    if callable(target):
        y = target(t)
        if not isinstance(y, Tensor):
            raise TypeError(f"{name}(t) must return a torch.Tensor")
        y = y.to(device=t.device, dtype=t.dtype)
    else:
        y = torch.full_like(t, fill_value=float(target))
    return _as_col(y, t.shape[0], name=name)


def boundary_loss(
    model: torch.nn.Module,
    x_left: float = 0.0,
    x_right: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: torch.device | str = "cpu",
    n_boundary: int = 256,
    n_initial: int = 512,
    *,
    dtype: torch.dtype | None = None,
    method: Literal["sobol", "uniform"] = "sobol",
    seed: int | None = None,
    bc_left: float | Callable[[Tensor], Tensor] = 0.0,
    bc_right: float | Callable[[Tensor], Tensor] = 0.0,
    ic: Callable[[Tensor], Tensor] | None = None,
    w_boundary: float = 1.0,
    w_initial: float = 1.0,
    # Optional generalizations (kept compatible with the experiment configs)
    bc_left_type: Literal["dirichlet", "neumann", "robin"] = "dirichlet",
    bc_right_type: Literal["dirichlet", "neumann", "robin"] = "dirichlet",
    robin_left: tuple[float, float] | None = None,  # (a,b): a·u + b·u_x = g_L(t)
    robin_right: tuple[float, float] | None = None,  # (a,b): a·u + b·u_x = g_R(t)
) -> Tensor:
    r"""Soft penalty for boundary and initial conditions.

    The default (Dirichlet + sine IC) matches the experiment scripts:

    * ``u(x_left,t)  = bc_left(t)``
    * ``u(x_right,t) = bc_right(t)``
    * ``u(x,t_min)   = ic(x)`` (defaults to :func:`sine_ic`)

    With ``bc_*_type == 'neumann'``, ``bc_*`` denotes the desired derivative:

    .. math:: u_x(\text{boundary}, t) = g(t).

    With ``bc_*_type == 'robin'``, provide coefficients ``(a,b)`` via
    ``robin_*`` and set ``bc_*`` to the target function ``g(t)``:

    .. math:: a\,u(\text{boundary}, t) + b\,u_x(\text{boundary}, t) = g(t).
    """
    if x_right <= x_left:
        raise ValueError("Expected x_right > x_left")
    if t_max <= t_min:
        raise ValueError("Expected t_max > t_min")
    if n_boundary < 0 or n_initial < 0:
        raise ValueError("n_boundary and n_initial must be non-negative")

    if dtype is None:
        dtype = torch.get_default_dtype()

    dev = torch.device(device)

    # Boundary samples (left/right)
    if n_boundary > 0:
        u = _unit_samples(n_boundary, 1, method=method, device=dev, dtype=dtype, seed=seed)
        t = (t_min + u * (t_max - t_min)).to(device=dev, dtype=dtype)  # (Nb,1)
        x_l = torch.full_like(t, fill_value=float(x_left))
        x_r = torch.full_like(t, fill_value=float(x_right))

        left = torch.cat([x_l, t], dim=1)
        right = torch.cat([x_r, t], dim=1)

        bc_left_type_n = bc_left_type.lower()
        bc_right_type_n = bc_right_type.lower()
        need_derivs = (bc_left_type_n != "dirichlet") or (bc_right_type_n != "dirichlet")
        if need_derivs:
            left = left.requires_grad_(True)
            right = right.requires_grad_(True)

        # Targets are always time functions (Dirichlet values, fluxes, or Robin RHS).
        target_l = _eval_target(bc_left, t, name="bc_left")
        target_r = _eval_target(bc_right, t, name="bc_right")

        pred_l = _check_scalar_field(model(left), n_boundary)
        pred_r = _check_scalar_field(model(right), n_boundary)

        def bc_residual(
            *,
            bc_type: str,
            coords: Tensor,
            pred: Tensor,
            target: Tensor,
            robin_ab: tuple[float, float] | None,
            side_name: str,
        ) -> Tensor:
            if bc_type == "dirichlet":
                return pred - target

            dudx = _grad(pred, coords)[:, 0:1]

            if bc_type == "neumann":
                return dudx - target

            if bc_type == "robin":
                if robin_ab is None:
                    raise ValueError(
                        f"robin_{side_name} must be provided when bc_{side_name}_type='robin'"
                    )
                a, b = robin_ab
                a_t = _as_tensor(a, device=coords.device, dtype=pred.dtype)
                b_t = _as_tensor(b, device=coords.device, dtype=pred.dtype)
                return a_t * pred + b_t * dudx - target

            raise ValueError(f"Unknown BC type {bc_type!r} for {side_name} boundary")

        r_l = bc_residual(
            bc_type=bc_left_type_n,
            coords=left,
            pred=pred_l,
            target=target_l,
            robin_ab=robin_left,
            side_name="left",
        )
        r_r = bc_residual(
            bc_type=bc_right_type_n,
            coords=right,
            pred=pred_r,
            target=target_r,
            robin_ab=robin_right,
            side_name="right",
        )
        # Mean over the *union* of left+right boundary samples.
        loss_bc = 0.5 * (r_l.square().mean() + r_r.square().mean())
    else:
        loss_bc = torch.zeros((), device=dev, dtype=dtype)

    # Initial condition samples (t=t_min)
    if n_initial > 0:
        seed_ic = None if seed is None else int(seed) + 7
        u = _unit_samples(n_initial, 1, method=method, device=dev, dtype=dtype, seed=seed_ic)
        x = (x_left + u * (x_right - x_left)).to(device=dev, dtype=dtype)
        ic_coords = torch.cat([x, torch.full_like(x, fill_value=float(t_min))], dim=1)

        _u_l, _u_r, target_ic = bc_ic_targets(
            x=x,
            t=torch.zeros_like(x),
            x_left=x_left,
            x_right=x_right,
            bc_left=bc_left,
            bc_right=bc_right,
            ic=ic,
        )
        target_ic = _as_col(target_ic.to(device=dev, dtype=dtype), n_initial, name="ic")

        pred_ic = _check_scalar_field(model(ic_coords), n_initial)
        loss_ic = (pred_ic - target_ic).square().mean()
    else:
        loss_ic = torch.zeros((), device=dev, dtype=dtype)

    w_b = _as_tensor(w_boundary, device=dev, dtype=dtype)
    w_i = _as_tensor(w_initial, device=dev, dtype=dtype)
    return w_b * loss_bc + w_i * loss_ic


# Extras (analytic baseline + Dirichlet masking)

@dataclass(frozen=True)
class Interval1D:
    """Convenience container for a rectangular 1-D space-time domain."""

    x_left: float = 0.0
    x_right: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    @property
    def length(self) -> float:
        return float(self.x_right - self.x_left)


def sine_solution(
    x: Tensor,
    t: Tensor,
    alpha: float | Tensor = 0.1,
    *,
    x_left: float = 0.0,
    x_right: float = 1.0,
    amplitude: float | Tensor = 1.0,
) -> Tensor:
    r"""Analytic solution for sine IC + homogeneous Dirichlet BCs (constant α).

    .. math::

        u(x,t) = A\,e^{-\alpha k^2 t}\,\sin\big(k(x-x_\mathrm{left})\big),
        \quad k = \pi/(x_\mathrm{right}-x_\mathrm{left}).

    This closed form is valid only for **constant** diffusivity ``α``.
    """
    if x_right <= x_left:
        raise ValueError("Expected x_right > x_left")
    if isinstance(alpha, Tensor) and alpha.ndim != 0:
        raise ValueError("sine_solution requires a scalar alpha (float or 0-D tensor)")

    x_b, t_b = torch.broadcast_tensors(x, t)

    l = _as_tensor(x_right - x_left, device=x_b.device, dtype=x_b.dtype)
    k = torch.pi / l

    a = _as_tensor(alpha, device=x_b.device, dtype=x_b.dtype)
    amp = _as_tensor(amplitude, device=x_b.device, dtype=x_b.dtype)

    return amp * torch.exp(-a * (k.square()) * t_b) * torch.sin(k * (x_b - x_left))


def make_dirichlet_mask_1d(x_left: float = 0.0, x_right: float = 1.0) -> Callable[[Tensor], Tensor]:
    r"""Return a smooth mask ``m(x,t)`` that vanishes at the Dirichlet boundaries.

    The mask is

    .. math:: m(x,t) = (x-x_\mathrm{left})(x_\mathrm{right}-x),

    so that ``û(x,t) = m(x,t)·v(x,t)`` satisfies homogeneous Dirichlet BCs at
    both ends for any unconstrained field ``v``.
    """
    if x_right <= x_left:
        raise ValueError("Expected x_right > x_left")

    def mask(coords: Tensor) -> Tensor:
        coords = _check_coords(coords)
        x = coords[:, 0:1]
        return (x - x_left) * (x_right - x)

    return mask


class MaskedModel(torch.nn.Module):
    r"""Wrap a base model to enforce homogeneous Dirichlet BCs by construction.

    The wrapped field is

    .. math:: \hat{u}(x,t) = m(x,t)\,\text{base}([x,t]),

    where ``m`` is a mask returned by :func:`make_dirichlet_mask_1d`.
    """

    def __init__(self, base: torch.nn.Module, mask: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.base = base
        self.mask = mask

    def forward(self, coords: Tensor) -> Tensor:  # pragma: no cover - thin wrapper
        return self.mask(coords) * self.base(coords)


__all__ = [
    "pde_residual",
    "residual_loss",
    "boundary_loss",
    "Interval1D",
    "sine_ic",
    "sine_solution",
    "make_dirichlet_mask_1d",
    "MaskedModel",
]
