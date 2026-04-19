"""Differentiable ("soft") quantitative semantics for Signal Temporal Logic.

This module implements a compact subset of *discrete-time* STL quantitative
semantics (a.k.a. robustness semantics) using smooth approximations of
``min``/``max``. The primary intended use is to:

* **Monitor** temporal logic specifications on time series / trajectories.
* **Differentiate** through those specifications (PyTorch autograd).
* **Regularize training** by converting robustness into a loss term.

Robustness convention
We follow the standard sign convention:

* ``robustness > 0``  => the specification is satisfied.
* ``robustness < 0``  => the specification is violated.
* ``robustness == 0`` => boundary.

Smooth extrema
We smooth max/min using LogSumExp with a positive *temperature* ``temp = τ``:

* ``softmax(x) = τ · log Σ_i exp(x_i / τ)``   (smooth max / over-approx of max)
* ``softmin(x) = -τ · log Σ_i exp(-x_i / τ)`` (smooth min / under-approx of min)

As ``τ -> 0+``, these converge to hard max/min. For a reduction over ``n``
elements, the approximation error is bounded by ``τ · log(n)`` (dimension-
dependent offset). This is often acceptable for optimization, but it *does*
mean that changing the number of reduced elements (e.g., the time horizon)
changes the bias.

Important: ``softmax``/``softmin`` here are *smooth extrema*, **not** the
probability-normalized softmax used in classification.

Windowed operators
The ``*_window`` operators compute robustness over sliding windows and return
a *robustness trace* (offline monitoring). To keep semantics unambiguous on
finite traces, these functions only return values where the entire window fits
in the trace.

Keepdim semantics for windowed operators
For historical reasons and API symmetry, ``*_window`` operators accept
``keepdim``. In this module it means:

* ``keepdim=False`` (default): output has the same number of dimensions as the
  input; the time dimension length becomes the number of valid windows.
* ``keepdim=True``: output includes an extra singleton dimension *immediately
  after the time dimension* (interpretable as the reduced "window" axis).

That is, if the input time dimension is at index ``time_dim`` (normalized to a
non-negative index), then with ``keepdim=True`` the output time dimension
remains at ``time_dim`` and an extra dimension of size 1 is inserted at
``time_dim + 1``.

"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

Tensor = torch.Tensor

__all__ = [
    # smooth extrema
    "softmin",
    "softmax",
    # boolean connectives
    "soft_and",
    "soft_or",
    "soft_not",
    "soft_implies",
    # atomic predicates
    "pred_leq",
    "pred_geq",
    "pred_abs_leq",
    "pred_linear_leq",
    # temporal operators (whole-trace)
    "always",
    "eventually",
    # temporal operators (windowed / trace)
    "always_window",
    "eventually_window",
    "until_window",
    "release_window",
    # utilities
    "shift_left",
    # past operators (windowed / trace)
    "once_window",
    "historically_window",
    # loss wrapper
    "STLPenaltyConfig",
    "STLPenalty",
]


# Helpers


def _normalize_dim(dim: int, ndim: int, *, name: str = "dim") -> int:
    """Normalize a possibly-negative dimension index to ``[0, ndim-1]``.

    This follows PyTorch's indexing rules and raises for out-of-range indices
    (unlike ``dim % ndim`` which can silently mask errors).
    """

    if ndim <= 0:
        raise ValueError(f"{name} normalization requires ndim > 0, got {ndim}.")
    dim_i = int(dim)
    if dim_i < -ndim or dim_i >= ndim:
        raise ValueError(f"{name}={dim_i} is out of range for tensor with {ndim} dims.")
    return dim_i % ndim


def _check_temp(temp: float) -> float:
    """Validate and return a positive, finite temperature as ``float``."""

    tau = float(temp)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"temp must be a finite positive float, got {temp!r}.")
    return tau


def _move_time_last(x: Tensor, time_dim: int) -> tuple[Tensor, int]:
    """Move ``time_dim`` to the last axis; return (moved_tensor, old_time_dim)."""

    td = _normalize_dim(time_dim, x.ndim, name="time_dim")
    if td == x.ndim - 1:
        return x, td
    return x.movedim(td, -1), td


def _restore_time_last(
    y: Tensor,
    *,
    old_time_dim: int,
    keepdim: bool,
) -> Tensor:
    """Restore time axis (and optional singleton "window" axis) after window ops.

    Internal convention:
      * keepdim=False: y has time axis at -1.
      * keepdim=True:  y has shape (..., L, 1) with time axis at -2.

    We place the time axis back at ``old_time_dim``. If keepdim=True, we insert
    the singleton axis at ``old_time_dim + 1``.
    """

    if not keepdim:
        if old_time_dim == y.ndim - 1:
            return y
        return y.movedim(-1, old_time_dim)

    # keepdim=True: move (time_axis, window_axis) = (-2, -1) to (old, old+1)
    return y.movedim((-2, -1), (old_time_dim, old_time_dim + 1))


def _validate_window(*, window: int, stride: int, T: int) -> None:
    if window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}.")
    if stride <= 0:
        raise ValueError(f"stride must be a positive integer, got {stride}.")
    # Keep behavior consistent with torch.Tensor.unfold, which raises RuntimeError
    # when size > length. Some unit tests rely on this exception type.
    if window > T:
        raise RuntimeError(
            f"window={window} is larger than time length T={T}; no valid windows."
        )


def _unfold_time(x: Tensor, *, window: int, stride: int) -> Tensor:
    """Create sliding windows along the last dimension.

    Returns a view of shape ``(..., L, window)``, where
    ``L = floor((T - window)/stride) + 1``.
    """

    return x.unfold(dimension=-1, size=window, step=stride)


# Smooth extrema


def softmax(x: Tensor, *, temp: float = 0.1, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Smooth approximation of ``max(x, dim)`` using LogSumExp.

    Note: this is a smooth *maximum*, not a probability-normalized softmax.
    """

    tau = _check_temp(temp)
    d = _normalize_dim(dim, x.ndim, name="dim")
    return tau * torch.logsumexp(x / tau, dim=d, keepdim=keepdim)


def softmin(x: Tensor, *, temp: float = 0.1, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Smooth approximation of ``min(x, dim)`` using LogSumExp."""

    tau = _check_temp(temp)
    d = _normalize_dim(dim, x.ndim, name="dim")
    return -tau * torch.logsumexp(-x / tau, dim=d, keepdim=keepdim)


# Boolean connectives (robust semantics)


def soft_and(a: Tensor, b: Tensor, *, temp: float = 0.1) -> Tensor:
    """Smooth conjunction: ``min(a, b)``."""

    a_b, b_b = torch.broadcast_tensors(a, b)
    return softmin(torch.stack((a_b, b_b), dim=-1), temp=temp, dim=-1)


def soft_or(a: Tensor, b: Tensor, *, temp: float = 0.1) -> Tensor:
    """Smooth disjunction: ``max(a, b)``."""

    a_b, b_b = torch.broadcast_tensors(a, b)
    return softmax(torch.stack((a_b, b_b), dim=-1), temp=temp, dim=-1)


def soft_not(r: Tensor) -> Tensor:
    """Negation: ``-r``."""

    return -r


def soft_implies(a: Tensor, b: Tensor, *, temp: float = 0.1) -> Tensor:
    """Implication: ``(¬a) ∨ b``."""

    return soft_or(soft_not(a), b, temp=temp)


# Predicates


def pred_leq(u: Tensor, c: float | Tensor) -> Tensor:
    """Predicate ``u ≤ c`` as robustness ``c - u``."""

    return c - u


def pred_geq(u: Tensor, c: float | Tensor) -> Tensor:
    """Predicate ``u ≥ c`` as robustness ``u - c``."""

    return u - c


def pred_abs_leq(u: Tensor, c: float | Tensor) -> Tensor:
    """Predicate ``|u| ≤ c`` as robustness ``c - |u|``."""

    return c - u.abs()


def pred_linear_leq(x: Tensor, a: Tensor, b: float | Tensor) -> Tensor:
    """Predicate ``a·x ≤ b`` as robustness ``b - a·x``.

    Expects ``x`` and ``a`` broadcastable with a feature dimension at the end.
    """

    return b - (a * x).sum(dim=-1)


# Temporal operators (whole trace)


def always(margins: Tensor, *, temp: float = 0.1, time_dim: int = -1, keepdim: bool = False) -> Tensor:
    """STL always (□): reduction over time using ``min``."""

    td = _normalize_dim(time_dim, margins.ndim, name="time_dim")
    return softmin(margins, temp=temp, dim=td, keepdim=keepdim)


def eventually(
    margins: Tensor, *, temp: float = 0.1, time_dim: int = -1, keepdim: bool = False
) -> Tensor:
    """STL eventually (◇): reduction over time using ``max``."""

    td = _normalize_dim(time_dim, margins.ndim, name="time_dim")
    return softmax(margins, temp=temp, dim=td, keepdim=keepdim)


# Temporal operators (windowed; robustness trace)


def _windowed_soft_agg(
    margins_last_time: Tensor,
    *,
    window: int,
    stride: int,
    temp: float,
    kind: Literal["min", "max"],
) -> Tensor:
    """Apply a smooth min/max over a sliding window on the last axis.

    Input:
        ``margins_last_time`` has time on the last axis.

    Output:
        shape ``(..., L)`` where ``L`` is the number of windows.
    """

    _validate_window(window=window, stride=stride, T=margins_last_time.shape[-1])
    xw = _unfold_time(margins_last_time, window=window, stride=stride)  # (..., L, window)
    if kind == "max":
        return softmax(xw, temp=temp, dim=-1)
    return softmin(xw, temp=temp, dim=-1)


def always_window(
    margins: Tensor,
    window: int,
    *,
    stride: int = 1,
    temp: float = 0.1,
    time_dim: int = -1,
    keepdim: bool = False,
) -> Tensor:
    """Bounded always (□_[0,window-1]) over a sliding window."""

    x, old_td = _move_time_last(margins, time_dim)
    T = x.shape[-1]
    _validate_window(window=window, stride=stride, T=T)

    y = _windowed_soft_agg(x, window=window, stride=stride, temp=temp, kind="min")  # (..., L)
    if keepdim:
        y = y.unsqueeze(-1)  # (..., L, 1)
    return _restore_time_last(y, old_time_dim=old_td, keepdim=keepdim)


def eventually_window(
    margins: Tensor,
    window: int,
    *,
    stride: int = 1,
    temp: float = 0.1,
    time_dim: int = -1,
    keepdim: bool = False,
) -> Tensor:
    """Bounded eventually (◇_[0,window-1]) over a sliding window."""

    x, old_td = _move_time_last(margins, time_dim)
    T = x.shape[-1]
    _validate_window(window=window, stride=stride, T=T)

    y = _windowed_soft_agg(x, window=window, stride=stride, temp=temp, kind="max")  # (..., L)
    if keepdim:
        y = y.unsqueeze(-1)  # (..., L, 1)
    return _restore_time_last(y, old_time_dim=old_td, keepdim=keepdim)


def _valid_hi(T: int, hi: int | None) -> int:
    if hi is None:
        return T - 1
    if hi < 0:
        raise ValueError("hi must be >= 0.")
    return int(hi)


def until_window(
    a: Tensor,
    b: Tensor,
    *,
    lo: int = 0,
    hi: int | None = None,
    temp: float = 0.1,
    time_dim: int = -1,
    keepdim: bool = False,
) -> Tensor:
    r"""Bounded until: ``a U_[lo,hi] b`` as a robustness trace.

    Discrete-time quantitative semantics (hard) at time ``t``:

    ``max_{k ∈ [lo,hi]} min( b[t+k],  min_{τ ∈ [t, t+k)} a[τ] )``

    where the inner min over an empty set (k=0) is defined as ``+∞``.

    This implementation uses smooth min/max (LogSumExp) and returns values only
    for start times where the full ``hi`` horizon fits in the trace.
    """

    if lo < 0:
        raise ValueError("lo must be >= 0.")

    a_m, old_td = _move_time_last(a, time_dim)
    b_m, _ = _move_time_last(b, time_dim)

    # Allow broadcasting across non-time dims (and also time if one side has T=1).
    # If tensors are not broadcastable (including mismatched time lengths),
    # raise ValueError for a clearer, API-level error.
    try:
        a_m, b_m = torch.broadcast_tensors(a_m, b_m)
    except RuntimeError as exc:
        raise ValueError(
            "a and b must be broadcastable after aligning time_dim (including matching time lengths)."
        ) from exc

    T = a_m.shape[-1]
    hi_i = min(_valid_hi(T, hi), T - 1)
    if lo > hi_i:
        raise ValueError("lo cannot be greater than hi.")

    tau = _check_temp(temp)
    W = hi_i + 1  # window length
    L = T - hi_i  # number of valid windows
    if L <= 0:
        raise ValueError(
            f"No valid windows: time length T={T}, hi={hi_i} implies output length {L}."
        )

    a_w = _unfold_time(a_m, window=W, stride=1)  # (..., L, W)
    b_w = _unfold_time(b_m, window=W, stride=1)  # (..., L, W)

    # prefix_softmin[k] ≈ min_{i<=k} a[i]
    z = -a_w / tau
    prefix_lse = torch.logcumsumexp(z, dim=-1)  # (..., L, W)
    prefix_softmin = -tau * prefix_lse

    # For each k, need min_{i < k} a[i]. Shift right by one and set k=0 to +∞.
    plus_inf = torch.full_like(prefix_softmin[..., :1], float("inf"))
    a_prefix_to_k_minus_1 = torch.cat((plus_inf, prefix_softmin[..., :-1]), dim=-1)

    # Candidate for each k: min( b[t+k],  prefix_min_a(t..t+k-1) ).
    cand = softmin(
        torch.stack((b_w, a_prefix_to_k_minus_1), dim=-1),
        temp=temp,
        dim=-1,
    )  # (..., L, W)

    # Enforce lower bound lo by masking invalid k with -∞ (neutral for max).
    if lo > 0:
        k = torch.arange(W, device=cand.device)
        mask = (k < lo).view(*([1] * (cand.ndim - 1)), W)
        cand = cand.masked_fill(mask, float("-inf"))

    # Outer max over k.
    out = softmax(cand, temp=temp, dim=-1)  # (..., L)
    if keepdim:
        out = out.unsqueeze(-1)  # (..., L, 1)
    return _restore_time_last(out, old_time_dim=old_td, keepdim=keepdim)


def release_window(
    a: Tensor,
    b: Tensor,
    *,
    lo: int = 0,
    hi: int | None = None,
    temp: float = 0.1,
    time_dim: int = -1,
    keepdim: bool = False,
) -> Tensor:
    r"""Bounded release: ``a R_[lo,hi] b`` via duality.

    ``a R b  ≡  ¬(¬a U ¬b)``
    """

    return soft_not(
        until_window(
            soft_not(a),
            soft_not(b),
            lo=lo,
            hi=hi,
            temp=temp,
            time_dim=time_dim,
            keepdim=keepdim,
        )
    )


# Utilities


def shift_left(
    x: Tensor,
    steps: int,
    *,
    time_dim: int = -1,
    pad_value: float = float("nan"),
) -> Tensor:
    """Shift time left by ``steps`` and pad the end with ``pad_value``.

    Example (time_dim=-1):
        [x0, x1, x2, x3] --steps=2--> [x2, x3, pad, pad]

    If ``steps >= T`` (i.e., shifting past the whole signal), the result is all
    padding.
    """

    if steps < 0:
        raise ValueError("steps must be >= 0.")

    x_m, old_td = _move_time_last(x, time_dim)
    T = x_m.shape[-1]
    if steps >= T:
        pad = torch.full_like(x_m, pad_value)
        if old_td == x.ndim - 1:
            return pad
        return pad.movedim(-1, old_td)
    if steps == 0:
        return x

    pad = torch.full_like(x_m[..., :steps], pad_value)
    y = torch.cat((x_m[..., steps:], pad), dim=-1)
    if old_td == x.ndim - 1:
        return y
    return y.movedim(-1, old_td)


def _flip_time(x: Tensor, *, time_dim: int) -> Tensor:
    """Reverse a signal along its time axis."""

    x_m, old_td = _move_time_last(x, time_dim)
    y = torch.flip(x_m, dims=(-1,))
    if old_td == x.ndim - 1:
        return y
    return y.movedim(-1, old_td)


# Past operators (windowed; trace)


def once_window(
    margins: Tensor,
    window: int,
    *,
    temp: float = 0.1,
    time_dim: int = -1,
    keepdim: bool = False,
) -> Tensor:
    """Past "once" (◇ in the past) over the previous ``window`` samples."""

    td = _normalize_dim(time_dim, margins.ndim, name="time_dim")
    rev = _flip_time(margins, time_dim=td)
    y = eventually_window(rev, window, temp=temp, time_dim=td, keepdim=keepdim)
    return _flip_time(y, time_dim=td)


def historically_window(
    margins: Tensor,
    window: int,
    *,
    temp: float = 0.1,
    time_dim: int = -1,
    keepdim: bool = False,
) -> Tensor:
    """Past "historically" (□ in the past) over the previous ``window`` samples."""

    td = _normalize_dim(time_dim, margins.ndim, name="time_dim")
    rev = _flip_time(margins, time_dim=td)
    y = always_window(rev, window, temp=temp, time_dim=td, keepdim=keepdim)
    return _flip_time(y, time_dim=td)


# Loss wrapper


@dataclass(frozen=True)
class STLPenaltyConfig:
    """Configuration for :class:`STLPenalty`.

    Attributes
    weight:
        Overall scale factor (often denoted λ in papers/notes).
    margin:
        Target robustness threshold. For strict satisfaction, use 0.0.
        A positive margin enforces a robustness buffer.
    kind:
        Shape of the penalty as a function of violation ``δ``:
        "softplus" / "logistic" are smooth hinges;
        "hinge" is ReLU; "sqhinge" is squared ReLU.
    beta:
        Sharpness parameter for the soft hinge (higher -> closer to ReLU).
    reduction:
        Reduction over batch/time/etc: "mean", "sum", or "none".
    """

    weight: float = 1.0
    margin: float = 0.0
    kind: Literal["softplus", "logistic", "hinge", "sqhinge"] = "softplus"
    beta: float = 10.0
    reduction: Literal["mean", "sum", "none"] = "mean"


class STLPenalty(torch.nn.Module):
    r"""Convert robustness into a differentiable penalty.

    Let ``ρ`` be a robustness value (positive means satisfied). Define
    ``δ = margin - ρ`` so that ``δ > 0`` indicates violation of the desired
    robustness threshold.

    We then use a smooth hinge:

    * ``penalty(ρ) = softplus(β · δ) / β``

    which behaves like ``max(0, δ)`` for large ``β``.

    The returned value is ``weight · reduce(penalty(ρ))``.
    """

    def __init__(
        self,
        *,
        weight: float = 1.0,
        margin: float = 0.0,
        kind: Literal["softplus", "logistic", "hinge", "sqhinge"] = "softplus",
        beta: float = 10.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.register_buffer("margin", torch.tensor(float(margin)))
        if kind not in {"softplus", "logistic", "hinge", "sqhinge"}:
            raise ValueError("kind must be 'softplus', 'logistic', 'hinge', or 'sqhinge'.")
        self.kind = kind
        if beta <= 0:
            raise ValueError("beta must be > 0.")
        self.beta = float(beta)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def forward(self, robustness: Tensor) -> Tensor:
        if self.weight == 0.0:
            # Return an exact zero *with* a grad_fn so autograd.grad(...) works.
            return robustness.sum() * 0.0

        # δ > 0 means: robustness < margin.
        delta = self.margin.to(device=robustness.device, dtype=robustness.dtype) - robustness

        if self.kind == "softplus":
            base = torch.nn.functional.softplus(self.beta * delta) / self.beta
        elif self.kind == "logistic":
            # Same function as softplus, written explicitly to match common references.
            base = torch.log1p(torch.exp(self.beta * delta)) / self.beta
        elif self.kind == "hinge":
            base = torch.clamp(delta, min=0.0)
        else:  # "sqhinge"
            base = torch.clamp(delta, min=0.0).square()

        if self.reduction == "mean":
            return self.weight * base.mean()
        if self.reduction == "sum":
            return self.weight * base.sum()
        # "none"
        return self.weight * base
