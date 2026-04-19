# tests/test_stl_soft.py
"""Unit tests for differentiable ("soft") STL robustness semantics.

This repository includes a small, self-contained implementation of common
Signal Temporal Logic (STL) quantitative semantics based on smooth
approximations of min/max using log-sum-exp.

These tests emphasize:

* mathematical sanity checks (bounds / identities)
* shape + dtype/device propagation
* differentiability (finite gradients, soft-weight properties)
* windowed temporal operators and their discrete-time alignment

Note: the repo demo scripts call
``pytest -q tests/test_stl_soft.py::test_soft_and_or_implies_behave_reasonably``.
Keep that test CPU-friendly and fast.
"""

from __future__ import annotations

import math
import pathlib

import pytest


# Skip these tests if PyTorch is not installed in the environment.
torch = pytest.importorskip("torch")


# Support running tests from a source checkout without installation.
try:  # pragma: no cover - import convenience
    from neural_pde_stl_strel.monitoring import stl_soft as stl  # type: ignore
except Exception:  # pragma: no cover - import convenience
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if SRC.exists():
        import sys

        sys.path.insert(0, str(SRC))
    from neural_pde_stl_strel.monitoring import stl_soft as stl  # type: ignore  # noqa: E402


TOL64 = 1e-12
TOL32 = 1e-6


def _devices():
    """Devices to validate dtype/device propagation.

    On Apple MPS, float64 support and/or numerical kernels can differ; we keep
    value-precision checks on CPU and use MPS only for simple propagation tests.
    """

    yield torch.device("cpu")
    if torch.cuda.is_available():
        yield torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        yield torch.device("mps")


def _atol_for(dtype: torch.dtype) -> float:
    return TOL32 if dtype == torch.float32 else TOL64


def _softmin_ref(x: torch.Tensor, *, tau: float, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Reference soft-min: -tau * logsumexp(-x/tau)."""

    return -(tau * torch.logsumexp(-x / tau, dim=dim, keepdim=keepdim))


def _softmax_ref(x: torch.Tensor, *, tau: float, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Reference soft-max: tau * logsumexp(x/tau)."""

    return tau * torch.logsumexp(x / tau, dim=dim, keepdim=keepdim)


def _windowed_ref(
    margins: torch.Tensor,
    *,
    window: int,
    stride: int,
    tau: float,
    kind: str,
) -> torch.Tensor:
    """Naive sliding-window reference for always_window/eventually_window.

    Assumes time is the last dimension.
    """

    assert kind in {"min", "max"}
    t = margins.shape[-1]
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if window > t:
        # Match torch.unfold behavior (RuntimeError), but keep this helper simple.
        raise RuntimeError("window larger than signal")

    outs = []
    for start in range(0, t - window + 1, stride):
        sl = margins[..., start : start + window]
        if kind == "min":
            outs.append(_softmin_ref(sl, tau=tau, dim=-1))
        else:
            outs.append(_softmax_ref(sl, tau=tau, dim=-1))
    return torch.stack(outs, dim=-1)


def _until_window_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    lo: int,
    hi: int | None,
    tau: float,
    time_dim: int = -1,
) -> torch.Tensor:
    """Naive discrete-time reference for `until_window` using the *same* soft ops.

    This mirrors the docstring semantics:

        (a U_[lo,hi] b)(t) = max_{k ∈ [lo, hi]} min(b[t+k], min_{τ ∈ [t, t+k)} a[τ])

    The nested max/min are implemented using smooth log-sum-exp approximations.

    Returns a tensor whose time axis has length L = T - hi and is placed at
    ``time_dim`` (matching :func:`neural_pde_stl_strel.monitoring.stl_soft.until_window`).
    """

    if lo < 0:
        raise ValueError("lo must be >= 0")

    old_dim = int(time_dim) % a.ndim
    a_m = a.movedim(old_dim, -1)
    b_m = b.movedim(old_dim, -1)
    if a_m.shape != b_m.shape:
        raise ValueError("a and b must have the same shape")

    t = a_m.shape[-1]
    if hi is None:
        hi_i = t - 1
    else:
        if hi < 0:
            raise ValueError("hi must be >= 0")
        hi_i = min(int(hi), t - 1)
    if lo > hi_i:
        raise ValueError("lo must be <= hi")

    w = hi_i + 1
    l = t - hi_i

    out_ts = []
    for start_t in range(l):
        cand_ks = []
        for k in range(w):
            if k < lo:
                cand_ks.append(torch.full_like(a_m[..., 0], float("-inf")))
                continue

            # prefix min over a[start_t : start_t + k] (empty for k=0)
            if k == 0:
                prefix = torch.full_like(a_m[..., 0], float("inf"))
            else:
                prefix = _softmin_ref(a_m[..., start_t : start_t + k], tau=tau, dim=-1)

            b_k = b_m[..., start_t + k]
            cand_ks.append(_softmin_ref(torch.stack([b_k, prefix], dim=-1), tau=tau, dim=-1))

        cand = torch.stack(cand_ks, dim=-1)
        out_ts.append(_softmax_ref(cand, tau=tau, dim=-1))

    out = torch.stack(out_ts, dim=-1)  # time last
    if old_dim != a.ndim - 1:
        out = out.movedim(-1, old_dim)
    return out


# Input validation


@pytest.mark.parametrize("bad_temp", [0.0, -1.0])
def test_temperature_must_be_positive(bad_temp: float) -> None:
    x = torch.tensor([0.0, 1.0], dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = stl.softmin(x, temp=bad_temp)
    with pytest.raises(ValueError):
        _ = stl.softmax(x, temp=bad_temp)


def test_shift_left_rejects_negative_steps() -> None:
    x = torch.arange(5, dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = stl.shift_left(x, steps=-1)


@pytest.mark.parametrize("window", [0, -3])
def test_windowed_ops_reject_nonpositive_window(window: int) -> None:
    x = torch.zeros(5, dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = stl.always_window(x, window=window)
    with pytest.raises(ValueError):
        _ = stl.eventually_window(x, window=window)


@pytest.mark.parametrize("stride", [0, -2])
def test_windowed_ops_reject_nonpositive_stride(stride: int) -> None:
    x = torch.zeros(5, dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = stl.always_window(x, window=2, stride=stride)
    with pytest.raises(ValueError):
        _ = stl.eventually_window(x, window=2, stride=stride)


# Predicates -> robustness margins


def test_pred_leq_basic_and_broadcast() -> None:
    u = torch.tensor([0.1, 0.4, 0.5], dtype=torch.float32)
    margins = stl.pred_leq(u, 0.5)
    expected = torch.tensor([0.4, 0.1, 0.0], dtype=torch.float32)
    assert torch.allclose(margins, expected, atol=TOL32)

    # Broadcasting: (2 x 3) - scalar
    u2 = torch.tensor(
        [[0.0, 1.0, 2.0], [0.5, 0.5, 0.5]],
        dtype=torch.float32,
    )
    margins2 = stl.pred_leq(u2, 1.0)
    expected2 = torch.tensor(
        [[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]],
        dtype=torch.float32,
    )
    assert torch.allclose(margins2, expected2, atol=TOL32)
    assert margins2.dtype == u2.dtype and margins2.device == u2.device


def test_pred_leq_broadcast_vector_threshold() -> None:
    # Broadcasting against a per-feature threshold vector
    u = torch.tensor([[0.0, 1.0, 2.0], [2.5, 2.0, 1.5]], dtype=torch.float64)
    c = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float64)  # shape (3,)
    margins = stl.pred_leq(u, c)  # expected (2,3) via broadcasting
    expected = torch.tensor([[1.0, 1.0, -0.5], [-1.5, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(margins, expected, atol=TOL64)
    assert margins.shape == u.shape


def test_pred_geq_is_negated_pred_leq() -> None:
    u = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float64)
    c = 1.25
    geq = stl.pred_geq(u, c)
    leq = stl.pred_leq(u, c)
    assert torch.allclose(geq, -leq, atol=TOL64)


def test_pred_abs_leq_matches_definition() -> None:
    u = torch.tensor([-2.0, -0.5, 0.0, 0.4, 1.5], dtype=torch.float32)
    margins = stl.pred_abs_leq(u, 1.0)
    expected = torch.tensor([-1.0, 0.5, 1.0, 0.6, -0.5], dtype=torch.float32)
    assert torch.allclose(margins, expected, atol=TOL32)


def test_pred_linear_leq_inner_product_over_last_dim() -> None:
    # a·x <= b  -> robustness = b - a·x
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    a = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    b = 1.25
    margins = stl.pred_linear_leq(x, a, b)
    expected = b - (x * a).sum(dim=-1)
    assert torch.allclose(margins, expected, atol=TOL64)
    assert margins.shape == (2,)


# Smooth min/max and Boolean connectives


@pytest.mark.parametrize("fn", [stl.softmin, stl.softmax])
def test_softmin_softmax_dtype_device_and_grad(fn) -> None:
    for dev in _devices():
        for dtype in (torch.float32, torch.float64):
            if dev.type == "mps" and dtype == torch.float64:
                continue

            x = torch.tensor([0.2, 0.8, 0.5], dtype=dtype, device=dev, requires_grad=True)
            y = fn(x, temp=0.1, dim=-1)

            assert y.dtype == x.dtype and y.device == x.device
            assert y.ndim == 0

            (g,) = torch.autograd.grad(y, x, retain_graph=False, allow_unused=False)
            assert g is not None
            assert torch.isfinite(g).all()


def test_softmin_softmax_bounds_and_singleton_identity() -> None:
    x = torch.tensor([0.2, 0.8, 0.5], dtype=torch.float32)
    tau = 0.05
    sm = stl.softmin(x, temp=tau)
    sx = stl.softmax(x, temp=tau)

    # For any tau > 0:
    #   softmin(x) <= min(x)
    #   softmax(x) >= max(x)
    assert sm.item() <= x.min().item() + 1e-7
    assert sx.item() >= x.max().item() - 1e-7

    # Singleton: exact identity for any temperature.
    y = torch.tensor([0.42], dtype=torch.float32)
    for t in (1e-3, 0.1, 1.0):
        assert torch.allclose(stl.softmin(y, temp=t), y, atol=TOL32)
        assert torch.allclose(stl.softmax(y, temp=t), y, atol=TOL32)


def test_softmin_softmax_keepdim_matches_unsqueezed() -> None:
    x = torch.tensor([[1.0, 0.0, 2.0], [-1.0, 3.0, 4.0]], dtype=torch.float64)
    tau = 0.2

    sm = stl.softmin(x, temp=tau, dim=-1)
    sm_k = stl.softmin(x, temp=tau, dim=-1, keepdim=True)
    assert sm_k.shape == (2, 1)
    assert torch.allclose(sm_k.squeeze(-1), sm, atol=TOL64)

    sx = stl.softmax(x, temp=tau, dim=-1)
    sx_k = stl.softmax(x, temp=tau, dim=-1, keepdim=True)
    assert sx_k.shape == (2, 1)
    assert torch.allclose(sx_k.squeeze(-1), sx, atol=TOL64)


def test_soft_not_is_exact_negation_and_preserves_grad() -> None:
    r = torch.tensor([2.0, -3.0, 0.1], dtype=torch.float64, requires_grad=True)
    out = stl.soft_not(r)
    assert torch.allclose(out, -r, atol=TOL64)

    loss = out.sum()
    (g,) = torch.autograd.grad(loss, r)
    assert torch.allclose(g, -torch.ones_like(r), atol=TOL64)


def test_soft_and_or_implies_behave_reasonably() -> None:
    """Smoke-test the core differentiable Boolean connectives.

    This is intentionally *demo-friendly* (fast on CPU) and checks:

    * AND ≈ min(a, b)
    * OR  ≈ max(a, b)
    * IMPLIES: a -> b ≡ ¬a ∨ b  ≈ max(-a, b)

    The approximation uses log-sum-exp smoothing:

        softmax(x) = tau * logsumexp(x / tau)
        softmin(x) = -tau * logsumexp(-x / tau)

    For n=2 values, the error band is bounded by tau*log(2).
    """

    tau = 0.05
    # Robustness values: positive = satisfied, negative = violated.
    a = torch.tensor([2.0, -3.0, 0.1], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([-0.5, -1.0, 0.2], dtype=torch.float64, requires_grad=True)

    r_and = stl.soft_and(a, b, temp=tau)
    r_or = stl.soft_or(a, b, temp=tau)
    r_imp = stl.soft_implies(a, b, temp=tau)

    assert r_and.shape == a.shape
    assert r_or.shape == a.shape
    assert r_imp.shape == a.shape

    hard_min = torch.minimum(a, b)
    hard_max = torch.maximum(a, b)
    hard_imp = torch.maximum(-a, b)

    tol = tau * math.log(2.0) + TOL64

    # AND ≈ softmin: never above hard min, and not too far below it.
    assert torch.all(r_and <= hard_min + TOL64)
    assert torch.all(hard_min - r_and <= tol)

    # OR ≈ softmax: never below hard max, and not too far above it.
    assert torch.all(r_or >= hard_max - TOL64)
    assert torch.all(r_or - hard_max <= tol)

    # IMPLIES behaves like max(-a, b) with the same softmax error band.
    assert torch.all(r_imp >= hard_imp - TOL64)
    assert torch.all(r_imp - hard_imp <= tol)

    # Qualitative spot-check:
    # antecedent true but consequent false -> implication violated (negative)
    assert r_imp[0].item() < 0
    # antecedent false -> implication satisfied by vacuity (positive)
    assert r_imp[1].item() > 0

    # Differentiability: gradients should exist and be finite.
    (r_and + r_or + r_imp).sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()


def test_softmin_softmax_gradients_are_soft_weights() -> None:
    # For f(x) = tau * logsumexp(x/tau), ∂f/∂x = softmax(x/tau).
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    tau = 0.25
    f = stl.softmax(x, temp=tau)
    (grad_f,) = torch.autograd.grad(f, x, create_graph=False)
    assert torch.all(grad_f >= 0) and torch.all(grad_f <= 1)
    assert torch.allclose(grad_f.sum(), torch.tensor(1.0, dtype=grad_f.dtype), atol=TOL64)
    assert grad_f.argmax().item() == x.argmax().item()

    # For g(x) = -tau * logsumexp(-x/tau), the gradient concentrates at the min.
    x2 = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64, requires_grad=True)
    g = stl.softmin(x2, temp=tau)
    (grad_g,) = torch.autograd.grad(g, x2, create_graph=False)
    assert torch.all(grad_g >= 0) and torch.all(grad_g <= 1)
    assert torch.allclose(grad_g.sum(), torch.tensor(1.0, dtype=grad_g.dtype), atol=TOL64)
    assert grad_g.argmax().item() == x2.argmin().item()


def test_softmin_softmax_temperature_sharpness() -> None:
    # Smaller temperature -> closer to hard min/max.
    x = torch.tensor([-1.0, 0.0, 3.0, 0.5], dtype=torch.float64)
    tau_small, tau_mid = 0.01, 0.25
    hard_min, hard_max = x.min(), x.max()

    sm_small = stl.softmin(x, temp=tau_small)
    sx_small = stl.softmax(x, temp=tau_small)
    sm_mid = stl.softmin(x, temp=tau_mid)
    sx_mid = stl.softmax(x, temp=tau_mid)

    assert abs(sm_small.item() - hard_min.item()) <= abs(sm_mid.item() - hard_min.item()) + TOL64
    assert abs(sx_small.item() - hard_max.item()) <= abs(sx_mid.item() - hard_max.item()) + TOL64


def test_numerical_stability_very_small_temperature() -> None:
    # Large dynamic range should not cause inf/nan thanks to log-sum-exp.
    x = torch.linspace(-50.0, 50.0, steps=101, dtype=torch.float32)
    for fn in (stl.softmin, stl.softmax):
        y = fn(x, temp=1e-3)
        assert torch.isfinite(y).all()


# Temporal operators (unbounded)


def test_temporal_always_eventually_reduce_over_time_dim_and_bound_true_min_max() -> None:
    margins = torch.tensor(
        [[0.0, 1.0, -1.0, 2.0], [-2.0, -0.5, 0.5, 3.0]],
        dtype=torch.float32,
    )

    g = stl.always(margins, temp=0.05)  # ~ min over time
    f = stl.eventually(margins, temp=0.05)  # ~ max over time
    assert g.shape == (2,) and f.shape == (2,)

    true_min = margins.min(dim=-1).values
    true_max = margins.max(dim=-1).values
    assert torch.all(g <= true_min + TOL32)
    assert torch.all(f >= true_max - TOL32)

    # Explicit time_dim should match.
    g0 = stl.always(margins.t(), temp=0.05, time_dim=0)
    f0 = stl.eventually(margins.t(), temp=0.05, time_dim=0)
    assert torch.allclose(g0, g, atol=TOL32)
    assert torch.allclose(f0, f, atol=TOL32)


def test_temporal_ops_singleton_time_axis_are_identity() -> None:
    margins = torch.tensor([[0.3], [-1.2]], dtype=torch.float32)  # shape (B, 1)
    g = stl.always(margins, time_dim=-1)
    f = stl.eventually(margins, time_dim=-1)
    assert torch.allclose(g, margins.squeeze(-1), atol=TOL32)
    assert torch.allclose(f, margins.squeeze(-1), atol=TOL32)


def test_always_eventually_match_softmin_softmax_on_same_dim() -> None:
    b, t, d = 2, 5, 3
    x = torch.arange(b * t * d, dtype=torch.float64).reshape(b, t, d) / 7.0 - 4.0

    g = stl.always(x, temp=0.1, time_dim=1)
    f = stl.eventually(x, temp=0.1, time_dim=1)
    g_ref = stl.softmin(x, temp=0.1, dim=1)
    f_ref = stl.softmax(x, temp=0.1, dim=1)
    assert torch.allclose(g, g_ref, atol=TOL64)
    assert torch.allclose(f, f_ref, atol=TOL64)
    assert g.shape == (b, d) and f.shape == (b, d)


def test_temporal_gradients_sum_to_one_over_time() -> None:
    x = torch.tensor([[-1.0, 0.0, 2.0, -0.5, 1.0]], dtype=torch.float64, requires_grad=True)
    tau = 0.2

    f = stl.eventually(x, temp=tau)  # (1,)
    (g_f,) = torch.autograd.grad(f, x, retain_graph=True)
    assert torch.all(g_f >= 0) and torch.all(g_f <= 1)
    assert torch.allclose(g_f.sum(dim=-1), torch.ones_like(f), atol=TOL64)

    g = stl.always(x, temp=tau)
    (g_g,) = torch.autograd.grad(g, x, retain_graph=False)
    assert torch.all(g_g >= 0) and torch.all(g_g <= 1)
    assert torch.allclose(g_g.sum(dim=-1), torch.ones_like(g), atol=TOL64)


def test_temporal_ops_handle_noncontiguous_tensors() -> None:
    # Construct a deterministic non-contiguous tensor via transpose + slice.
    x = (torch.arange(3 * 8 * 5, dtype=torch.float32).reshape(3, 8, 5) / 17.0) - 2.0
    x_nc = x.transpose(0, 1)[1:, :, :]  # shape (7, 3, 5), non-contiguous

    g = stl.always(x_nc, temp=0.123, time_dim=0)
    f = stl.eventually(x_nc, temp=0.123, time_dim=0)
    assert g.shape == (3, 5)
    assert f.shape == (3, 5)

    g_ref = stl.softmin(x_nc, temp=0.123, dim=0)
    f_ref = stl.softmax(x_nc, temp=0.123, dim=0)
    assert torch.allclose(g, g_ref, atol=1e-7)
    assert torch.allclose(f, f_ref, atol=1e-7)


# Temporal operators (bounded / windowed)


def test_always_eventually_window_identity_when_window_is_one() -> None:
    x = torch.tensor([1.0, -2.0, 3.0, 0.5], dtype=torch.float64)
    tau = 0.3

    g1 = stl.always_window(x, window=1, temp=tau)
    f1 = stl.eventually_window(x, window=1, temp=tau)
    assert g1.shape == x.shape
    assert f1.shape == x.shape
    assert torch.allclose(g1, x, atol=TOL64)
    assert torch.allclose(f1, x, atol=TOL64)


def test_always_eventually_window_match_naive_reference() -> None:
    margins = torch.tensor(
        [[0.0, 1.0, -1.0, 2.0, 3.0], [-2.0, -0.5, 0.5, 3.0, -4.0]],
        dtype=torch.float64,
    )
    window = 3
    stride = 1
    tau = 0.2

    g = stl.always_window(margins, window=window, stride=stride, temp=tau)
    f = stl.eventually_window(margins, window=window, stride=stride, temp=tau)
    assert g.shape == (2, margins.shape[-1] - window + 1)
    assert f.shape == g.shape

    g_ref = _windowed_ref(margins, window=window, stride=stride, tau=tau, kind="min")
    f_ref = _windowed_ref(margins, window=window, stride=stride, tau=tau, kind="max")
    assert torch.allclose(g, g_ref, atol=1e-12)
    assert torch.allclose(f, f_ref, atol=1e-12)


def test_eventually_window_stride_changes_alignment() -> None:
    # Example (STL-friendly): F_[0,1](temp <= 25) evaluated every 2 steps.
    temp = torch.tensor([50.0, 30.0, 24.0, 40.0, 23.0, 22.0], dtype=torch.float64)
    margins = stl.pred_leq(temp, 25.0)  # robustness margins for predicate temp <= 25
    tau = 0.1

    # Sliding window of length 2 (next 2 samples), but only at t=0,2,4.
    out = stl.eventually_window(margins, window=2, stride=2, temp=tau)
    ref = _windowed_ref(margins, window=2, stride=2, tau=tau, kind="max")
    assert torch.allclose(out, ref, atol=TOL64)
    assert out.shape == (3,)


def test_windowed_ops_error_when_window_exceeds_signal_length() -> None:
    x = torch.zeros(3, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _ = stl.always_window(x, window=4)
    with pytest.raises(RuntimeError):
        _ = stl.eventually_window(x, window=4)


# Until / release (bounded)


def test_until_window_hi_zero_reduces_to_b_identity() -> None:
    # With hi=0, the only admissible k is 0, and the prefix over a is empty (+inf).
    # So: a U_[0,0] b == b.
    a = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float64)
    b = torch.tensor([-1.0, 2.0, -3.0, 4.0], dtype=torch.float64)
    tau = 0.2
    out = stl.until_window(a, b, lo=0, hi=0, temp=tau)
    assert out.shape == b.shape
    assert torch.allclose(out, b, atol=TOL64)


def test_until_window_matches_naive_reference_for_smooth_semantics() -> None:
    # a and b are robustness margins at each time step.
    # We compare the vectorized implementation against a small, explicit loop.
    a = torch.tensor([[1.0, -2.0, 3.0, 0.5, -1.5]], dtype=torch.float64)  # (B=1, T=5)
    b = torch.tensor([[-1.0, 2.0, -3.0, 4.0, 0.0]], dtype=torch.float64)
    tau = 0.3
    lo, hi = 1, 3

    out = stl.until_window(a, b, lo=lo, hi=hi, temp=tau, time_dim=-1)
    ref = _until_window_ref(a, b, lo=lo, hi=hi, tau=tau, time_dim=-1)
    assert torch.allclose(out, ref, atol=TOL64)
    assert out.shape == (1, a.shape[-1] - hi)


def test_until_window_time_dim_argument_is_respected() -> None:
    # Same data, but time is the first dimension.
    a = torch.tensor(
        [[1.0, -2.0], [3.0, 0.5], [-1.5, 2.5], [0.0, -0.25]],
        dtype=torch.float64,
    )  # (T=4, B=2)
    b = torch.tensor(
        [[-1.0, 2.0], [-3.0, 4.0], [0.0, 1.0], [2.0, -2.0]],
        dtype=torch.float64,
    )
    tau = 0.2
    lo, hi = 0, 2

    out = stl.until_window(a, b, lo=lo, hi=hi, temp=tau, time_dim=0)
    ref = _until_window_ref(a, b, lo=lo, hi=hi, tau=tau, time_dim=0)
    assert torch.allclose(out, ref, atol=TOL64)
    assert out.shape == (a.shape[0] - hi, a.shape[1])


def test_until_window_rejects_bad_bounds_and_mismatched_shapes() -> None:
    a = torch.zeros(5, dtype=torch.float32)
    b = torch.zeros(4, dtype=torch.float32)

    with pytest.raises(ValueError):
        _ = stl.until_window(a, b, lo=0, hi=1)

    with pytest.raises(ValueError):
        _ = stl.until_window(a, a, lo=-1, hi=1)
    with pytest.raises(ValueError):
        _ = stl.until_window(a, a, lo=2, hi=1)
    with pytest.raises(ValueError):
        _ = stl.until_window(a, a, lo=0, hi=-3)


def test_release_window_matches_duality_definition() -> None:
    # release_window(a,b) = ¬(¬a U ¬b) = - until_window(-a, -b)
    a = torch.tensor([0.5, -1.0, 2.0, 0.0], dtype=torch.float64)
    b = torch.tensor([-0.25, 3.0, -4.0, 1.0], dtype=torch.float64)
    tau = 0.15
    lo, hi = 1, 3

    rel = stl.release_window(a, b, lo=lo, hi=hi, temp=tau)
    dual = -stl.until_window(-a, -b, lo=lo, hi=hi, temp=tau)
    assert torch.allclose(rel, dual, atol=TOL64)


# Shift and past-time operators


def test_shift_left_basic_and_time_dim() -> None:
    x = torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], dtype=torch.float32)

    # time_dim=-1 (default)
    y = stl.shift_left(x, steps=1, pad_value=999.0)
    expected = torch.tensor([[1.0, 2.0, 999.0], [11.0, 12.0, 999.0]], dtype=torch.float32)
    assert torch.allclose(y, expected, atol=TOL32)

    # time_dim=0
    y0 = stl.shift_left(x.t(), steps=1, time_dim=0, pad_value=999.0)
    assert torch.allclose(y0, expected.t(), atol=TOL32)


def test_shift_left_steps_greater_than_length_pads_everything() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    y = stl.shift_left(x, steps=99, pad_value=-7.0)
    assert torch.allclose(y, torch.full_like(x, -7.0), atol=TOL64)


def test_once_and_historically_window_match_naive_past_windows() -> None:
    # Past operators are implemented by reversing time and applying future windowed ops.
    # They return outputs aligned to times t = window-1 .. T-1 (length T-window+1).
    margins = torch.tensor([1.0, -2.0, 3.0, 0.5, -1.5], dtype=torch.float64)
    window = 3
    tau = 0.2

    once = stl.once_window(margins, window=window, temp=tau)
    hist = stl.historically_window(margins, window=window, temp=tau)
    assert once.shape == (margins.numel() - window + 1,)
    assert hist.shape == once.shape

    # Naive past-window reference: at time t (>= window-1), aggregate margins[t-window+1:t+1].
    once_ref = []
    hist_ref = []
    for t in range(window - 1, margins.numel()):
        sl = margins[t - window + 1 : t + 1]
        once_ref.append(_softmax_ref(sl, tau=tau, dim=-1))
        hist_ref.append(_softmin_ref(sl, tau=tau, dim=-1))
    once_ref_t = torch.stack(once_ref, dim=-1)
    hist_ref_t = torch.stack(hist_ref, dim=-1)

    assert torch.allclose(once, once_ref_t, atol=TOL64)
    assert torch.allclose(hist, hist_ref_t, atol=TOL64)


# Penalty module


def test_stlpenalty_behavior_and_margin_shift() -> None:
    # For large positive robustness, penalty ~ 0; for large negative, penalty grows.
    penalty = stl.STLPenalty(weight=1.0, margin=0.0)
    high = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    low = torch.tensor([-10.0], dtype=torch.float64, requires_grad=True)
    v_high = penalty(high)
    v_low = penalty(low)
    assert v_high.item() < 1e-6
    assert v_low.item() > 1.0

    # Monotone decreasing in robustness (elementwise).
    penalty_elem = stl.STLPenalty(weight=1.0, margin=0.0, reduction="none")
    r = torch.tensor([-2.0, -1.0, 0.0, 1.0], dtype=torch.float64)
    vals = penalty_elem(r)
    assert vals.shape == r.shape
    assert torch.all(vals[:-1] >= vals[1:])

    # Weight=0 yields exact 0 output, with gradients either None or exactly zero.
    penalty_zero = stl.STLPenalty(weight=0.0, margin=0.0)
    z = torch.tensor([-1.0, 1.0], dtype=torch.float64, requires_grad=True)
    out = penalty_zero(z)
    assert float(out.detach()) == 0.0
    (g,) = torch.autograd.grad(out, z, retain_graph=False, allow_unused=True)
    if g is not None:
        assert torch.allclose(g, torch.zeros_like(z), atol=TOL64)

    # Margin acts as a shift: at robustness == margin, softplus(0) = log(2).
    p = stl.STLPenalty(weight=1.0, margin=1.0, beta=1.0)
    val_at_margin = p(torch.tensor([1.0], dtype=torch.float64))
    assert abs(val_at_margin.item() - math.log(2.0)) < 1e-6


@pytest.mark.parametrize("kind", ["softplus", "logistic", "hinge", "sqhinge"])
def test_stlpenalty_kinds_match_reference_formulas(kind: str) -> None:
    w, m, beta = 3.5, -0.25, 7.0
    r = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
    penalty = stl.STLPenalty(weight=w, margin=m, beta=beta, kind=kind)
    out = penalty(r)

    delta = (m - r)
    if kind == "softplus":
        base = torch.nn.functional.softplus(beta * delta) / beta
    elif kind == "logistic":
        base = torch.log1p(torch.exp(beta * delta)) / beta
    elif kind == "hinge":
        base = torch.clamp(delta, min=0.0)
    elif kind == "sqhinge":
        base = torch.clamp(delta, min=0.0).square()
    else:  # pragma: no cover
        raise AssertionError("unreachable")

    expected = w * base.mean()
    assert torch.allclose(out, expected, atol=TOL64)


def test_stlpenalty_reduction_sum_and_none() -> None:
    r = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64)
    m = 0.5
    beta = 2.0

    none = stl.STLPenalty(weight=1.0, margin=m, beta=beta, reduction="none")(r)
    assert none.shape == r.shape

    summ = stl.STLPenalty(weight=1.0, margin=m, beta=beta, reduction="sum")(r)
    assert torch.allclose(summ, none.sum(), atol=TOL64)


def test_stlpenalty_rejects_invalid_kind_and_reduction() -> None:
    with pytest.raises(ValueError):
        _ = stl.STLPenalty(kind="not-a-kind")
    with pytest.raises(ValueError):
        _ = stl.STLPenalty(reduction="not-a-reduction")
