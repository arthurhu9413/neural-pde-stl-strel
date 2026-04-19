#!/usr/bin/env python3
"""
Evaluate (robust) satisfaction of STL specifications on diffusion rollouts.

This script is used throughout the repo to *monitor* properties of a 1-D diffusion
PINN rollout saved as a ``*_field.pt`` artifact (see ``results/``). The artifact
typically contains a dense field ``u(x,t)`` (2-D tensor) and mesh-grids ``X, T``
with the same shape.

Data-flow (what is monitored)
1) Load a PDE field rollout ``u(x,t)`` from the checkpoint.
2) Reduce the spatial dimensions to a **single scalar signal** ``s(t)`` using an
   aggregation operator (default: ``amax``, i.e. worst-case over space).
3) Monitor an STL specification over the resulting scalar signal ``s(t)`` and
   report its robustness.

Two convenient entry points
The report/demo uses the short commands below, which auto-select the latest
checkpoint for the requested tag and write JSON summaries to ``results/``:

    python scripts/eval_diffusion_rtamt.py --baseline
    python scripts/eval_diffusion_rtamt.py --stl

Explicit checkpoint path (e.g. from a custom run folder):

    python scripts/eval_diffusion_rtamt.py --ckpt results/diffusion1d--baseline--*/diffusion1d_baseline_field.pt \
        --agg amax --op always --spec upper --u-max 1.0 --json results/my_eval.json

Custom STL formula (RTAMT required)
For custom formulas, the monitored signal name is **``s``** (after aggregation):

    python scripts/eval_diffusion_rtamt.py --ckpt <...> --agg amax --semantics dense --dt 0.015873 \
        --formula "eventually[0:1](s <= 0.2)" --t0 0.5 --t1 1.0

Notes
-----
- If RTAMT is not available, we provide an *exact* fallback for the built-in
  predicate forms (upper/lower/range) with outer operator always/eventually.
  Arbitrary custom formulas require RTAMT.
- Time windows can be specified either in *absolute time* (``--t0/--t1`` using
  ``T`` from the checkpoint or ``--dt``) or by *indices* (``--idx0/--idx1``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np

# Allow running from a source checkout without requiring the user to set PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from neural_pde_stl_strel.monitoring.rtamt_monitor import (  # noqa: E402
    evaluate_series as _rtamt_eval_series,
)
from neural_pde_stl_strel.monitoring.rtamt_monitor import (  # noqa: E402
    satisfied as _rtamt_satisfied,
)
from neural_pde_stl_strel.monitoring.rtamt_monitor import (  # noqa: E402
    stl_always_upper_bound as _rtamt_stl_always_upper,
)

_STL_RUN_SENTINEL = "__SELECT_STL_RUN__"


def _rtamt_available() -> bool:
    """Return True if RTAMT is importable in the current environment."""
    try:
        import rtamt  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class Args:
    # Input selection
    ckpt: str | None
    results_dir: str
    baseline: bool
    stl: str | None  # optional arg; no value => select STL run; value => legacy formula string

    # Field extraction + reduction
    var: str
    stl_var: str
    time_axis: int
    agg: str
    p: float | None
    q: float | None
    temp: float

    # Monitoring semantics
    semantics: str
    dt: float | None

    # Windowing
    t0: float | None
    t1: float | None
    idx0: int | None
    idx1: int | None

    # Spec
    op: str
    spec: str
    u_max: float | None
    u_min: float | None
    formula: str | None  # preferred custom formula string

    # Outputs
    json_out: str | None
    dump_series: str | None
    plot: str | None

    verbose: bool


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate STL robustness on a diffusion rollout (RTAMT optional).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument(
        "--ckpt",
        "--field",
        dest="ckpt",
        default=None,
        help="Path to a saved diffusion field checkpoint (*.pt). Alias: --field.",
    )
    ap.add_argument(
        "--results-dir",
        default=str(_REPO_ROOT / "results"),
        help="Results directory used for --baseline/--stl auto-discovery.",
    )
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="Auto-select the most recent baseline field checkpoint in results/.",
    )

    # NOTE: Backwards compatibility + report convenience:
    #   - `--stl` (no value): select the STL-trained run (report Appendix B).
    #   - `--stl "<formula>"`: legacy custom formula string (deprecated; use --formula).
    ap.add_argument(
        "--stl",
        nargs="?",
        const=_STL_RUN_SENTINEL,
        default=None,
        help=(
            "If provided without a value, auto-select the most recent STL-trained run. "
            "If provided WITH a value, treat it as a legacy custom STL formula string "
            "(deprecated; prefer --formula)."
        ),
    )

    ap.add_argument(
        "--var",
        default="u",
        help="Checkpoint key containing the PDE field to monitor (e.g. 'u').",
    )
    ap.add_argument(
        "--stl-var",
        default="s",
        help="Signal name used inside STL formulas after spatial aggregation.",
    )
    ap.add_argument(
        "--time-axis",
        type=int,
        default=-1,
        help="Axis of the field tensor that corresponds to time; -1 means infer.",
    )
    ap.add_argument(
        "--agg",
        choices=["mean", "amax", "amin", "median", "quantile", "lp", "softmax"],
        default="amax",
        help="Spatial reduction used to map u(x,t) to a scalar signal s(t).",
    )
    ap.add_argument(
        "--p",
        type=float,
        default=None,
        help="p for lp aggregation: (mean |x|^p)^(1/p).",
    )
    ap.add_argument(
        "--q",
        type=float,
        default=None,
        help="q for quantile aggregation (e.g. 0.95).",
    )
    ap.add_argument(
        "--temp",
        type=float,
        default=0.1,
        help="Temperature for softmax aggregation; higher≈harder max. Must be > 0.",
    )

    ap.add_argument(
        "--semantics",
        choices=["dense", "discrete"],
        default="discrete",
        help="RTAMT time interpretation. Fallback monitoring ignores this for simple specs.",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Sampling period if checkpoint lacks an explicit T vector (needed for dense time).",
    )

    ap.add_argument("--t0", type=float, default=None, help="Start time (inclusive).")
    ap.add_argument("--t1", type=float, default=None, help="End time (inclusive).")
    ap.add_argument(
        "--idx0",
        type=int,
        default=None,
        help="Start index on the reduced series (inclusive). Applied after t0/t1 slicing.",
    )
    ap.add_argument(
        "--idx1",
        type=int,
        default=None,
        help="End index on the reduced series (exclusive). Applied after t0/t1 slicing.",
    )

    ap.add_argument(
        "--op",
        choices=["always", "eventually"],
        default="always",
        help="Temporal operator applied to the predicate (built-in specs only).",
    )
    ap.add_argument(
        "--spec",
        choices=["upper", "lower", "range"],
        default="upper",
        help="Predicate form for built-in monitoring.",
    )
    ap.add_argument("--u-max", type=float, default=None, help="Upper bound for 'upper'/'range' specs.")
    ap.add_argument("--u-min", type=float, default=None, help="Lower bound for 'lower'/'range' specs.")
    ap.add_argument(
        "--formula",
        type=str,
        default=None,
        help="Custom RTAMT STL formula string (uses --stl-var, default 's'). Requires RTAMT.",
    )

    ap.add_argument("--json", dest="json_out", default=None, help="Write summary JSON to this path.")
    ap.add_argument(
        "--dump-series",
        default=None,
        help="Write the reduced signal (time,value) as a CSV for plotting.",
    )
    ap.add_argument(
        "--plot",
        default=None,
        help="Optional: save a simple plot of the reduced signal to this path (e.g. .png).",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging to stderr.")

    return ap


def _fatal(msg: str, code: int = 2) -> NoReturn:
    print(f"[error] {msg}", file=sys.stderr)
    raise SystemExit(code)


def _load_ckpt(path: Path) -> Mapping[str, Any]:
    """Load a torch checkpoint safely-ish (prefers weights_only when supported)."""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        _fatal(f"PyTorch is required to load {path}: {e}")

    if not path.exists():
        _fatal(f"Checkpoint not found: {path}")

    # Newer PyTorch versions warn about pickle safety; weights_only=True reduces risk,
    # but not all artifacts are weights-only, so we fall back if needed.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        # Older torch without weights_only parameter
        return torch.load(path, map_location="cpu")
    except Exception:
        # Some artifacts include non-tensor objects; fall back (trusted local file).
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]


def _as_tensor(x: Any):
    import torch

    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _find_latest_field_checkpoint(tag: str, results_dir: Path) -> Path:
    """Find the newest diffusion1d_<tag>_field.pt under results/."""
    flat = results_dir / f"diffusion1d_{tag}_field.pt"
    if flat.is_file():
        return flat

    pattern = f"diffusion1d--{tag}--*/diffusion1d_{tag}_field.pt"
    candidates = list(results_dir.glob(pattern))
    if not candidates:
        _fatal(
            f"Could not find any field checkpoints for tag='{tag}'. "
            f"Looked for {flat} and {results_dir.as_posix()}/{pattern}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _infer_time_axis(u_shape: Sequence[int], user_axis: int, T: Any | None, U: Any) -> int:
    """Infer which axis corresponds to time.

    Heuristics:
    - If user specifies an axis, use it.
    - If T is 1-D, match its length against a unique axis in u_shape.
    - If T is a mesh-grid with the same shape as U, pick the axis along which a
      1-D slice is monotone and has the most unique values.
    - Fallback: last axis.
    """
    ndim = len(u_shape)
    if ndim == 0:
        _fatal("Field variable has scalar shape; expected at least 1-D.")

    if user_axis != -1:
        ax = user_axis % ndim
        return ax

    if T is None:
        return ndim - 1

    t = _as_tensor(T).detach().cpu()
    try:
        u = _as_tensor(U)
    except Exception:
        u = None

    # 1-D time vector: match length to an axis
    if t.ndim == 1:
        tlen = int(t.numel())
        matches = [i for i, s in enumerate(u_shape) if int(s) == tlen]
        if len(matches) == 1:
            return matches[0]
        return ndim - 1

    # Mesh-grid matching U shape: detect axis with varying time
    if u is not None and hasattr(u, "shape") and tuple(int(s) for s in t.shape) == tuple(int(s) for s in u.shape):
        best_ax = None
        best_score = -1
        for ax in range(t.ndim):
            idx = [0] * t.ndim
            idx[ax] = slice(None)
            vec = t[tuple(idx)].reshape(-1).numpy()
            if vec.size < 2:
                continue
            uniq = np.unique(vec).size
            if uniq <= 1:
                continue
            if np.all(np.diff(vec) >= 0) or np.all(np.diff(vec) <= 0):
                if uniq > best_score:
                    best_score = uniq
                    best_ax = ax
        if best_ax is not None:
            return int(best_ax)

    return ndim - 1


def _reduce_spatial(U: Any, *, time_axis: int, agg: str, p: float | None, q: float | None, temp: float) -> np.ndarray:
    """Reduce a field tensor to a scalar series s(t)."""
    import torch

    u = _as_tensor(U).detach().cpu().float()

    if u.ndim == 0:
        _fatal("Field tensor is scalar; expected at least 1-D.")
    if u.ndim == 1:
        return u.numpy()

    ax = time_axis % u.ndim
    u = u.movedim(ax, -1)
    nt = u.shape[-1]
    spatial = u.reshape(-1, nt)

    if agg == "mean":
        s = spatial.mean(dim=0)
    elif agg == "amax":
        s = spatial.max(dim=0).values
    elif agg == "amin":
        s = spatial.min(dim=0).values
    elif agg == "median":
        s = spatial.median(dim=0).values
    elif agg == "quantile":
        qv = 0.95 if q is None else float(q)
        if not (0.0 <= qv <= 1.0):
            _fatal(f"--q must be in [0,1], got {qv}")
        s = torch.quantile(spatial, q=qv, dim=0)
    elif agg == "lp":
        pv = 2.0 if p is None else float(p)
        if pv <= 0:
            _fatal(f"--p must be > 0 for lp aggregation, got {pv}")
        s = spatial.abs().pow(pv).mean(dim=0).pow(1.0 / pv)
    elif agg == "softmax":
        tau = float(temp)
        if not (tau > 0):
            _fatal(f"--temp must be > 0 for softmax aggregation, got {tau}")
        s = (spatial * tau).logsumexp(dim=0) / tau
    else:  # pragma: no cover
        _fatal(f"Unknown agg: {agg}")

    return s.numpy()


def _extract_time_vector(T: Any, *, nt: int, time_axis: int, U: Any) -> np.ndarray:
    """Extract a 1-D time vector aligned with the reduced signal length."""
    t = _as_tensor(T).detach().cpu().float()

    if t.ndim == 0:
        _fatal("Checkpoint time tensor T is scalar; expected vector or mesh-grid.")

    if t.ndim == 1:
        tt = t.numpy()
        if tt.size != nt:
            m = min(tt.size, nt)
            tt = tt[:m]
            if m < nt:
                if m >= 2:
                    dt = float(np.mean(np.diff(tt)))
                else:
                    dt = 1.0
                extra = tt[-1] + dt * np.arange(1, nt - m + 1, dtype=tt.dtype)
                tt = np.concatenate([tt, extra])
        return tt.astype(float)

    try:
        u_shape = tuple(int(s) for s in _as_tensor(U).shape)
    except Exception:
        u_shape = None
    if u_shape is not None and tuple(int(s) for s in t.shape) == u_shape:
        ax = time_axis % t.ndim
        idx = [0] * t.ndim
        idx[ax] = slice(None)
        tt = t[tuple(idx)].reshape(-1).numpy()
        if tt.size != nt:
            tt = tt[:nt]
        return tt.astype(float)

    axes = [ax for ax, s in enumerate(t.shape) if int(s) == int(nt)]
    if axes:
        ax = time_axis % t.ndim
        if ax not in axes:
            ax = axes[-1]
        idx = [0] * t.ndim
        idx[ax] = slice(None)
        tt = t[tuple(idx)].reshape(-1).numpy()
        return tt.astype(float)

    flat = t.reshape(-1).numpy()
    return flat[:nt].astype(float)


def _build_time_vector_and_dt(
    *,
    T: Any | None,
    dt_arg: float | None,
    nt: int,
    time_axis: int,
    U: Any,
    semantics: str,
    needs_time_window: bool,
) -> tuple[np.ndarray, float]:
    """Return (tvec, dt). dt is always returned as a finite float."""
    if T is not None:
        tvec = _extract_time_vector(T, nt=nt, time_axis=time_axis, U=U)
        if tvec.size >= 2:
            diffs = np.diff(tvec)
            dt = float(np.mean(diffs))
            if not math.isfinite(dt) or dt <= 0:
                dt = float(dt_arg) if dt_arg is not None else 1.0
        else:
            dt = float(dt_arg) if dt_arg is not None else 1.0
        return tvec, dt

    if dt_arg is None:
        if semantics == "dense":
            _fatal("Dense-time semantics requested but checkpoint has no T; provide --dt.")
        if needs_time_window:
            _fatal("t0/t1 provided but checkpoint has no T; provide --dt or use idx0/idx1.")
        dt = 1.0
    else:
        dt = float(dt_arg)
        if not math.isfinite(dt) or dt <= 0:
            _fatal(f"--dt must be a finite positive number, got {dt_arg}")

    tvec = (np.arange(nt, dtype=float) * dt).astype(float)
    return tvec, dt


def _apply_window(
    series: np.ndarray,
    tvec: np.ndarray,
    *,
    t0: float | None,
    t1: float | None,
    idx0: int | None,
    idx1: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice (series, tvec) by time window and/or index window."""
    if series.ndim != 1:
        _fatal("Internal error: series is not 1-D.")
    if tvec.ndim != 1:
        _fatal("Internal error: time vector is not 1-D.")
    if tvec.size != series.size:
        m = min(tvec.size, series.size)
        tvec = tvec[:m]
        series = series[:m]

    if t0 is not None or t1 is not None:
        lo = -np.inf if t0 is None else float(t0)
        hi = np.inf if t1 is None else float(t1)
        mask = (tvec >= lo) & (tvec <= hi)
        series = series[mask]
        tvec = tvec[mask]

    if idx0 is not None or idx1 is not None:
        i0 = 0 if idx0 is None else int(idx0)
        i1 = series.size if idx1 is None else int(idx1)
        series = series[i0:i1]
        tvec = tvec[i0:i1]

    return series, tvec


_TIME_TOKEN_RE = r"(?:inf|INF|[0-9.+\-eE]+)"


def _normalize_timed_bounds(formula: str, *, semantics: str) -> str:
    """Normalize RTAMT time-interval syntax in common cases.

    RTAMT examples commonly use:
      - dense time:     op[a:b]
      - discrete time: op[a,b]

    Different versions may accept both, but normalizing avoids avoidable parse errors
    when users copy/paste formulas across semantics.
    """
    sem = semantics.lower().strip()
    if sem == "dense":
        return re.sub(
            rf"\b(always|eventually)\[\s*({_TIME_TOKEN_RE})\s*,\s*({_TIME_TOKEN_RE})\s*\]",
            r"\1[\2:\3]",
            formula,
        )
    if sem == "discrete":
        return re.sub(
            rf"\b(always|eventually)\[\s*({_TIME_TOKEN_RE})\s*:\s*({_TIME_TOKEN_RE})\s*\]",
            r"\1[\2,\3]",
            formula,
        )
    return formula


def _build_preset_spec(
    *,
    stl_var: str,
    op: str,
    spec_kind: str,
    semantics: str,
    u_min: float | None,
    u_max: float | None,
) -> Any | None:
    """Build an RTAMT spec for built-in predicate forms. Returns None if unavailable."""
    if not _rtamt_available():
        return None

    spec_kind = spec_kind.lower()
    op = op.lower()
    semantics = semantics.lower()

    if op == "always" and spec_kind == "upper":
        umax = 1.0 if u_max is None else float(u_max)
        return _rtamt_stl_always_upper(var=stl_var, u_max=umax, time_semantics=semantics)

    try:
        import rtamt  # type: ignore
    except Exception:
        return None

    SpecCls = None
    if semantics == "dense":
        SpecCls = (
            getattr(rtamt, "StlDenseTimeSpecification", None)
            or getattr(rtamt, "StlDenseTimeOfflineSpecification", None)
        )
    else:
        SpecCls = (
            getattr(rtamt, "StlDiscreteTimeSpecification", None)
            or getattr(rtamt, "StlDiscreteTimeOfflineSpecification", None)
        )

    if SpecCls is None:
        return None

    spec = SpecCls()
    spec.declare_var(stl_var, "float")

    umin = 0.0 if u_min is None else float(u_min)
    umax = 1.0 if u_max is None else float(u_max)

    if spec_kind == "upper":
        pred = f"({stl_var} <= {umax})"
    elif spec_kind == "lower":
        pred = f"({stl_var} >= {umin})"
    elif spec_kind == "range":
        pred = f"({stl_var} >= {umin}) and ({stl_var} <= {umax})"
    else:  # pragma: no cover
        _fatal(f"Unknown spec kind: {spec_kind}")

    if op == "always":
        spec.spec = f"always {pred}"
    elif op == "eventually":
        spec.spec = f"eventually {pred}"
    else:  # pragma: no cover
        _fatal(f"Unknown op: {op}")

    spec.parse()
    return spec


def _build_custom_spec(*, stl_var: str, semantics: str, formula: str) -> Any:
    """Build an RTAMT spec for a custom STL formula; exits if RTAMT is missing."""
    if not _rtamt_available():
        _fatal("Custom formulas require RTAMT. Install with: pip install rtamt (Python <3.12).")

    try:
        import rtamt  # type: ignore
    except Exception as e:  # pragma: no cover
        _fatal(f"RTAMT is not importable: {e}")

    semantics = semantics.lower()
    SpecCls = None
    if semantics == "dense":
        SpecCls = (
            getattr(rtamt, "StlDenseTimeSpecification", None)
            or getattr(rtamt, "StlDenseTimeOfflineSpecification", None)
        )
    else:
        SpecCls = (
            getattr(rtamt, "StlDiscreteTimeSpecification", None)
            or getattr(rtamt, "StlDiscreteTimeOfflineSpecification", None)
        )

    if SpecCls is None:
        _fatal("RTAMT STL specification class not found; check your RTAMT installation.")

    spec = SpecCls()
    spec.declare_var(stl_var, "float")
    spec.spec = formula
    spec.parse()
    return spec


def _fallback_predicate_robustness(
    series: np.ndarray,
    *,
    spec_kind: str,
    u_min: float | None,
    u_max: float | None,
) -> np.ndarray:
    """Return per-time-step predicate robustness r(t) for simple predicate forms."""
    spec_kind = spec_kind.lower()
    if spec_kind == "upper":
        umax = 1.0 if u_max is None else float(u_max)
        return umax - series
    if spec_kind == "lower":
        umin = 0.0 if u_min is None else float(u_min)
        return series - umin
    if spec_kind == "range":
        umin = 0.0 if u_min is None else float(u_min)
        umax = 1.0 if u_max is None else float(u_max)
        return np.minimum(series - umin, umax - series)
    _fatal(f"Unsupported fallback spec_kind: {spec_kind}")
    return series  # unreachable


def _fallback_monitor(
    series: np.ndarray,
    *,
    op: str,
    spec_kind: str,
    u_min: float | None,
    u_max: float | None,
) -> tuple[float, bool]:
    """Exact fallback for always/eventually over simple predicates."""
    r = _fallback_predicate_robustness(series, spec_kind=spec_kind, u_min=u_min, u_max=u_max)
    if r.size == 0:
        _fatal("Empty evaluation window after slicing; nothing to monitor.")
    op = op.lower()
    if op == "always":
        rob = float(np.min(r))
    elif op == "eventually":
        rob = float(np.max(r))
    else:  # pragma: no cover
        _fatal(f"Unsupported op for fallback: {op}")
    return rob, rob >= 0.0


def _evaluate(
    *,
    series: np.ndarray,
    dt: float,
    semantics: str,
    stl_var: str,
    op: str,
    spec_kind: str,
    u_min: float | None,
    u_max: float | None,
    custom_formula: str | None,
) -> tuple[float, bool, str]:
    """Return (robustness, satisfied, backend)."""
    if custom_formula is None:
        spec = _build_preset_spec(
            stl_var=stl_var,
            op=op,
            spec_kind=spec_kind,
            semantics=semantics,
            u_min=u_min,
            u_max=u_max,
        )
        if spec is None:
            rob, sat = _fallback_monitor(series, op=op, spec_kind=spec_kind, u_min=u_min, u_max=u_max)
            return rob, sat, "fallback"
        rob = float(_rtamt_eval_series(spec, series.tolist(), dt=dt))
        return rob, bool(_rtamt_satisfied(rob)), "rtamt"

    spec = _build_custom_spec(stl_var=stl_var, semantics=semantics, formula=custom_formula)
    rob = float(_rtamt_eval_series(spec, series.tolist(), dt=dt))
    return rob, bool(_rtamt_satisfied(rob)), "rtamt"


def _series_stats(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def _maybe_save_plot(
    *,
    out_path: Path,
    tvec: np.ndarray,
    series: np.ndarray,
    spec_kind: str,
    u_min: float | None,
    u_max: float | None,
    title: str,
) -> None:
    """Save a simple time-series plot (optional)."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-friendly
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        _fatal(f"Plot requested but matplotlib is unavailable: {e}")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(tvec, series, label="s(t)")
    ax.set_xlabel("t")
    ax.set_ylabel("s")

    spec_kind = spec_kind.lower()
    if spec_kind in {"upper", "range"}:
        umax = 1.0 if u_max is None else float(u_max)
        ax.axhline(umax, linestyle="--", linewidth=1.0, label="u_max")
    if spec_kind in {"lower", "range"}:
        umin = 0.0 if u_min is None else float(u_min)
        ax.axhline(umin, linestyle="--", linewidth=1.0, label="u_min")

    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = build_argparser()
    ns = parser.parse_args()
    args = Args(**vars(ns))

    run_tag: str | None = None
    legacy_formula: str | None = None
    if args.stl == _STL_RUN_SENTINEL:
        run_tag = "stl"
    elif args.stl is not None:
        legacy_formula = str(args.stl)

    if args.baseline:
        if run_tag is not None:
            _fatal("Use only one of --baseline or --stl (run selector).")
        run_tag = "baseline"

    custom_formula = args.formula if args.formula is not None else legacy_formula
    if custom_formula is not None:
        custom_formula = _normalize_timed_bounds(custom_formula, semantics=args.semantics)

    results_dir = Path(args.results_dir)
    if args.ckpt is not None and run_tag is not None:
        _fatal("Use either --ckpt/--field (explicit) or --baseline/--stl (auto), not both.")

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    else:
        if run_tag is None:
            _fatal("Provide --ckpt/--field OR select a packaged run via --baseline/--stl.")
        ckpt_path = _find_latest_field_checkpoint(run_tag, results_dir)

    json_out = args.json_out
    if json_out is None and run_tag is not None:
        json_out = str(results_dir / f"diffusion1d_{run_tag}_rtamt.json")

    data = _load_ckpt(ckpt_path)
    if args.var not in data:
        _fatal(f"Key {args.var!r} not found in checkpoint. Available keys: {sorted(data.keys())}")

    U = data[args.var]
    T = data.get("T", None)
    if T is None:
        T = data.get("t", None)

    u_shape = tuple(int(s) for s in _as_tensor(U).shape)
    time_axis = _infer_time_axis(u_shape, args.time_axis, T, U)

    full_series = _reduce_spatial(U, time_axis=time_axis, agg=args.agg, p=args.p, q=args.q, temp=args.temp)
    nt_full = int(full_series.size)

    needs_time_window = (args.t0 is not None) or (args.t1 is not None)
    tvec_full, dt = _build_time_vector_and_dt(
        T=T,
        dt_arg=args.dt,
        nt=nt_full,
        time_axis=time_axis,
        U=U,
        semantics=args.semantics,
        needs_time_window=needs_time_window,
    )

    series, tvec = _apply_window(full_series, tvec_full, t0=args.t0, t1=args.t1, idx0=args.idx0, idx1=args.idx1)

    stl_var = args.stl_var

    umin_eff: float | None = None
    umax_eff: float | None = None
    if custom_formula is None:
        if args.spec in {"lower", "range"}:
            umin_eff = 0.0 if args.u_min is None else float(args.u_min)
        if args.spec in {"upper", "range"}:
            umax_eff = 1.0 if args.u_max is None else float(args.u_max)

        if args.spec == "upper":
            pred_text = f"({stl_var} <= {umax_eff})"
        elif args.spec == "lower":
            pred_text = f"({stl_var} >= {umin_eff})"
        else:
            pred_text = f"({stl_var} >= {umin_eff}) and ({stl_var} <= {umax_eff})"

        spec_text = f"{args.op} {pred_text}"
    else:
        spec_text = custom_formula

    rob, sat, backend = _evaluate(
        series=series,
        dt=dt,
        semantics=args.semantics,
        stl_var=stl_var,
        op=args.op,
        spec_kind=args.spec,
        u_min=umin_eff,
        u_max=umax_eff,
        custom_formula=custom_formula,
    )

    if args.dump_series is not None:
        outp = Path(args.dump_series)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", stl_var])
            for ti, si in zip(tvec.tolist(), series.tolist(), strict=False):
                w.writerow([float(ti), float(si)])

    if args.plot is not None:
        title = f"{spec_text}  (backend={backend}, rob={rob:.4g})"
        _maybe_save_plot(
            out_path=Path(args.plot),
            tvec=tvec,
            series=series,
            spec_kind=args.spec,
            u_min=umin_eff,
            u_max=umax_eff,
            title=title,
        )

    series_stats = _series_stats(series)
    full_stats = _series_stats(full_series)

    predicate_summary: dict[str, Any] | None = None
    if custom_formula is None:
        r = _fallback_predicate_robustness(series, spec_kind=args.spec, u_min=umin_eff, u_max=umax_eff)
        if args.op == "always":
            k = int(np.argmin(r))
        else:
            k = int(np.argmax(r))
        predicate_summary = {
            "robustness_min": float(np.min(r)),
            "robustness_max": float(np.max(r)),
            "violations": int(np.sum(r < 0.0)),
            "violation_fraction": float(np.mean(r < 0.0)),
            "extreme": {
                "op": args.op,
                "idx": k,
                "t": float(tvec[k]),
                "s": float(series[k]),
                "r": float(r[k]),
            },
        }

    try:
        ckpt_record = ckpt_path.resolve().relative_to(_REPO_ROOT.resolve()).as_posix()
    except Exception:
        ckpt_record = ckpt_path.as_posix()

    summary: dict[str, Any] = {
        "ckpt": ckpt_record,
        "run_tag": run_tag,
        "field_var": args.var,
        "stl_var": stl_var,
        "shape": list(u_shape),
        "time_axis": int(time_axis),
        "nt_full": int(nt_full),
        "nt_eval": int(series.size),
        "semantics": args.semantics,
        "dt": float(dt),
        "backend": backend,
        "agg": {"kind": args.agg, "p": args.p, "q": args.q, "temp": args.temp},
        "window": {"t0": args.t0, "t1": args.t1, "idx0": args.idx0, "idx1": args.idx1},
        "spec": {
            "op": args.op, "kind": args.spec, "text": spec_text,
            "formula": custom_formula, "u_min": umin_eff, "u_max": umax_eff,
        },
        "signal_stats": series_stats,
        "signal_stats_full": full_stats,
        "predicate_summary": predicate_summary,
        "robustness": float(rob),
        "satisfied": bool(sat),
    }

    if json_out is not None:
        outp = Path(json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        if args.verbose:
            print(f"[ok] wrote {outp}", file=sys.stderr)

    print(f"[result] robustness={rob:.6g}  satisfied={sat}  backend={backend}")

    if args.verbose:
        print(f"[info] ckpt={ckpt_path}", file=sys.stderr)
        print(f"[info] run_tag={run_tag}", file=sys.stderr)
        print(f"[info] field_var={args.var!r} stl_var={stl_var!r}", file=sys.stderr)
        print(f"[info] spec: {spec_text}", file=sys.stderr)
        print(f"[info] shape={u_shape} time_axis={time_axis} nt_full={nt_full} nt_eval={series.size}", file=sys.stderr)
        print(f"[info] semantics={args.semantics} dt={dt:.6g} backend={backend}", file=sys.stderr)
        if args.t0 is not None or args.t1 is not None or args.idx0 is not None or args.idx1 is not None:
            print(
                f"[info] window: "
                f"t∈[{args.t0 if args.t0 is not None else '-inf'}, "
                f"{args.t1 if args.t1 is not None else 'inf'}] "
                f"idx∈[{args.idx0 if args.idx0 is not None else 0}:{args.idx1 if args.idx1 is not None else nt_full}]",
                file=sys.stderr,
            )
        print(f"[info] signal(full) min/max = {full_stats['min']:.6g}/{full_stats['max']:.6g}", file=sys.stderr)
        print(f"[info] signal(eval)  min/max = {series_stats['min']:.6g}/{series_stats['max']:.6g}", file=sys.stderr)


if __name__ == "__main__":
    main()
