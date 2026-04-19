from __future__ import annotations

"""neural_pde_stl_strel.monitors.moonlight_strel_hello

MoonLight STREL "hello world".

This module provides a tiny, CPU-friendly, end-to-end demonstration of
**spatio-temporal monitoring** using MoonLight (STREL) on a 2-D grid.

The demo monitors the containment property used throughout this repository
(see ``scripts/specs/contain_hotspot.mls``):

    nowhere_hot     = !(somewhere[0, 100](hot));
    quench          = globally(nowhere_hot);
    contain         = eventually(quench);
    contain_hotspot = contain;

Intuition:
    A transient "hot" region appears somewhere in space. The property holds if
    **eventually** the system reaches a state where **globally** there are no
    hot cells anywhere (i.e., the hotspot is quenched and stays quenched).

Data flow (high level):

    u(x,y,t)  ── threshold ──▶  hot(x,y,t)  ── reshape ──▶  signal[loc][t][1]
       │                                                    │
       └─ grid(nx,ny) ── edges ──▶ graph_edges ─▶ MoonLight monitor ─▶ output

Public API:
    :func:`strel_hello` runs the demo and returns MoonLight's raw monitor output
    as a NumPy array (dtype ``float``) so it can be printed, tested, or plotted.

Optional dependency:
    MoonLight is optional (Java backend). If it is not available, this module
    raises a clear ``RuntimeError``.
"""

import contextlib
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np


# Prefer the shared MoonLight helpers (keeps behavior consistent across the repo).
# Importing these does *not* import MoonLight itself; the helper lazily imports it.
try:  # pragma: no cover - optional dependency path
    from neural_pde_stl_strel.monitoring.moonlight_helper import (
        get_monitor,
        load_script_from_file,
        load_script_from_text,
        set_domain,
    )
except Exception:  # pragma: no cover
    get_monitor = None  # type: ignore[assignment]
    load_script_from_file = None  # type: ignore[assignment]
    load_script_from_text = None  # type: ignore[assignment]
    set_domain = None  # type: ignore[assignment]


# Repository-relative paths

_MLS_RELATIVE = ("scripts", "specs", "contain_hotspot.mls")
_ASSET_FIELD_RELATIVE = ("assets", "heat2d_scalar", "field_xy_t.npy")
_ASSET_META_RELATIVE = ("assets", "heat2d_scalar", "meta.json")


# Inline fallback spec

_MLS_INLINE = (
    "signal { bool hot; }\n\n"
    "space {\n"
    "  edges { real dist; }\n"
    "}\n\n"
    "domain boolean;\n\n"
    "formula nowhere_hot = !(somewhere[0, 100](hot));\n"
    "formula quench = globally(nowhere_hot);\n"
    "formula contain = eventually(quench);\n"
    "formula contain_hotspot = contain;\n\n"
    "formula quench_for(real tau) = globally[0, tau](nowhere_hot);\n"
    "formula contain_within(real deadline, real tau) = "
    "eventually[0, deadline](globally[0, tau](nowhere_hot));\n"
)


def _normalize_domain(raw: str) -> str | None:
    """Normalize a domain selector.

    Returns:
        ``"boolean"`` or ``"minmax"`` when a known domain is requested, or
        ``None`` to keep whatever the script specifies.
    """
    val = raw.strip().lower()
    if val in {"", "script", "default", "keep"}:
        return None
    if val in {"boolean", "bool"}:
        return "boolean"
    if val in {"minmax", "min_max", "quant", "quantitative", "robust", "robustness"}:
        return "minmax"
    raise ValueError(
        f"Unsupported MoonLight domain {raw!r}. Use 'boolean', 'minmax', or 'script'."
    )


def _env_flag(name: str, *, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _resolve_repo_file(relative_parts: tuple[str, ...]) -> Path | None:
    """Best-effort resolution of a repo-relative file.

    This repo is sometimes used as an installed package where the top-level
    ``scripts/`` and ``assets/`` folders may not be present.

    Precedence:
      1) If an absolute path is provided via an env var (handled elsewhere).
      2) Walk upward from this file looking for the target path.
      3) Try relative to the current working directory.
    """
    here = Path(__file__).resolve()

    # Walk upward from this file.
    for parent in here.parents:
        candidate = parent.joinpath(*relative_parts)
        if candidate.is_file():
            return candidate

    # Try relative to CWD for ad-hoc execution.
    cwd_candidate = Path.cwd().joinpath(*relative_parts)
    if cwd_candidate.is_file():
        return cwd_candidate

    return None


def _resolve_spec_file() -> Path | None:
    """Resolve the MoonLight ``.mls`` script file, if available."""
    env = os.environ.get("NEURAL_PDE_STL_STREL_MLS_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    return _resolve_repo_file(_MLS_RELATIVE)


def _resolve_asset_field_file() -> Path | None:
    """Resolve the optional 2-D heat field asset, if available."""
    env = os.environ.get("NEURAL_PDE_STL_STREL_HEAT2D_FIELD")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    return _resolve_repo_file(_ASSET_FIELD_RELATIVE)


def _resolve_asset_meta_file() -> Path | None:
    """Resolve the optional heat field meta.json, if available."""
    return _resolve_repo_file(_ASSET_META_RELATIVE)


def _build_grid_edges_triples(nx: int, ny: int, *, weight: float = 1.0) -> list[list[float]]:
    """Build a 4-neighbor undirected grid as a list of directed edge triples.

    MoonLight expects graphs as ``[ [src, dst, w], ... ]`` with numeric entries.
    We add both directions for each undirected adjacency.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError(f"nx and ny must be positive; got nx={nx}, ny={ny}")
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError(f"weight must be finite and > 0; got {weight!r}")

    def idx(i: int, j: int) -> int:
        return i * ny + j

    edges: list[list[float]] = []
    w = float(weight)
    for i in range(nx):
        for j in range(ny):
            u = idx(i, j)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    v = idx(ii, jj)
                    edges.append([float(u), float(v), w])
    return edges


def _synthesize_heat_trace(
    nx: int,
    ny: int,
    nt: int,
    *,
    alpha: float = 0.25,
    hotspot_value: float = 1.0,
) -> np.ndarray:
    """Synthesize a tiny diffusion-like trace ``u[x, y, t]``.

    The update is a simple explicit smoothing step with Dirichlet (0) padding.
    It is *not* intended as a physically exact solver--just a deterministic toy
    trace with a hotspot that decays over time.
    """
    if nx <= 0 or ny <= 0 or nt <= 1:
        raise ValueError(f"Expected nx>0, ny>0, nt>1; got {(nx, ny, nt)}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
    if not np.isfinite(hotspot_value) or hotspot_value <= 0:
        raise ValueError(f"hotspot_value must be finite and > 0; got {hotspot_value!r}")

    u = np.zeros((nx, ny, nt), dtype=float)
    u[nx // 2, ny // 2, 0] = float(hotspot_value)

    a = float(alpha)
    for t in range(1, nt):
        prev = u[:, :, t - 1]
        pad = np.pad(prev, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)
        neigh = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        u[:, :, t] = (1.0 - a) * prev + (a / 4.0) * neigh
    return u


def _field_to_moonlight_signals(
    u: np.ndarray,
    *,
    threshold: float,
) -> tuple[list[list[list[bool]]], list[list[list[bool]]]]:
    """Convert ``u[x,y,t]`` to MoonLight signal layouts.

    MoonLight's spatio-temporal monitor consumes values in **node × time** form:

        signal_node_major[loc][t][feature]

    Some legacy wrappers (and some convenience methods) use **time × node**:

        signal_time_major[t][loc][feature]

    We return both.
    """
    if u.ndim != 3:
        raise ValueError(f"Expected u to have shape (nx, ny, nt); got {u.shape}")
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite; got {threshold!r}")

    nx, ny, nt = u.shape
    n_nodes = nx * ny

    flat = u.reshape(n_nodes, nt)
    hot = (flat >= float(threshold))

    # Node-major: [node][time][1]
    hot_list_node = hot.tolist()  # list[list[bool]]
    node_major = [[[b] for b in series] for series in hot_list_node]

    # Time-major: [time][node][1]
    hot_list_time = hot.T.tolist()  # list[list[bool]]
    time_major = [[[b] for b in series] for series in hot_list_time]

    return node_major, time_major


@contextlib.contextmanager
def _suppress_output(enabled: bool) -> Iterator[None]:
    """Optionally suppress stdout/stderr at the file-descriptor level.

    MoonLight may print JVM initialization logs to stdout/stderr. For demos and
    unit tests, this can be distracting. This context manager redirects both
    streams to ``os.devnull`` while active.
    """
    if not enabled:
        yield
        return

    # Best-effort: if anything goes wrong, fall back to no suppression.
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
    except Exception:
        yield
        return

    try:
        yield
    finally:
        try:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
        finally:
            for fd in (devnull_fd, stdout_fd, stderr_fd):
                try:
                    os.close(fd)
                except Exception:
                    pass


def _get_monitor_any(mls: Any, name: str) -> Any:
    """Get a monitor by name, using the shared helper when available."""
    if get_monitor is not None:
        return get_monitor(mls, name)
    return mls.getMonitor(name)


def _monitor_strel(
    mon: Any,
    *,
    graph_edges: list[list[float]],
    signal_times: list[float],
    signal_node_major: list[list[list[bool]]],
    signal_time_major: list[list[list[bool]]],
) -> Any:
    """Call into MoonLight using the most compatible available API."""
    last_err: Exception | None = None

    # 1) Canonical spatio-temporal API: monitor(graph_times, graph_seq, signal_times, signal_values)
    fn = getattr(mon, "monitor", None)
    if callable(fn):
        graph_times = [float(signal_times[0])]
        graph_seq = [graph_edges]
        try:
            return fn(graph_times, graph_seq, signal_times, signal_node_major)
        except TypeError as e:
            last_err = e
        except Exception as e:  # pragma: no cover
            raise RuntimeError("MoonLight STREL monitor() call failed") from e

    # 2) Some interfaces expose monitor_static/monitorStatic for a fixed graph.
    for name in ("monitor_static", "monitorStatic"):
        fn = getattr(mon, name, None)
        if callable(fn):
            try:
                return fn(graph_edges, signal_times, signal_node_major)
            except Exception as e:  # pragma: no cover
                last_err = e

    # 3) Legacy 2-arg convenience wrappers.
    for name in ("monitor_graph_time_series", "monitorGraphTimeSeries"):
        fn = getattr(mon, name, None)
        if callable(fn):
            try:
                return fn(graph_edges, signal_time_major)
            except TypeError as e:
                # Some wrappers want a one-element graph sequence.
                try:
                    return fn([graph_edges], signal_time_major)
                except Exception as e2:  # pragma: no cover
                    last_err = e2
            except Exception as e:  # pragma: no cover
                last_err = e

    msg = (
        "MoonLight STREL monitor: no compatible method found. Tried: "
        "monitor(graph_times, graph_seq, signal_times, signal_values), "
        "monitor_static/monitorStatic, and monitorGraphTimeSeries variants."
    )
    if last_err is None:
        raise RuntimeError(msg)
    raise RuntimeError(msg) from last_err


def _to_float_ndarray(out: Any) -> np.ndarray:
    """Convert MoonLight output to a ``float`` NumPy array when possible."""
    try:
        return np.asarray(out, dtype=float)
    except Exception:
        # MoonLight may return a mapping {loc -> signal}.
        if isinstance(out, dict) and out:
            try:
                items = sorted(out.items(), key=lambda kv: kv[0])
            except Exception:
                items = list(out.items())
            arrs = [_to_float_ndarray(v) for _, v in items]
            try:
                return np.stack(arrs, axis=0)
            except Exception:
                return np.asarray(arrs, dtype=float)
        raise


def _strel_hello_impl() -> tuple[np.ndarray, list[float], np.ndarray]:
    """Implementation backend returning ``(u, times, out)``."""
    if load_script_from_file is None:
        raise RuntimeError("MoonLight is not available; cannot run STREL hello.")

    # Load .mls script
    spec_path = _resolve_spec_file()
    if spec_path is not None:
        mls = load_script_from_file(str(spec_path))  # type: ignore[arg-type]
    else:
        if load_script_from_text is None:
            # Defensive fallback: the helper should exist if MoonLight exists.
            from moonlight import ScriptLoader  # type: ignore

            mls = ScriptLoader.loadFromText(_MLS_INLINE)
        else:
            mls = load_script_from_text(_MLS_INLINE)  # type: ignore[arg-type]

    # Domain selection (safe to ignore if the binding doesn't expose setters).
    domain_raw = os.environ.get("NEURAL_PDE_STL_STREL_STREL_DOMAIN", "boolean")
    domain = _normalize_domain(domain_raw)
    if domain is not None and set_domain is not None:
        try:
            set_domain(mls, domain)
        except Exception:
            # Some bindings may not expose the domain setters.
            pass

    # Choose trace data
    use_asset = _env_flag("NEURAL_PDE_STL_STREL_STREL_USE_ASSET", default=False)
    asset_path = _resolve_asset_field_file() if use_asset else None
    if asset_path is not None:
        u = np.load(asset_path, allow_pickle=False)
        if u.ndim != 3:
            raise ValueError(f"Expected asset field shape (nx, ny, nt); got {u.shape}")

        # Use meta.json dt when present; otherwise fall back to unit steps.
        dt = 1.0
        meta_path = _resolve_asset_meta_file()
        if meta_path is not None:
            try:
                import json

                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                dt = float(meta.get("dt", dt))
            except Exception:
                pass
        signal_times = [float(k) * dt for k in range(u.shape[2])]

        # Threshold: explicit value wins; else quantile (default 0.995).
        th_env = os.environ.get("NEURAL_PDE_STL_STREL_STREL_THRESHOLD")
        if th_env is not None:
            threshold = float(th_env)
            if not np.isfinite(threshold):
                raise ValueError(f"NEURAL_PDE_STL_STREL_STREL_THRESHOLD must be finite; got {th_env!r}")
        else:
            q = float(os.environ.get("NEURAL_PDE_STL_STREL_STREL_QUANTILE", "0.995"))
            if not (0.0 < q < 1.0):
                raise ValueError(f"Quantile must be in (0, 1); got {q}")
            threshold = float(np.quantile(u, q))
    else:
        # Self-contained toy trace.
        nx = int(os.environ.get("NEURAL_PDE_STL_STREL_STREL_NX", "5"))
        ny = int(os.environ.get("NEURAL_PDE_STL_STREL_STREL_NY", "5"))
        nt = int(os.environ.get("NEURAL_PDE_STL_STREL_STREL_NT", "20"))
        alpha = float(os.environ.get("NEURAL_PDE_STL_STREL_STREL_ALPHA", "0.25"))
        u = _synthesize_heat_trace(nx, ny, nt, alpha=alpha)
        signal_times = [float(k) for k in range(nt)]
        threshold = float(os.environ.get("NEURAL_PDE_STL_STREL_STREL_THRESHOLD", "0.3"))
        if not np.isfinite(threshold):
            raise ValueError(
                "NEURAL_PDE_STL_STREL_STREL_THRESHOLD must be finite; got "
                f"{os.environ.get('NEURAL_PDE_STL_STREL_STREL_THRESHOLD')!r}"
            )

    # Build spatial model and signals
    nx, ny, nt = u.shape
    graph_edges = _build_grid_edges_triples(nx, ny, weight=1.0)
    signal_node, signal_time = _field_to_moonlight_signals(u, threshold=threshold)

    # Run the monitor
    formula = os.environ.get("NEURAL_PDE_STL_STREL_STREL_FORMULA", "contain_hotspot").strip()
    fallbacks = ("contain",)
    mon: Any
    try:
        mon = _get_monitor_any(mls, formula)
    except Exception:
        # Best-effort fallback for older scripts.
        mon = None
        for fb in fallbacks:
            try:
                mon = _get_monitor_any(mls, fb)
                break
            except Exception:
                pass
        if mon is None:
            raise

    quiet = not _env_flag("NEURAL_PDE_STL_STREL_MOONLIGHT_VERBOSE", default=False)
    with _suppress_output(quiet):
        out = _monitor_strel(
            mon,
            graph_edges=graph_edges,
            signal_times=signal_times,
            signal_node_major=signal_node,
            signal_time_major=signal_time,
        )

    return u, signal_times, _to_float_ndarray(out)


def strel_hello() -> np.ndarray:
    """Run a tiny STREL monitoring demo and return the raw MoonLight output.

    The return value is a NumPy ``float`` array view of MoonLight's monitor
    output. For spatio-temporal formulas, MoonLight typically returns an
    array-like object shaped like:

      * ``(n_locations, n_knots, 2)`` where the last axis is ``[time, value]``;
        or
      * ``(n_knots, 2)`` for some degenerate/summarized cases.

    The exact format can vary across MoonLight versions and semantics.
    """
    _, _, out = _strel_hello_impl()
    return out


def _main() -> None:  # pragma: no cover
    """Small CLI for ad-hoc local checking."""
    u, t, out = _strel_hello_impl()
    print("✅ MoonLight STREL hello ran")
    print(f"u shape: {u.shape}   time steps: {len(t)}")
    print(f"output shape: {out.shape}   dtype: {out.dtype}")

    if not _env_flag("NEURAL_PDE_STL_STREL_STREL_PLOT", default=False):
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("(plot) matplotlib is not installed; skipping plots")
        return

    outdir = Path(os.environ.get("NEURAL_PDE_STL_STREL_STREL_PLOT_DIR", "figs"))
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Field snapshots.
    for k in (0, u.shape[2] // 2, u.shape[2] - 1):
        fig, ax = plt.subplots()
        ax.imshow(u[:, :, k].T, origin="lower")
        ax.set_title(f"u(x,y,t) @ t={t[k]:.3g}")
        fig.tight_layout()
        fig.savefig(outdir / f"strel_hello_u_t{k:03d}.png", dpi=200)
        plt.close(fig)

    # 2) Simple robustness/satisfaction summary plot when output looks like [loc][knot][time,val].
    if out.ndim == 3 and out.shape[2] >= 2:
        times = out[0, :, 0]
        vals = out[:, :, 1]

        fig, ax = plt.subplots()
        ax.plot(times, np.min(vals, axis=0), label="min over locations")
        ax.plot(times, np.mean(vals, axis=0), label="mean over locations")
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "strel_hello_monitor_summary.png", dpi=200)
        plt.close(fig)

    print(f"🖼️  wrote plots to {outdir}")


if __name__ == "__main__":  # pragma: no cover
    _main()


__all__ = ["strel_hello"]
