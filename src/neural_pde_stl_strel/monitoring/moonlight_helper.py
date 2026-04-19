from __future__ import annotations

"""MoonLight (STREL) helpers.

This module centralizes interaction with the optional :pypi:`moonlight` package
(MoonLightSuite) so the rest of this repository can remain importable even when
MoonLight/Java are not available.

Design principles
* Optional dependency: importing this file must *not* import :mod:`moonlight`
  (or start a JVM) unless you call :func:`load_script_from_text` or
  :func:`load_script_from_file`.
* Version resilience: the MoonLight Python interface has had small API
  differences across releases. This module targets the PyPI package
  ``moonlight>=0.3`` (as declared in the optional extras / ``requirements-extra.txt``)
  but includes a few fallbacks for older method names.
* Shape clarity: MoonLight's spatio-temporal monitor expects the signal in
  node-major layout ``[location][time][feature]`` and requires explicit time
  arrays. These helpers accept either node-major or time-major
  ``[time][location][feature]`` inputs and convert as needed.

Data flow (typical use in this repo)
The 2D heat/diffusion examples commonly follow this pipeline:

    PDE field u(x, y, t)  ->  field_to_signal(...)  ->  signal_values
           grid graph     ->  build_grid_graph(...) ->  graph_snapshot
                                      |
                                      v
                           monitor.monitor(...)

For spatio-temporal monitoring, the upstream API is (per the official wiki):

    monitor.monitor(location_times, graph_seq, signal_times, signal_values)

where:

* ``location_times`` is a list of timestamps at which the spatial model changes;
* ``graph_seq`` is a list of graph snapshots (adjacency matrix or edge triples);
* ``signal_times`` is a list of timestamps for the signal samples; and
* ``signal_values`` is node-major ``[N][T][F]``.

This module provides helpers to normalize graphs/signals into that form, and a
small adapter (returned by :func:`get_monitor`) that lets older 2-argument call
sites keep working (``monitor(graph, signal)`` / ``monitorGraphTimeSeries``).

References
MoonLightSuite wiki (Python examples, including spatio-temporal monitoring):
https://github.com/MoonLightSuite/moonlight/wiki/Python
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias, overload

import numpy as np

# Type aliases (best-effort documentation; runtime does not depend on typing)

AdjacencyMatrix: TypeAlias = list[list[float]]  # shape: [N][N]
EdgeTriple: TypeAlias = list[float]  # [u, v, w] with u,v integer-like floats
EdgeTriples: TypeAlias = list[EdgeTriple]  # shape: [E][3]
GraphSnapshot: TypeAlias = AdjacencyMatrix | EdgeTriples
GraphSequence: TypeAlias = list[GraphSnapshot]
GraphTimeSeries: TypeAlias = tuple[list[float], GraphSequence]

# MoonLight expects node-major values: [location][time][feature]
SignalNodeMajor: TypeAlias = list[list[list[float]]]
SignalTimeSeries: TypeAlias = tuple[list[float], SignalNodeMajor]

SignalMajor = Literal["node", "time"]
Domain = Literal["boolean", "minmax", "robustness"]

# Optional dependency handling

_SCRIPT_LOADER: Any | None = None


def _import_script_loader() -> Any:
    """Import and cache ``moonlight.ScriptLoader``.

    The :pypi:`moonlight` package uses :pypi:`pyjnius` under the hood, which
    starts a JVM at import time. We therefore keep the import lazy.
    """
    global _SCRIPT_LOADER
    if _SCRIPT_LOADER is not None:
        return _SCRIPT_LOADER

    try:
        from moonlight import ScriptLoader  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "MoonLight is not available. Install it with 'pip install moonlight' and "
            "ensure a Java runtime is installed "
            "(MoonLightSuite currently requires Java 21+).\n\n"
            "Upstream docs: https://github.com/MoonLightSuite/moonlight/wiki/Python"
        ) from e

    _SCRIPT_LOADER = ScriptLoader
    return ScriptLoader


# Script loading + monitor retrieval


def load_script_from_text(script: str) -> Any:
    """Load a MoonLight script from a string."""
    ScriptLoader = _import_script_loader()
    return ScriptLoader.loadFromText(script)


def load_script_from_file(path: str | Path) -> Any:
    """Load a MoonLight script from a ``.mls`` file."""
    ScriptLoader = _import_script_loader()
    return ScriptLoader.loadFromFile(str(path))


def set_domain(mls: Any, domain: Domain | None) -> None:
    """Set (or override) the MoonLight evaluation domain for a loaded script.

    Parameters
    mls:
        A MoonLight script object, as returned by :func:`load_script_from_text`
        or :func:`load_script_from_file`.
    domain:
        * ``"boolean"``: boolean semantics (satisfaction / falsification)
        * ``"minmax"`` or ``"robustness"``: real-valued robustness semantics
        * ``None``: do nothing
    """
    if domain is None:
        return

    if domain == "boolean":
        fn = getattr(mls, "setBooleanDomain", None)
        if not callable(fn):
            raise AttributeError("MoonLight script object lacks setBooleanDomain().")
        fn()
        return

    if domain in {"minmax", "robustness"}:
        fn = getattr(mls, "setMinMaxDomain", None)
        if not callable(fn):
            raise AttributeError("MoonLight script object lacks setMinMaxDomain().")
        fn()
        return

    raise ValueError(f"Unknown domain: {domain!r}")


def _raw_get_monitor(mls: Any, name: str) -> Any:
    """Get the raw monitor object from the MoonLight script."""
    # Preferred naming per wiki: getMonitor("formula")
    fn = getattr(mls, "getMonitor", None)
    if callable(fn):
        return fn(name)

    # Older variants sometimes used snake_case.
    fn = getattr(mls, "get_monitor", None)
    if callable(fn):  # pragma: no cover
        return fn(name)

    raise AttributeError(
        "MoonLight script object exposes neither getMonitor() nor get_monitor()."
    )


def _looks_like_spatiotemporal_monitor(raw_monitor: Any) -> bool:
    """Best-effort check for a spatio-temporal monitor signature."""
    mon = getattr(raw_monitor, "monitor", None)
    code = getattr(mon, "__code__", None)
    if code is None:
        return False
    # TemporalScriptComponent.monitor(self, time, values, parameters=None) -> 4
    # SpatialTemporalScriptComponent.monitor(self, locationTimeArray, graph,
    #                                        signalTimeArray, signalValues,
    #                                        parameters=None) -> 6
    return int(getattr(code, "co_argcount", 0)) >= 6


class MonitorAdapter:
    """A thin wrapper that adds convenience methods and shape adaptation.

    The upstream PyPI package exposes spatial monitoring only via the 4-argument
    call:

        ``monitor(location_times, graph_seq, signal_times, signal_values)``

    Within this repo (and some older examples), it is convenient to work with:

    * ``monitor(graph, signal)`` (static graph, implicit time index), or
    * ``monitorGraphTimeSeries(graph, signal)``

    Instances of this adapter behave like the underlying monitor, but add:

    * :meth:`monitor_graph_time_series` / :meth:`monitorGraphTimeSeries`
    * a permissive :meth:`monitor` that transparently upgrades the 2-arg form
      into the 4-arg spatio-temporal call when needed.
    """

    __slots__ = ("_raw", "_is_spatiotemporal")

    def __init__(self, raw_monitor: Any) -> None:
        self._raw = raw_monitor
        self._is_spatiotemporal = _looks_like_spatiotemporal_monitor(raw_monitor)

    @property
    def raw(self) -> Any:
        """Return the underlying MoonLight monitor object."""
        return self._raw

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MonitorAdapter({self._raw!r})"

    def monitor_graph_time_series(
        self,
        graph: Any,
        signal: Any,
        *,
        graph_times: Sequence[float] | None = None,
        signal_times: Sequence[float] | None = None,
        dt: float | None = None,
        t0: float = 0.0,
        signal_major: SignalMajor | None = None,
        parameters: Any | None = None,
    ) -> Any:
        return monitor_graph_time_series(
            self._raw,
            graph,
            signal,
            graph_times=graph_times,
            signal_times=signal_times,
            dt=dt,
            t0=t0,
            signal_major=signal_major,
            parameters=parameters,
        )

    # Historical CamelCase alias used in some older notebooks/examples.
    def monitorGraphTimeSeries(  # noqa: N802
        self,
        graph: Any,
        signal: Any,
        **kwargs: Any,
    ) -> Any:
        return self.monitor_graph_time_series(graph, signal, **kwargs)

    def monitor(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying monitor, with a convenience fallback for STREL.

        If the wrapped monitor is a spatio-temporal one (STREL/SSTL) and you call
        it with the 2-argument form ``monitor(graph, signal)``, we automatically:

        * construct the missing time arrays, and
        * convert the signal layout to MoonLight's expected node-major form.
        """
        raw = self._raw

        try:
            return raw.monitor(*args, **kwargs)
        except TypeError:
            if not self._is_spatiotemporal:
                raise

            # Adapt only the 2-arg (or 2-arg + positional parameters) form to
            # avoid masking genuine user errors.
            if len(args) not in {2, 3}:
                raise

            graph = args[0]
            signal = args[1]

            parameters = kwargs.get("parameters", None)
            if len(args) == 3 and parameters is None:
                parameters = args[2]

            return self.monitor_graph_time_series(graph, signal, parameters=parameters)


def get_monitor(mls: Any, name: str) -> MonitorAdapter:
    """Return a monitor for the named formula, wrapped in :class:`MonitorAdapter`."""
    return MonitorAdapter(_raw_get_monitor(mls, name))


# Graph helpers


def _grid_adjacency(nx: int, ny: int, weight: float) -> AdjacencyMatrix:
    if nx <= 0 or ny <= 0:
        raise ValueError(f"grid dimensions must be positive, got nx={nx}, ny={ny}")
    if not np.isfinite(weight):
        raise ValueError(f"edge weight must be finite, got {weight!r}")
    weight = float(weight)

    n = nx * ny
    adj: AdjacencyMatrix = [[0.0] * n for _ in range(n)]

    def idx(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            u = idx(i, j)
            if i > 0:
                v = idx(i - 1, j)
                adj[u][v] = weight
                adj[v][u] = weight
            if i + 1 < nx:
                v = idx(i + 1, j)
                adj[u][v] = weight
                adj[v][u] = weight
            if j > 0:
                v = idx(i, j - 1)
                adj[u][v] = weight
                adj[v][u] = weight
            if j + 1 < ny:
                v = idx(i, j + 1)
                adj[u][v] = weight
                adj[v][u] = weight
    return adj


def _grid_triples(nx: int, ny: int, weight: float) -> EdgeTriples:
    """Return a list of ``[u, v, w]`` triples for a 4-neighbor grid graph."""
    if nx <= 0 or ny <= 0:
        raise ValueError(f"grid dimensions must be positive, got nx={nx}, ny={ny}")
    if not np.isfinite(weight):
        raise ValueError(f"edge weight must be finite, got {weight!r}")
    weight = float(weight)

    triples: EdgeTriples = []

    def idx(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            u = idx(i, j)
            if i + 1 < nx:
                v = idx(i + 1, j)
                triples.append([float(u), float(v), weight])
                triples.append([float(v), float(u), weight])
            if j + 1 < ny:
                v = idx(i, j + 1)
                triples.append([float(u), float(v), weight])
                triples.append([float(v), float(u), weight])
    return triples


def build_grid_graph(
    n_x: int,
    n_y: int,
    *,
    weight: float = 1.0,
    return_format: Literal["adjacency", "triples", "nodes_edges"] = "adjacency",
) -> GraphSnapshot | tuple[np.ndarray, np.ndarray]:
    """Construct a simple 2D grid graph in various formats.

    Parameters
    n_x, n_y:
        Grid size along x and y (both positive).
    weight:
        Edge weight for cardinal neighbors (interpreted as distance for STREL).
    return_format:
        * ``"adjacency"``: dense (N×N) matrix. Supported by MoonLight, but large
          for big grids.
        * ``"triples"``: list of edge triples ``[u, v, weight]``. Recommended;
          matches the official MoonLight wiki examples.
        * ``"nodes_edges"``: convenience form for other tooling: a (n_x×n_y)
          integer node grid and an (#edges×2) array of directed edges (u, v).
    """
    if return_format == "adjacency":
        return _grid_adjacency(n_x, n_y, weight)
    if return_format == "triples":
        return _grid_triples(n_x, n_y, weight)
    if return_format == "nodes_edges":
        nodes = np.arange(n_x * n_y, dtype=np.int64).reshape(n_x, n_y)
        edges: list[tuple[int, int]] = []
        for i in range(n_x):
            for j in range(n_y):
                u = int(nodes[i, j])
                if i + 1 < n_x:
                    v = int(nodes[i + 1, j])
                    edges.append((u, v))
                    edges.append((v, u))
                if j + 1 < n_y:
                    v = int(nodes[i, j + 1])
                    edges.append((u, v))
                    edges.append((v, u))
        return nodes, np.asarray(edges, dtype=np.int64)
    raise ValueError(f"Unknown return_format: {return_format!r}")


def _looks_like_adjacency(mat: Any) -> bool:
    """Heuristic check for a dense adjacency matrix."""
    if not isinstance(mat, list) or not mat:
        return False
    if not all(isinstance(row, list) for row in mat):
        return False
    n = len(mat)
    return all(len(row) == n for row in mat)


def _infer_num_locations(graph: GraphSnapshot) -> int:
    """Infer the number of graph locations (nodes)."""
    if graph == []:
        return 0

    if _looks_like_adjacency(graph):
        return len(graph)

    # Otherwise treat as edge triples.
    max_idx = -1
    for triple in graph:  # type: ignore[assignment]
        if not isinstance(triple, (list, tuple)) or len(triple) < 2:
            continue
        try:
            u = int(float(triple[0]))
            v = int(float(triple[1]))
        except Exception:
            continue
        max_idx = max(max_idx, u, v)
    return max_idx + 1


def as_graph_time_series(
    graph: Any,
    times: Sequence[float] | None = None,
) -> GraphTimeSeries:
    """Normalize graph input to MoonLight's ``(location_times, graph_seq)``.

    Accepted ``graph`` forms
    * A single graph snapshot (adjacency matrix or edge triples)
    * A sequence of snapshots (``[snapshot0, snapshot1, ...]``)
    * A pre-wrapped tuple ``(times, graph_seq)``

    Notes
    -----
    MoonLight expects ``len(location_times) == len(graph_seq)``.
    """
    # Allow callers to pass (times, graph_seq) directly.
    if isinstance(graph, tuple) and len(graph) == 2:
        if times is not None:
            raise ValueError(
                "If graph is a (times, graph_seq) tuple, do not pass times=."
            )

        t_raw, g_raw = graph
        t_list = [float(t) for t in t_raw]  # type: ignore[assignment]
        g_list = list(g_raw)  # type: ignore[arg-type]

        if not t_list:
            raise ValueError("Graph time array must not be empty.")
        if len(t_list) != len(g_list):
            raise ValueError(
                f"Graph time series mismatch: len(times)={len(t_list)} != "
                f"len(graph_seq)={len(g_list)}."
            )
        return t_list, g_list

    if isinstance(graph, np.ndarray):
        graph = graph.tolist()

    # Detect a graph sequence: list of snapshots. A snapshot is itself a list of
    # lists (adjacency rows or edge triples), so a sequence is 3-level nested.
    graph_seq: GraphSequence
    if (
        isinstance(graph, list)
        and graph
        and isinstance(graph[0], list)
        and graph[0]
        and isinstance(graph[0][0], list)
    ):
        graph_seq = graph  # type: ignore[assignment]
    else:
        graph_seq = [graph]  # type: ignore[list-item]

    if not graph_seq:
        raise ValueError("Graph sequence must not be empty.")

    if times is None:
        t_list = [float(i) for i in range(len(graph_seq))]
    else:
        t_list = [float(t) for t in times]
        if not t_list:
            raise ValueError("Graph time array must not be empty.")
        if len(t_list) != len(graph_seq):
            raise ValueError(
                f"Graph time series mismatch: len(times)={len(t_list)} != "
                f"len(graph_seq)={len(graph_seq)}."
            )

    return t_list, graph_seq


# Signal helpers


def field_to_signal(
    u: np.ndarray,
    threshold: float | None = None,
    *,
    layout: Literal["xy_t", "t_xy"] = "xy_t",
    major: SignalMajor = "node",
) -> list[list[list[float]]]:
    """Convert a 3D grid field to a MoonLight-friendly nested list signal.

    Parameters
    u:
        A 3-D array representing a scalar field on a grid.

        * ``layout="xy_t"``: shape ``(nx, ny, nt)`` (default)
        * ``layout="t_xy"``: shape ``(nt, nx, ny)``
    threshold:
        If provided, values ``>= threshold`` map to ``1.0`` and others to ``0.0``
        (useful when the MoonLight script declares boolean variables). If
        ``None``, raw real values are passed through as floats.
    layout:
        Input array layout.
    major:
        Output layout.

        * ``"node"`` (default): ``[location][time][feature]`` (MoonLight's
          expected spatio-temporal format).
        * ``"time"``: ``[time][location][feature]`` (occasionally convenient
          for plotting / older helper code). When used with
          :func:`monitor_graph_time_series`, it will be transposed automatically.

    Returns
    -------
    Nested lists of floats with a singleton feature dimension (F=1).
    """
    a = np.asarray(u)
    if a.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {a.shape!r}")

    if layout == "xy_t":
        nx, ny, nt = a.shape
        node_time = a.reshape(nx * ny, nt)  # (N, T)
    elif layout == "t_xy":
        nt, nx, ny = a.shape
        node_time = a.reshape(nt, nx * ny).T  # (N, T)
    else:
        raise ValueError(f"Unknown layout: {layout!r}")

    if threshold is not None:
        node_time = (node_time >= float(threshold)).astype(float, copy=False)
    else:
        node_time = node_time.astype(float, copy=False)

    if major == "node":
        out = node_time[:, :, None]  # (N, T, 1)
    elif major == "time":
        out = node_time.T[:, :, None]  # (T, N, 1)
    else:  # pragma: no cover
        raise ValueError(f"Unknown major: {major!r}")

    # Ensure we return Python floats (not numpy scalars) for pyjnius marshalling.
    return np.asarray(out, dtype=float, order="C").tolist()


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _coerce_times(times: Sequence[float]) -> list[float]:
    t_list = [float(t) for t in times]
    if not t_list:
        raise ValueError("Time array must not be empty.")
    return t_list


def _ensure_feature_dim(values: Any) -> list[list[list[float]]]:
    """Ensure nested list values have an explicit feature dimension."""
    if not isinstance(values, list):
        raise TypeError("signal values must be a (nested) Python list.")
    if not values:
        raise ValueError("signal values must not be empty.")
    if not isinstance(values[0], list) or not values[0]:
        raise ValueError("signal values must be a non-empty nested list.")

    # If innermost elements are numbers, add a singleton feature dimension.
    if _is_number(values[0][0]):  # type: ignore[index]
        return [[[float(v)] for v in row] for row in values]  # type: ignore[arg-type]

    # Otherwise assume already has a feature dimension.
    return [
        [[float(v) for v in feat] for feat in row]
        for row in values
    ]  # type: ignore[arg-type]


def _transpose_time_to_node(
    time_major: list[list[list[float]]],
) -> list[list[list[float]]]:
    """Transpose ``[T][N][F]`` to ``[N][T][F]``."""
    t_len = len(time_major)
    n_len = len(time_major[0])
    if any(len(frame) != n_len for frame in time_major):
        raise ValueError("time-major signal has inconsistent location dimension.")

    out: list[list[list[float]]] = [[[] for _ in range(t_len)] for _ in range(n_len)]
    for t, frame in enumerate(time_major):
        for n, feat in enumerate(frame):
            out[n][t] = [float(v) for v in feat]
    return out


def _infer_signal_major(values: Any, *, n_locations: int | None) -> SignalMajor:
    """Infer whether the top-level dimension is node or time."""
    if n_locations is None:
        return "node"

    if isinstance(values, np.ndarray):
        if values.ndim >= 2:
            if values.shape[0] == n_locations:
                return "node"
            if values.shape[1] == n_locations:
                return "time"
        return "node"

    if isinstance(values, list) and values:
        if len(values) == n_locations:
            return "node"
        if isinstance(values[0], list) and len(values[0]) == n_locations:
            return "time"
    return "node"


def as_signal_time_series(
    signal: Any,
    *,
    times: Sequence[float] | None = None,
    dt: float | None = None,
    t0: float = 0.0,
    major: SignalMajor | None = None,
    n_locations: int | None = None,
) -> SignalTimeSeries:
    """Normalize a signal to MoonLight's ``(signal_times, signal_values)``.

    The returned ``signal_values`` are node-major ``[N][T][F]``.

    Accepted input forms
    * ``signal`` as a tuple ``(signal_times, signal_values)``
    * NumPy arrays with shape ``(N, T)``, ``(T, N)``, ``(N, T, F)``, or ``(T, N, F)``
    * Nested Python lists in either node-major ``[N][T][F]`` or time-major
      ``[T][N][F]`` form (feature dimension may be omitted).
    """
    # Allow callers to pass (times, values) directly.
    if isinstance(signal, tuple) and len(signal) == 2:
        times = signal[0]
        signal = signal[1]

    # Convert values to node-major [N][T][F] Python floats.
    if isinstance(signal, np.ndarray):
        arr = np.asarray(signal)
        if arr.ndim < 2:
            raise ValueError(
                "spatio-temporal monitoring expects at least 2-D signal values."
            )

        if major is None:
            major = _infer_signal_major(arr, n_locations=n_locations)

        if arr.ndim == 2:
            node_major = arr if major == "node" else arr.T
            node_major = node_major[:, :, None]  # (N, T, 1)
        elif arr.ndim == 3:
            node_major = arr if major == "node" else np.transpose(arr, (1, 0, 2))
        else:
            raise ValueError(f"Unsupported signal array shape {arr.shape!r}")

        values_node: list[list[list[float]]] = np.asarray(
            node_major,
            dtype=float,
        ).tolist()
    else:
        if not isinstance(signal, list):
            raise TypeError(
                "signal values must be a numpy array or nested Python lists."
            )

        if major is None:
            major = _infer_signal_major(signal, n_locations=n_locations)

        if major == "node":
            values_node = _ensure_feature_dim(signal)
        else:
            time_major = _ensure_feature_dim(signal)
            values_node = _transpose_time_to_node(time_major)

    n = len(values_node)
    if n_locations is not None and n != n_locations:
        raise ValueError(f"signal has {n} locations but graph expects {n_locations}.")

    t_len = len(values_node[0])
    if any(len(loc) != t_len for loc in values_node):
        raise ValueError(
            "node-major signal has inconsistent time dimension across locations."
        )

    if times is not None:
        t_list = _coerce_times(times)
        if len(t_list) != t_len:
            raise ValueError(
                f"Signal time series mismatch: len(signal_times)={len(t_list)} != "
                f"samples={t_len}."
            )
    else:
        step = 1.0 if dt is None else float(dt)
        if step <= 0.0:
            raise ValueError("dt must be > 0 when constructing implicit signal times.")
        t_list = [float(t0 + k * step) for k in range(t_len)]

    return t_list, values_node


# Monitor invocation (version-tolerant)


@overload
def monitor_graph_time_series(
    mon: Any,
    graph: Any,
    sig: Any,
    *,
    graph_times: Sequence[float] | None = None,
    signal_times: Sequence[float] | None = None,
    dt: float | None = None,
    t0: float = 0.0,
    signal_major: SignalMajor | None = None,
    parameters: Any | None = None,
) -> Any: ...


def monitor_graph_time_series(
    mon: Any,
    graph: Any,
    sig: Any,
    *,
    graph_times: Sequence[float] | None = None,
    signal_times: Sequence[float] | None = None,
    dt: float | None = None,
    t0: float = 0.0,
    signal_major: SignalMajor | None = None,
    parameters: Any | None = None,
) -> Any:
    """Invoke a spatio-temporal monitor with best-effort compatibility.

    This helper accepts a variety of graph and signal layouts and calls the
    upstream 4-argument API:

        ``monitor(location_times, graph_seq, signal_times, signal_values)``

    If a monitor exposes older 2-argument wrapper names (``monitorGraphTimeSeries``),
    those are tried first.
    """
    # Unwrap our adapter to avoid infinite recursion when probing method names.
    raw_mon = getattr(mon, "raw", mon)

    for name in ("monitor_graph_time_series", "monitorGraphTimeSeries"):
        fn = getattr(raw_mon, name, None)
        if callable(fn):
            try:
                return fn(graph, sig)
            except TypeError:
                break  # fall back to the 4-arg call

    location_times, graph_seq = as_graph_time_series(graph, times=graph_times)
    n_locations = _infer_num_locations(graph_seq[0])

    sig_times, sig_values = as_signal_time_series(
        sig,
        times=signal_times,
        dt=dt,
        t0=t0,
        major=signal_major,
        n_locations=n_locations,
    )

    fn = getattr(raw_mon, "monitor", None)
    if not callable(fn):
        raise AttributeError("Monitor object lacks a monitor(...) method.")

    try:
        if parameters is None:
            return fn(location_times, graph_seq, sig_times, sig_values)
        return fn(
            location_times,
            graph_seq,
            sig_times,
            sig_values,
            parameters=parameters,
        )
    except TypeError as e:  # pragma: no cover - fallback for older bindings
        try:
            if parameters is None:
                return fn(graph_seq, location_times, sig_values, sig_times)
            return fn(
                graph_seq,
                location_times,
                sig_values,
                sig_times,
                parameters=parameters,
            )
        except TypeError:
            raise RuntimeError(
                "MoonLight monitor did not accept a compatible spatio-temporal "
                "signature. Expected monitor(location_times, graph_seq, "
                "signal_times, signal_values)."
            ) from e


__all__ = [
    "load_script_from_text",
    "load_script_from_file",
    "set_domain",
    "get_monitor",
    "build_grid_graph",
    "field_to_signal",
    "as_graph_time_series",
    "as_signal_time_series",
    "monitor_graph_time_series",
    "MonitorAdapter",
]
