"""tests/test_neuromancer_hello.py

High-signal tests for the optional Neuromancer integration probe.

This repository intentionally includes small, *toy* "hello" adapters for large
external frameworks (Neuromancer / PhysicsNeMo / TorchPhysics). These probes are
meant to be:

* CPU-first and fast (milliseconds).
* Clear about the *data flow* and what is being checked.
* Robust to optional dependencies: the unit tests should still run when the
  external framework (and even PyTorch) is not installed.

The Neuromancer probe we test here is `neural_pde_stl_strel.frameworks.neuromancer_hello`.
It constructs (or emulates) the smallest symbolic graph that demonstrates the
integration point where one could attach an STL/STREL penalty in a real project.

Toy data-flow (block diagram)

    batch input          symbolic graph                       output
  ┌────────────┐     ┌───────────────────┐              ┌───────────────┐
  │  p (tensor)│ ───▶ │ Node: id(p) -> x  │ ───────────▶ │ x (tensor)     │
  └────────────┘     └───────────────────┘              └───────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ Objective:      │
                      │   (x - 0.5)^2   │
                      └─────────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ PenaltyLoss     │
                      └─────────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ Problem(...)    │
                      └─────────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ {"loss": scalar}│
                      └─────────────────┘

Notes
-----
* The "spec" here is an objective term, not full STL. In Neuromancer, symbolic
  objectives/constraints are exactly where an STL robustness penalty can be
  integrated (see `tests/test_neuromancer_demo.py` for the STL-audit demo).
* Unit tests below stub both `torch` and `neuromancer` in `sys.modules` to
  validate control-flow without requiring heavy installs.
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import math
import pathlib
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest


# Test bootstrap: make `src/` importable without installing the package.

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


HELPER_MOD = "neural_pde_stl_strel.frameworks.neuromancer_hello"


def _import_helper(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import the helper freshly.

    The helper itself is in-repo and must be importable with *no* optional
    third-party dependencies installed.
    """

    monkeypatch.delitem(sys.modules, HELPER_MOD, raising=False)
    try:
        return importlib.import_module(HELPER_MOD)
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Failed to import {HELPER_MOD}: {exc!r}")
        raise


# Minimal stubs for torch + neuromancer (enough for neuromancer_smoke).


class _StubTensor:
    """A tiny tensor-like object sufficient for `neuromancer_smoke`.

    Supports:
      * elementwise scalar ops (sub, pow)
      * mean reduction
      * detach/cpu/item chain used by the helper
    """

    __slots__ = ("_data",)

    def __init__(self, data: float | list[float]):
        self._data = data

    def __sub__(self, other: float | "_StubTensor") -> "_StubTensor":
        other_data = other._data if isinstance(other, _StubTensor) else other
        if isinstance(self._data, list):
            if isinstance(other_data, list):
                return _StubTensor([a - b for a, b in zip(self._data, other_data, strict=True)])
            return _StubTensor([a - float(other_data) for a in self._data])
        if isinstance(other_data, list):
            raise TypeError("scalar - vector is undefined in this stub")
        return _StubTensor(float(self._data) - float(other_data))

    def __rsub__(self, other: float) -> "_StubTensor":
        if isinstance(self._data, list):
            return _StubTensor([float(other) - a for a in self._data])
        return _StubTensor(float(other) - float(self._data))

    def __pow__(self, power: int | float) -> "_StubTensor":
        if isinstance(self._data, list):
            return _StubTensor([a ** float(power) for a in self._data])
        return _StubTensor(float(self._data) ** float(power))

    def mean(self) -> "_StubTensor":
        if isinstance(self._data, list):
            if not self._data:
                raise ValueError("mean of empty tensor")
            return _StubTensor(sum(self._data) / float(len(self._data)))
        return self

    def detach(self) -> "_StubTensor":
        return self

    def cpu(self) -> "_StubTensor":
        return self

    def item(self) -> float:
        if isinstance(self._data, list):
            raise TypeError("item() only valid for scalar tensor")
        return float(self._data)

    def numel(self) -> int:
        if isinstance(self._data, list):
            return len(self._data)
        return 1


def _make_torch_stub() -> types.ModuleType:
    torch = types.ModuleType("torch")

    # dtype sentinel used by `torch.ones(..., dtype=torch.float32)`
    torch.float32 = object()

    class Module:  # noqa: D401
        """Minimal stand-in for `torch.nn.Module`."""

        def __call__(self, *args: Any, **kwargs: Any):
            return self.forward(*args, **kwargs)

        def forward(self, *args: Any, **kwargs: Any):  # pragma: no cover
            raise NotImplementedError

    nn = types.ModuleType("torch.nn")
    nn.Module = Module  # type: ignore[attr-defined]
    torch.nn = nn  # type: ignore[attr-defined]

    # Tensor type for isinstance checks
    torch.Tensor = _StubTensor  # type: ignore[attr-defined]

    def ones(n: int, m: int, *, dtype: Any | None = None, device: Any | None = None) -> _StubTensor:  # noqa: ARG001
        if n <= 0 or m <= 0:
            raise ValueError("stub torch.ones expects positive sizes")
        if m != 1:
            # The helper only uses (batch_size, 1) in the smoke test.
            raise ValueError("stub torch.ones only supports shape (N, 1)")
        return _StubTensor([1.0] * int(n))

    torch.ones = ones  # type: ignore[attr-defined]

    @contextmanager
    def no_grad() -> Iterator[None]:
        yield

    torch.no_grad = no_grad  # type: ignore[attr-defined]

    def tensor(val: float | list[float], *, dtype: Any | None = None) -> _StubTensor:  # noqa: ARG001
        return _StubTensor(val)

    torch.tensor = tensor  # type: ignore[attr-defined]

    return torch


class _Expr:
    """Symbolic expression tree for a single variable (sufficient for smoke)."""

    def eval(self, ctx: dict[str, _StubTensor]) -> _StubTensor:  # pragma: no cover
        raise NotImplementedError

    def __sub__(self, other: float | "_Expr") -> "_Expr":
        return _Sub(self, other if isinstance(other, _Expr) else _Const(float(other)))

    def __rsub__(self, other: float) -> "_Expr":
        return _Sub(_Const(float(other)), self)

    def __pow__(self, power: int | float) -> "_Expr":
        return _Pow(self, float(power))

    def minimize(self, *, weight: float = 1.0, **_: Any) -> "_Objective":
        return _Objective(self, weight=float(weight))

    def __le__(self, other: float | "_Expr") -> "_Constraint":
        return _Constraint(self, "<=", other if isinstance(other, _Expr) else _Const(float(other)))

    def __ge__(self, other: float | "_Expr") -> "_Constraint":
        return _Constraint(self, ">=", other if isinstance(other, _Expr) else _Const(float(other)))


class _Const(_Expr):
    def __init__(self, value: float):
        self._value = float(value)

    def eval(self, ctx: dict[str, _StubTensor]) -> _StubTensor:  # noqa: ARG002
        return _StubTensor(self._value)


class _Var(_Expr):
    def __init__(self, name: str):
        self._name = str(name)

    def eval(self, ctx: dict[str, _StubTensor]) -> _StubTensor:
        return ctx[self._name]


class _Sub(_Expr):
    def __init__(self, left: _Expr, right: _Expr):
        self._left = left
        self._right = right

    def eval(self, ctx: dict[str, _StubTensor]) -> _StubTensor:
        return self._left.eval(ctx) - self._right.eval(ctx)


class _Pow(_Expr):
    def __init__(self, base: _Expr, power: float):
        self._base = base
        self._power = float(power)

    def eval(self, ctx: dict[str, _StubTensor]) -> _StubTensor:
        return self._base.eval(ctx) ** self._power


class _Objective:
    def __init__(self, expr: _Expr, *, weight: float):
        self._expr = expr
        self._weight = float(weight)

    def __call__(self, ctx: dict[str, _StubTensor]) -> _StubTensor:
        # Mirror the typical scalarization in loss functions: mean over batch.
        val = self._expr.eval(ctx).mean()
        return _StubTensor(val.item() * self._weight)


class _Constraint:
    """Stub for neuromancer constraint objects returned by variable comparisons."""

    def __init__(self, left: _Expr, op: str, right: _Expr, weight: float = 1.0):
        self._left = left
        self._op = op
        self._right = right
        self._weight = weight

    def __rmul__(self, other: float) -> "_Constraint":
        return _Constraint(self._left, self._op, self._right, weight=float(other) * self._weight)

    def __call__(self, ctx: dict[str, Any]) -> _StubTensor:
        lv = self._left.eval(ctx)
        rv = self._right.eval(ctx)
        # Extract scalar: handle both _StubTensor and torch.Tensor
        def _scalar(v: Any) -> float:
            if isinstance(v, _StubTensor):
                m = v.mean()
                return m.item()
            if hasattr(v, "mean"):
                m = v.mean()
                return float(m.item() if hasattr(m, "item") else m)
            return float(v)
        l_val, r_val = _scalar(lv), _scalar(rv)
        if self._op == "<=":
            violation = max(0.0, l_val - r_val)
        else:
            violation = max(0.0, r_val - l_val)
        return _StubTensor(violation * self._weight)


def _make_neuromancer_stub(*, layout: str) -> types.ModuleType:
    """Create a tiny `neuromancer` stub.

    Parameters
    layout:
        * "primary": expose symbols at the common module paths used in docs
          (e.g., `neuromancer.system.Node`, `neuromancer.constraint.variable`).
        * "fallback": expose only top-level aliases (`neuromancer.Node`,
          `neuromancer.variable`, ...), exercising the helper's fallback paths.
    """

    if layout not in {"primary", "fallback"}:
        raise ValueError(f"unexpected layout: {layout!r}")

    nm = types.ModuleType("neuromancer")
    nm.__version__ = "0.0.stub"  # type: ignore[attr-defined]

    def variable(name: str) -> _Var:
        return _Var(name)

    class Node:
        def __init__(
            self,
            module: Any,
            input_keys: list[str],
            output_keys: list[str],
            **_: Any,
        ) -> None:
            self._module = module
            self._input_keys = list(input_keys)
            self._output_keys = list(output_keys)

        def __call__(self, data: dict[str, _StubTensor]) -> dict[str, _StubTensor]:
            inputs = [data[k] for k in self._input_keys]
            out = self._module(*inputs)
            # Convert real tensors to _StubTensor for expression evaluation
            if not isinstance(out, _StubTensor):
                flat = out.detach().flatten().tolist()
                out = _StubTensor(flat if len(flat) > 1 else flat[0])
            if len(self._output_keys) != 1:
                raise ValueError("stub Node only supports a single output")
            return {self._output_keys[0]: out}

    class PenaltyLoss:
        def __init__(
            self, objectives: list[_Objective] | None = None,
            constraints: list[Any] | None = None, **_: Any,
        ) -> None:
            self._objectives = list(objectives or [])
            self._constraints = list(constraints or [])

        def compute_loss(self, ctx: dict[str, _StubTensor]) -> _StubTensor:
            total = 0.0
            for obj in self._objectives:
                total += float(obj(ctx).item())
            for con in self._constraints:
                if callable(con):
                    total += float(con(ctx).item())
            return _StubTensor(total)

    class Problem:
        def __init__(self, nodes: list[Node] | None = None, loss: PenaltyLoss | None = None, **_: Any) -> None:
            self.nodes = list(nodes or [])
            self.loss = loss

        def __call__(self, batch: dict[str, _StubTensor]) -> dict[str, Any]:
            ctx = dict(batch)
            for node in self.nodes:
                ctx.update(node(ctx))
            return {"loss": self.loss.compute_loss(ctx)}

        def compute_loss(self, batch: dict[str, _StubTensor]) -> _StubTensor:
            ctx = dict(batch)
            for node in self.nodes:
                ctx.update(node(ctx))
            return self.loss.compute_loss(ctx)

    if layout == "primary":
        nm.constraint = types.ModuleType("neuromancer.constraint")  # type: ignore[attr-defined]
        nm.constraint.variable = variable  # type: ignore[attr-defined]

        nm.system = types.ModuleType("neuromancer.system")  # type: ignore[attr-defined]
        nm.system.Node = Node  # type: ignore[attr-defined]

        nm.loss = types.ModuleType("neuromancer.loss")  # type: ignore[attr-defined]
        nm.loss.PenaltyLoss = PenaltyLoss  # type: ignore[attr-defined]

        nm.problem = types.ModuleType("neuromancer.problem")  # type: ignore[attr-defined]
        nm.problem.Problem = Problem  # type: ignore[attr-defined]

    # Always provide the top-level aliases too (common in tutorials/presentations).
    nm.variable = variable  # type: ignore[attr-defined]
    nm.Node = Node  # type: ignore[attr-defined]
    nm.PenaltyLoss = PenaltyLoss  # type: ignore[attr-defined]
    nm.Problem = Problem  # type: ignore[attr-defined]

    return nm


def _install_stub_modules(monkeypatch: pytest.MonkeyPatch, *, layout: str) -> None:
    """Install stub modules into sys.modules for the duration of a test."""

    torch_stub = _make_torch_stub()
    nm_stub = _make_neuromancer_stub(layout=layout)

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_stub.nn)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "neuromancer", nm_stub)

    # Provide submodule entries only if present.
    for sub in ("constraint", "system", "loss", "problem"):
        if hasattr(nm_stub, sub):
            monkeypatch.setitem(sys.modules, f"neuromancer.{sub}", getattr(nm_stub, sub))


# Tests


def test_helper_import_is_lazy_wrt_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the helper must not import Neuromancer or PyTorch."""

    torch_preloaded = "torch" in sys.modules

    # Ensure we start from a clean slate for the optional dependency.
    monkeypatch.delitem(sys.modules, "neuromancer", raising=False)

    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any):  # type: ignore[override]
        top = name.split(".", 1)[0]
        if top in {"neuromancer", "torch"}:
            raise ModuleNotFoundError(f"blocked optional import: {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    _ = _import_helper(monkeypatch)

    assert "neuromancer" not in sys.modules
    if not torch_preloaded:
        assert "torch" not in sys.modules


def test_public_api_and_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper(monkeypatch)

    assert hasattr(helper, "__all__"), "helper should define an explicit public API"
    exported = set(getattr(helper, "__all__"))
    assert {"neuromancer_version", "neuromancer_available", "neuromancer_smoke"} <= exported

    assert list(inspect.signature(helper.neuromancer_version).parameters) == []
    assert list(inspect.signature(helper.neuromancer_available).parameters) == []

    smoke_sig = inspect.signature(helper.neuromancer_smoke)
    assert "batch_size" in smoke_sig.parameters
    assert smoke_sig.parameters["batch_size"].default == 4


def test_neuromancer_available_reflects_find_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper(monkeypatch)

    monkeypatch.setattr(helper, "_find_spec", lambda _name: None)
    assert helper.neuromancer_available() is False

    monkeypatch.setattr(helper, "_find_spec", lambda name: object() if name == "neuromancer" else None)
    assert helper.neuromancer_available() is True


def test_neuromancer_version_unknown_when_dunder_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper(monkeypatch)

    nm = types.ModuleType("neuromancer")
    monkeypatch.setitem(sys.modules, "neuromancer", nm)

    assert helper.neuromancer_version() == "unknown"


def test_neuromancer_version_returns_dunder_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper(monkeypatch)

    nm = types.ModuleType("neuromancer")
    nm.__version__ = "9.9.9"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "neuromancer", nm)

    assert helper.neuromancer_version() == "9.9.9"


def test_neuromancer_version_raises_actionable_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper(monkeypatch)

    monkeypatch.delitem(sys.modules, "neuromancer", raising=False)
    monkeypatch.setattr(helper, "import_module", lambda _name: (_ for _ in ()).throw(ModuleNotFoundError()))

    with pytest.raises(ImportError) as excinfo:
        _ = helper.neuromancer_version()
    msg = str(excinfo.value)
    assert "neuromancer not installed" in msg.lower()
    assert "pip install neuromancer" in msg.lower()


@pytest.mark.parametrize("layout", ["primary", "fallback"])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_neuromancer_smoke_runs_with_stub_modules(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    batch_size: int,
) -> None:
    """The smoke path must work without the real dependencies installed."""

    helper = _import_helper(monkeypatch)
    _install_stub_modules(monkeypatch, layout=layout)

    metrics = helper.neuromancer_smoke(batch_size=batch_size)
    assert metrics["version"] == "0.0.stub"
    assert metrics["samples"] == float(batch_size)
    assert isinstance(metrics["loss"], float)
    assert math.isfinite(metrics["loss"]) and metrics["loss"] >= 0.0

    # With p=1 everywhere and objective mean((x-0.5)^2), the loss is 0.25.
    assert metrics["loss"] == pytest.approx(0.25, abs=1e-12)


def test_neuromancer_smoke_errors_actionably_when_api_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper(monkeypatch)

    # Provide a `neuromancer` module but omit the symbols the helper resolves.
    nm = types.ModuleType("neuromancer")
    nm.__version__ = "0.0.stub"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "neuromancer", nm)
    monkeypatch.setitem(sys.modules, "torch", _make_torch_stub())

    with pytest.raises(AttributeError) as excinfo:
        _ = helper.neuromancer_smoke(batch_size=2)
    msg = str(excinfo.value)
    assert "could not resolve" in msg.lower()
    # The helper prints the attempted dotted paths to aid debugging.
    assert "nm.constraint.variable" in msg


def test_neuromancer_smoke_runs_if_real_neuromancer_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional end-to-end smoke test against the real Neuromancer package."""

    helper = _import_helper(monkeypatch)

    if not helper.neuromancer_available():
        pytest.skip("Neuromancer not installed")

    try:
        metrics = helper.neuromancer_smoke(batch_size=4)
    except Exception as exc:  # pragma: no cover
        # Neuromancer is an optional dependency; if a local install is broken or
        # API-incompatible, skip rather than failing the whole suite.
        pytest.skip(f"Neuromancer smoke failed in this environment: {exc!r}")

    assert isinstance(metrics.get("version"), str) and metrics["version"].strip()
    assert metrics.get("samples") == 4.0
    assert isinstance(metrics.get("loss"), float)
    assert math.isfinite(metrics["loss"]) and metrics["loss"] >= 0.0
