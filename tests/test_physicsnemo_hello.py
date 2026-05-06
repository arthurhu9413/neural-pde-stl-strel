# tests/test_physicsnemo_hello.py
"""PhysicsNeMo helper tests (optional dependency safe).

This test file targets :mod:`neural_pde_stl_strel.frameworks.physicsnemo_hello`.

Principles (aligned with this repository's demo/CI goals):
- **Zero required third-party deps**: the suite must run even when neither
  PhysicsNeMo nor PyTorch are installed.
- **Lazy imports**: importing the helper must not import heavy optional stacks
  (``physicsnemo`` / ``torch``).
- **Actionable failures**: when PhysicsNeMo is missing, public helpers should
  raise a clear :class:`ImportError` (or return ``False`` for availability).

We exercise both the negative path (missing dependency) and the positive path by
injecting small *stub* modules into ``sys.modules``--this keeps tests fast and
hermetic while still validating the helper's control-flow and return values.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# Test bootstrap: ensure the in-repo package is importable without installation.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_HELPER_MOD = "neural_pde_stl_strel.frameworks.physicsnemo_hello"
_PHYSICSNEMO = "physicsnemo"
_TORCH = "torch"


def _import_helper_fresh(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import the helper module with a clean import cache."""
    monkeypatch.delitem(sys.modules, _HELPER_MOD, raising=False)
    spec = importlib.util.find_spec(_HELPER_MOD)
    if spec is None:
        pytest.fail(f"Expected helper module to exist: {_HELPER_MOD}")
    return importlib.import_module(_HELPER_MOD)


def _make_module(name: str, *, is_pkg: bool = False) -> types.ModuleType:
    """Create a tiny module (optionally package-like) for stubbing imports."""
    mod = types.ModuleType(name)
    if is_pkg:
        # Mark as a package so submodules can be imported.
        mod.__path__ = []  # type: ignore[attr-defined]
    return mod


class _NoGrad:
    """Minimal context manager emulating ``torch.no_grad()``."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> bool:
        return False


class _DummyTensor:
    """Minimal tensor-like object carrying only a ``shape`` tuple."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def _install_stub_torch(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a minimal stub ``torch`` module into ``sys.modules``."""
    torch = _make_module(_TORCH)

    def manual_seed(seed: int) -> None:
        _ = seed

    def device(name: str) -> str:
        return name

    def randn(batch: int, in_features: int, *, device: str | None = None) -> _DummyTensor:
        _ = device
        return _DummyTensor((int(batch), int(in_features)))

    def no_grad() -> _NoGrad:
        return _NoGrad()

    torch.manual_seed = manual_seed  # type: ignore[attr-defined]
    torch.device = device  # type: ignore[attr-defined]
    torch.randn = randn  # type: ignore[attr-defined]
    torch.no_grad = no_grad  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, _TORCH, torch)
    return torch


def _install_stub_physicsnemo_core(
    monkeypatch: pytest.MonkeyPatch, *, version: str | None = "0.0-stub"
) -> None:
    """Install a minimal package tree for ``physicsnemo`` core imports."""
    pn = _make_module(_PHYSICSNEMO, is_pkg=True)
    if version is not None:
        pn.__version__ = version  # type: ignore[attr-defined]

    pn_models = _make_module(f"{_PHYSICSNEMO}.models", is_pkg=True)
    pn_models_mlp = _make_module(f"{_PHYSICSNEMO}.models.mlp", is_pkg=True)
    pn_fc = _make_module(f"{_PHYSICSNEMO}.models.mlp.fully_connected")

    class FullyConnected:
        def __init__(self, *, in_features: int, out_features: int) -> None:
            self._in = int(in_features)
            self._out = int(out_features)

        def to(self, device: str) -> "FullyConnected":
            _ = device
            return self

        def __call__(self, x: _DummyTensor) -> _DummyTensor:
            return _DummyTensor((int(x.shape[0]), int(self._out)))

    pn_fc.FullyConnected = FullyConnected  # type: ignore[attr-defined]

    for name, mod in [
        (_PHYSICSNEMO, pn),
        (f"{_PHYSICSNEMO}.models", pn_models),
        (f"{_PHYSICSNEMO}.models.mlp", pn_models_mlp),
        (f"{_PHYSICSNEMO}.models.mlp.fully_connected", pn_fc),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def _install_stub_physicsnemo_sym(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a minimal package tree for ``physicsnemo.sym`` imports."""
    pn_sym = _make_module(f"{_PHYSICSNEMO}.sym", is_pkg=True)
    pn_eq = _make_module(f"{_PHYSICSNEMO}.sym.eq", is_pkg=True)
    pn_pdes = _make_module(f"{_PHYSICSNEMO}.sym.eq.pdes", is_pkg=True)
    pn_ns = _make_module(f"{_PHYSICSNEMO}.sym.eq.pdes.navier_stokes")

    class NavierStokes:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)

        def pprint(self) -> None:
            print("NavierStokes(nu=..., rho=..., dim=...)")
            print("eq1: ...")
            print("eq2: ...")
            print("eq3: ...")

    pn_ns.NavierStokes = NavierStokes  # type: ignore[attr-defined]

    for name, mod in [
        (f"{_PHYSICSNEMO}.sym", pn_sym),
        (f"{_PHYSICSNEMO}.sym.eq", pn_eq),
        (f"{_PHYSICSNEMO}.sym.eq.pdes", pn_pdes),
        (f"{_PHYSICSNEMO}.sym.eq.pdes.navier_stokes", pn_ns),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def _block_imports(monkeypatch: pytest.MonkeyPatch, top_level: str) -> None:
    """Force imports of ``top_level`` to fail, even if installed."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name.split(".", 1)[0] == top_level:
            raise ModuleNotFoundError(f"No module named '{top_level}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


# Tests
def test_helper_import_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the helper should not import PhysicsNeMo/PyTorch implicitly."""
    # Start from a clean slate for the modules we care about.
    monkeypatch.delitem(sys.modules, _HELPER_MOD, raising=False)
    monkeypatch.delitem(sys.modules, _PHYSICSNEMO, raising=False)

    torch_preloaded = _TORCH in sys.modules

    _ = _import_helper_fresh(monkeypatch)

    assert _PHYSICSNEMO not in sys.modules
    if not torch_preloaded:
        assert _TORCH not in sys.modules


def test_public_api_and_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_fresh(monkeypatch)

    assert hasattr(helper, "__all__")
    exported = set(getattr(helper, "__all__"))
    expected = {
        "PHYSICSNEMO_DIST_NAME",
        "PHYSICSNEMO_MODULE_NAME",
        "physicsnemo_version",
        "physicsnemo_available",
        "physicsnemo_smoke",
        "physicsnemo_pde_summary",
    }
    assert expected.issubset(exported)

    assert len(inspect.signature(helper.physicsnemo_version).parameters) == 0
    assert len(inspect.signature(helper.physicsnemo_available).parameters) == 0

    smoke_sig = inspect.signature(helper.physicsnemo_smoke)
    assert set(smoke_sig.parameters) == {"batch", "in_features", "out_features", "seed"}

    pde_sig = inspect.signature(helper.physicsnemo_pde_summary)
    assert len(pde_sig.parameters) == 0


def test_available_false_when_import_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_fresh(monkeypatch)

    # Ensure nothing cached leaks through.
    monkeypatch.delitem(sys.modules, _PHYSICSNEMO, raising=False)
    _block_imports(monkeypatch, _PHYSICSNEMO)

    assert helper.physicsnemo_available() is False


def test_version_raises_clean_importerror_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_fresh(monkeypatch)

    monkeypatch.delitem(sys.modules, _PHYSICSNEMO, raising=False)
    _block_imports(monkeypatch, _PHYSICSNEMO)

    with pytest.raises(ImportError) as ei:
        _ = helper.physicsnemo_version()

    msg = str(ei.value)
    assert "PhysicsNeMo is not installed" in msg
    assert "pip install" in msg


def test_version_unknown_when_dunder_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_fresh(monkeypatch)

    pn = _make_module(_PHYSICSNEMO, is_pkg=True)  # no __version__
    monkeypatch.setitem(sys.modules, _PHYSICSNEMO, pn)

    assert helper.physicsnemo_version() == "unknown"


def test_version_returns_dunder_version_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_fresh(monkeypatch)

    pn = _make_module(_PHYSICSNEMO, is_pkg=True)
    pn.__version__ = "9.9.9"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _PHYSICSNEMO, pn)

    assert helper.physicsnemo_version() == "9.9.9"


def test_smoke_runs_with_stub_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_fresh(monkeypatch)

    # Install stub optional deps (PhysicsNeMo + Torch) so the smoke path can run
    # deterministically and CPU-only even in minimal environments.
    _install_stub_torch(monkeypatch)
    _install_stub_physicsnemo_core(monkeypatch, version="0.0-stub")

    metrics = helper.physicsnemo_smoke(batch=4, in_features=3, out_features=2, seed=123)
    assert metrics["version"] == "0.0-stub"
    assert tuple(metrics["out_shape"]) == (4, 2)
    assert metrics["out_batch"] == 4.0
    assert metrics["out_dim"] == 2.0


def test_pde_summary_runs_with_stub_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_fresh(monkeypatch)

    _install_stub_physicsnemo_core(monkeypatch)
    _install_stub_physicsnemo_sym(monkeypatch)

    summary = helper.physicsnemo_pde_summary()
    assert isinstance(summary, list)
    assert len(summary) >= 1
    assert all(isinstance(line, str) and line.strip() for line in summary)
    # The helper returns at most 3 lines.
    assert len(summary) <= 3
