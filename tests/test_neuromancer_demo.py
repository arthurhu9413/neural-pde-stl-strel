"""Smoke tests for the lightweight Neuromancer × STL demo.

These tests target ``neural_pde_stl_strel.frameworks.neuromancer_stl_demo``.

Why this file exists
The repository guidance for this project emphasizes:
- concrete, actually-run examples,
- explicit STL/STREL specifications (written out, not implied), and
- CPU-friendly, reproducible demos that degrade gracefully when optional stacks
  are absent.

The Neuromancer demo module is the smallest end-to-end example in the repo that
exercises this integration path:

  *a simple model*  +  *an STL safety spec*  ->  *training + post-hoc monitoring*

These tests therefore check, in order:
1) the demo source is present and documents the STL template ``G(y <= bound)``;
2) the public API contract is stable (symbols, config dataclass fields);
3) the demo can run quickly on CPU and returns sane, finite metrics without
   requiring Neuromancer/RTAMT to be installed;
4) the STL helper semantics are correct for discrete-time robustness.

Design goals (in order): correctness -> speed -> portability.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import pathlib
import re
import sys
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from types import ModuleType
from typing import Any

import pytest


# Test-time import setup (support running from a source checkout with src/).

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

# In a source checkout, add the src/ directory so imports work without
# requiring an editable install.
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MOD_PATH = "neural_pde_stl_strel.frameworks.neuromancer_stl_demo"
_DEMO_SRC = _SRC / "neural_pde_stl_strel" / "frameworks" / "neuromancer_stl_demo.py"


def _configure_torch_determinism_best_effort() -> None:
    """Make CPU results as deterministic as reasonably possible.

    The repository already sets seeds/threads in tests/conftest.py, but these
    additional calls keep this file robust when run in isolation.

    We intentionally treat failures as non-fatal (portability over strictness).
    """

    try:
        import torch  # type: ignore
    except Exception:
        return

    # Keep CPU execution stable and fast for CI.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # Can only be called once and before inter-op work starts; ignore failures.
    try:
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Prefer deterministic ops on CPU; warn-only when supported.
    try:
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[call-arg]
            except TypeError:
                torch.use_deterministic_algorithms(True)  # type: ignore[call-arg]
    except Exception:
        pass


def _import_demo_module() -> ModuleType:
    """Import the demo module, failing loudly in a source checkout."""

    spec = importlib.util.find_spec(MOD_PATH)
    if spec is None:
        pytest.skip(f"{MOD_PATH} is not importable in this environment")

    try:
        return importlib.import_module(MOD_PATH)
    except Exception as exc:
        # If we are running from a source checkout (the demo source exists), an
        # import error indicates a real regression and should fail CI.
        if _DEMO_SRC.exists():
            pytest.fail(
                f"Unexpected import error for {MOD_PATH} in source checkout: "
                f"{type(exc).__name__}: {exc}"
            )
        pytest.skip(f"Failed to import {MOD_PATH}: {type(exc).__name__}: {exc}")


def _is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


# Tests


def test_source_file_documents_api_and_stl_template() -> None:
    """The demo should be self-documenting and write the STL template explicitly."""

    if not _DEMO_SRC.exists():
        pytest.skip("demo source file not present (likely running from an installed package)")

    text = _DEMO_SRC.read_text(encoding="utf-8", errors="ignore")

    # Public API names should be present in the source.
    for name in ("DemoConfig", "train_demo", "stl_violation", "stl_offline_robustness"):
        assert name in text

    # The canonical safety template should appear in the module docs.
    # We allow arbitrary whitespace to avoid brittleness.
    assert re.search(r"G\s*\(\s*y\s*<=\s*bound\s*\)", text) is not None

    # The module should explicitly define its public surface.
    assert "__all__" in text


def test_public_api_contract() -> None:
    """The demo module should expose a small, stable API."""

    mod = _import_demo_module()

    expected_all = {"DemoConfig", "train_demo", "stl_violation", "stl_offline_robustness"}
    exported = set(getattr(mod, "__all__", []))
    assert exported == expected_all

    DemoConfig = getattr(mod, "DemoConfig")
    assert is_dataclass(DemoConfig)

    # The config is expected to be immutable (frozen dataclass) to make runs
    # reproducible and easy to pass around.
    assert getattr(DemoConfig, "__dataclass_params__").frozen is True

    field_names = {f.name for f in dataclass_fields(DemoConfig)}
    # These fields are part of the minimal contract used by the demo + tests.
    required_fields = {
        "n",
        "epochs",
        "lr",
        "device",
        "seed",
        "bound",
        "weight",
        "soft_beta",
        "use_soft_stl_in_loss",
        "nm_batch_size",
        "nm_epochs",
        "nm_lr",
    }
    assert required_fields.issubset(field_names)

    assert callable(getattr(mod, "train_demo"))
    assert callable(getattr(mod, "stl_violation"))
    assert callable(getattr(mod, "stl_offline_robustness"))


def test_train_demo_smoke_cpu_returns_sane_metrics() -> None:
    """Run a tiny CPU-only demo and validate returned metrics."""

    mod = _import_demo_module()
    _configure_torch_determinism_best_effort()

    # train_demo uses torch internally; skip cleanly if torch isn't available.
    pytest.importorskip("torch")

    DemoConfig = getattr(mod, "DemoConfig")
    train_demo = getattr(mod, "train_demo")

    # Keep the run tiny for CI speed.
    cfg = DemoConfig(
        n=64,
        epochs=5,
        nm_epochs=1,  # exercise the optional neuromancer path if installed
        nm_batch_size=32,
        device="cpu",
        seed=0,
        lr=1e-3,
        nm_lr=1e-3,
        # Make the penalty moderate to avoid numerical explosions in very short runs.
        weight=25.0,
    )

    out = train_demo(cfg)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"pytorch", "neuromancer"}

    pt = out["pytorch"]
    assert isinstance(pt, dict)

    for key in ("final_mse", "max_violation", "stl_robustness"):
        assert key in pt
        assert isinstance(pt[key], float)
        assert math.isfinite(pt[key])

    assert pt["final_mse"] >= 0.0
    assert pt["max_violation"] >= 0.0

    # Robustness/violation consistency for the predicate y <= bound:
    # - If robustness >= 0, there is no pointwise violation (mean violation == 0).
    # - If robustness < 0, there is at least one violating time (mean violation > 0).
    tol = 1e-12
    if pt["stl_robustness"] >= -tol:
        assert pt["max_violation"] <= 1e-8
    else:
        assert pt["max_violation"] > 0.0

    nm = out["neuromancer"]
    if nm is not None:
        assert isinstance(nm, dict)
        for key in ("final_mse", "max_violation", "stl_robustness"):
            assert key in nm
            assert isinstance(nm[key], float)
            assert math.isfinite(nm[key])

        assert nm["final_mse"] >= 0.0
        assert nm["max_violation"] >= 0.0

        if nm["stl_robustness"] >= -tol:
            assert nm["max_violation"] <= 1e-8
        else:
            assert nm["max_violation"] > 0.0


def test_train_demo_reproducible_best_effort_cpu() -> None:
    """Same config + seed should give the same metrics on CPU (within tolerance)."""

    mod = _import_demo_module()
    _configure_torch_determinism_best_effort()
    pytest.importorskip("torch")

    DemoConfig = getattr(mod, "DemoConfig")
    train_demo = getattr(mod, "train_demo")

    cfg = DemoConfig(n=32, epochs=2, nm_epochs=0, device="cpu", seed=42)
    out1 = train_demo(cfg)
    out2 = train_demo(cfg)

    mse1 = float(out1["pytorch"]["final_mse"])
    mse2 = float(out2["pytorch"]["final_mse"])

    assert _is_finite_number(mse1)
    assert _is_finite_number(mse2)

    # Loose tolerance: deterministic *should* be exact on CPU, but keep robust
    # against minor library/version differences.
    assert abs(mse1 - mse2) <= max(1e-8, 1e-4 * (1.0 + abs(mse1)))


def test_stl_helpers_semantics_and_monotonicity() -> None:
    """Check the discrete-time robustness semantics for G(y <= bound)."""

    mod = _import_demo_module()
    torch = pytest.importorskip("torch")

    stl_offline_robustness = getattr(mod, "stl_offline_robustness")
    stl_violation = getattr(mod, "stl_violation")

    # Tiny signal with both safe and unsafe points.
    u = torch.tensor([-0.2, 0.1, 0.9, 0.0], dtype=torch.float32)
    bound = 0.8

    # Exact robustness for phi := G(y <= bound):  rho = min_t(bound - u[t]).
    rho = stl_offline_robustness(u, bound)
    assert isinstance(rho, float)
    assert math.isfinite(rho)

    expected = float((bound - u).min().item())
    assert abs(rho - expected) <= 1e-12 * (1.0 + abs(expected))

    # Pointwise predicate violation is ReLU(u - bound).
    v = stl_violation(u, bound)
    assert torch.allclose(v, torch.relu(u - bound))

    # Identity: rho = bound - max(u)  =>  -rho = max(u - bound).
    max_unclipped = float((u - bound).max().item())
    assert abs(-rho - max_unclipped) <= 1e-7 * (1.0 + abs(max_unclipped))

    # Monotonicity: increasing the bound cannot decrease robustness.
    rho_hi = stl_offline_robustness(u, bound + 0.1)
    rho_lo = stl_offline_robustness(u, bound - 0.1)
    assert rho_lo <= rho_hi + 1e-12

    # Monotonicity: increasing the bound cannot increase the (mean) violation.
    v_hi = float(stl_violation(u, bound + 0.1).mean().item())
    v_lo = float(stl_violation(u, bound - 0.1).mean().item())
    assert v_hi <= v_lo + 1e-12
