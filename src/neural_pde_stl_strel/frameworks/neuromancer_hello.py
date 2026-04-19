from __future__ import annotations

"""neural_pde_stl_strel.frameworks.neuromancer_hello

A tiny, CPU-first, zero-training "hello" helper for the optional NeuroMANCER
(`neuromancer`) dependency.

This repository treats NeuroMANCER as an **optional extra** (see
`requirements-extra.txt`). For CI and for lightweight installs we want:

- importing this repository to stay fast and not import NeuroMANCER or PyTorch
  unless explicitly requested
- a crisp install check that exercises the core symbolic API on CPU

High-level data flow (smoke test)
    {"p": Tensor} ──▶ Node(p -> x) ──▶ {"x": Tensor, ...}
                              │
                              ├──▶ Objective:   (x - 0.5)^2
                              └──▶ Constraints: (optional trivial bounds)
                                      │
                                      ▼
                              PenaltyLoss(objectives, constraints)
                                      │
                                      ▼
                             Problem(nodes, loss)(batch) -> out["loss"]

Design goals
- Optional-dependency safe: no `neuromancer` / `torch` import at module import time.
- Version-tolerant: tolerate small namespace moves between NeuroMANCER releases.
- Fast: deterministic tensors, CPU-only, no training.
"""

from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec as _find_spec
from types import ModuleType
from typing import Any, Final

NEUROMANCER_MODULE_NAME: Final[str] = "neuromancer"
NEUROMANCER_DIST_NAME: Final[str] = "neuromancer"

_NEUROMANCER_IMPORT_HINT: Final[str] = (
    "neuromancer not installed. Try one of:\n"
    "  * pip install neuromancer\n"
    "  * pip install -r requirements-extra.txt"
)

_TORCH_IMPORT_HINT: Final[str] = (
    "torch is required for neuromancer_smoke. Try one of:\n"
    "  * pip install torch\n"
    "  * pip install -r requirements.txt"
)

_SENTINEL = object()


def _require_neuromancer() -> Any:
    """Import and return the top-level ``neuromancer`` module.

    This is kept inside a function so importing this helper module stays cheap.
    """

    try:
        return import_module(NEUROMANCER_MODULE_NAME)
    except Exception as exc:  # pragma: no cover
        raise ImportError(_NEUROMANCER_IMPORT_HINT) from exc


def _resolve(
    nm: Any,
    dotted_alternatives: tuple[tuple[str, ...], ...],
    *,
    root_alias: str = "nm",
) -> tuple[Any, str]:
    """Resolve an attribute from ``neuromancer`` given dotted candidates.

    The resolver is robust to submodules not being imported in
    ``neuromancer.__init__``: if an attribute is missing and the current object
    is a module, we also try to import ``<module>.<name>``.

    Returns
    -------
    (obj, path)
        ``obj`` is the resolved attribute. ``path`` is the chosen dotted path,
        e.g. ``"nm.loss.PenaltyLoss"``.
    """

    for parts in dotted_alternatives:
        obj: Any = nm
        ok = True
        for name in parts:
            try:
                val = getattr(obj, name, _SENTINEL)
            except Exception:
                val = _SENTINEL

            if val is not _SENTINEL and val is not None:
                obj = val
                continue

            if isinstance(obj, ModuleType):
                try:
                    obj = import_module(f"{obj.__name__}.{name}")
                    continue
                except Exception:
                    ok = False
                    break

            ok = False
            break

        if ok:
            return obj, f"{root_alias}.{'.'.join(parts)}"

    alts = " | ".join(f"{root_alias}.{'.'.join(p)}" for p in dotted_alternatives)
    raise AttributeError(f"Could not resolve any of: {alts}")


def neuromancer_available() -> bool:
    """Return True iff ``neuromancer`` appears importable (without importing it)."""

    return _find_spec(NEUROMANCER_MODULE_NAME) is not None


def neuromancer_version() -> str:
    """Return ``neuromancer.__version__`` or ``\"unknown\"`` if absent."""

    nm = _require_neuromancer()
    ver = getattr(nm, "__version__", None)
    return ver if isinstance(ver, str) and ver else "unknown"


def neuromancer_smoke(
    batch_size: int = 4,
    *,
    include_trivial_constraints: bool = True,
) -> dict[str, Any]:
    """Construct and evaluate a minimal NeuroMANCER Problem on CPU.

    The goal is a smoke test ("does the symbolic API execute end-to-end?"), not
    a benchmark.
    """

    if not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an int; got {type(batch_size).__name__}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")

    nm = _require_neuromancer()

    try:
        import torch  # defer heavy import
    except Exception as exc:  # pragma: no cover
        raise ImportError(_TORCH_IMPORT_HINT) from exc

    variable_obj, variable_path = _resolve(
        nm,
        (
            ("constraint", "variable"),
            ("constraints", "variable"),
            ("variable",),
        ),
    )
    Node, node_path = _resolve(nm, (("system", "Node"), ("Node",)))
    PenaltyLoss, loss_path = _resolve(nm, (("loss", "PenaltyLoss"), ("PenaltyLoss",)))
    Problem, problem_path = _resolve(nm, (("problem", "Problem"), ("Problem",)))

    variable: Callable[[str], Any] = variable_obj  # type: ignore[assignment]

    class _Id(torch.nn.Module):
        def forward(self, p: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return p

    try:
        node = Node(_Id(), ["p"], ["x"], name="id_map")
    except TypeError:
        # Extremely defensive: tolerate older signatures that omit `name`.
        node = Node(_Id(), ["p"], ["x"])

    x = variable("x")
    expr = (x - 0.5) ** 2
    try:
        obj = expr.minimize(weight=1.0, name="obj")
    except TypeError:
        obj = expr.minimize(weight=1.0)

    constraints: list[Any] = []
    if include_trivial_constraints:
        # These are satisfied for p == 1.0, so they should contribute ~0 penalty
        # while still exercising constraint construction and evaluation.
        try:
            constraints = [100.0 * (x <= 1.0), 100.0 * (x >= 0.0)]
        except Exception:
            constraints = [(x <= 1.0), (x >= 0.0)]

    try:
        loss = PenaltyLoss(objectives=[obj], constraints=constraints)
    except TypeError:
        loss = PenaltyLoss([obj], constraints)

    try:
        problem = Problem(nodes=[node], loss=loss)
    except TypeError:
        problem = Problem([node], loss)

    p = torch.ones(batch_size, 1, dtype=torch.float32, device="cpu")
    batch = {"p": p}

    def _extract_loss(out: Any) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            return out[0]
        if isinstance(out, dict):
            for k in (
                "loss",
                "train_loss",
                "dev_loss",
                "mean_train_loss",
                "mean_dev_loss",
            ):
                if k in out and isinstance(out[k], torch.Tensor):
                    return out[k]
            scalar_tensors = [
                v for v in out.values() if isinstance(v, torch.Tensor) and v.numel() == 1
            ]
            if len(scalar_tensors) == 1:
                return scalar_tensors[0]

        if hasattr(problem, "compute_loss"):
            maybe = problem.compute_loss(batch)  # type: ignore[attr-defined]
            if isinstance(maybe, torch.Tensor):
                return maybe

        raise TypeError(f"Unexpected Problem output type: {type(out).__name__}")

    with torch.no_grad():
        out = problem(batch)

    loss_tensor = _extract_loss(out)
    if loss_tensor.numel() != 1:
        loss_tensor = loss_tensor.mean()
    loss_value = float(loss_tensor.detach().cpu().item())

    return {
        "version": neuromancer_version(),
        "loss": loss_value,
        "samples": batch_size,
        "device": "cpu",
        "constraints": bool(include_trivial_constraints),
        "resolved": {
            "variable": variable_path,
            "Node": node_path,
            "PenaltyLoss": loss_path,
            "Problem": problem_path,
        },
    }


__all__ = [
    "NEUROMANCER_DIST_NAME",
    "NEUROMANCER_MODULE_NAME",
    "neuromancer_available",
    "neuromancer_version",
    "neuromancer_smoke",
]
