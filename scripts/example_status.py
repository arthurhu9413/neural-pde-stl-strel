#!/usr/bin/env python3
"""Print a compact inventory of examples and their status.

This script exists for one purpose: make it obvious to a reader
what is currently a committed result, what is a runnable demo, and what is still
only a probe or draft integration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Status = Literal["reported-result", "runnable-demo", "probe", "planned"]


@dataclass(frozen=True)
class ExampleRow:
    name: str
    logic: str
    training_time: bool
    backend: str
    status: Status
    evidence: tuple[str, ...]


ROWS: tuple[ExampleRow, ...] = (
    ExampleRow(
        name="1D diffusion baseline",
        logic="none",
        training_time=False,
        backend="optional RTAMT audit",
        status="reported-result",
        evidence=("results/diffusion1d_main_summary.json",),
    ),
    ExampleRow(
        name="1D diffusion + STL regularization",
        logic="STL",
        training_time=True,
        backend="RTAMT + exact sampled audit",
        status="reported-result",
        evidence=(
            "results/diffusion1d_main_summary.json",
            "figs/diffusion1d_comparison.png",
            "figs/diffusion1d_training_dynamics.png",
        ),
    ),
    ExampleRow(
        name="1D diffusion lambda ablation",
        logic="STL",
        training_time=True,
        backend="exact sampled audit",
        status="reported-result",
        evidence=("results/diffusion1d_ablation_summary.json", "figs/diffusion1d_lambda_ablation.png"),
    ),
    ExampleRow(
        name="2D heat rollout + STREL-style monitoring",
        logic="STREL-style containment",
        training_time=False,
        backend="MoonLight-oriented audit",
        status="reported-result",
        evidence=("results/heat2d_strel_monitoring.json", "figs/heat2d_field_evolution.png"),
    ),
    ExampleRow(
        name="2D heat + scalar STL safety",
        logic="STL",
        training_time=True,
        backend="repo smooth STL",
        status="runnable-demo",
        evidence=("configs/heat2d_stl_safe.yaml",),
    ),
    ExampleRow(
        name="2D heat + scalar STL eventually-cool",
        logic="STL",
        training_time=True,
        backend="repo smooth STL",
        status="runnable-demo",
        evidence=("configs/heat2d_stl_eventually.yaml",),
    ),
    ExampleRow(
        name="Neuromancer toy demo",
        logic="STL",
        training_time=True,
        backend="repo smooth STL",
        status="runnable-demo",
        evidence=("scripts/train_neuromancer_stl.py",),
    ),
    ExampleRow(
        name="TorchPhysics Burgers draft",
        logic="STL planned",
        training_time=False,
        backend="none yet",
        status="planned",
        evidence=("scripts/train_burgers_torchphysics.py",),
    ),
    ExampleRow(
        name="PhysicsNeMo hello probe",
        logic="none",
        training_time=False,
        backend="none",
        status="probe",
        evidence=("src/neural_pde_stl_strel/frameworks/physicsnemo_hello.py",),
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate() -> list[str]:
    root = _repo_root()
    missing: list[str] = []
    for row in ROWS:
        for rel in row.evidence:
            if not (root / rel).exists():
                missing.append(rel)
    return missing


def _render_table() -> str:
    headers = ["status", "example", "logic", "train?", "backend"]
    rows: list[list[str]] = [headers]
    for row in ROWS:
        rows.append([
            row.status,
            row.name,
            row.logic,
            "yes" if row.training_time else "no",
            row.backend,
        ])
    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]

    def fmt(r: list[str]) -> str:
        return "  ".join(r[i].ljust(widths[i]) for i in range(len(headers)))

    out = [fmt(rows[0]), fmt(["-" * w for w in widths])]
    out.extend(fmt(r) for r in rows[1:])
    return "\n".join(out)


def _payload() -> dict[str, object]:
    missing = _validate()
    return {
        "summary": {
            "reported_results": sum(r.status == "reported-result" for r in ROWS),
            "runnable_demos": sum(r.status == "runnable-demo" for r in ROWS),
            "probes": sum(r.status == "probe" for r in ROWS),
            "planned": sum(r.status == "planned" for r in ROWS),
        },
        "rows": [asdict(r) for r in ROWS],
        "missing_evidence": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    payload = _payload()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("neural-pde-stl-strel example status")
        print(_render_table())
        print()
        print(json.dumps(payload, indent=2))
    return 1 if payload["missing_evidence"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
