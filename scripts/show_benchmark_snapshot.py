#!/usr/bin/env python3
"""Print the committed runtime / memory snapshot used by the repository.

This script is intentionally lightweight and *does not* rerun training jobs.
It simply surfaces ``results/benchmark_training.csv`` in a human-readable form
so readers can inspect the committed cost table that backs the README and
figure artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "results" / "benchmark_training.csv"


def load_rows() -> list[dict[str, Any]]:
    if not BENCH.exists():
        raise FileNotFoundError(f"Missing benchmark snapshot: {BENCH}")
    with BENCH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Benchmark snapshot is empty: {BENCH}")
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    headers = ["experiment", "config", "epochs", "wall_time_s", "peak_memory_mb", "device"]
    widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    header = "  ".join(f"{h:<{widths[h]}}" for h in headers)
    rule = "  ".join("-" * widths[h] for h in headers)
    body = ["  ".join(f"{str(r.get(h, '')):<{widths[h]}}" for h in headers) for r in rows]
    return "\n".join([header, rule, *body])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Show the committed benchmark_training.csv snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a text table.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    rows = load_rows()
    payload = {
        "source": str(BENCH.relative_to(ROOT)),
        "note": (
            "This is the committed CPU-first runtime/memory snapshot used by the repo. "
            "It is a reproducibility artifact, not a universal performance claim."
        ),
        "rows": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("Committed benchmark snapshot")
    print(payload["note"])
    print()
    print(format_table(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
