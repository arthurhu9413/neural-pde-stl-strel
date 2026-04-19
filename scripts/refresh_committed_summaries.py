#!/usr/bin/env python3
"""Refresh or verify summary artifacts derived from committed diffusion artifacts.

The repository keeps a small set of JSON/CSV summary files under ``results/`` so
that the README, docs, and plotting code can cite a stable source of truth.

By default, this script treats the committed dense-field sidecars as the stable
source of truth and refreshes only the text summaries:
- results/diffusion1d_main_summary.json
- results/diffusion1d_baseline_rtamt.json
- results/diffusion1d_stl_rtamt.json
- results/diffusion1d_ablation_summary.json
- results/diffusion1d_ablation_summary.csv
- results/heat2d_threshold_analysis.csv

That default matters. Re-evaluating a checkpoint to regenerate a dense field can
introduce tiny, platform-dependent floating-point drift, which is enough to make
tracked ``.pt``/``.npz`` sidecars differ byte-for-byte in CI even when the
underlying result is semantically unchanged.

Use ``--check`` for the read-only CI path. Use ``--prefer-checkpoints`` (and,
if desired, ``--rewrite-main-fields-from-checkpoints``) only when you
intentionally want to rebuild the main diffusion field sidecars from the raw
checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neural_pde_stl_strel.models.mlp import MLP  # noqa: E402

RESULTS = ROOT / "results"
LAMBDAS = (0, 2, 4, 6, 8, 10)
FLOAT_ATOL = 1e-6
FLOAT_RTOL = 1e-9


class ArtifactDriftError(RuntimeError):
    """Raised when a committed artifact does not match the expected payload."""


def _torch_load(path: Path) -> Any:
    """Load a PyTorch artifact compatibly across torch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not _is_bool(value)


def _semantically_equal(a: Any, b: Any, *, atol: float = FLOAT_ATOL, rtol: float = FLOAT_RTOL) -> bool:
    """Compare two nested payloads with a small tolerance on floats."""

    if _is_bool(a) or _is_bool(b):
        return a == b

    try:
        if pd.isna(a) and pd.isna(b):  # type: ignore[arg-type]
            return True
    except Exception:
        pass

    if _is_number(a) and _is_number(b):
        return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_semantically_equal(a[key], b[key], atol=atol, rtol=rtol) for key in a)

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_semantically_equal(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b))

    if a is None or b is None:
        return a is b

    if isinstance(a, str) and isinstance(b, str):
        return a == b

    return a == b


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_if_needed(path: Path, payload: Any, *, check: bool) -> str:
    if path.exists():
        existing = _read_json(path)
        if _semantically_equal(existing, payload):
            return "unchanged"
    if check:
        raise ArtifactDriftError(f"{path.relative_to(ROOT)} is out of date.")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return "updated"


def _write_csv_if_needed(path: Path, frame: pd.DataFrame, *, check: bool) -> str:
    if path.exists():
        existing = pd.read_csv(path)
        same_columns = list(existing.columns) == list(frame.columns)
        if same_columns and _semantically_equal(existing.to_dict(orient="records"), frame.to_dict(orient="records")):
            return "unchanged"
    if check:
        raise ArtifactDriftError(f"{path.relative_to(ROOT)} is out of date.")
    frame.to_csv(path, index=False)
    return "updated"


def _load_checkpoint_config(ckpt_path: Path) -> dict[str, Any]:
    payload = _torch_load(ckpt_path)
    return dict(payload["config"])


def load_ckpt_field(ckpt_path: Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = _torch_load(ckpt_path)
    cfg = dict(payload["config"])
    model = MLP(2, 1, hidden=cfg["hidden"], activation=cfg["activation"], out_activation=cfg.get("out_act"))
    model.load_state_dict(payload["model"])
    model.eval()

    x = torch.linspace(cfg["x_min"], cfg["x_max"], cfg["n_x"])
    t = torch.linspace(cfg["t_min"], cfg["t_max"], cfg["n_t"])
    X, T = torch.meshgrid(x, t, indexing="ij")
    inp = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

    with torch.no_grad():
        U = model(inp).reshape(cfg["n_x"], cfg["n_t"])
    return cfg, X, T, U


def load_field_sidecar(field_path: Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = _torch_load(field_path)
    cfg = dict(payload.get("config") or {})
    if "alpha" not in cfg and "alpha" in payload:
        cfg["alpha"] = float(payload["alpha"])
    return cfg, payload["X"].detach().cpu(), payload["T"].detach().cpu(), payload["u"].detach().cpu()


def load_diffusion_artifact(stem: str, *, prefer_checkpoints: bool) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    field_pt = RESULTS / f"{stem}_field.pt"
    field_npz = RESULTS / f"{stem}_field.npz"
    ckpt = RESULTS / f"{stem}.pt"

    if prefer_checkpoints and ckpt.exists():
        return load_ckpt_field(ckpt)

    if field_pt.exists():
        return load_field_sidecar(field_pt)

    if field_npz.exists():
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Found {field_npz.relative_to(ROOT)} but no matching checkpoint/config at {ckpt.relative_to(ROOT)}."
            )
        cfg = _load_checkpoint_config(ckpt)
        arr = np.load(field_npz, allow_pickle=False)
        return (
            cfg,
            torch.as_tensor(arr["X"]),
            torch.as_tensor(arr["T"]),
            torch.as_tensor(arr["u"]),
        )

    if ckpt.exists():
        return load_ckpt_field(ckpt)

    raise FileNotFoundError(f"Could not find a committed field or checkpoint for {stem!r}.")


def rewrite_main_field_sidecars_from_checkpoints() -> list[str]:
    rewritten: list[str] = []
    for tag in ("baseline", "stl"):
        stem = f"diffusion1d_{tag}"
        cfg, X, T, U = load_ckpt_field(RESULTS / f"{stem}.pt")
        torch.save(
            {
                "u": U.detach().cpu(),
                "X": X.detach().cpu(),
                "T": T.detach().cpu(),
                "u_max": 1.0,
                "alpha": float(cfg["alpha"]),
                "config": cfg,
            },
            RESULTS / f"{stem}_field.pt",
        )
        np.savez_compressed(
            RESULTS / f"{stem}_field.npz",
            u=U.detach().cpu().numpy(),
            X=X.detach().cpu().numpy(),
            T=T.detach().cpu().numpy(),
        )
        rewritten.extend(
            [
                f"results/{stem}_field.pt",
                f"results/{stem}_field.npz",
            ]
        )
    return rewritten


def diffusion_metrics(U: torch.Tensor, X: torch.Tensor, T: torch.Tensor, *, alpha: float) -> dict[str, Any]:
    u = U.detach().cpu().numpy()
    x = X.detach().cpu().numpy()
    t = T.detach().cpu().numpy()
    ref = np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)
    s = u.max(axis=0)
    time_axis = t[0]
    cool_mask = (time_axis >= 0.8 - 1e-12) & (time_axis <= 1.0 + 1e-12)
    center = u[u.shape[0] // 2]

    return {
        "pde_mse": float(np.mean((u - ref) ** 2)),
        "peak_value": float(u.max()),
        "final_spatial_max": float(s[-1]),
        "safety_rho": float(np.min(1.0 - s)),
        "safety_satisfied": bool(np.min(1.0 - s) >= 0.0),
        "cool_tight_rho": float(np.max(0.3 - s[cool_mask])),
        "cool_tight_satisfied": bool(np.max(0.3 - s[cool_mask]) >= 0.0),
        "cool_loose_rho": float(np.max(0.4 - s[cool_mask])),
        "cool_loose_satisfied": bool(np.max(0.4 - s[cool_mask]) >= 0.0),
        "centerline_cool_tight_rho": float(np.max(0.3 - center[cool_mask])),
        "grid_shape": [int(u.shape[0]), int(u.shape[1])],
        "dt": float(time_axis[1] - time_axis[0]),
        "alpha": float(alpha),
    }


def build_main_summary(*, prefer_checkpoints: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    main_summary: dict[str, Any] = {"experiment": "diffusion1d_main", "cases": {}}
    rtamt_payloads: dict[str, Any] = {}

    for tag in ("baseline", "stl"):
        cfg, X, T, U = load_diffusion_artifact(f"diffusion1d_{tag}", prefer_checkpoints=prefer_checkpoints)
        metrics = diffusion_metrics(U, X, T, alpha=float(cfg["alpha"]))
        metrics["checkpoint"] = f"results/diffusion1d_{tag}.pt"
        metrics["field_artifact"] = f"results/diffusion1d_{tag}_field.pt"
        metrics["epochs"] = int(cfg["epochs"])
        metrics["stl_weight"] = float(cfg["stl_weight"])
        main_summary["cases"][tag] = metrics

        rtamt_payloads[f"diffusion1d_{tag}_rtamt.json"] = {
            "ckpt": f"results/diffusion1d_{tag}_field.pt",
            "var": "u",
            "shape": metrics["grid_shape"],
            "time_axis": 1,
            "nt_full": metrics["grid_shape"][1],
            "nt_eval": metrics["grid_shape"][1],
            "dt": metrics["dt"],
            "semantics": "discrete",
            "agg": {"name": "amax", "p": 2.0, "q": 0.95, "temp": 0.1},
            "window": {"t0": None, "t1": None, "idx0": None, "idx1": None},
            "spec": {"kind": "upper", "formula": None, "u_min": None, "u_max": 1.0},
            "robustness": metrics["safety_rho"],
            "satisfied": metrics["safety_satisfied"],
            "backend": "exact_discrete_fallback",
        }

    return main_summary, rtamt_payloads


def build_ablation_summary(*, prefer_checkpoints: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    bench = pd.read_csv(RESULTS / "benchmark_training.csv")
    baseline_wall = float(
        bench.loc[
            (bench["experiment"] == "diffusion1d") & (bench["config"] == "baseline"),
            "wall_time_s",
        ].iloc[0]
    )
    default_wall = float(
        bench.loc[
            (bench["experiment"] == "diffusion1d") & (bench["config"] == "stl_default"),
            "wall_time_s",
        ].iloc[0]
    )
    slope = (default_wall - baseline_wall) / 5.0

    ablation_summary: dict[str, Any] = {
        "experiment": "diffusion1d_lambda_ablation",
        "derived_from": {
            "checkpoints": [f"results/diffusion1d_abl_w{lam}_s0.pt" for lam in LAMBDAS],
            "analytic_reference": "u(x,t) = sin(pi x) exp(-alpha pi^2 t)",
            "safety_spec": "G_[0,1](max_x u(x,t) <= 1.0)",
            "cooling_spec_tight": "F_[0.8,1](max_x u(x,t) <= 0.3)",
            "cooling_spec_loose": "F_[0.8,1](max_x u(x,t) <= 0.4)",
        },
        "notes": [
            "The ablation sweep is separate from the main baseline/STL comparison.",
            "Metrics are recomputed from committed dense-field artifacts when available, with checkpoint evaluation reserved for explicit refresh/fallback paths.",
            "Wall time is estimated from the two diffusion rows in results/benchmark_training.csv and then interpolated/extrapolated across lambda values.",
        ],
        "sweeps": [],
    }

    for lam in LAMBDAS:
        cfg, X, T, U = load_diffusion_artifact(f"diffusion1d_abl_w{lam}_s0", prefer_checkpoints=prefer_checkpoints)
        metrics = diffusion_metrics(U, X, T, alpha=float(cfg["alpha"]))
        ablation_summary["sweeps"].append(
            {
                "lambda": int(lam),
                "pde_mse": metrics["pde_mse"],
                "safety_rho": metrics["safety_rho"],
                "cooling_rho_tight": metrics["cool_tight_rho"],
                "cooling_rho_loose": metrics["cool_loose_rho"],
                "peak_value": metrics["peak_value"],
                "final_spatial_max": metrics["final_spatial_max"],
                "wall_time_s_estimate": float(baseline_wall + slope * lam),
                "source_checkpoint": f"results/diffusion1d_abl_w{lam}_s0.pt",
                "source_field": f"results/diffusion1d_abl_w{lam}_s0_field.pt",
            }
        )

    sweeps = ablation_summary["sweeps"]
    first_ok = next((i for i, row in enumerate(sweeps) if row["cooling_rho_tight"] >= 0), None)
    ablation_summary["conclusion"] = {
        "cooling_tight_first_satisfied_at_lambda": sweeps[first_ok]["lambda"] if first_ok is not None else None,
        "recommended_reference_lambda": 4,
        "rationale": "lambda=4 is the first setting with a comfortably positive cooling margin while remaining noticeably closer to the analytic diffusion solution than the larger penalties.",
    }

    frame = pd.DataFrame(
        [
            {
                "lambda": row["lambda"],
                "pde_mse": row["pde_mse"],
                "safety_rho": row["safety_rho"],
                "cooling_rho_tight": row["cooling_rho_tight"],
                "cooling_rho_loose": row["cooling_rho_loose"],
                "peak_value": row["peak_value"],
                "final_spatial_max": row["final_spatial_max"],
                "wall_time_s_estimate": row["wall_time_s_estimate"],
            }
            for row in sweeps
        ]
    )
    return ablation_summary, frame


def build_heat_threshold_frame() -> pd.DataFrame:
    heat = _read_json(RESULTS / "heat2d_strel_monitoring.json")
    return pd.DataFrame(
        [
            {
                "threshold_rule": name,
                "theta": item["theta"],
                "satisfied": item["satisfied"],
                "first_quench_time": item["first_quench_time"],
                "robustness_at_final": item["robustness_at_final"],
            }
            for name, item in heat["threshold_analysis"].items()
        ]
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Read-only validation mode for CI. Exit non-zero if a committed summary artifact is stale.",
    )
    parser.add_argument(
        "--prefer-checkpoints",
        action="store_true",
        help="Recompute diffusion summaries from checkpoints instead of committed field sidecars.",
    )
    parser.add_argument(
        "--rewrite-main-fields-from-checkpoints",
        action="store_true",
        help=(
            "Rewrite results/diffusion1d_{baseline,stl}_field.{pt,npz} from the raw checkpoints before refreshing "
            "the text summaries. This is intentionally opt-in because it can introduce tiny cross-platform drift."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    if args.check and args.rewrite_main_fields_from_checkpoints:
        raise SystemExit("--check cannot be combined with --rewrite-main-fields-from-checkpoints")

    rewritten_fields: list[str] = []
    if args.rewrite_main_fields_from_checkpoints:
        rewritten_fields = rewrite_main_field_sidecars_from_checkpoints()
        args.prefer_checkpoints = True

    main_summary, rtamt_payloads = build_main_summary(prefer_checkpoints=args.prefer_checkpoints)
    ablation_summary, ablation_frame = build_ablation_summary(prefer_checkpoints=args.prefer_checkpoints)
    heat_frame = build_heat_threshold_frame()

    actions: list[tuple[str, str]] = []
    actions.append(
        ("results/diffusion1d_main_summary.json", _write_json_if_needed(RESULTS / "diffusion1d_main_summary.json", main_summary, check=args.check))
    )
    for name, payload in rtamt_payloads.items():
        actions.append((f"results/{name}", _write_json_if_needed(RESULTS / name, payload, check=args.check)))
    actions.append(
        (
            "results/diffusion1d_ablation_summary.json",
            _write_json_if_needed(RESULTS / "diffusion1d_ablation_summary.json", ablation_summary, check=args.check),
        )
    )
    actions.append(
        (
            "results/diffusion1d_ablation_summary.csv",
            _write_csv_if_needed(RESULTS / "diffusion1d_ablation_summary.csv", ablation_frame, check=args.check),
        )
    )
    actions.append(
        (
            "results/heat2d_threshold_analysis.csv",
            _write_csv_if_needed(RESULTS / "heat2d_threshold_analysis.csv", heat_frame, check=args.check),
        )
    )

    updated = [path for path, status in actions if status == "updated"]
    unchanged = [path for path, status in actions if status == "unchanged"]

    if args.check:
        print(
            "Verified "
            f"{len(actions)} summary artifacts against committed field sidecars"
            f"{' / checkpoints' if args.prefer_checkpoints else ''}."
        )
        return 0

    print(
        "Committed summaries refreshed "
        f"({len(updated)} updated, {len(unchanged)} unchanged)."
    )
    if rewritten_fields:
        print("Rewrote main diffusion field sidecars from checkpoints:")
        for rel in rewritten_fields:
            print(f"  - {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
