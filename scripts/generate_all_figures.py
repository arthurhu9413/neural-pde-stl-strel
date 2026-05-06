#!/usr/bin/env python3
"""Generate repository figures from committed results.

This script serves two purposes:

1. Rebuild the figures committed under ``figs/`` for the README/report.
2. Refresh the lightweight compatibility images under ``assets/`` used by the
   top-level README.

Unlike the earlier version of this script, the plots below are tied directly to
committed results artifacts whenever possible. Conceptual diagrams remain
explicitly schematic and are labeled as such.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT
FIGS = OUTPUT_ROOT / "figs"
ASSETS = OUTPUT_ROOT / "assets"
RESULTS = ROOT / "results"
HEAT_FIELD = ROOT / "assets" / "heat2d_scalar" / "field_xy_t.npy"

EXPECTED_OUTPUTS: tuple[str, ...] = (
    "figs/architecture_diagram.png",
    "figs/training_pipeline.png",
    "figs/stl_semantics_overview.png",
    "figs/method_landscape.png",
    "figs/summary_results.png",
    "figs/comprehensive_results.png",
    "figs/diffusion1d_training_dynamics.png",
    "figs/diffusion1d_lambda_ablation.png",
    "figs/diffusion1d_comparison.png",
    "figs/heat2d_field_evolution.png",
    "figs/heat2d_stl_traces.png",
    "figs/pde_fields_overview.png",
    "figs/quality_dashboard.png",
    "figs/diffusion1d_ablations.png",
    "figs/benchmark_cost.png",
    "figs/diffusion1d_ablations_summary.csv",
    "assets/diffusion1d_training_loss.png",
    "assets/diffusion1d_training_robustness.png",
    "assets/diffusion1d_training_loss_components_stl.png",
    "assets/diffusion1d_robust_vs_lambda.png",
    "assets/diffusion1d_baseline_field.png",
    "assets/diffusion1d_stl_field.png",
)


def configure_output_root(output_root: Path | None) -> Path:
    """Set the directory used for generated ``figs/`` and ``assets/`` outputs."""

    global OUTPUT_ROOT, FIGS, ASSETS

    OUTPUT_ROOT = (output_root or ROOT).resolve()
    FIGS = OUTPUT_ROOT / "figs"
    ASSETS = OUTPUT_ROOT / "assets"
    FIGS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT


configure_output_root(None)

PALETTE = {
    "blue": "#2563eb",
    "red": "#dc2626",
    "green": "#16a34a",
    "orange": "#ea580c",
    "purple": "#7c3aed",
    "teal": "#0f766e",
    "slate": "#334155",
    "gray": "#64748b",
    "light": "#e2e8f0",
    "gold": "#ca8a04",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig: plt.Figure, rel_path: str | Path, dpi: int) -> None:
    path = OUTPUT_ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUTPUT_ROOT)}")


def _load_pt_field(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if torch is None:
        raise RuntimeError("PyTorch is not available to load .pt field artifacts.")
    payload = torch.load(path, map_location="cpu")
    u = payload["u"].detach().cpu().numpy()
    x = payload["X"].detach().cpu().numpy()
    t = payload["T"].detach().cpu().numpy()
    return u, x, t


def _load_diffusion_case(tag: str) -> dict:
    summary = json.loads((RESULTS / "diffusion1d_main_summary.json").read_text())["cases"][tag]
    npz_path = RESULTS / f"diffusion1d_{tag}_field.npz"
    if npz_path.exists():
        arr = np.load(npz_path)
        u, x, t = arr["u"], arr["X"], arr["T"]
    else:
        u, x, t = _load_pt_field(RESULTS / f"diffusion1d_{tag}_field.pt")
    return {"summary": summary, "u": u, "x": x, "t": t}


def _load_ablation_df() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "diffusion1d_ablation_summary.csv").sort_values("lambda")


def _load_training_csv(tag: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"diffusion1d_{tag}.csv")


def _load_heat_field() -> np.ndarray:
    return np.load(HEAT_FIELD)


def _load_heat_traces() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "heat2d_spatial_max.csv")


def _load_heat_thresholds() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "heat2d_threshold_analysis.csv")


def _analytic_field(x: np.ndarray, t: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)


# ---------------------------------------------------------------------------
# Schematic figures
# ---------------------------------------------------------------------------

def fig_architecture_diagram(dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    ax.set_title("System overview: PDE model + logic monitor + reproducibility artifacts", fontsize=13, fontweight="bold")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.12",
            facecolor=color,
            edgecolor=PALETTE["slate"],
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.31, body, ha="center", va="center", fontsize=8)

    def arrow(x1: float, y1: float, x2: float, y2: float, *, text: str | None = None) -> None:
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12, linewidth=1.5, color=PALETTE["gray"])
        ax.add_patch(arr)
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, fontsize=7.5, ha="center", color=PALETTE["gray"])

    box(0.3, 4.3, 2.0, 1.0, "Inputs", "PDE, IC/BC,\nnetwork, spec, λ", "#dbeafe")
    box(2.9, 4.3, 2.4, 1.0, "Neural surrogate", "PINN / neural field\nûθ(x,t[,y])", "#bfdbfe")
    box(5.9, 4.3, 2.3, 1.0, "Training-time regularizer", "soft STL robustness\nloss term", "#fecaca")
    box(8.8, 4.3, 2.1, 1.0, "Offline audit", "RTAMT or MoonLight\npost-hoc check", "#ddd6fe")

    box(0.6, 2.2, 2.5, 1.0, "Implemented case study A", "1D diffusion PINN\nSTL-regularized training", "#dcfce7")
    box(3.7, 2.2, 2.5, 1.0, "Implemented case study B", "2D heat rollout\nSTREL monitoring", "#fef3c7")
    box(6.8, 2.2, 2.6, 1.0, "Framework probes", "Neuromancer demo\nTorchPhysics helper\nPhysicsNeMo smoke test", "#e0f2fe")

    box(2.2, 0.5, 2.8, 1.0, "Outputs", "checkpoints, CSV logs,\nrobustness JSON, figures", "#e2e8f0")
    box(6.1, 0.5, 3.0, 1.0, "Paper/report layer", "formal specs, diagrams,\nreproducible commands", "#fee2e2")

    arrow(2.3, 4.8, 2.9, 4.8)
    arrow(5.3, 4.8, 5.9, 4.8)
    arrow(8.2, 4.8, 8.8, 4.8)
    arrow(4.1, 4.3, 1.8, 3.2, text="same monitoring core")
    arrow(7.0, 4.3, 4.9, 3.2, text="same robustness semantics")
    arrow(4.9, 2.2, 3.6, 1.5)
    arrow(8.1, 2.2, 7.6, 1.5)
    arrow(5.0, 1.0, 6.1, 1.0)

    ax.text(0.25, 0.1, "Conceptual block diagram (qualitative; not to scale).", fontsize=8, color=PALETTE["gray"])
    _save(fig, "figs/architecture_diagram.png", dpi)


def fig_training_pipeline(dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("Training data-flow for the 1D diffusion example", fontsize=12.5, fontweight="bold")

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10", facecolor=color, edgecolor=PALETTE["slate"], linewidth=1.1)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, fontweight="bold")

    def arrow(x1: float, y1: float, x2: float, y2: float, label: str | None = None) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color=PALETTE["gray"]))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, fontsize=7.2, ha="center", color=PALETTE["gray"])

    box(0.2, 2.9, 1.7, 0.8, "1. sample\ncollocation, BC,\nIC points", "#dbeafe")
    box(2.2, 2.9, 1.8, 0.8, "2. evaluate\nMLP field\nûθ(x,t)", "#bfdbfe")
    box(4.3, 2.9, 1.8, 0.8, "3. autograd PDE\nresidual + BC/IC\nlosses", "#fde68a")
    box(6.4, 2.9, 1.8, 0.8, "4. reduce over x\ns(t)=maxx û or\nmeanx û", "#e0e7ff")
    box(8.5, 2.9, 2.0, 0.8, "5. temporal logic\nrobustness ρ(φ)\n→ penalty", "#fecaca")
    box(3.7, 0.9, 3.8, 1.0, "6. total objective:  LPDE + LBC/IC + λ·LSTL", "#dcfce7")
    box(7.9, 0.9, 2.4, 1.0, "7. checkpoint +\npost-hoc audit", "#e2e8f0")

    arrow(1.9, 3.3, 2.2, 3.3)
    arrow(4.0, 3.3, 4.3, 3.3)
    arrow(6.1, 3.3, 6.4, 3.3)
    arrow(8.2, 3.3, 8.5, 3.3)
    arrow(5.2, 2.9, 5.5, 1.9, "physics loss")
    arrow(9.5, 2.9, 6.0, 1.9, "logic term")
    arrow(7.5, 1.4, 7.9, 1.4)

    ax.text(0.2, 0.15, "In the committed configs, λ corresponds to stl.weight in YAML.", fontsize=8, color=PALETTE["gray"])
    _save(fig, "figs/training_pipeline.png", dpi)


def fig_stl_semantics_overview(dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))
    fig.suptitle("Differentiable STL ingredients used in the training-time penalty", fontsize=12.5, fontweight="bold")

    # (a) penalty shapes
    ax = axes[0]
    delta = np.linspace(-1.0, 1.4, 300)
    beta = 10.0
    softplus = np.log1p(np.exp(beta * delta)) / beta
    hinge = np.maximum(delta, 0.0)
    sqhinge = np.maximum(delta, 0.0) ** 2
    ax.plot(delta, softplus, linewidth=2, label="softplus")
    ax.plot(delta, hinge, "--", linewidth=2, label="hinge")
    ax.plot(delta, sqhinge, ":", linewidth=2, label="squared hinge")
    ax.axvline(0.0, color=PALETTE["gray"], linestyle=":", linewidth=1)
    ax.set_xlabel("margin − ρ")
    ax.set_ylabel("penalty")
    ax.set_title("(a) Robustness-to-loss mapping")
    ax.legend(framealpha=0.9)

    # (b) temperature
    ax = axes[1]
    vals = np.array([0.7, -0.2, 0.1, 0.4, -0.4])
    taus = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
    softmins = [-tau * np.log(np.sum(np.exp(-vals / tau))) for tau in taus]
    ax.plot(taus, softmins, "o-", linewidth=2)
    ax.axhline(vals.min(), color=PALETTE["red"], linestyle="--", linewidth=1.5, label=f"hard min = {vals.min():.1f}")
    ax.set_xscale("log")
    ax.set_xlabel("temperature τ")
    ax.set_ylabel("softmin value")
    ax.set_title("(b) Smooth approximation")
    ax.legend(framealpha=0.9)

    # (c) always/eventually illustration
    ax = axes[2]
    t = np.linspace(0, 1, 100)
    s = 0.85 * np.exp(-3.0 * t) + 0.05 * np.sin(8 * np.pi * t)
    th = 0.45
    ax.plot(t, s, linewidth=2, label="s(t)")
    ax.axhline(th, color=PALETTE["red"], linestyle="--", linewidth=1.5, label="threshold")
    ax.fill_between(t, s, th, where=s <= th, alpha=0.20, label="positive margin")
    ax.fill_between(t, s, th, where=s > th, alpha=0.15, color=PALETTE["red"], label="negative margin")
    ax.set_xlabel("time")
    ax.set_ylabel("signal")
    ax.set_title("(c) Predicate margins over time")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "figs/stl_semantics_overview.png", dpi)


def fig_method_landscape(dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.set_title("Qualitative method landscape (schematic, not to scale)", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("training-time use of logic")
    ax.set_ylabel("spatial expressiveness")

    points = [
        ("RTAMT", 0.4, 0.1, PALETTE["blue"]),
        ("MoonLight / STREL", 0.5, 0.9, PALETTE["purple"]),
        ("STLnet", 1.6, 0.2, PALETTE["orange"]),
        ("STLCG++", 1.8, 0.25, PALETTE["orange"]),
        ("GradSTL", 1.9, 0.25, PALETTE["orange"]),
        ("Neuromancer", 1.2, 0.35, PALETTE["green"]),
        ("TorchPhysics", 1.2, 0.45, PALETTE["green"]),
        ("PhysicsNeMo", 1.35, 0.45, PALETTE["green"]),
        ("This repo", 1.75, 0.82, PALETTE["red"]),
    ]
    for label, x, y, c in points:
        ax.scatter([x], [y], s=90 if label == "This repo" else 70, color=c, edgecolors="white", linewidth=0.8, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=8)

    ax.set_xlim(0.0, 2.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([0.3, 1.2, 1.9])
    ax.set_xticklabels(["audit only", "framework hook", "direct regularization"])
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.set_yticklabels(["scalar trace", "field summary", "spatial / STREL"])
    ax.grid(True, alpha=0.22)

    ax.text(
        0.03,
        0.02,
        "The plot is a positioning aid: it summarizes how the tools are used in this repo,\nnot a benchmark of quality or correctness.",
        transform=ax.transAxes,
        fontsize=8,
        color=PALETTE["gray"],
    )
    _save(fig, "figs/method_landscape.png", dpi)


# ---------------------------------------------------------------------------
# Data-driven figures
# ---------------------------------------------------------------------------

def fig_summary_results(dpi: int) -> None:
    baseline = _load_diffusion_case("baseline")["summary"]
    stl = _load_diffusion_case("stl")["summary"]
    heat = _load_heat_thresholds()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    fig.suptitle("Committed result summary", fontsize=12.5, fontweight="bold")

    ax = axes[0]
    labels = ["safety\nG[0,1](max u ≤ 1.0)", "cooling\nF[0.8,1](max u ≤ 0.3)"]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, [baseline["safety_rho"], baseline["cool_tight_rho"]], width, label="baseline")
    ax.bar(x + width / 2, [stl["safety_rho"], stl["cool_tight_rho"]], width, label=f"STL-reg (λ={stl['stl_weight']:.0f})")
    ax.axhline(0.0, color=PALETTE["gray"], linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("robustness ρ")
    ax.set_title("(a) 1D diffusion audits")
    ax.legend(framealpha=0.9)

    ax = axes[1]
    colors = [PALETTE["red"] if not sat else PALETTE["green"] for sat in heat["satisfied"]]
    ax.bar(np.arange(len(heat)), heat["robustness_at_final"], color=colors)
    ax.axhline(0.0, color=PALETTE["gray"], linestyle=":", linewidth=1)
    ax.set_xticks(np.arange(len(heat)))
    ax.set_xticklabels(heat["threshold_rule"], rotation=22, ha="right")
    ax.set_ylabel("final-time margin")
    ax.set_title("(b) 2D heat threshold analysis")
    for i, row in heat.reset_index(drop=True).iterrows():
        txt = "sat" if bool(row["satisfied"]) else "unsat"
        ax.text(i, row["robustness_at_final"] + (0.015 if row["robustness_at_final"] >= 0 else -0.06), txt, ha="center", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "figs/summary_results.png", dpi)


def fig_comprehensive_results(dpi: int) -> None:
    df = _load_ablation_df()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    fig.suptitle("1D diffusion ablation analysis", fontsize=12.5, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df["lambda"], df["pde_mse"], "o-", linewidth=2, label="PDE MSE")
    ax.set_xlabel("λ")
    ax.set_ylabel("mean squared error")
    ax.set_title("(a) PDE fidelity")
    ax.axvline(4, color=PALETTE["gold"], linestyle="--", linewidth=1.4)

    ax = axes[0, 1]
    ax.plot(df["lambda"], df["safety_rho"], "s-", linewidth=2, label="safety ρ")
    ax.plot(df["lambda"], df["cooling_rho_tight"], "D-", linewidth=2, label="cooling ρ (0.3)")
    ax.axhline(0.0, color=PALETTE["gray"], linestyle=":", linewidth=1)
    ax.set_xlabel("λ")
    ax.set_ylabel("robustness ρ")
    ax.set_title("(b) specification margins")
    ax.legend(framealpha=0.9)

    ax = axes[1, 0]
    sc = ax.scatter(df["pde_mse"], df["safety_rho"], c=df["lambda"], cmap="viridis", s=85, edgecolors="white", linewidth=0.8)
    for _, row in df.iterrows():
        ax.annotate(f"λ={int(row['lambda'])}", (row["pde_mse"], row["safety_rho"]), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    plt.colorbar(sc, ax=ax, label="λ")
    ax.set_xlabel("PDE MSE")
    ax.set_ylabel("safety ρ")
    ax.set_title("(c) accuracy/robustness trade-off")

    ax = axes[1, 1]
    ax.bar(df["lambda"].astype(str), df["wall_time_s_estimate"])
    ax.set_xlabel("λ")
    ax.set_ylabel("estimated wall time (s)")
    ax.set_title("(d) cost trend in committed benchmark")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "figs/comprehensive_results.png", dpi)


def fig_diffusion1d_training_dynamics(dpi: int) -> None:
    base = _load_training_csv("baseline")
    stl = _load_training_csv("stl")
    stl_summary = _load_diffusion_case("stl")["summary"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    fig.suptitle("1D diffusion training dynamics", fontsize=12.5, fontweight="bold")

    ax = axes[0]
    ax.semilogy(base["epoch"], np.maximum(base["loss"], 1e-8), linewidth=1.6, label="baseline")
    ax.semilogy(stl["epoch"], np.maximum(stl["loss"], 1e-8), linewidth=1.6, label=f"STL-reg (λ={stl_summary['stl_weight']:.0f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss")
    ax.set_title("(a) Objective value")
    ax.legend(framealpha=0.9)

    ax = axes[1]
    ax.semilogy(stl["epoch"], np.maximum(stl["loss_pde"], 1e-8), linewidth=1.6, label="PDE")
    ax.semilogy(stl["epoch"], np.maximum(stl["loss_bcic"], 1e-8), linewidth=1.6, label="BC/IC")
    ax.semilogy(stl["epoch"], np.maximum(stl["loss_stl"], 1e-8), linewidth=1.6, label="STL")
    ax.set_xlabel("epoch")
    ax.set_ylabel("component loss")
    ax.set_title("(b) STL-run loss breakdown")
    ax.legend(framealpha=0.9)

    ax = axes[2]
    ax.plot(base["epoch"], base["robustness"], linewidth=1.4, label="baseline")
    ax.plot(stl["epoch"], stl["robustness"], linewidth=1.4, label=f"STL-reg (λ={stl_summary['stl_weight']:.0f})")
    ax.axhline(0.0, color=PALETTE["gray"], linestyle=":", linewidth=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training robustness")
    ax.set_title("(c) robustness signal")
    ax.legend(framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "figs/diffusion1d_training_dynamics.png", dpi)

    # Compatibility assets
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.4))
    ax2.semilogy(base["epoch"], np.maximum(base["loss"], 1e-8), linewidth=1.8, label="baseline")
    ax2.semilogy(stl["epoch"], np.maximum(stl["loss"], 1e-8), linewidth=1.8, label="STL-reg")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.legend(framealpha=0.9)
    ax2.set_title("Training loss")
    _save(fig2, "assets/diffusion1d_training_loss.png", dpi)

    fig3, ax3 = plt.subplots(figsize=(5.2, 3.4))
    ax3.plot(base["epoch"], base["robustness"], linewidth=1.8, label="baseline")
    ax3.plot(stl["epoch"], stl["robustness"], linewidth=1.8, label="STL-reg")
    ax3.axhline(0.0, color=PALETTE["gray"], linestyle=":", linewidth=1)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("robustness")
    ax3.legend(framealpha=0.9)
    ax3.set_title("Training robustness")
    _save(fig3, "assets/diffusion1d_training_robustness.png", dpi)

    fig4, ax4 = plt.subplots(figsize=(5.2, 3.4))
    ax4.semilogy(stl["epoch"], np.maximum(stl["loss_pde"], 1e-8), linewidth=1.8, label="PDE")
    ax4.semilogy(stl["epoch"], np.maximum(stl["loss_bcic"], 1e-8), linewidth=1.8, label="BC/IC")
    ax4.semilogy(stl["epoch"], np.maximum(stl["loss_stl"], 1e-8), linewidth=1.8, label="STL")
    ax4.set_xlabel("epoch")
    ax4.set_ylabel("component loss")
    ax4.legend(framealpha=0.9)
    ax4.set_title("STL loss components")
    _save(fig4, "assets/diffusion1d_training_loss_components_stl.png", dpi)


def fig_diffusion1d_lambda_ablation(dpi: int) -> None:
    df = _load_ablation_df()
    fig, ax1 = plt.subplots(figsize=(7.3, 4.6))
    ax1.set_title("λ sweep on the committed ablation checkpoints", fontsize=12.5, fontweight="bold")
    ax1.plot(df["lambda"], df["pde_mse"], "o-", linewidth=2, label="PDE MSE")
    ax1.set_xlabel("λ")
    ax1.set_ylabel("PDE MSE")
    ax1.axvline(4, color=PALETTE["gold"], linestyle="--", linewidth=1.4)
    ax1.annotate("reference λ=4", xy=(4, float(df.loc[df["lambda"] == 4, "pde_mse"].iloc[0])), xytext=(5.1, float(df["pde_mse"].max()) * 0.85), arrowprops=dict(arrowstyle="->"), fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(df["lambda"], df["safety_rho"], "s--", linewidth=2, label="safety ρ")
    ax2.plot(df["lambda"], df["cooling_rho_tight"], "D-.", linewidth=2, label="cooling ρ")
    ax2.axhline(0.0, color=PALETTE["gray"], linestyle=":", linewidth=1)
    ax2.set_ylabel("robustness ρ")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, framealpha=0.9, loc="lower right")
    fig.tight_layout()
    _save(fig, "figs/diffusion1d_lambda_ablation.png", dpi)

    fig2, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(df["lambda"], df["safety_rho"], "o-", linewidth=2, label="safety ρ")
    ax.plot(df["lambda"], df["cooling_rho_tight"], "s-", linewidth=2, label="cooling ρ")
    ax.axhline(0.0, color=PALETTE["gray"], linestyle=":", linewidth=1)
    ax.set_xlabel("λ")
    ax.set_ylabel("robustness")
    ax.legend(framealpha=0.9)
    ax.set_title("Robustness vs λ")
    _save(fig2, "assets/diffusion1d_robust_vs_lambda.png", dpi)


def fig_diffusion1d_comparison(dpi: int) -> None:
    baseline = _load_diffusion_case("baseline")
    stl = _load_diffusion_case("stl")
    x = baseline["x"]
    t = baseline["t"]
    analytic = _analytic_field(x, t, alpha=float(baseline["summary"]["alpha"]))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), constrained_layout=True)
    fig.suptitle("1D diffusion fields on the committed checkpoints", fontsize=12.5, fontweight="bold")
    fields = [
        (analytic, "analytic reference"),
        (baseline["u"], "baseline checkpoint"),
        (stl["u"], "STL-regularized checkpoint"),
    ]
    vmin = min(np.min(arr) for arr, _ in fields)
    vmax = max(np.max(arr) for arr, _ in fields)
    for ax, (u, title) in zip(axes, fields):
        im = ax.pcolormesh(t, x, u, shading="auto", cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.86, label="u(x,t)")
    _save(fig, "figs/diffusion1d_comparison.png", dpi)

    # Compatibility single-field assets
    for tag, data in [("baseline", baseline), ("stl", stl)]:
        fig2, ax2 = plt.subplots(figsize=(4.6, 3.6))
        im2 = ax2.pcolormesh(data["t"], data["x"], data["u"], shading="auto", cmap="RdYlBu_r")
        ax2.set_xlabel("t")
        ax2.set_ylabel("x")
        ax2.set_title(f"{tag} field")
        fig2.colorbar(im2, ax=ax2, shrink=0.84)
        _save(fig2, f"assets/diffusion1d_{tag}_field.png", dpi)


def fig_heat2d_field_evolution(dpi: int) -> None:
    field = _load_heat_field()
    nt = field.shape[-1]
    dt = 0.05
    times = [0, 10, 20, 30, 40, nt - 1]

    fig, axes = plt.subplots(1, len(times), figsize=(13.8, 2.9), constrained_layout=True)
    fig.suptitle("2D heat rollout committed with the repository", fontsize=12.5, fontweight="bold")
    vmin, vmax = float(field.min()), float(field.max())
    for ax, idx in zip(axes, times):
        im = ax.imshow(field[:, :, idx].T, origin="lower", cmap="hot", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"t = {idx * dt:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.86, label="u(x,y,t)")
    _save(fig, "figs/heat2d_field_evolution.png", dpi)


def fig_heat2d_stl_traces(dpi: int) -> None:
    traces = _load_heat_traces()
    th = _load_heat_thresholds()

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    fig.suptitle("2D heat monitoring views", fontsize=12.5, fontweight="bold")

    ax = axes[0]
    ax.plot(traces["time"], traces["spatial_max"], linewidth=2, label="max")
    ax.plot(traces["time"], traces["spatial_mean"], linewidth=2, label="mean")
    ax.plot(traces["time"], traces["spatial_min"], linewidth=2, label="min")
    for _, row in th.iterrows():
        ax.axhline(row["theta"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("aggregated temperature")
    ax.set_title("(a) spatially reduced traces")
    ax.legend(framealpha=0.9)

    ax = axes[1]
    ypos = np.arange(len(th))
    colors = [PALETTE["green"] if sat else PALETTE["red"] for sat in th["satisfied"]]
    ax.barh(ypos, th["theta"], color=colors, alpha=0.85)
    ax.set_yticks(ypos)
    ax.set_yticklabels(th["threshold_rule"])
    ax.set_xlabel("θ threshold")
    ax.set_title("(b) thresholds used in STREL audit")
    for i, row in th.reset_index(drop=True).iterrows():
        q = "sat" if bool(row["satisfied"]) else "unsat"
        q2 = row["first_quench_time"]
        extra = f", tq={q2:.2f}" if pd.notna(q2) else ""
        ax.text(float(row["theta"]) + 0.01, i, q + extra, va="center", fontsize=7.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "figs/heat2d_stl_traces.png", dpi)


def fig_pde_fields_overview(dpi: int) -> None:
    baseline = _load_diffusion_case("baseline")
    stl = _load_diffusion_case("stl")
    heat = _load_heat_field()

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    fig.suptitle("Field-level visual overview", fontsize=12.5, fontweight="bold")

    im = axes[0, 0].pcolormesh(baseline["t"], baseline["x"], baseline["u"], shading="auto", cmap="RdYlBu_r")
    axes[0, 0].set_title("1D diffusion baseline")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("x")

    im = axes[0, 1].pcolormesh(stl["t"], stl["x"], stl["u"], shading="auto", cmap="RdYlBu_r")
    axes[0, 1].set_title("1D diffusion STL-reg")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("x")

    axes[1, 0].imshow(heat[:, :, 0].T, origin="lower", cmap="hot")
    axes[1, 0].set_title("2D heat at t=0")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    axes[1, 1].imshow(heat[:, :, -1].T, origin="lower", cmap="hot")
    axes[1, 1].set_title("2D heat at final time")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "figs/pde_fields_overview.png", dpi)


def fig_quality_dashboard(dpi: int) -> None:
    py_files = list((ROOT / "src").rglob("*.py"))
    script_files = list((ROOT / "scripts").glob("*.py"))
    test_files = list((ROOT / "tests").glob("test_*.py"))
    test_functions = 0
    for path in test_files:
        text = path.read_text(encoding="utf-8")
        test_functions += len(re.findall(r"^def test_", text, flags=re.M))

    result_files = [p for p in RESULTS.glob("*") if p.is_file()]
    fig_files = [p for p in FIGS.glob("*.png")]

    pytest_summary_path = RESULTS / "pytest_summary.json"
    pytest_pass = None
    if pytest_summary_path.exists():
        try:
            summary = json.loads(pytest_summary_path.read_text())
            pytest_pass = summary.get("pytest", {}).get("passed", summary.get("passed"))
        except Exception:
            pytest_pass = None

    metrics = [
        ("library .py files", len(py_files)),
        ("scripts", len(script_files)),
        ("test files", len(test_files)),
        ("test functions", test_functions),
        ("result artifacts", len(result_files)),
        ("figure files", len(fig_files)),
        ("optional frameworks", 3),
        ("logic backends", 2),
    ]
    if pytest_pass is not None:
        metrics.insert(4, ("pytest passed", int(pytest_pass)))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.set_title("Repository health snapshot", fontsize=12.5, fontweight="bold")
    names = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    bars = ax.barh(np.arange(len(metrics)), values)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, str(val), va="center", fontsize=8)
    ax.text(
        0.01,
        -0.13,
        "Most counts are derived from the working tree; passed-test counts come from the committed summary sidecar.",
        transform=ax.transAxes,
        fontsize=8,
        color=PALETTE["gray"],
    )
    fig.tight_layout()
    _save(fig, "figs/quality_dashboard.png", dpi)


def fig_diffusion1d_ablations(dpi: int) -> None:
    df = _load_ablation_df()
    raw = pd.read_csv(RESULTS / "diffusion1d_ablations.csv", header=None, names=["lambda", "reported_robustness"])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    fig.suptitle("Ablation views tied to committed artifacts", fontsize=12.5, fontweight="bold")

    ax = axes[0]
    ax.plot(df["lambda"], df["safety_rho"], "o-", linewidth=2, label="recomputed safety ρ")
    ax.plot(df["lambda"], df["cooling_rho_tight"], "s-", linewidth=2, label="recomputed cooling ρ")
    ax.axhline(0.0, color=PALETTE["gray"], linestyle=":", linewidth=1)
    ax.set_xlabel("λ")
    ax.set_ylabel("robustness ρ")
    ax.set_title("(a) recomputed from checkpoint fields")
    ax.legend(framealpha=0.9)

    ax = axes[1]
    ax.plot(raw["lambda"], raw["reported_robustness"], "o-", linewidth=2)
    ax.set_xlabel("λ")
    ax.set_ylabel("reported robustness")
    ax.set_title("(b) original sweep CSV")
    ax.text(0.03, 0.05, "This panel is shown as a provenance trace.\nThe left panel is what the README/report should cite.", transform=ax.transAxes, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "figs/diffusion1d_ablations.png", dpi)
    df.to_csv(FIGS / "diffusion1d_ablations_summary.csv", index=False)


def fig_benchmark_cost(dpi: int) -> None:
    bench = pd.read_csv(RESULTS / "benchmark_training.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_title("Committed benchmark rows", fontsize=12.5, fontweight="bold")
    labels = [f"{r.experiment}\n{r.config}" for r in bench.itertuples()]
    ax.bar(np.arange(len(bench)), bench["wall_time_s"])
    ax.set_xticks(np.arange(len(bench)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("wall time (s)")
    fig.tight_layout()
    _save(fig, "figs/benchmark_cost.png", dpi)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate repository figures from committed results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional directory to write figs/ and assets/ into instead of the repository root.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest describing the generated artifacts.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if any expected artifact is missing or empty after generation.",
    )
    return parser



def build_manifest(*, dpi: int) -> dict[str, object]:
    outputs: list[dict[str, object]] = []
    missing: list[str] = []
    for rel in EXPECTED_OUTPUTS:
        path = OUTPUT_ROOT / rel
        size = path.stat().st_size if path.exists() else 0
        record = {"path": rel, "exists": path.exists(), "bytes": size}
        outputs.append(record)
        if size <= 0:
            missing.append(rel)
    return {
        "source_root": str(ROOT),
        "output_root": str(OUTPUT_ROOT),
        "dpi": dpi,
        "expected_count": len(EXPECTED_OUTPUTS),
        "generated_count": sum(1 for item in outputs if item["exists"]),
        "outputs": outputs,
        "missing": missing,
        "ok": not missing,
    }



def write_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path



def main() -> int:
    args = build_argparser().parse_args()
    configure_output_root(args.output_root)

    generators = [
        fig_architecture_diagram,
        fig_training_pipeline,
        fig_stl_semantics_overview,
        fig_method_landscape,
        fig_summary_results,
        fig_comprehensive_results,
        fig_diffusion1d_training_dynamics,
        fig_diffusion1d_lambda_ablation,
        fig_diffusion1d_comparison,
        fig_heat2d_field_evolution,
        fig_heat2d_stl_traces,
        fig_pde_fields_overview,
        fig_quality_dashboard,
        fig_diffusion1d_ablations,
        fig_benchmark_cost,
    ]
    print(f"Generating figures into {OUTPUT_ROOT}...")
    for fn in generators:
        fn(args.dpi)

    manifest = build_manifest(dpi=args.dpi)
    if args.manifest is not None:
        manifest_path = write_manifest(args.manifest, manifest)
        print(f"  wrote {manifest_path}")

    if args.check:
        if manifest["missing"]:
            print("Missing or empty figure artifacts:")
            for rel in manifest["missing"]:
                print(f"  - {rel}")
            return 1
        print(f"Validated {manifest['generated_count']} generated artifacts.")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
