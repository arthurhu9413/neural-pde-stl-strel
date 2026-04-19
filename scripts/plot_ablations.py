#!/usr/bin/env python3
"""plot_ablations.py

Plot ablation sweeps from one or more CSV files.

This script is intentionally dependency-light (no pandas / numpy required) and is
used in this repo to visualize how an ablation parameter (often the STL penalty
weight λ) affects a scalar metric (often STL robustness ρ).

Typical inputs

1) A *summary* CSV with a header row, e.g.:

    lambda,robustness
    0,0.14
    2,0.21
    4,0.25

2) A header-less 2-column CSV, e.g.:

    0,0.14
    2,0.21
    4,0.25

3) Multiple CSVs (one per method / configuration). In that case we plot each as
   a separate series on the same axes.

Aggregation

If the same x-value appears multiple times in a file, we aggregate those repeats
and report mean ± (std/sem/95%-CI) error bars.

Notes
-----

* If all error bars are exactly zero (common when there is only one repeat per
  x value), we omit error bars and draw a simple line plot.
* When the plotted metric looks like STL robustness (column name contains
  "robust" or "rho"), we draw a horizontal y=0 reference line (the satisfaction
  boundary under standard robustness semantics).

"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

# Headless / CI-safe.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


ErrMode = Literal["none", "std", "sem", "ci95"]
BestMode = Literal["max", "min"]
AutoScale = Literal["none", "auto", "logx"]
LegendMode = Literal["auto", "none", "inside", "outside"]


# Common column names we want to recognize.
_X_CANDIDATES = (
    "lambda",
    "lam",
    "stl_weight",
    "lambda_stl",
    "weight",
    "x",
)
_Y_CANDIDATES = (
    "robustness",
    "rho",
    "rho_soft",
    "rho_hard",
    "loss",
    "y",
)


# Two-tailed 95% t critical values for df=1..30 (t_{0.975, df}).
# Rounded to 6 decimals; for df>30 we fall back to ~1.96 (normal approx).
_T_CRIT_975_DF_1_30 = (
    12.706205,
    4.302653,
    3.182446,
    2.776445,
    2.570582,
    2.446912,
    2.364624,
    2.306004,
    2.262157,
    2.228139,
    2.200985,
    2.178813,
    2.160369,
    2.144787,
    2.13145,
    2.119905,
    2.109816,
    2.100922,
    2.093024,
    2.085963,
    2.079614,
    2.073873,
    2.068658,
    2.063899,
    2.059539,
    2.055529,
    2.051831,
    2.048407,
    2.04523,
    2.042272,
)


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    x_name: str
    y_name: str
    x: list[float]
    y_mean: list[float]
    y_err: list[float]
    n: list[int]


def _to_float(cell: str) -> float | None:
    """Parse a float cell; return None if it is empty or non-numeric."""

    s = cell.strip()
    if not s:
        return None
    try:
        # Support common numeric spellings.
        s = s.replace("∞", "inf").replace("-∞", "-inf")
        return float(s)
    except ValueError:
        return None


def _sniff_csv_dialect(sample: str) -> csv.Dialect:
    """Best-effort CSV dialect detection.

    The stdlib csv.Sniffer can be noisy for purely numeric files; we keep it but
    fall back to sane defaults if it fails.
    """

    try:
        # Prioritize the common delimiters we use in this repo.
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        return csv.get_dialect("excel")


def _has_header(sample: str, dialect: csv.Dialect) -> bool:
    """Heuristic header detection.

    csv.Sniffer.has_header() is only a heuristic and can be wrong for numeric
    files. We combine it with a simple: "can the first row parse as floats?"
    check.
    """

    # csv.Sniffer.has_header may throw on some inputs.
    try:
        sniffer_says_header = bool(csv.Sniffer().has_header(sample))
    except csv.Error:
        sniffer_says_header = False

    # Read the first non-empty non-comment row and see if most entries are
    # numeric.
    reader = csv.reader(sample.splitlines(), dialect)
    first_row: list[str] | None = None
    for row in reader:
        if not row:
            continue
        joined = "".join(row).strip()
        if not joined:
            continue
        if joined.lstrip().startswith("#"):
            continue
        first_row = row
        break

    if first_row is None:
        return False

    numeric = sum(1 for c in first_row if _to_float(c) is not None)
    # If nearly everything parses as numeric, treat it as data.
    first_row_is_numeric = numeric >= max(1, len(first_row) - 1)

    return sniffer_says_header and not first_row_is_numeric


def _normalize_col_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _infer_col_indices(
    header: list[str] | None,
    xcol: str | None,
    ycol: str | None,
    max_cols: int,
) -> tuple[int, int, str, str]:
    """Pick x/y column indices and return (x_idx, y_idx, x_name, y_name)."""

    if header:
        normalized = [_normalize_col_name(h) for h in header]

        def find_idx(user: str | None, candidates: tuple[str, ...], fallback: int) -> int:
            if user is None:
                for cand in candidates:
                    if cand in normalized:
                        return normalized.index(cand)
                return fallback
            key = _normalize_col_name(user)
            if key.isdigit():
                i = int(key)
                if 0 <= i < max_cols:
                    return i
                raise ValueError(f"Column index out of range: {user}")
            if key in normalized:
                return normalized.index(key)
            raise ValueError(f"Column '{user}' not found. Available: {header}")

        xi = find_idx(xcol, _X_CANDIDATES, 0)
        yi = find_idx(ycol, _Y_CANDIDATES, 1 if max_cols > 1 else 0)
        return xi, yi, header[xi], header[yi]

    # No header: accept numeric indices (as strings) or default to first two cols.
    def parse_idx(user: str | None, default: int) -> int:
        if user is None:
            return default
        if user.strip().isdigit():
            i = int(user.strip())
            if 0 <= i < max_cols:
                return i
            raise ValueError(f"Column index out of range: {user}")
        raise ValueError(
            "CSV has no header; please pass xcol/ycol as integer indices (e.g. --xcol 0 --ycol 1)."
        )

    xi = parse_idx(xcol, 0)
    yi = parse_idx(ycol, 1 if max_cols > 1 else 0)
    return xi, yi, "x", "y"


def _t_crit_95(df: int) -> float:
    """Return t_{0.975, df} for a two-tailed 95% confidence interval."""

    if df <= 0:
        return float("nan")
    if df <= 30:
        return _T_CRIT_975_DF_1_30[df - 1]
    # Normal approximation.
    return 1.959964


def _aggregate_xy(
    rows: list[tuple[float, float]],
    err_mode: ErrMode,
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Group by x and compute mean and error bars."""

    by_x: dict[float, list[float]] = {}
    for x, y in rows:
        by_x.setdefault(x, []).append(y)

    xs = sorted(by_x.keys())
    y_mean: list[float] = []
    y_err: list[float] = []
    ns: list[int] = []

    for x in xs:
        ys = by_x[x]
        n = len(ys)
        ns.append(n)
        mu = statistics.fmean(ys)
        y_mean.append(mu)

        if err_mode == "none" or n <= 1:
            y_err.append(0.0)
            continue

        # Sample standard deviation (ddof=1) is appropriate when repeats are
        # treated as samples.
        try:
            s = statistics.stdev(ys)
        except statistics.StatisticsError:
            y_err.append(0.0)
            continue

        if err_mode == "std":
            y_err.append(s)
        else:
            sem = s / math.sqrt(n)
            if err_mode == "sem":
                y_err.append(sem)
            elif err_mode == "ci95":
                y_err.append(_t_crit_95(n - 1) * sem)
            else:
                raise ValueError(f"Unknown err_mode: {err_mode}")

    return xs, y_mean, y_err, ns


def _humanize_label(raw: str) -> str:
    """Turn a filename-ish label into something presentation-friendly."""

    s = raw.strip()
    s = s.replace("_", " ").replace("-", " ")
    # Insert spaces between letters and digits: diffusion1d -> diffusion 1d.
    s = re.sub(r"(?<=\D)(?=\d)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    tokens = s.split(" ") if s else []

    special = {
        "stl": "STL",
        "strel": "STREL",
        "pde": "PDE",
        "odes": "ODEs",
        "ode": "ODE",
        "pinn": "PINN",
        "nn": "NN",
        "ml": "ML",
        "cpu": "CPU",
        "gpu": "GPU",
        "rtamt": "RTAMT",
    }

    out: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in special:
            out.append(special[low])
            continue
        if re.fullmatch(r"\d+d", low):
            out.append(tok[:-1] + "D")
            continue
        if tok.isupper():
            out.append(tok)
            continue
        out.append(tok.capitalize())

    return " ".join(out) if out else raw


def _default_series_label(path: Path) -> str:
    stem = path.stem
    tokens = re.split(r"[\s_-]+", stem)

    # Strip generic ablation/sweep tokens at the end.
    strip_tail = {"abl", "ablation", "ablations", "sweep", "sweeps", "results"}
    while tokens and tokens[-1].lower() in strip_tail:
        tokens.pop()

    # Strip generic prefixes at the front.
    while tokens and tokens[0].lower() in {"results", "figs", "fig"}:
        tokens.pop(0)

    if not tokens:
        tokens = [stem]

    return _humanize_label(" ".join(tokens))


def _guess_names_for_headerless(path: Path, x_name: str, y_name: str) -> tuple[str, str]:
    """Heuristic naming when the CSV has no header.

    In this repo, a common pattern is a 2-column ablation file like
    diffusion1d_ablations.csv, which is (λ, robustness). We use filename
    heuristics to name axes sensibly without requiring an explicit header.
    """

    if x_name != "x" or y_name != "y":
        return x_name, y_name

    stem = path.stem.lower()

    if any(k in stem for k in ("lambda", "stl_weight", "weight", "abl", "ablation")):
        x_name = "lambda"

    if "loss" in stem:
        y_name = "loss"
    elif any(k in stem for k in ("robust", "rho")):
        y_name = "robustness"
    elif any(k in stem for k in ("abl", "ablation")):
        # Repo default for ablations: plot robustness vs λ.
        y_name = "robustness"

    return x_name, y_name


def _pretty_axis_label(name: str) -> str:
    """Map internal column names to presentation labels."""

    key = _normalize_col_name(name)

    # x-axis (ablation parameter)
    if key in {"lambda", "lam", "stl_weight", "lambda_stl", "weight"}:
        return "STL weight λ"

    # y-axis (metric)
    if key in {"robustness", "rho", "rho_soft", "rho_hard"}:
        return "robustness"

    # Reasonable defaults for common losses.
    if key in {"loss", "l", "l_total", "total_loss"}:
        return "loss"

    return name.strip() if name.strip() else name


def _looks_like_robustness(metric_name: str, ylabel: str) -> bool:
    """Return True if we should draw a y=0 satisfaction boundary line."""

    name = _normalize_col_name(metric_name)
    lab = _normalize_col_name(ylabel)
    return any(k in name for k in ("robust", "rho")) or any(k in lab for k in ("robust", "rho"))


def _path_list(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        # Expand ~ and environment variables first.
        expanded = os.path.expandvars(str(Path(pattern).expanduser()))

        # pathlib.Path.glob() does not support absolute patterns; glob.glob() does.
        if any(ch in expanded for ch in "*?["):
            matches = sorted(glob.glob(expanded, recursive=True))
            if matches:
                out.extend(Path(m) for m in matches)
            else:
                # Keep the pattern so we can fail with a clear FileNotFoundError later.
                out.append(Path(expanded))
        else:
            out.append(Path(expanded))
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    return uniq


def _load_series(path: Path, label: str | None, xcol: str | None, ycol: str | None, err: ErrMode) -> SeriesSpec:
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8", errors="replace")
    sample = "\n".join(text.splitlines()[:50])
    dialect = _sniff_csv_dialect(sample)
    header_present = _has_header(sample, dialect)

    reader = csv.reader(text.splitlines(), dialect)
    header: list[str] | None = None

    rows_raw: list[list[str]] = []
    for row in reader:
        if not row:
            continue
        joined = "".join(row).strip()
        if not joined:
            continue
        if joined.lstrip().startswith("#"):
            continue

        if header is None and header_present:
            header = row
            continue

        rows_raw.append(row)

    if not rows_raw:
        raise ValueError(f"No data rows found in {path}")

    max_cols = max(len(r) for r in rows_raw) if rows_raw else 0
    xi, yi, x_name, y_name = _infer_col_indices(header, xcol, ycol, max_cols)

    if header is None:
        x_name, y_name = _guess_names_for_headerless(path, x_name, y_name)

    rows_num: list[tuple[float, float]] = []
    for r in rows_raw:
        if xi >= len(r) or yi >= len(r):
            continue
        x = _to_float(r[xi])
        y = _to_float(r[yi])
        if x is None or y is None:
            continue
        rows_num.append((float(x), float(y)))

    if not rows_num:
        raise ValueError(
            f"No numeric x/y pairs found in {path} using xcol={xcol or xi}, ycol={ycol or yi}"
        )

    xs, y_mean, y_err, ns = _aggregate_xy(rows_num, err)

    lbl = label if label is not None else _default_series_label(path)
    return SeriesSpec(label=lbl, x_name=x_name, y_name=y_name, x=xs, y_mean=y_mean, y_err=y_err, n=ns)


def _maybe_log_x(ax: plt.Axes, xs: list[float], autoscale: AutoScale) -> None:
    if autoscale == "none":
        return

    # Log-scale requires all x > 0.
    if any(x <= 0 for x in xs):
        return

    if autoscale == "logx":
        ax.set_xscale("log")
        return

    # autoscale == "auto": log if values span 2+ orders of magnitude.
    lo = min(xs)
    hi = max(xs)
    if lo <= 0:
        return
    if hi / lo >= 100:
        ax.set_xscale("log")


def _maybe_set_integer_xticks(ax: plt.Axes, xs: list[float], *, max_ticks: int = 20) -> None:
    if not xs:
        return

    # If all x are near-integers and the unique set is small, set integer tick
    # labels. This is helpful for ablations like λ ∈ {0,2,4,...} but would be
    # disastrous for dense integer x like epoch=0..N.
    if not all(abs(x - round(x)) < 1e-9 for x in xs):
        return

    ticks = sorted({int(round(x)) for x in xs})
    if len(ticks) <= max_ticks:
        ax.set_xticks(ticks)


def _default_title(series: list[SeriesSpec], xlabel: str, ylabel: str) -> str:
    if not series:
        return ""
    if len(series) == 1:
        return f"{series[0].label} - {ylabel} vs {xlabel}"
    return f"{ylabel} vs {xlabel}"


def _write_summary_csv(path: Path, series: list[SeriesSpec], best_mode: BestMode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["series", "metric", "best_x", "best_y_mean", "y_err", "n"])

        for s in series:
            if not s.x:
                continue
            if best_mode == "max":
                best_idx = max(range(len(s.y_mean)), key=lambda i: s.y_mean[i])
            else:
                best_idx = min(range(len(s.y_mean)), key=lambda i: s.y_mean[i])
            w.writerow(
                [
                    s.label,
                    s.y_name,
                    s.x[best_idx],
                    s.y_mean[best_idx],
                    s.y_err[best_idx],
                    s.n[best_idx],
                ]
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Plot ablation sweeps (mean ± error) from CSV file(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "inputs",
        nargs="+",
        help="One or more CSV paths (or glob patterns). Each is plotted as a separate series.",
    )

    p.add_argument(
        "-o",
        "--out",
        default="ablations",
        help="Output base path (without extension).",
    )

    p.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output format(s) supported by Matplotlib (e.g. png pdf svg).",
    )

    p.add_argument("--title", default=None, help="Figure title (auto if omitted).")
    p.add_argument("--xlabel", default=None, help="X-axis label (auto if omitted).")
    p.add_argument("--ylabel", default=None, help="Y-axis label (auto if omitted).")

    p.add_argument(
        "--label",
        nargs="*",
        default=None,
        help="Legend label(s) for each input file (order matters).",
    )

    p.add_argument(
        "--xcol",
        default=None,
        help="X column name (if CSV has header) or index (if no header).",
    )
    p.add_argument(
        "--ycol",
        default=None,
        help="Y column name (if CSV has header) or index (if no header).",
    )

    p.add_argument(
        "--err",
        choices=["none", "std", "sem", "ci95"],
        default="ci95",
        help="Error bar type for repeated x values.",
    )

    p.add_argument(
        "--autoscale",
        choices=["none", "auto", "logx"],
        default="auto",
        help="Axis autoscaling heuristics.",
    )

    p.add_argument(
        "--legend",
        choices=["auto", "none", "inside", "outside"],
        default="auto",
        help="Legend placement. 'auto' hides legend for a single series.",
    )

    p.add_argument(
        "--summary",
        default=None,
        help="Optional CSV path to write a per-series 'best setting' summary.",
    )

    p.add_argument(
        "--best",
        choices=["max", "min"],
        default=None,
        help="Whether 'best' means max or min. Defaults to max unless y looks like a loss.",
    )

    p.add_argument("--dpi", type=int, default=160, help="Output dpi for raster formats.")
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=(7.5, 5.0),
        help="Figure size in inches.",
    )

    args = p.parse_args(argv)

    paths = _path_list(args.inputs)
    if not paths:
        raise SystemExit("No inputs matched.")

    labels: list[str | None] = []
    if args.label is None or len(args.label) == 0:
        labels = [None] * len(paths)
    else:
        # If fewer labels than inputs, pad with None.
        labels = list(args.label) + [None] * max(0, len(paths) - len(args.label))

    series: list[SeriesSpec] = []
    for path, lbl in zip(paths, labels, strict=False):
        series.append(_load_series(path, lbl, args.xcol, args.ycol, args.err))

    # Derive axis labels from the first series unless overridden.
    x_name = series[0].x_name
    y_name = series[0].y_name

    xlabel = args.xlabel if args.xlabel is not None else _pretty_axis_label(x_name)
    ylabel = args.ylabel if args.ylabel is not None else _pretty_axis_label(y_name)

    title = args.title if args.title is not None else _default_title(series, xlabel, ylabel)

    # Decide best-mode.
    if args.best is not None:
        best_mode: BestMode = args.best
    else:
        best_mode = "min" if "loss" in _normalize_col_name(y_name) else "max"

    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)

    # Plot each series.
    for s in series:
        has_err = any(e > 0 for e in s.y_err)
        if has_err:
            ax.errorbar(
                s.x,
                s.y_mean,
                yerr=s.y_err,
                marker="o",
                linewidth=1.8,
                markersize=4,
                capsize=4,
                capthick=1,
                label=s.label,
            )
        else:
            ax.plot(
                s.x,
                s.y_mean,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=s.label,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.35)

    all_x = [x for s in series for x in s.x]
    _maybe_log_x(ax, all_x, args.autoscale)
    _maybe_set_integer_xticks(ax, all_x)

    # Draw y=0 satisfaction boundary for robustness-like metrics.
    # IMPORTANT: Do not let the reference line change the y-limits (it can
    # otherwise compress plots where all robustness values are strictly
    # positive).
    if _looks_like_robustness(y_name, ylabel):
        ylim = ax.get_ylim()
        ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6, zorder=0)
        ax.set_ylim(ylim)

    # Legend.
    legend_mode: LegendMode = args.legend
    if legend_mode == "auto":
        legend_mode = "none" if len(series) <= 1 else "inside"

    if legend_mode == "inside":
        ax.legend(loc="best", frameon=False)
    elif legend_mode == "outside":
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    elif legend_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown legend mode: {legend_mode}")

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # Use tight layout for nicer spacing.
    fig.tight_layout()

    save_kwargs: dict[str, object] = {"dpi": args.dpi}
    # Only use bbox_inches="tight" when we *must* (legend outside), because it
    # can change the final pixel dimensions of raster outputs depending on text
    # extents.
    if legend_mode == "outside":
        save_kwargs["bbox_inches"] = "tight"

    for fmt in args.formats:
        out_path = out_base.with_suffix("." + fmt)
        fig.savefig(out_path, **save_kwargs)
        print(f"Wrote {out_path}")

    if args.summary is not None:
        _write_summary_csv(Path(args.summary), series, best_mode)
        print(f"Wrote summary {args.summary}")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
