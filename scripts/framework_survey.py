#!/usr/bin/env python3
"""scripts/framework_survey.py

Purpose
-------
This is a **CPU-first**, **offline** helper that prints a compact, reproducible
survey of the core third-party stacks discussed for the project:

* physics-ML / scientific-ML frameworks (Neuromancer, TorchPhysics, PhysicsNeMo)
* Temporal & spatial specification tooling (RTAMT, MoonLight, SpaTiaL)

This script provides a clear accounting of:

* **what software frameworks are being used**, and
* **what is installed / runnable** on a given machine,
* along with key **runtime prerequisites** (e.g., Java for MoonLight).

This script is intended to generate a small table you can drop into a report or
append to a repo doc as an "experimental setup / software" appendix.

What it outputs
By default it prints a fixed-width text table. You can also output:

* Markdown (`--format md`) - suitable for READMEs / reports
* JSON (`--format json`) - suitable for tooling

In all formats, each row contains:

* Display name
* Installed? (based on `importlib.metadata` and, optionally, import probes)
* Version (distribution version if available)
* Import module name
* Recommended install string
* Notes (platform/runtime constraints)

Safety & performance
* The default mode does **not import** heavy frameworks.
* With `--deep`, module imports are probed in a **subprocess with a timeout**
  so this script can't hang your shell due to a slow import.
* All probing is best-effort: failures are captured as notes rather than
  crashing.

File output
If you pass `--out path/to/file.md`, the script writes an auto-generated block.

* If the file does not exist, it is created.
* If it exists and you do **not** pass `--overwrite`, the script **updates** an
  auto-generated block delimited by HTML comments, preserving the rest of the
  document.
* If you pass `--overwrite`, the file is replaced entirely.

Examples
--------
    # Plain text table
    python scripts/framework_survey.py

    # Markdown appendix (table + system info)
    python scripts/framework_survey.py --format md

    # Markdown appendix + runtime probes (Java/CUDA/importability)
    python scripts/framework_survey.py --format md --deep

    # Write (or update) an appendix block inside docs/FRAMEWORK_SURVEY.md
    python scripts/framework_survey.py --format md --deep --out docs/FRAMEWORK_SURVEY.md

    # JSON (machine-readable)
    python scripts/framework_survey.py --format json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Literal


Category = Literal["framework", "stl"]


@dataclass(frozen=True)
class Component:
    """A third-party component we care about for this project."""

    name: str
    category: Category
    desc: str
    # Distribution names to probe via importlib.metadata (dash/underscore variants are also tried).
    dist_names: tuple[str, ...]
    # Module names to try importing (deep probe). Use multiple entries when naming changed upstream.
    import_names: tuple[str, ...]
    # Recommended pip install target (may include extras or a direct URL).
    install: str
    homepage: str
    license: str


COMPONENTS: tuple[Component, ...] = (

    Component(
        name="Neuromancer",
        category="framework",
        desc="PNNL PyTorch framework for differentiable programming, control, and SciML.",
        dist_names=("neuromancer",),
        import_names=("neuromancer",),
        install="neuromancer",
        homepage="https://github.com/pnnl/neuromancer",
        license="BSD-3-Clause",
    ),
    Component(
        name="TorchPhysics",
        category="framework",
        desc="Bosch PyTorch library for PINNs / DeepRitz style PDE/ODE solving.",
        dist_names=("torchphysics",),
        import_names=("torchphysics",),
        install="torchphysics",
        homepage="https://github.com/boschresearch/torchphysics",
        license="Apache-2.0",
    ),
    Component(
        name="PhysicsNeMo",
        category="framework",
        desc="NVIDIA PhysicsNeMo (successor naming of Modulus) for Physics AI pipelines.",
        # Some older environments still have the Modulus distribution installed.
        dist_names=("nvidia-physicsnemo", "nvidia-modulus"),
        # Import renamed with the rebrand; probe both.
        import_names=("physicsnemo", "modulus"),
        # Official docs recommend [all] for examples; core install is `nvidia-physicsnemo`.
        install="nvidia-physicsnemo[all]",
        homepage="https://github.com/NVIDIA/physicsnemo",
        license="Apache-2.0",
    ),

    Component(
        name="RTAMT",
        category="stl",
        desc="STL monitoring library (discrete and dense time; optional C++ backend).",
        dist_names=("rtamt",),
        import_names=("rtamt",),
        # Pinned in this repo; also conditionally installed for Python < 3.12.
        install="rtamt==0.3.5",
        homepage="https://github.com/nickovic/rtamt",
        license="BSD-3-Clause",
    ),
    Component(
        name="MoonLight",
        category="stl",
        desc="Runtime monitoring for STL and spatial logics (STREL).",
        dist_names=("moonlight",),
        import_names=("moonlight",),
        install="moonlight",
        homepage="https://github.com/MoonLightSuite/moonlight",
        license="Apache-2.0",
    ),
    Component(
        name="SpaTiaL (spatial-spec)",
        category="stl",
        desc="Python spatial spec/monitoring helpers (object/geometry oriented).",
        dist_names=("spatial-spec",),
        import_names=("spatial_spec",),
        install="spatial-spec",
        homepage="https://github.com/KTH-RPL-Planiacs/SpaTiaL",
        license="MIT",
    ),
    Component(
        name="SpaTiaL (spatial-lib)",
        category="stl",
        desc="SpaTiaL core library (git subdir) for STREL-style spatial reasoning/planning.",
        dist_names=("spatial",),
        import_names=("spatial",),
        install="spatial @ git+https://github.com/KTH-RPL-Planiacs/SpaTiaL.git#subdirectory=spatial-lib",
        homepage="https://github.com/KTH-RPL-Planiacs/SpaTiaL",
        license="MIT",
    ),
)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _norm_dist_name(name: str) -> tuple[str, ...]:
    """Return common normalization variants for a distribution name."""

    # importlib.metadata applies PEP503 normalization internally in many cases,
    # but trying a couple of obvious variants is cheap and improves robustness.
    a = name
    b = name.replace("-", "_")
    c = name.replace("_", "-")

    # Preserve order, remove duplicates.
    out: list[str] = []
    for x in (a, b, c):
        if x not in out:
            out.append(x)
    return tuple(out)


def _dist_version(dist_names: Iterable[str]) -> str | None:
    """Return the first distribution version found for any of the given names."""

    for raw in dist_names:
        for name in _norm_dist_name(raw):
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
            except Exception:
                # Be conservative: treat unexpected errors as "not found".
                continue
    return None


def _probe_import_version(
    import_names: Iterable[str], *, timeout_s: float
) -> tuple[bool, str | None, str | None, str | None]:
    """Try importing a module in a subprocess and return (ok, name, version, err).

    Why subprocess?
      * Importing heavyweight frameworks can be slow.
      * Subprocess + timeout keeps this script responsive.

    The probe tries each candidate import name in order.
    """

    names = [n for n in import_names if n]
    if not names:
        return False, None, None, "no import names"

    # Emit exactly one line of JSON so we can parse robustly.
    code = r"""
import importlib, json, sys

names = json.loads(sys.argv[1])


def _stringify(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    try:
        return str(v)
    except Exception:
        return None


for name in names:
    try:
        m = importlib.import_module(name)
        ver = None
        for attr in ("__version__", "version", "VERSION"):
            ver = getattr(m, attr, None)
            if ver is not None:
                break
        out = {"ok": True, "import": name, "version": _stringify(ver)}
        print(json.dumps(out, ensure_ascii=False))
        raise SystemExit(0)
    except Exception as e:
        last_err = repr(e)

print(
    json.dumps(
        {
            "ok": False,
            "import": None,
            "version": None,
            "err": last_err if "last_err" in globals() else "import failed",
        }
    )
)
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, json.dumps(names)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, None, None, f"import probe timed out after {timeout_s:.1f}s"
    except Exception as e:
        return False, None, None, f"import probe failed: {e!r}"

    line = (proc.stdout or "").strip().splitlines()[-1:]  # last line only
    if not line:
        err = (proc.stderr or "").strip() or "no stdout from probe"
        return False, None, None, err

    try:
        payload = json.loads(line[0])
    except Exception:
        err = (proc.stderr or "").strip() or f"unparseable probe output: {line[0]!r}"
        return False, None, None, err

    ok = bool(payload.get("ok"))
    imp = payload.get("import") if isinstance(payload.get("import"), str) else None
    ver = payload.get("version") if isinstance(payload.get("version"), str) else None
    err = payload.get("err") if isinstance(payload.get("err"), str) else None
    return ok, imp, ver, err


def _java_version(timeout_s: float) -> str | None:
    """Return Java version string from `java -version`, or None if not available."""

    try:
        proc = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except Exception:
        return None

    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # Typical output includes: openjdk version "17.0.11" 2024-04-16
    m = re.search(r'version\s+"([^"]+)"', blob)
    if m:
        return m.group(1)

    # Fall back to first non-empty line.
    for ln in blob.splitlines():
        s = ln.strip()
        if s:
            return s
    return None


def _java_major(version_str: str) -> int | None:
    """Parse the Java *major* version from a `java -version` style string."""

    # Java 8 style: "1.8.0_202" => major 8
    m = re.match(r"^(\d+)(?:\.(\d+))?", version_str.strip())
    if not m:
        return None

    first = m.group(1)
    second = m.group(2)
    try:
        major = int(first)
        if major == 1 and second is not None:
            return int(second)
        return major
    except Exception:
        return None


def _torch_info() -> dict[str, Any]:
    """Minimal torch + CUDA availability snapshot."""

    try:
        import torch  # noqa: WPS433 (optional import)

        gpus: list[str] = []
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    gpus.append(torch.cuda.get_device_name(i))
            except Exception:
                gpus = []

        return {
            "installed": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpus": gpus,
        }
    except Exception:
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpus": [],
        }


def _platform_note() -> str:
    sysname = platform.system()
    if sysname == "Windows":
        return "Windows (consider WSL2 for PhysicsNeMo/MONA-based tooling)"
    if sysname == "Darwin":
        return "macOS"
    if sysname == "Linux":
        return "Linux"
    return sysname


def _python_major_minor() -> tuple[int, int]:
    return int(sys.version_info.major), int(sys.version_info.minor)


def _component_notes(
    *,
    c: Component,
    deep: bool,
    import_ok: bool | None,
    import_err: str | None,
    java_ver: str | None,
) -> str:
    """Create a short, human-readable notes string for a component."""

    notes: list[str] = []
    sysname = platform.system()
    py_mm = _python_major_minor()

    if c.name == "PhysicsNeMo":
        if sysname != "Linux":
            notes.append("Linux/WSL-first (non-Linux installs may be unsupported)")
        if py_mm < (3, 10):
            notes.append("requires Python ≥ 3.10")
        notes.append("pip: nvidia-physicsnemo (Modulus rename)")

    if c.name == "RTAMT":
        if py_mm >= (3, 12):
            notes.append("repo installs RTAMT only on Python < 3.12")
        notes.append("optional C++ backend/build (Boost/CMake) for speed")

    if c.name == "MoonLight":
        if deep:
            if java_ver is None:
                notes.append("requires Java ≥ 21 (java not detected)")
            else:
                mj = _java_major(java_ver)
                if mj is not None and mj < 21:
                    notes.append(f"requires Java ≥ 21 (detected {java_ver})")
        else:
            notes.append("requires Java ≥ 21")

    if c.name.startswith("SpaTiaL"):
        if sysname == "Windows":
            notes.append("planning stack easiest via WSL (MONA/ltlf2dfa)")
        else:
            notes.append("planning uses MONA via ltlf2dfa (may not work on Windows)")

    if deep and import_ok is False and import_err:
        notes.append(f"import probe failed: {import_err}")

    # De-duplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for n in notes:
        if n not in seen:
            seen.add(n)
            uniq.append(n)

    return " · ".join(uniq)


def survey(
    *,
    deep: bool,
    only: Category | None,
    import_timeout_s: float,
) -> dict[str, Any]:
    """Run the survey and return a dict suitable for formatting."""

    java_ver = _java_version(timeout_s=2.0) if deep else None

    rows: list[dict[str, Any]] = []
    for c in COMPONENTS:
        if only is not None and c.category != only:
            continue

        dist_ver = _dist_version(c.dist_names)

        import_ok: bool | None = None
        import_name: str | None = c.import_names[0] if c.import_names else None
        import_ver: str | None = None
        import_err: str | None = None

        if deep and c.import_names:
            ok, imp, ver, err = _probe_import_version(c.import_names, timeout_s=import_timeout_s)
            import_ok = ok
            if imp:
                import_name = imp
            import_ver = ver
            import_err = err

        installed = dist_ver is not None or bool(import_ok)
        version = dist_ver or import_ver or "not installed"

        notes = _component_notes(
            c=c,
            deep=deep,
            import_ok=import_ok,
            import_err=import_err,
            java_ver=java_ver,
        )

        rows.append(
            {
                "name": c.name,
                "category": c.category,
                "desc": c.desc,
                "homepage": c.homepage,
                "license": c.license,
                "installed": installed,
                "version": version,
                "dist_version": dist_ver,
                "import": import_name,
                "import_version": import_ver,
                "install": c.install,
                "notes": notes,
            }
        )

    sysinfo: dict[str, Any] = {
        "generated_at_utc": _now_utc_iso(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "os": _platform_note(),
        "cpu_count": os.cpu_count(),
        "torch": _torch_info(),
    }
    if deep:
        sysinfo["java"] = java_ver or "not detected"

    return {"rows": rows, "sys": sysinfo}


def _truncate(s: str, width: int) -> str:
    if width <= 0:
        return ""
    return s if len(s) <= width else (s[: max(0, width - 1)] + "...")


def format_text_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Package", "Installed", "Version", "Import", "Install", "Notes"]
    # Soft column caps (keeps the table readable on narrow terminals).
    caps = [24, 9, 18, 16, 26, 60]

    cols: list[list[str]] = [
        [str(r["name"]) for r in rows],
        [("yes" if r["installed"] else "no") for r in rows],
        [str(r["version"]) for r in rows],
        [str(r["import"]) for r in rows],
        [str(r["install"]) for r in rows],
        [str(r["notes"]) for r in rows],
    ]

    widths: list[int] = []
    for i, h in enumerate(headers):
        max_data = max([len(h)] + [len(x) for x in cols[i]])
        widths.append(min(max_data, caps[i]))

    def line(values: list[str]) -> str:
        return "  ".join(_truncate(values[i], widths[i]).ljust(widths[i]) for i in range(len(values)))

    out: list[str] = []
    out.append(line(headers))
    out.append(line(["-" * w for w in widths]))
    for r in rows:
        out.append(
            line(
                [
                    str(r["name"]),
                    "yes" if r["installed"] else "no",
                    str(r["version"]),
                    str(r["import"]),
                    str(r["install"]),
                    str(r["notes"]),
                ]
            )
        )
    return "\n".join(out)


def format_md_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Package", "Installed", "Version", "Import", "Install", "Notes"]
    md: list[str] = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        notes = str(r["notes"]).replace("|", "\\|")
        md.append(
            "| "
            + " | ".join(
                [
                    str(r["name"]),
                    "✅" if r["installed"] else "❌",
                    str(r["version"]),
                    (f"`{r['import']}`" if r["import"] else ""),
                    f"`{r['install']}`",
                    notes,
                ]
            )
            + " |"
        )
    return "\n".join(md)


def format_md_appendix(*, rows: list[dict[str, Any]], sysinfo: dict[str, Any]) -> str:
    """A compact Markdown appendix block (table + system info)."""

    out: list[str] = []
    out.append("## Auto-generated software availability")
    out.append("")
    out.append(
        f"Generated by `scripts/framework_survey.py` at **{sysinfo.get('generated_at_utc', '')}** (UTC)."
    )
    out.append("")
    out.append(format_md_table(rows))
    out.append("")
    out.append("### System summary")
    out.append("")
    out.append(f"- Python: `{sysinfo.get('python')}`")
    out.append(f"- Platform: `{sysinfo.get('platform')}`")
    out.append(f"- CPU cores: `{sysinfo.get('cpu_count')}`")

    torch = sysinfo.get("torch", {}) or {}
    if torch.get("installed"):
        cuda = "CPU"
        if torch.get("cuda_available"):
            cuda = f"CUDA {torch.get('cuda_version') or '?'}"
        out.append(f"- torch: `{torch.get('version')}` ({cuda})")
        gpus = torch.get("gpus") or None
        if gpus:
            out.append(f"  - GPUs: {', '.join(str(x) for x in gpus)}")
    else:
        out.append("- torch: not installed")

    if "java" in sysinfo:
        out.append(f"- Java: `{sysinfo.get('java')}`")

    return "\n".join(out).rstrip() + "\n"


_BLOCK_BEGIN = "<!-- BEGIN AUTO-GENERATED: framework_survey.py -->"
_BLOCK_END = "<!-- END AUTO-GENERATED: framework_survey.py -->"


def _wrap_as_markdown_block(payload: str, *, fmt: str) -> str:
    fence = "json" if fmt == "json" else "text"
    return f"```{fence}\n{payload.rstrip()}\n```\n"


def write_output(
    *,
    out_path: str,
    payload: str,
    fmt: str,
    overwrite: bool,
) -> None:
    """Write payload to stdout or to a file.

    For Markdown outputs to an existing .md file, we update/replace a delimited
    block so the rest of the document is preserved.
    """

    if out_path == "-":
        sys.stdout.write(payload)
        if not payload.endswith("\n"):
            sys.stdout.write("\n")
        return

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    is_markdown_target = path.suffix.lower() in {".md", ".markdown", ".mdown"}
    if not path.exists() or overwrite or not is_markdown_target:
        # For non-Markdown targets (e.g., .json), always overwrite.
        path.write_text(payload, encoding="utf-8")
        return

    # Update-in-place for Markdown targets.
    existing = path.read_text(encoding="utf-8")

    insert = payload
    if fmt in {"text", "json"}:
        insert = _wrap_as_markdown_block(payload, fmt=fmt)

    block = f"{_BLOCK_BEGIN}\n{insert.rstrip()}\n{_BLOCK_END}\n"

    if _BLOCK_BEGIN in existing and _BLOCK_END in existing:
        pre, rest = existing.split(_BLOCK_BEGIN, 1)
        _, post = rest.split(_BLOCK_END, 1)
        updated = pre.rstrip() + "\n\n" + block + post.lstrip("\n")
    else:
        updated = existing.rstrip() + "\n\n" + block

    path.write_text(updated, encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Survey availability/versions of frameworks + STL/STREL tooling used in neural-pde-stl-strel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument(
        "--format",
        choices=["auto", "text", "md", "json"],
        default="auto",
        help=(
            "Output format. 'auto' infers from --out: Markdown for *.md, JSON for *.json, otherwise text."
        ),
    )
    ap.add_argument(
        "--deep",
        action="store_true",
        help="Probe importability in a subprocess (timeout-protected) and detect Java.",
    )
    ap.add_argument(
        "--import-timeout",
        type=float,
        default=4.0,
        help="Per-module import probe timeout (seconds) when using --deep.",
    )
    ap.add_argument(
        "--only",
        choices=["all", "framework", "stl"],
        default="all",
        help="Filter rows by category.",
    )
    ap.add_argument(
        "--show-install",
        action="store_true",
        help="Include install commands for missing components.",
    )
    ap.add_argument(
        "--out",
        default="-",
        help="Write output to a file (use '-' for stdout). For existing .md files, updates an auto-generated block.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --out file entirely (instead of updating an auto-generated block).",
    )

    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Infer a sane default output format if the user asked for --format auto.
    fmt = str(args.format)
    if fmt == "auto":
        if args.out != "-":
            suf = Path(str(args.out)).suffix.lower()
            if suf in {".md", ".markdown", ".mdown"}:
                fmt = "md"
            elif suf == ".json":
                fmt = "json"
            else:
                fmt = "text"
        else:
            fmt = "text"

    only: Category | None
    if args.only == "all":
        only = None
    else:
        only = args.only

    result = survey(deep=bool(args.deep), only=only, import_timeout_s=float(args.import_timeout))
    rows: list[dict[str, Any]] = result["rows"]
    sysinfo: dict[str, Any] = result["sys"]

    missing = [r for r in rows if not r["installed"]]
    install_cmds = [f"pip install {shlex.quote(str(r['install']))}" for r in missing]

    payload: str

    if fmt == "json":
        out_obj: dict[str, Any] = {"rows": rows, "sys": sysinfo}
        if args.show_install:
            out_obj["install_commands"] = install_cmds
        payload = json.dumps(out_obj, indent=2, sort_keys=True)

    elif fmt == "md":
        payload = format_md_appendix(rows=rows, sysinfo=sysinfo)
        if args.show_install:
            payload += "\n### Install commands for missing components\n\n"
            if install_cmds:
                payload += "\n".join(f"- `{c}`" for c in install_cmds) + "\n"
            else:
                payload += "- (none; everything above is installed)\n"

    else:
        payload = format_text_table(rows)
        payload += "\n\nSystem:\n"
        payload += f"  Python:   {sysinfo.get('python')}\n"
        payload += f"  Platform: {sysinfo.get('platform')}\n"
        payload += f"  CPU cores:{sysinfo.get('cpu_count')}\n"

        torch = sysinfo.get("torch", {}) or {}
        if torch.get("installed"):
            cuda = "CPU"
            if torch.get("cuda_available"):
                cuda = f"CUDA {torch.get('cuda_version') or '?'}"
            payload += f"  torch:    {torch.get('version')} · {cuda}\n"
        if args.deep:
            payload += f"  Java:     {sysinfo.get('java')}\n"

        if args.show_install:
            payload += "\nInstall commands for missing components:\n"
            if install_cmds:
                payload += "\n".join(f"  {c}" for c in install_cmds) + "\n"
            else:
                payload += "  (none; everything above is installed)\n"

    write_output(out_path=str(args.out), payload=payload, fmt=fmt, overwrite=bool(args.overwrite))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
