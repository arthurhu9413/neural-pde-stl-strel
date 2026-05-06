#!/usr/bin/env python3
"""neural-pde-stl-strel -- environment check.

This script is used for the neural-pde-stl-strel repository to quickly answer:

1) What machine am I running on (OS / Python / CPU / RAM / GPU)?
2) Which optional stacks are available (frameworks + STL/STREL tooling)?

It is intentionally **CPU-first** and **robust**: it should never crash.
Missing optional components are reported with actionable installation hints.

Outputs:
- Default: human-friendly summary (colored if stdout is a TTY).
- --md: Markdown dependency table (easy to paste into README/issues).
- --json: JSON blob with system information + dependency probe results
          (print to stdout and redirect if you want to save it).

Exit codes:
- 0: all core requirements are satisfied (Python >= 3.10, NumPy >= 1.24, PyYAML >= 6.0).
- 1: one or more core requirements are missing or below minimum version.

Usage:
  python scripts/check_env.py
  python scripts/check_env.py --json
  python scripts/check_env.py --md
  python scripts/check_env.py --quick  # avoid external commands (java, nvidia-smi, mona)

"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _supports_color(stream: Any) -> bool:
    """Return True iff ANSI color codes should be enabled for the given stream."""
    try:
        if not getattr(stream, "isatty", lambda: False)():
            return False
    except Exception:
        return False
    # Respect the de-facto standard "NO_COLOR" opt-out.
    if os.environ.get("NO_COLOR") is not None:
        return False
    # Some CI environments set TERM=dumb.
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    return True


class _Color:
    """Minimal ANSI color helper (no external deps)."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _c(self, code: str) -> str:
        return code if self.enabled else ""

    @property
    def reset(self) -> str:
        return self._c("\033[0m")

    @property
    def dim(self) -> str:
        return self._c("\033[2m")

    @property
    def bold(self) -> str:
        return self._c("\033[1m")

    @property
    def green(self) -> str:
        return self._c("\033[32m")

    @property
    def red(self) -> str:
        return self._c("\033[31m")

    @property
    def yellow(self) -> str:
        return self._c("\033[33m")

    @property
    def blue(self) -> str:
        return self._c("\033[34m")

    @property
    def magenta(self) -> str:
        return self._c("\033[35m")


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run(cmd: Sequence[str], *, timeout: float = 2.0) -> Tuple[int, str, str]:
    """Run a small external command; never raises; returns (code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            list(cmd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except Exception as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"


def _is_wsl() -> bool:
    """Best-effort WSL detection (Linux kernel with Microsoft signature)."""
    if platform.system() != "Linux":
        return False
    rel = platform.release().lower()
    if "microsoft" in rel or "wsl" in rel:
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().lower()
        return "microsoft" in txt or "wsl" in txt
    except Exception:
        return False


def _is_docker() -> bool:
    """Best-effort container detection (no false-positive guarantees)."""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8", errors="ignore") as f:
            cg = f.read().lower()
        return any(tok in cg for tok in ("docker", "containerd", "kubepods"))
    except Exception:
        return False


def _bytes_to_gib(n_bytes: Optional[int]) -> Optional[float]:
    if n_bytes is None:
        return None
    return n_bytes / (1024**3)


def _na_token(*, ascii_only: bool) -> str:
    return "-" if ascii_only else "--"


def _format_gib(n_bytes: Optional[int], *, ascii_only: bool = False) -> str:
    gib = _bytes_to_gib(n_bytes)
    if gib is None:
        return _na_token(ascii_only=ascii_only)
    if gib < 10:
        return f"{gib:.2f} GiB"
    if gib < 100:
        return f"{gib:.1f} GiB"
    return f"{gib:.0f} GiB"


def _cpu_brand(*, quick: bool) -> Optional[str]:
    """Return a best-effort CPU brand/model string (cross-platform)."""
    # 1) Prefer py-cpuinfo if available (optional dependency in requirements-extra/dev).
    try:
        import cpuinfo  # type: ignore

        info = cpuinfo.get_cpu_info()
        for key in ("brand_raw", "brand", "arch_string_raw"):
            brand = info.get(key)
            if isinstance(brand, str) and brand.strip():
                return brand.strip()
    except Exception:
        pass

    sysname = platform.system()

    # 2) Linux: /proc/cpuinfo usually has "model name".
    if sysname == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    low = line.lower()
                    if low.startswith("model name") and ":" in line:
                        _, val = line.split(":", 1)
                        val = val.strip()
                        if val:
                            return val
                    if low.startswith("hardware") and ":" in line:
                        _, val = line.split(":", 1)
                        val = val.strip()
                        if val:
                            return val
                    # Some ARM systems include "Processor : ..."
                    if low.startswith("processor") and ":" in line:
                        _, val = line.split(":", 1)
                        val = val.strip()
                        if val and not val.isdigit():
                            return val
        except Exception:
            pass

    # 3) macOS: sysctl (fast, but still an external command).
    if sysname == "Darwin" and not quick:
        sysctl = _which("sysctl")
        if sysctl:
            code, out, _err = _run([sysctl, "-n", "machdep.cpu.brand_string"], timeout=0.8)
            if code == 0 and out.strip():
                return out.strip()

    # 4) Windows: environment variable often contains useful info.
    if sysname == "Windows":
        ident = os.environ.get("PROCESSOR_IDENTIFIER")
        if ident and ident.strip():
            return ident.strip()

    # 5) Last resort: platform.processor() (can be empty on some platforms).
    try:
        proc = platform.processor()
        if proc and proc.strip():
            return proc.strip()
    except Exception:
        pass

    return None


def _cpu_cores() -> Tuple[Optional[int], Optional[int]]:
    """Return (physical_cores, logical_cores) where known."""
    logical = os.cpu_count()
    physical: Optional[int] = None
    try:
        import psutil  # type: ignore

        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical = None
    return physical, logical


def _memory_total_bytes(*, quick: bool) -> Optional[int]:
    """Return total physical memory in bytes, when detectable."""
    # 1) Prefer psutil if available (optional dependency in requirements-extra/dev).
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        pass

    sysname = platform.system()

    # 2) Linux: /proc/meminfo
    if sysname == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            return int(parts[1]) * 1024  # kB -> B
        except Exception:
            return None

    # 3) macOS: sysctl hw.memsize
    if sysname == "Darwin" and not quick:
        sysctl = _which("sysctl")
        if sysctl:
            code, out, _err = _run([sysctl, "-n", "hw.memsize"], timeout=0.8)
            if code == 0 and out.strip().isdigit():
                return int(out.strip())

    # 4) Windows: ctypes GlobalMemoryStatusEx
    if sysname == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return int(stat.ullTotalPhys)
        except Exception:
            pass

    return None


def _nvidia_smi_summary(*, quick: bool) -> Optional[str]:
    """Return a one-line GPU/driver summary from nvidia-smi, if available."""
    if quick:
        return None
    nvsmi = _which("nvidia-smi")
    if not nvsmi:
        return None
    code, out, err = _run(
        [nvsmi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
        timeout=1.2,
    )
    if code == 0 and out:
        return out.splitlines()[0].strip()
    if err:
        return f"(error) {err.splitlines()[0].strip()}"
    return None


def _collect_system_info(*, quick: bool) -> Dict[str, Any]:
    """Collect a small, reproducibility-oriented system information blob."""
    hostname = socket.gethostname()

    physical, logical = _cpu_cores()
    mem_total = _memory_total_bytes(quick=quick)

    info: Dict[str, Any] = {
        "hostname": hostname,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "is_wsl": _is_wsl(),
        "is_docker": _is_docker(),
        "cpu": {
            "brand": _cpu_brand(quick=quick),
            "cores_physical": physical,
            "cores_logical": logical,
        },
        "memory": {
            "total_bytes": mem_total,
            "total_gib": (round(_bytes_to_gib(mem_total), 3) if mem_total is not None else None),
        },
        "gpu": {},
        "env": {},
    }

    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        info["env"]["conda_env"] = conda_env
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        info["env"]["virtual_env"] = venv

    nvsmi = _nvidia_smi_summary(quick=quick)
    if nvsmi:
        info["gpu"]["nvidia_smi"] = nvsmi

    return info


def _version_of_distribution(names: Iterable[str]) -> Optional[str]:
    """Return a version string for any of the provided candidates.

    Strategy:
    1) use importlib.metadata (distribution version), then
    2) fall back to importing the module and reading __version__.
    """
    lowered = {n.lower() for n in names}
    if "python" in lowered:
        return platform.python_version()

    try:
        from importlib import metadata as md  # Python 3.8+
    except Exception:  # pragma: no cover
        md = None  # type: ignore[assignment]
        try:
            import importlib_metadata as md  # type: ignore[no-redef]
        except Exception:
            md = None  # type: ignore[assignment]

    if md is not None:
        for n in names:
            try:
                return md.version(n)
            except Exception:
                pass

    for n in names:
        try:
            mod = importlib.import_module(n)
            v = getattr(mod, "__version__", None)
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:
            continue

    return None


def _parse_version(v: str) -> Tuple[Any, ...]:
    """Lenient version parser used for >= checks.

    If packaging is installed, we delegate to packaging.version.Version.
    Otherwise we fall back to a simple digit/text tuple.
    """
    try:
        from packaging.version import Version  # type: ignore

        return (Version(v),)
    except Exception:
        parts = re.split(r"[^0-9A-Za-z]+", v)
        norm: List[Any] = []
        for p in parts:
            if not p:
                continue
            norm.append(int(p) if p.isdigit() else p.lower())
        return tuple(norm)


def _meets(v: Optional[str], min_v: Optional[str]) -> bool:
    if min_v is None:
        return True
    if v is None:
        return False
    return _parse_version(v) >= _parse_version(min_v)


ApplicableFn = Callable[[], Tuple[bool, str]]
ExtraProbeFn = Callable[[Optional[ModuleType], bool], Mapping[str, Any]]


@dataclass(frozen=True)
class Dependency:
    """Metadata describing a dependency we want to probe."""

    display: str
    import_names: Tuple[str, ...]
    pip_names: Tuple[str, ...]
    required: bool = False
    min_version: Optional[str] = None
    notes: str = ""
    applicable: Optional[ApplicableFn] = None
    install_hint: Optional[str] = None
    extra_probe: Optional[ExtraProbeFn] = None


@dataclass
class ProbeResult:
    applicable: bool
    present: bool
    imported: bool
    version: Optional[str]
    message: str
    extra: Dict[str, Any]


def _probe(dep: Dependency, *, quick: bool) -> Tuple[Dependency, ProbeResult]:
    applicable = True
    reason = ""
    if dep.applicable is not None:
        try:
            applicable, reason = dep.applicable()
        except Exception as exc:
            applicable = False
            reason = f"applicability check failed: {type(exc).__name__}: {exc}"

    version = _version_of_distribution((*dep.pip_names, *dep.import_names))

    if not applicable:
        msg = f"n/a: {reason}" if reason else "n/a"
        return dep, ProbeResult(
            applicable=False,
            present=False,
            imported=False,
            version=version,
            message=msg,
            extra={"reason": reason} if reason else {},
        )

    present = False
    spec_name: Optional[str] = None
    for name in dep.import_names:
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is not None:
            present = True
            spec_name = name
            break

    imported = False
    module: Optional[ModuleType] = None
    message = ""
    extra: Dict[str, Any] = {}

    if present and spec_name is not None:
        try:
            module = importlib.import_module(spec_name)
            imported = True
            message = "OK"
        except Exception as exc:
            message = f"import error: {type(exc).__name__}: {exc}"
    else:
        message = "module not found"

    if dep.min_version:
        if version is None:
            if imported:
                message = f"version unknown; needs >= {dep.min_version}"
        elif not _meets(version, dep.min_version):
            message = f"needs >= {dep.min_version}; found {version}"

    if dep.extra_probe is not None:
        try:
            extra.update(dict(dep.extra_probe(module, quick)))
        except Exception as exc:
            extra.setdefault("warning", f"extra probe failed: {type(exc).__name__}: {exc}")

    return dep, ProbeResult(
        applicable=True,
        present=present,
        imported=imported,
        version=version,
        message=message,
        extra=extra,
    )


def _probe_torch(mod: Optional[ModuleType], quick: bool) -> Mapping[str, Any]:
    """Extra PyTorch info (CUDA/MPS presence, device name, etc.)."""
    info: Dict[str, Any] = {}
    torch = None

    if mod is not None:
        torch = mod  # type: ignore[assignment]
    else:
        try:
            import torch as _torch  # type: ignore

            torch = _torch
        except Exception as exc:  # pragma: no cover
            return {"error": f"torch import failed: {type(exc).__name__}: {exc}"}

    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["cuda_available"] = False

    try:
        mps_backend = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps_backend and mps_backend.is_available())
    except Exception:
        info["mps_available"] = False

    info["build_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)

    cudnn_v: Optional[int] = None
    try:
        cudnn_v = torch.backends.cudnn.version()
    except Exception:
        cudnn_v = None
    info["cudnn"] = cudnn_v

    if info.get("cuda_available"):
        try:
            info["gpu_count"] = int(torch.cuda.device_count())
            info["gpu_name0"] = str(torch.cuda.get_device_name(0))
            cap = torch.cuda.get_device_capability(0)
            info["capability0"] = ".".join(str(x) for x in cap)
        except Exception:
            pass

    if not quick and _which("nvidia-smi") and info.get("cuda_available"):
        code, out, err = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            timeout=1.5,
        )
        if code == 0 and out:
            info["nvidia_smi"] = out.splitlines()[0].strip()
        elif err:
            info["nvidia_smi"] = f"(error) {err.splitlines()[0].strip()}"

    return info


def _parse_java_major(raw: str) -> Optional[int]:
    """Parse the Java major version from `java -version` output.

    Handles both modern version strings (e.g. 21.0.2) and legacy Java 8 style
    (e.g. 1.8.0_362, where major is 8).
    """
    m = re.search(r'version\s+"([^"]+)"', raw)
    if m:
        ver = m.group(1)
        parts = [p for p in re.split(r"[^0-9]+", ver) if p]
        if not parts:
            return None
        if parts[0] == "1" and len(parts) >= 2:
            return int(parts[1])
        return int(parts[0])

    m = re.search(r"\b(\d+)(?:\.(\d+))?", raw)
    if not m:
        return None
    if m.group(1) == "1" and m.group(2):
        return int(m.group(2))
    return int(m.group(1))


def _probe_java(_mod: Optional[ModuleType], quick: bool) -> Mapping[str, Any]:
    """Probe Java availability/version for MoonLight (STREL)."""
    info: Dict[str, Any] = {}
    jpath = _which("java")
    if not jpath:
        info["java_present"] = False
        return info

    info["java_present"] = True
    info["java_path"] = jpath

    if quick:
        return info

    code, out, err = _run([jpath, "-version"], timeout=1.5)
    raw = (out + "\n" + err).strip()
    if raw:
        first = raw.splitlines()[0].strip()
        info["java_raw"] = first
        major = _parse_java_major(raw)
        if major is not None:
            info["java_major"] = major
            info["java_ok_for_moonlight"] = major >= 21
    return info


def _probe_mona(_mod: Optional[ModuleType], quick: bool) -> Mapping[str, Any]:
    """Probe MONA + ltlf2dfa presence for SpaTiaL-style tooling."""
    info: Dict[str, Any] = {}

    mpath = _which("mona")
    if mpath:
        info["mona_available"] = True
        info["mona_path"] = mpath
        if not quick:
            code, out, err = _run([mpath, "-v"], timeout=1.5)
            raw = (out or err or "").strip()
            if raw:
                info["mona_raw"] = raw.splitlines()[0].strip()
    else:
        info["mona_available"] = False

    ltlf = _which("ltlf2dfa")
    if ltlf:
        info["ltlf2dfa_available"] = True
        info["ltlf2dfa_path"] = ltlf
    else:
        info["ltlf2dfa_available"] = False

    return info


def _applicable_linux_only() -> Tuple[bool, str]:
    sysname = platform.system()
    if sysname == "Linux":
        return True, ""
    return False, f"Linux/WSL only (current OS: {sysname})"


def _applicable_not_windows() -> Tuple[bool, str]:
    sysname = platform.system()
    if sysname != "Windows":
        return True, ""
    return False, "Not supported on Windows (use WSL/Linux)."


def _applicable_rtamt() -> Tuple[bool, str]:
    if sys.version_info < (3, 12):
        return True, ""
    return (
        False,
        f"Python {platform.python_version()} >= 3.12; RTAMT imports typing.io (removed). Use Python 3.11.",
    )


CORE: List[Dependency] = [
    Dependency(
        display="Python",
        import_names=("sys",),
        pip_names=("python",),
        required=True,
        min_version="3.10",
        notes="Project requires Python >= 3.10.",
    ),
    Dependency(
        display="NumPy",
        import_names=("numpy",),
        pip_names=("numpy",),
        required=True,
        min_version="1.24.0",
        notes="Core PDE sandbox + array ops.",
    ),
    Dependency(
        display="PyYAML",
        import_names=("yaml",),
        pip_names=("pyyaml",),
        required=True,
        min_version="6.0.0",
        notes="YAML config parsing for experiment and figure scripts.",
    ),
]

FRAMEWORKS: List[Dependency] = [
    Dependency(
        display="PyTorch",
        import_names=("torch",),
        pip_names=("torch",),
        notes="Training backend; CPU-only is sufficient.",
        extra_probe=_probe_torch,
    ),
    Dependency(
        display="NeuroMANCER",
        import_names=("neuromancer",),
        pip_names=("neuromancer",),
        notes="PNNL Neuromancer (PyTorch) control + physics-ML.",
    ),
    Dependency(
        display="PhysicsNeMo",
        import_names=("physicsnemo",),
        pip_names=("nvidia-physicsnemo",),
        notes="NVIDIA PhysicsNeMo (Linux/WSL).",
        applicable=_applicable_linux_only,
        install_hint="python -m pip install nvidia-physicsnemo",
    ),
    Dependency(
        display="PhysicsNeMo-Sym",
        import_names=("physicsnemo.sym",),
        pip_names=("nvidia-physicsnemo.sym", "nvidia-physicsnemo-sym"),
        notes="PhysicsNeMo symbolic module (Linux/WSL).",
        applicable=_applicable_linux_only,
        install_hint=(
            'python -m pip install Cython && python -m pip install'
            ' "nvidia-physicsnemo.sym" --no-build-isolation'
        ),
    ),
    Dependency(
        display="TorchPhysics",
        import_names=("torchphysics",),
        pip_names=("torchphysics",),
        notes="Bosch TorchPhysics (PINNs).",
    ),
]

STL_TOOLS: List[Dependency] = [
    Dependency(
        display="RTAMT (STL)",
        import_names=("rtamt",),
        pip_names=("rtamt",),
        notes="Real-time STL monitoring (Nickovic et al.).",
        applicable=_applicable_rtamt,
    ),
    Dependency(
        display="MoonLight (STREL)",
        import_names=("moonlight",),
        pip_names=("moonlight",),
        notes="MoonLightSuite STREL; requires Java >= 21 on PATH.",
        extra_probe=_probe_java,
    ),
    Dependency(
        display="SpaTiaL / spatial-spec",
        import_names=("spatial_spec",),
        pip_names=("spatial-spec",),
        notes="SpaTiaL-style spatial specs; needs MONA + ltlf2dfa on PATH.",
        applicable=_applicable_not_windows,
        extra_probe=_probe_mona,
    ),
]

EVERYTHING: List[Dependency] = [*CORE, *FRAMEWORKS, *STL_TOOLS]


def _status_icon(status: str, c: _Color, *, ascii_only: bool) -> str:
    if ascii_only:
        if status == "ok":
            return "OK"
        if status == "na":
            return "NA"
        return "NO"
    if status == "ok":
        return f"{c.green}✔{c.reset}"
    if status == "na":
        return f"{c.dim}--{c.reset}"
    return f"{c.red}✖{c.reset}"


def _render_row(dep: Dependency, res: ProbeResult, c: _Color, *, ascii_only: bool) -> str:
    if not res.applicable:
        status = "na"
        ok = False
    else:
        ok = res.imported and (dep.min_version is None or _meets(res.version, dep.min_version))
        status = "ok" if ok else "warn"

    icon = _status_icon(status, c, ascii_only=ascii_only)
    ver = res.version or _na_token(ascii_only=ascii_only)
    if not res.applicable:
        msg = res.message
    else:
        msg = dep.notes if ok and dep.notes else res.message

    extras = []
    for k in (
        "cuda_available", "mps_available", "build_cuda", "gpu_name0",
        "java_major", "mona_available", "ltlf2dfa_available",
    ):
        if k in res.extra:
            extras.append(f"{k}={res.extra[k]}")
    extra_str = f"  [{', '.join(extras)}]" if extras else ""
    return f"{icon}  {dep.display}  {ver}  {msg}{extra_str}"


def _gpu_line(system_info: Mapping[str, Any], results: Mapping[str, Tuple[Dependency, ProbeResult]]) -> Optional[str]:
    # Prefer PyTorch-derived info if available.
    torch_pair = results.get("PyTorch")
    if torch_pair is not None:
        _dep, res = torch_pair
        if res.imported and res.extra:
            cuda = res.extra.get("cuda_available")
            mps = res.extra.get("mps_available")
            if cuda:
                name = res.extra.get("gpu_name0") or system_info.get("gpu", {}).get("nvidia_smi")
                return f"GPU: {name}" if name else "GPU: CUDA available"
            if mps:
                return "GPU: Apple MPS available"

    nvsmi = system_info.get("gpu", {}).get("nvidia_smi")
    if nvsmi:
        return f"GPU: {nvsmi}"
    return None


def _print_human(
    system_info: Mapping[str, Any],
    results: Mapping[str, Tuple[Dependency, ProbeResult]],
    *,
    ascii_only: bool,
    extended: bool,
) -> None:
    c = _Color(enabled=_supports_color(sys.stdout) and not ascii_only)
    sep = "-" * 79
    mid = " * " if not ascii_only else " | "
    title_sep = " * " if not ascii_only else " - "

    print(c.bold + f"neural-pde-stl-strel{title_sep}environment check" + c.reset)
    print(sep)

    host = system_info.get("hostname", "(unknown)")
    plat = system_info.get("platform", platform.platform())
    machine = system_info.get("machine", platform.machine())
    py = system_info.get("python_version", platform.python_version())
    impl = system_info.get("python_implementation", platform.python_implementation())

    cpu = system_info.get("cpu", {})
    cpu_brand = cpu.get("brand")
    cores_ph = cpu.get("cores_physical")
    cores_lo = cpu.get("cores_logical")

    mem = system_info.get("memory", {})
    mem_total = mem.get("total_bytes")

    env = system_info.get("env", {})
    conda_env = env.get("conda_env")
    venv = env.get("virtual_env")

    tags: List[str] = []
    if system_info.get("is_wsl"):
        tags.append("WSL")
    if system_info.get("is_docker"):
        tags.append("container")
    tag_str = f" ({', '.join(tags)})" if tags else ""

    print(c.dim + f"Host: {host}{tag_str}" + c.reset)
    print(c.dim + f"OS: {plat}{mid}Arch: {machine}" + c.reset)
    print(c.dim + f"Python: {py} ({impl})" + c.reset)

    if cpu_brand or cores_lo:
        core_str = ""
        if cores_ph and cores_lo:
            core_str = f"{mid}Cores: {cores_ph} physical / {cores_lo} logical"
        elif cores_lo:
            core_str = f"{mid}Cores: {cores_lo} logical"
        cpu_str = cpu_brand or "(unknown CPU)"
        print(c.dim + f"CPU: {cpu_str}{core_str}" + c.reset)

    if mem_total is not None:
        print(c.dim + f"RAM: {_format_gib(int(mem_total), ascii_only=ascii_only)}" + c.reset)

    gpu_line = _gpu_line(system_info, results)
    if gpu_line:
        print(c.dim + gpu_line + c.reset)

    env_bits: List[str] = []
    if conda_env:
        env_bits.append(f"conda:{conda_env}")
    if venv:
        env_bits.append("venv")
    if env_bits:
        print(c.dim + "env: " + ", ".join(env_bits) + c.reset)

    print(sep)

    def block(title: str, deps: Sequence[Dependency]) -> None:
        print(c.blue + title + c.reset)
        for d in deps:
            _dep, res = results[d.display]
            print("  " + _render_row(d, res, c, ascii_only=ascii_only))
        print(sep)

    block("Core", CORE)
    block("Physics-ML frameworks", FRAMEWORKS)
    block("STL / STREL", STL_TOOLS)

    # Install/upgrade hints

    hints: List[str] = []

    for dep, res in results.values():
        if not res.applicable:
            continue

        # Version too old -> upgrade hint (special-case Python).
        if dep.min_version and not _meets(res.version, dep.min_version):
            if dep.display == "Python":
                hints.append(f"Upgrade Python to >= {dep.min_version} (current: {res.version or 'unknown'})")
            elif dep.pip_names:
                hints.append(f"python -m pip install --upgrade {dep.pip_names[0]}")
            continue

        # Missing or import-failing packages -> install hint.
        if not res.imported:
            if dep.install_hint:
                hints.append(dep.install_hint)
            elif dep.pip_names:
                hints.append(f"python -m pip install {dep.pip_names[0]}")

    # Targeted hints for external toolchains.
    sysname = platform.system()

    # MoonLight: Java 21+
    ml = results.get("MoonLight (STREL)")
    if ml is not None:
        _dep, res = ml
        j = res.extra
        need_java = False
        if not j.get("java_present", False):
            need_java = True
        elif j.get("java_major") is not None and int(j["java_major"]) < 21:
            need_java = True

        if need_java:
            if sysname == "Darwin":
                hints.append("brew install openjdk@21    # MoonLight requires Java >= 21")
            elif sysname == "Windows":
                hints.append("winget install --id Microsoft.OpenJDK.21 -e    # MoonLight requires Java >= 21")
            else:
                hints.append(
                    "sudo apt-get update && sudo apt-get install -y"
                    " openjdk-21-jre    # MoonLight requires Java >= 21"
                )

    # SpaTiaL-style tooling: MONA + ltlf2dfa
    sp = results.get("SpaTiaL / spatial-spec")
    if sp is not None:
        _dep, res = sp
        if res.applicable:
            er = res.extra
            if er.get("mona_available") is False:
                if sysname == "Linux":
                    hints.append("sudo apt-get install -y mona    # needed for DFA construction")
                elif sysname == "Darwin":
                    hints.append("Install MONA (https://www.brics.dk/mona/) and put it on PATH")
            if er.get("ltlf2dfa_available") is False:
                hints.append("python -m pip install ltlf2dfa")

    if hints:
        print(c.magenta + "Install hints:" + c.reset)
        seen: set[str] = set()
        bullet = "*" if not ascii_only else "-"
        for h in hints:
            if h in seen:
                continue
            seen.add(h)
            print(f"  {bullet} {h}")

    if extended:
        print(sep)
        print(c.yellow + "Details:" + c.reset)
        for dep, res in results.values():
            if res.extra:
                print(f"  {dep.display}:")
                for k, v in sorted(res.extra.items()):
                    print(f"    - {k}: {v}")
        print(sep)


def _print_markdown(results: Mapping[str, Tuple[Dependency, ProbeResult]]) -> None:
    headers = ["Status", "Package", "Version", "Notes"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for dep, res in results.values():
        if not res.applicable:
            status = "N/A"
        else:
            ok = res.imported and (dep.min_version is None or _meets(res.version, dep.min_version))
            status = "OK" if ok else "Missing"

        ver = res.version or "--"
        note = dep.notes if (status == "OK" and dep.notes) else res.message
        print(f"| {status} | {dep.display} | {ver} | {note} |")


def _print_json(system_info: Mapping[str, Any], results: Mapping[str, Tuple[Dependency, ProbeResult]]) -> None:
    payload = {
        "system": dict(system_info),
        "dependencies": {
            name: {
                "applicable": res.applicable,
                "present": res.present,
                "imported": res.imported,
                "version": res.version,
                "message": res.message,
                "extra": res.extra,
                "required": dep.required,
                "min_version": dep.min_version,
            }
            for name, (dep, res) in results.items()
        },
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    print()


# Main


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check environment for neural-pde-stl-strel.")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    parser.add_argument("--md", action="store_true", help="Markdown dependency table output")
    parser.add_argument("--plain", action="store_true", help="ASCII-only output (no colors/emoji)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip external command execution (e.g., java -version, nvidia-smi, mona -v)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="include extra probe details in human output",
    )
    args = parser.parse_args(argv)

    system_info = _collect_system_info(quick=args.quick)

    # Probe everything deterministically.
    results: Dict[str, Tuple[Dependency, ProbeResult]] = {}
    for dep in EVERYTHING:
        d, r = _probe(dep, quick=args.quick)
        results[dep.display] = (d, r)

    if args.json:
        _print_json(system_info, results)
    elif args.md:
        _print_markdown(results)
    else:
        _print_human(system_info, results, ascii_only=args.plain, extended=args.extended)

    # Exit 0 only if core requirements are present and meet minimum versions.
    missing_core = [
        dep.display
        for dep in CORE
        if not (results[dep.display][1].imported and _meets(results[dep.display][1].version, dep.min_version))
    ]
    return 0 if not missing_core else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
