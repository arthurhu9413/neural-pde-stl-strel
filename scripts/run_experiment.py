#!/usr/bin/env python3
"""scripts/run_experiment.py

A small, dependency-light **experiment launcher** for the neural-pde-stl-strel
repository.

This script is intentionally conservative: it does *not* assume Hydra, MLflow,
or any external experiment manager. Instead, it provides a reproducible run
directory layout and delegates the heavy lifting to experiment implementations
in ``src/neural_pde_stl_strel/experiments``.

Key behaviors
1) **YAML config** with a tiny ``include:`` mechanism (deep-merged).
2) **Typed overrides** via ``--set KEY=VALUE`` (VALUE parsed as YAML).
3) **Optional Cartesian sweeps** via a top-level ``sweep:`` mapping.
4) **Per-run directories** under a results root (default: ``./results``):

   - ``config.effective.yaml``: the exact config passed to the runner.
   - ``env.json``: hardware/software snapshot (CPU/GPU/RAM, Python, Torch, git).
   - ``run.json``: timing + status + CLI provenance.
   - ``metrics.json`` (optional): persisted if the runner returns a mapping.

5) **Artifact isolation**: before calling the experiment runner, we override
   ``io.results_dir`` to point at the run directory so that experiment outputs
   land *inside the run directory* by default.

6) **Convenience aliases** (single-run only, by default): we create/update
   *stable* files in the results root (hardlinks when possible, else copies)
   for any file in the run directory whose name starts with
   ``{experiment}_{tag}``. This preserves the repo's "stable filename"
   workflow (e.g., ``results/diffusion1d_baseline.csv``) while keeping the
   full run self-contained.

Examples
--------

List experiments:

  python scripts/run_experiment.py --list

Run a config:

  python scripts/run_experiment.py -c configs/diffusion1d_stl.yaml

Override values (YAML-typed):

  python scripts/run_experiment.py -c configs/diffusion1d_stl.yaml \
      --set seed=1 --set optim.epochs=50

Small sweep (2×3 = 6 runs):

  python scripts/run_experiment.py -c configs/diffusion1d_stl.yaml \
      --set sweep.model.activation=['tanh','gelu'] \
      --set sweep.optim.epochs=[200,400,800] -j 2

Notes
-----
* This script uses PyYAML. If missing, install with ``pip install pyyaml``.
* The experiment runners are responsible for config validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import pkgutil
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


# YAML loader (with crisp errors) + minimal `include:` + deep merge


def _require_yaml():
    """Import PyYAML with a friendly error message."""

    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: pyyaml. Install it with:\n"
            "  pip install pyyaml\n"
            "or via the repo extras file:\n"
            "  pip install -r requirements-extra.txt"
        ) from e
    return yaml


def _expand_path(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        raise SystemExit(f"YAML file not found: {path}") from e


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings (b overrides a)."""

    out: dict[str, Any] = dict(a)
    for k, bv in b.items():
        av = out.get(k)
        if isinstance(av, dict) and isinstance(bv, dict):
            out[k] = _deep_merge(av, bv)
        else:
            out[k] = bv
    return out


def load_yaml(path: str) -> dict[str, Any]:
    """Load YAML with support for a lightweight, deep-merged ``include:``.

    ``include:`` may be a string path or a list of string paths. Included files
    are loaded first; the including file overrides them.
    """

    yaml = _require_yaml()

    def _load_one(p: str, stack: list[str]) -> dict[str, Any]:
        p = os.path.abspath(_expand_path(p))
        if p in stack:
            cycle = " -> ".join(stack + [p])
            raise SystemExit(f"YAML include cycle detected: {cycle}")

        stack.append(p)
        try:
            base_dir = os.path.dirname(p)
            raw = _read_text(p)
            data = yaml.safe_load(raw) or {}
            if not isinstance(data, dict):
                raise SystemExit(
                    f"Top-level YAML must be a mapping in {p} (got {type(data)})"
                )

            inc_val = data.pop("include", None)
            merged: dict[str, Any] = {}
            if inc_val:
                inc_paths = inc_val if isinstance(inc_val, list) else [inc_val]
                for inc in inc_paths:
                    if not isinstance(inc, str):
                        raise SystemExit(f"Each 'include' entry must be a string path (in {p})")
                    inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
                    merged = _deep_merge(merged, _load_one(inc_path, stack))

            merged = _deep_merge(merged, data)
            return merged
        finally:
            stack.pop()

    merged = _load_one(path, [])

    # Expand env vars and ~ inside strings *post-merge*.
    def _expand(obj: Any) -> Any:
        if isinstance(obj, str):
            return _expand_path(obj)
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_expand(v) for v in obj)
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        return obj

    return _expand(merged)


# Experiment discovery (prefers the package registry if present)


def _ensure_src_on_path() -> None:
    """Allow running from a git clone without installing the package."""

    try:
        import neural_pde_stl_strel  # type: ignore  # noqa: F401

        return
    except ModuleNotFoundError as e:
        # Only fall back to ./src when *this package* is missing.
        # If an internal import failed, surface the real error.
        if e.name != "neural_pde_stl_strel":
            raise

    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    src = os.path.join(repo_root, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def _repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))


@dataclass(frozen=True)
class ExpInfo:
    name: str
    module: str
    run_candidates: tuple[str, ...]


def _discover_via_registry() -> list[ExpInfo] | None:
    _ensure_src_on_path()
    try:
        import neural_pde_stl_strel.experiments as exps  # type: ignore
    except Exception:
        return None

    names_fn: Callable[[], list[str]] | None = getattr(exps, "names", None)  # type: ignore[assignment]
    if names_fn is None:
        return None

    infos: list[ExpInfo] = []
    for n in names_fn():
        module = f"neural_pde_stl_strel.experiments.{n}"
        infos.append(ExpInfo(name=n, module=module, run_candidates=(f"run_{n}", "run")))
    return sorted(infos, key=lambda i: i.name)


def discover_experiments() -> list[ExpInfo]:
    reg = _discover_via_registry()
    if reg is not None:
        return reg

    _ensure_src_on_path()
    try:
        pkg = importlib.import_module("neural_pde_stl_strel.experiments")
    except Exception as e:
        raise SystemExit(
            "Cannot import 'neural_pde_stl_strel.experiments'. If running from a clone,\n"
            "ensure the repository root's 'src/' is on PYTHONPATH or install the\n"
            "package (e.g., 'pip install -e .').\n\n"
            f"Original error: {e}"
        ) from e

    infos: list[ExpInfo] = []
    for modinfo in pkgutil.iter_modules(pkg.__path__):  # type: ignore[attr-defined]
        name = modinfo.name
        module = f"neural_pde_stl_strel.experiments.{name}"
        infos.append(ExpInfo(name=name, module=module, run_candidates=(f"run_{name}", "run")))
    return sorted(infos, key=lambda i: i.name)


def get_runner(exp_name: str):
    infos = {i.name: i for i in discover_experiments()}
    if exp_name not in infos:
        available = ", ".join(sorted(infos))
        raise SystemExit(f"Unknown experiment '{exp_name}'. Available: [{available}]")

    info = infos[exp_name]
    mod = importlib.import_module(info.module)
    for fn in info.run_candidates:
        if hasattr(mod, fn):
            return getattr(mod, fn)

    raise SystemExit(f"No runnable function found in {info.module}. Tried: {info.run_candidates}")


# Config utilities: dotted overrides + tiny sweep helper


def _parse_override(s: str) -> tuple[list[str], Any]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("--set expects KEY=VALUE (use quotes for lists)")
    key, val = s.split("=", 1)
    keys = [k for k in key.split(".") if k]
    if not keys:
        raise argparse.ArgumentTypeError(f"Invalid key in override: {s}")

    yaml = _require_yaml()
    value = yaml.safe_load(val)
    return keys, value


def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    cur[keys[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    for o in overrides:
        keys, value = _parse_override(o)
        _set_nested(cfg, keys, value)
    return cfg


def iter_sweep_cfgs(base: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (suffix, cfg) for each sweep combination.

    The sweep is driven by a top-level ``sweep:`` mapping in the config:

        sweep:
          model.activation: [tanh, gelu]
          optim.epochs: [200, 400]

    If no sweep is present, yields exactly one item: ("", base).
    """

    sweep_raw = base.get("sweep")
    if not sweep_raw:
        # Always deepcopy so the worker can safely annotate cfg.io without
        # mutating the caller's config object.
        yield "", deepcopy(base)
        return


    if not isinstance(sweep_raw, dict):
        raise SystemExit("'sweep' must be a mapping")

    # Support both of these equivalent sweep syntaxes:
    #
    #   sweep:
    #     model.activation: [tanh, gelu]
    #     optim.epochs: [200, 400]
    #
    # and (useful with CLI overrides like --set sweep.model.activation=[...]):
    #
    #   sweep:
    #     model:
    #       activation: [tanh, gelu]
    #     optim:
    #       epochs: [200, 400]
    def _flatten_sweep(prefix: str, obj: Any, out: dict[str, Any]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                k_str = str(k)
                new_prefix = f"{prefix}.{k_str}" if prefix else k_str
                _flatten_sweep(new_prefix, v, out)
        else:
            out[prefix] = obj

    sweep: dict[str, Any] = {}
    for k, v in sweep_raw.items():
        k_str = str(k)
        if isinstance(v, dict):
            _flatten_sweep(k_str, v, sweep)
        else:
            sweep[k_str] = v

    items: list[tuple[list[str], list[Any]]] = []
    for k, v in sweep.items():
        keys = [p for p in str(k).split(".") if p]
        if not keys:
            raise SystemExit(f"Invalid sweep key: {k!r}")
        if not isinstance(v, list) or len(v) == 0:
            raise SystemExit(f"Each sweep entry must be a non-empty list: {k}")
        items.append((keys, v))

    # Cartesian product.
    from itertools import product

    for combo in product(*[vals for _, vals in items]):
        cfg = deepcopy(base)
        parts: list[str] = []
        for (keys, _), v in zip(items, combo, strict=True):
            _set_nested(cfg, keys, v)
            parts.append(f"{'.'.join(keys)}={_compact_value(v)}")
        yield "__".join(parts), cfg


def _compact_value(v: Any) -> str:
    """Compact, stable string form for sweep values."""

    if v is None:
        return "none"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v
    # Fallback: short hash of a JSON-ish representation.
    try:
        s = json.dumps(v, sort_keys=True, default=str)
    except Exception:
        s = repr(v)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


# Seeding and run directory handling


def try_set_seed(seed: int | None) -> None:
    """Best-effort seeding across random/numpy/torch.

    Experiments may still introduce nondeterminism (e.g., multithreading,
    non-deterministic ops). The goal here is to provide a sensible baseline.
    """

    if seed is None:
        return

    try:
        import random

        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # no-op if CUDA absent
        if hasattr(torch, "use_deterministic_algorithms"):
            # Keep the default fast/compatible behavior; experiments can opt-in
            # to stricter determinism if desired.
            torch.use_deterministic_algorithms(False)
    except Exception:
        pass


_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _slugify(s: str, *, max_len: int = 80) -> str:
    """Convert to a filesystem-friendly slug."""

    s = s.strip()
    if not s:
        return "run"
    s = _SLUG_RE.sub("-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        return "run"
    return s[:max_len]


def _cfg_tag(cfg: dict[str, Any]) -> str:
    """Resolve the run tag from a config.

    We prefer a top-level ``tag`` when present, but fall back to ``io.tag`` so
    YAML configs can stay compact while still producing descriptive run
    directories and stable alias filenames.
    """

    tag = cfg.get("tag", None)
    if tag is None:
        io_cfg = cfg.get("io", {})
        if isinstance(io_cfg, dict):
            tag = io_cfg.get("tag", None)

    tag_str = "" if tag is None else str(tag).strip()
    return tag_str or "run"


def _unique_timestamp() -> str:
    # Include microseconds to avoid collisions in parallel runs.
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def make_run_dir(cfg: dict[str, Any], *, sweep_suffix: str | None = None) -> tuple[str, str]:
    """Create a new run directory and update cfg.io.* for reproducibility.

    Returns (run_dir, results_root).
    """

    io_cfg = cfg.setdefault("io", {})
    if not isinstance(io_cfg, dict):
        raise SystemExit("Config key 'io' must be a mapping if present")

    # Treat io.results_dir (or top-level results_dir) as the *results root*.
    results_root = str(io_cfg.get("results_dir", cfg.get("results_dir", "results")))
    results_root = _expand_path(results_root)
    if not os.path.isabs(results_root):
        # Interpret relative output dirs relative to the repository root (not CWD).
        results_root = os.path.join(_repo_root(), results_root)
    results_root = os.path.abspath(results_root)
    os.makedirs(results_root, exist_ok=True)

    exp = str(cfg.get("experiment", "exp")).strip() or "exp"
    tag = _cfg_tag(cfg)

    dir_exp = _slugify(exp, max_len=60)
    dir_tag = _slugify(tag, max_len=60)
    ts = _unique_timestamp()

    suffix_part = ""
    if sweep_suffix:
        suffix_slug = _slugify(sweep_suffix, max_len=60)
        suffix_hash = hashlib.sha1(sweep_suffix.encode("utf-8")).hexdigest()[:8]
        suffix_part = f"__{suffix_slug}--{suffix_hash}" if suffix_slug else f"__{suffix_hash}"

    base_name = f"{dir_exp}--{dir_tag}--{ts}{suffix_part}"
    base_path = os.path.join(results_root, base_name)

    # Be robust if multiple processes collide (same timestamp).
    run_dir = base_path
    i = 1
    while True:
        try:
            os.makedirs(run_dir)
            break
        except FileExistsError:
            i += 1
            run_dir = f"{base_path}-{i}"

    # Record both the root and the run directory.
    io_cfg["results_root"] = results_root
    io_cfg["run_dir"] = run_dir

    # Ensure experiment artifacts land in the run directory.
    io_cfg["results_dir"] = run_dir

    return run_dir, results_root


# JSON helpers


def _json_default(obj: Any) -> Any:  # noqa: ANN401 - json hook
    """Best-effort JSON serializer for common ML types."""

    # pathlib.Path
    try:
        from pathlib import Path

        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass

    # numpy types
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # torch types
    try:
        import torch  # type: ignore

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    # Fallback: repr
    return repr(obj)


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)


def dump_effective_config(run_dir: str, cfg: dict[str, Any]) -> None:
    yaml = _require_yaml()
    path = os.path.join(run_dir, "config.effective.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# Environment capture (hardware + software)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _cpu_model() -> str | None:
    """Best-effort CPU model string."""

    # Linux: /proc/cpuinfo
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    k = k.strip().lower()
                    if k not in {"model name", "hardware", "processor"}:
                        continue
                    v = v.strip()
                    # Guard against the common "processor: 0" entry which is
                    # an index, not a model string.
                    if k == "processor" and v.isdigit():
                        continue
                    return v or None
    except Exception:
        pass

    # macOS: sysctl
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            out = out.strip()
            return out or None
    except Exception:
        pass

    # Fallback
    try:
        out = platform.processor().strip()
        return out or None
    except Exception:
        return None


def _total_ram_bytes() -> int | None:
    """Best-effort total physical RAM in bytes."""

    # Linux: /proc/meminfo
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            kb = int(parts[1])
                            return kb * 1024
    except Exception:
        pass

    # POSIX: sysconf
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(pages, int) and page_size > 0 and pages > 0:
            return int(page_size) * int(pages)
    except Exception:
        pass

    # macOS: sysctl
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return int(out.strip())
    except Exception:
        pass

    # Windows: GlobalMemoryStatusEx via ctypes
    try:
        if platform.system() == "Windows":
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
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
    except Exception:
        pass

    return None


def _env_summary() -> dict[str, Any]:
    """Collect a dependency-tolerant environment/hardware snapshot."""

    info: dict[str, Any] = {
        "time_utc": _utc_now_iso(),
        "hostname": socket.gethostname(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {
            "count_logical": os.cpu_count(),
            "model": _cpu_model(),
        },
        "memory": {},
        "env": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
            if os.environ.get(k) is not None
        },
    }

    mem = _total_ram_bytes()
    if mem is not None:
        info["memory"] = {
            "total_bytes": mem,
            "total_gib": round(mem / (1024**3), 3),
        }

    # Optional: torch / CUDA
    try:
        import torch  # type: ignore

        torch_info: dict[str, Any] = {
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        }
        if torch.cuda.is_available():
            devices: list[dict[str, Any]] = []
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    devices.append(
                        {
                            "index": i,
                            "name": torch.cuda.get_device_name(i),
                            "capability": tuple(torch.cuda.get_device_capability(i)),
                            "total_memory_bytes": int(getattr(props, "total_memory", 0)),
                        }
                    )
                except Exception:
                    devices.append({"index": i, "name": torch.cuda.get_device_name(i)})
            torch_info["devices"] = devices
            torch_info["num_gpus"] = len(devices)
        else:
            torch_info["num_gpus"] = 0
        info["torch"] = torch_info
    except Exception:
        pass

    # Best-effort git info
    try:
        root = _repo_root()
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
        info["git"] = {"rev": sha, "branch": branch, "dirty": dirty}
    except Exception:
        pass

    return info


# Artifact aliasing (stable files in results root)


def _link_or_copy(src: str, dst: str) -> str:
    """Create/replace ``dst`` pointing at ``src``.

    Prefers a hardlink (no duplication). Falls back to a file copy.

    Returns the method used: "hardlink", "copy", or "skip" (if src == dst).
    """

    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)
    if src_abs == dst_abs:
        return "skip"

    # Remove existing dst.
    if os.path.lexists(dst_abs):
        try:
            os.unlink(dst_abs)
        except IsADirectoryError:
            shutil.rmtree(dst_abs)

    # Prefer hardlinks (no duplication) when possible.
    try:
        os.link(src_abs, dst_abs)
        return "hardlink"
    except Exception:
        pass

    shutil.copy2(src_abs, dst_abs)
    return "copy"


def alias_run_artifacts_to_root(
    *,
    run_dir: str,
    results_root: str,
    exp: str,
    tag: str,
) -> list[dict[str, str]]:
    """Alias run artifacts to stable filenames in the results root.

    Returns a list of alias records (src, dst, method).
    """

    created: list[dict[str, str]] = []
    prefix = f"{exp}_{tag}"

    try:
        names = sorted(os.listdir(run_dir))
    except Exception:
        return created

    for name in names:
        if not name.startswith(prefix):
            continue
        src = os.path.join(run_dir, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(results_root, name)
        try:
            method = _link_or_copy(src, dst)
            created.append({"src": src, "dst": dst, "method": method})
        except Exception:
            # Aliasing is best-effort; the run directory remains the source of truth.
            pass

    return created


# CLI


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generic runner for neural-pde-stl-strel experiments (YAML-driven).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", "-c", required=False, help="Path to YAML config.")
    p.add_argument("--list", action="store_true", help="List available experiments and exit.")
    p.add_argument("--describe", metavar="EXP", help="Describe one experiment and exit.")

    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config with KEY=VALUE (VALUE parsed as YAML). Repeatable.",
    )
    p.add_argument("--show-config", action="store_true", help="Print resolved config and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")

    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Parallel worker processes for sweeps (use 1 to run sequentially).",
    )
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="If a run fails, continue the sweep instead of aborting.",
    )
    p.add_argument(
        "--alias",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Whether to create stable alias files in the results root for artifacts "
            "in the run directory. 'auto' aliases only when there is a single run."
        ),
    )

    return p


# Worker (picklable for multiprocessing)


def _run_worker(
    exp: str,
    subcfg: dict[str, Any],
    *,
    sweep_suffix: str | None,
    argv: list[str],
    config_path: str,
    alias_mode: str,
    total_runs: int,
) -> dict[str, Any]:
    """Run a single experiment instance. Returns a small result dict."""

    start_perf = time.perf_counter()
    start_utc = _utc_now_iso()
    run_dir: str | None = None
    results_root: str | None = None
    out: Any = None

    try:
        try_set_seed(subcfg.get("seed"))
        # Ensure the experiment name is present in the config passed to the runner.
        subcfg["experiment"] = exp

        run_dir, results_root = make_run_dir(subcfg, sweep_suffix=sweep_suffix)

        # Persist provenance early.
        dump_effective_config(run_dir, subcfg)
        _write_json(os.path.join(run_dir, "env.json"), _env_summary())

        runner = get_runner(exp)
        out = runner(subcfg)  # type: ignore[misc]
        elapsed_s = time.perf_counter() - start_perf

        # Persist metrics if the runner returned a mapping.
        if isinstance(out, dict):
            _write_json(os.path.join(run_dir, "metrics.json"), out)

        # Convenience aliases in the results root.
        aliased: list[dict[str, str]] = []
        if results_root is not None:
            do_alias = (
                alias_mode == "always"
                or (alias_mode == "auto" and total_runs == 1)
            )
            if do_alias:
                aliased = alias_run_artifacts_to_root(
                    run_dir=run_dir,
                    results_root=results_root,
                    exp=exp,
                    tag=_cfg_tag(subcfg),
                )

        # Always write run.json.
        _write_json(
            os.path.join(run_dir, "run.json"),
            {
                "ok": True,
                "experiment": exp,
                "tag": _cfg_tag(subcfg),
                "sweep_suffix": sweep_suffix or "",
                "config_path": config_path,
                "argv": argv,
                "start_time_utc": start_utc,
                "end_time_utc": _utc_now_iso(),
                "elapsed_s": elapsed_s,
                "run_dir": run_dir,
                "results_root": results_root,
                "aliased": aliased,
                "output": out,
            },
        )

        return {
            "ok": True,
            "run_dir": run_dir,
            "results_root": results_root,
            "elapsed_s": elapsed_s,
            "out": out,
        }

    except Exception as e:
        elapsed_s = time.perf_counter() - start_perf
        tb = traceback.format_exc()

        # Best-effort error logging.
        if run_dir is not None:
            try:
                with open(os.path.join(run_dir, "error.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{e!r}\n\n{tb}")
            except Exception:
                pass

            try:
                _write_json(
                    os.path.join(run_dir, "run.json"),
                    {
                        "ok": False,
                        "experiment": exp,
                        "tag": _cfg_tag(subcfg),
                        "sweep_suffix": sweep_suffix or "",
                        "config_path": config_path,
                        "argv": argv,
                        "start_time_utc": start_utc,
                        "end_time_utc": _utc_now_iso(),
                        "elapsed_s": elapsed_s,
                        "run_dir": run_dir,
                        "results_root": results_root,
                        "error": repr(e),
                        "traceback": tb,
                        "output": out,
                    },
                )
            except Exception:
                pass

        return {
            "ok": False,
            "error": repr(e),
            "traceback": tb,
            "elapsed_s": elapsed_s,
            "run_dir": run_dir,
        }


# Entry point


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    # --list and --describe do not require a config.
    if args.list:
        try:
            _ensure_src_on_path()
            import neural_pde_stl_strel.experiments as exps  # type: ignore

            print(exps.about())  # type: ignore[attr-defined]
        except Exception:
            infos = discover_experiments()
            print("Available experiments:")
            for i in infos:
                print(f"  - {i.name}  (module: {i.module})")
        return

    if args.describe:
        _ensure_src_on_path()
        try:
            import neural_pde_stl_strel.experiments as exps  # type: ignore

            if hasattr(exps, "describe"):
                print(exps.describe(args.describe))  # type: ignore[attr-defined]
            else:
                info = {i.name: i for i in discover_experiments()}.get(args.describe)
                if not info:
                    raise SystemExit(f"Unknown experiment: {args.describe}")
                mod = importlib.import_module(info.module)
                print((mod.__doc__ or "").strip() or f"No description for {args.describe}")
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"Failed to describe '{args.describe}': {e}") from e
        return

    if not args.config:
        raise SystemExit("--config is required unless using --list/--describe")

    # Resolve config path:
    #  * if the given path exists as-is (relative to CWD), use it
    #  * else try relative to repo root (handy when launching from elsewhere)
    cfg_arg = _expand_path(args.config)
    cfg_cwd = os.path.abspath(cfg_arg)
    cfg_repo = os.path.abspath(os.path.join(_repo_root(), cfg_arg))
    if os.path.isfile(cfg_cwd):
        config_path = cfg_cwd
    elif os.path.isfile(cfg_repo):
        config_path = cfg_repo
    else:
        raise SystemExit(f"Config file not found: {args.config}")

    cfg = load_yaml(config_path)
    cfg = apply_overrides(cfg, args.overrides)

    # Experiment name: allow inference from filename.
    exp = str(cfg.get("experiment", "")).strip().lower()
    if not exp:
        base = os.path.basename(os.path.splitext(config_path)[0])
        exp = base.split("_")[0].lower()
        cfg["experiment"] = exp

    # Print the resolved config. By default this exits; if combined with
    # --dry-run, we also print the run plan below.
    if args.show_config:
        yaml = _require_yaml()
        print("# --- Resolved config ---")
        print(yaml.safe_dump(cfg, sort_keys=False))
        if not args.dry_run:
            return 0

    # Materialize sweep items up-front.
    sweep_items = list(iter_sweep_cfgs(cfg))
    total = len(sweep_items)
    if total == 0:
        print("No runs were scheduled (empty sweep?).")
        return 0

    if args.dry_run:
        print(f"[DRY-RUN] {total} run(s) planned:")
        for suffix, subcfg in sweep_items:
            tag = _cfg_tag(subcfg)
            print(f"  * {exp}  tag='{tag}'  sweep='{suffix or '-'}'")
        return

    # Build worker arguments.
    argv_list = list(sys.argv if argv is None else [sys.argv[0], *argv])

    # Sequential (default) or parallel execution.
    jobs = max(1, int(args.jobs))
    ran_any = False

    if jobs == 1 or total == 1:
        for idx, (suffix, subcfg) in enumerate(sweep_items, start=1):
            print(f"[{idx}/{total}] {exp}  tag='{_cfg_tag(subcfg)}'  sweep='{suffix or '-'}'")
            res = _run_worker(
                exp,
                subcfg,
                sweep_suffix=(suffix or None),
                argv=argv_list,
                config_path=config_path,
                alias_mode=args.alias,
                total_runs=total,
            )
            if res.get("ok"):
                print(f"  -> done in {res['elapsed_s']:.1f}s -> {res['run_dir']}")
                ran_any = True
            else:
                print(f"  -> FAILED: {res.get('error')}")
                if not args.keep_going:
                    raise SystemExit("Aborting on failure. Use --keep-going to continue.")
    else:
        print(f"Launching {total} run(s) with {jobs} worker process(es)...")
        # Use 'spawn' for better isolation with scientific/PyTorch stacks.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            fut2desc = {
                ex.submit(
                    _run_worker,
                    exp,
                    subcfg,
                    sweep_suffix=(suffix or None),
                    argv=argv_list,
                    config_path=config_path,
                    alias_mode=args.alias,
                    total_runs=total,
                ): (suffix, subcfg)
                for suffix, subcfg in sweep_items
            }

            for i, fut in enumerate(as_completed(fut2desc), start=1):
                suffix, subcfg = fut2desc[fut]
                print(f"[{i}/{total}] {exp}  tag='{_cfg_tag(subcfg)}'  sweep='{suffix or '-'}'")
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"  -> FAILED (executor): {e!r}")
                    if not args.keep_going:
                        raise
                    continue

                if res.get("ok"):
                    print(f"  -> done in {res['elapsed_s']:.1f}s -> {res['run_dir']}")
                    ran_any = True
                else:
                    print(f"  -> FAILED: {res.get('error')}")
                    if not args.keep_going:
                        raise SystemExit("Aborting on failure. Use --keep-going to continue.")

    if not ran_any:
        print("No runs were executed.")


if __name__ == "__main__":  # pragma: no cover
    main()
