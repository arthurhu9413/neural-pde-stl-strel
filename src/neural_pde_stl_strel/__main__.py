"""Command-line entry point for :mod:`neural_pde_stl_strel`.

This module is intentionally lightweight: it only imports small helper
functions from the top-level package so that ``python -m neural_pde_stl_strel``
works even when heavy optional dependencies (Neuromancer / PhysicsNeMo /
TorchPhysics, RTAMT / MoonLight / SpaTiaL, ...) are not installed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, cast

# Import only the *lightweight* utilities from the package.
# These do not pull heavy optional dependencies on import.
from . import __version__, _MIN_REQUIRED_VERSION, about as _about_text, optional_dependencies
from ._versioning import version_satisfies_minimum

_JSON_SCHEMA_VERSION = 1


# Helpers


def _table(rows: Sequence[Sequence[str]], headers: Sequence[str] | None = None) -> str:
    """Return a compact monospace table without external dependencies.

    Each cell is left-aligned; column widths are derived from the widest cell
    content. This helper is intentionally small so that the CLI stays portable.
    """

    table_rows: list[list[str]] = []
    if headers is not None:
        table_rows.append([str(h) for h in headers])

    for row in rows:
        table_rows.append([str(cell) for cell in row])

    if not table_rows:
        return ""

    n_cols = max(len(r) for r in table_rows)
    for r in table_rows:
        if len(r) < n_cols:
            r.extend([""] * (n_cols - len(r)))

    widths = [max(len(r[c]) for r in table_rows) for c in range(n_cols)]

    def fmt_row(row: Sequence[str]) -> str:
        return "  ".join(row[c].ljust(widths[c]) for c in range(n_cols))

    lines: list[str] = []
    for i, row in enumerate(table_rows):
        lines.append(fmt_row(row))
        if headers is not None and i == 0:
            lines.append("  ".join("-" * w for w in widths))

    return "\n".join(lines)


def _env_summary() -> dict[str, str]:
    """Return a small, dependency-free environment summary."""

    impl = platform.python_implementation()
    pyver = platform.python_version()
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    proc = platform.processor() or machine
    cpu_count = os.cpu_count()

    return {
        "python": f"{impl} {pyver}",
        "platform": f"{system} {release} ({machine})",
        "processor": proc,
        "cpus": str(cpu_count or ""),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }


def _emit_json(obj: object) -> None:
    """Print *obj* as stable, pretty JSON."""

    print(json.dumps(obj, indent=2, sort_keys=True))


def _warn(message: str) -> None:
    """Emit a warning message to stderr (keeps JSON output clean)."""

    print(f"warning: {message}", file=sys.stderr)


def _requirement_groups(available_names: Iterable[str]) -> dict[str, set[str]]:
    """Return named requirement groups intersected with *available_names*."""

    names = set(available_names)

    def keep(want: set[str]) -> set[str]:
        return {n for n in want if n in names}

    return {
        # Keep this aligned with the package's declared base runtime dependencies.
        "core": keep({"numpy", "pyyaml"}),
        # PhysicsNeMo rename note: include BOTH 'physicsnemo' and legacy 'modulus'
        # as satisfying the *framework present* intent.
        "physics": keep({"neuromancer", "physicsnemo", "modulus", "torchphysics"}),
        "stl": keep({"rtamt", "moonlight", "spatial_spec"}),
        "all": set(names),
    }


def _version_satisfies_minimum(found_version: str | None, minimum_version: str | None) -> bool:
    """Return whether *found_version* satisfies *minimum_version*.

    Prefer :mod:`packaging` when available, but retain a lightweight built-in
    fallback so clean installs still enforce the repository's supported floors.
    """

    return version_satisfies_minimum(found_version, minimum_version)


def _default_policy_for_group(group: str, fallback: str) -> str:
    """Return the implicit policy for *group* when the user omitted one.

    ``physics`` and ``stl`` are naturally satisfied by *any* installed member,
    while ``core`` and ``all`` are only intuitive when every member is present.
    Respect an explicit ``--policy all`` override, but otherwise upgrade the
    default ``any`` policy for the stricter groups.
    """

    if fallback == "any" and group in {"core", "all"}:
        return "all"
    return fallback


def _evaluate_requirements(
    require_specs: Sequence[str],
    report: Mapping[str, Mapping[str, Any]],
    *,
    default_policy: str,
) -> tuple[int, list[dict[str, Any]], list[str]]:
    """Evaluate ``--require`` specs against an ``optional_dependencies`` report.

    Returns
    -------
    (rc, evaluations, warnings)
        ``rc`` is 0 on success, 1 if requirements are unmet, and 2 for invalid
        input (unknown group/policy).
    """

    groups = _requirement_groups(report.keys())
    known_groups = ", ".join(sorted(groups.keys()))
    known_policies = {"any", "all"}

    warnings: list[str] = []
    evaluations: list[dict[str, Any]] = []
    invalid = False
    unmet = False
    version_warning_seen: set[tuple[str, str, str]] = set()

    for raw in require_specs:
        spec = (raw or "").strip()
        if not spec:
            continue

        if ":" in spec:
            group_part, policy_part = spec.split(":", 1)
            group = group_part.strip().lower()
            policy = policy_part.strip().lower()
        else:
            group = spec.strip().lower()
            policy = _default_policy_for_group(group, default_policy)

        if group not in groups:
            warnings.append(f"unknown group '{group}' (known: {known_groups})")
            invalid = True
            continue
        if policy not in known_policies:
            warnings.append(
                f"unknown policy '{policy}' in spec '{spec}' (expected: any, all)"
            )
            invalid = True
            continue

        want = groups[group]
        have: set[str] = set()
        for name in want:
            item = report.get(name, {})
            if not bool(item.get("available")):
                continue

            found_version = item.get("version")
            found_version_str = None if found_version is None else str(found_version)
            minimum_version = _MIN_REQUIRED_VERSION.get(name)
            if minimum_version and found_version_str and not _version_satisfies_minimum(
                found_version_str, minimum_version
            ):
                warning_key = (name, found_version_str, minimum_version)
                if warning_key not in version_warning_seen:
                    warnings.append(
                        f"'{name}' version {found_version_str} is below the required minimum "
                        f"{minimum_version}; treating it as unavailable for requirement checks."
                    )
                    version_warning_seen.add(warning_key)
                continue

            have.add(name)

        if policy == "any":
            satisfied = bool(have)
            missing = sorted(want) if not satisfied else []
        else:  # policy == "all"
            missing_set = want - have
            satisfied = not missing_set
            missing = sorted(missing_set)

        evaluations.append(
            {
                "spec": spec,
                "group": group,
                "policy": policy,
                "want": sorted(want),
                "have": sorted(have),
                "missing": missing,
                "satisfied": satisfied,
            }
        )
        if not satisfied:
            unmet = True

    rc = 2 if invalid else (1 if unmet else 0)
    return rc, evaluations, warnings


# Commands


def cmd_about(args: argparse.Namespace) -> int:
    if getattr(args, "brief", False):
        # Keep this one-liner *fast* and stable for demos.
        print(f"neural_pde_stl_strel {__version__}")
        return 0

    info = _about_text()

    if getattr(args, "json", False):
        report = optional_dependencies(include_pip_hints=True)
        payload = {
            "schema": _JSON_SCHEMA_VERSION,
            "tool": "neural_pde_stl_strel",
            "command": "about",
            "version": __version__,
            "env": _env_summary(),
            "optional_deps": report,
            "about": info,
        }
        _emit_json(payload)
        return 0

    print(info)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    report = optional_dependencies(
        refresh=args.refresh,
        include_pip_hints=not args.no_pip_hints,
    )
    env = _env_summary()
    requested_specs = list(args.require)
    require_specs = requested_specs or ["core"]

    names = sorted(report.keys())
    rows: list[list[str]] = []
    for name in names:
        item = report[name]
        avail = bool(item.get("available"))
        ver = str(item.get("version") or "-")
        pip_hint = "" if avail else str(item.get("pip") or "")
        rows.append([name, "yes" if avail else "no", ver, pip_hint])

    notes: list[str] = []
    if report.get("modulus", {}).get("available") and not report.get(
        "physicsnemo", {}
    ).get("available"):
        notes.append(
            "'modulus' is installed; NVIDIA renamed it to 'PhysicsNeMo' "
            "(import: 'physicsnemo')."
        )

    req_rc, req_evals, req_warnings = _evaluate_requirements(
        require_specs,
        report,
        default_policy=args.policy,
    )
    rc = req_rc

    if args.json:
        payload: dict[str, Any] = {
            "schema": _JSON_SCHEMA_VERSION,
            "tool": "neural_pde_stl_strel",
            "command": "doctor",
            "version": __version__,
            "env": env,
            "optional_deps": report,
            "notes": notes,
            "requirements": {
                "default_policy": args.policy,
                "specs": require_specs,
                "requested_specs": requested_specs,
                "evaluations": req_evals,
            },
            "warnings": req_warnings,
            "exit_code": rc,
        }
        _emit_json(payload)

        # Keep stdout *only JSON*; warnings go to stderr.
        for w in req_warnings:
            _warn(w)

        return rc

    # Human-readable output -------------------------------------------------
    print(f"neural_pde_stl_strel {__version__}")
    print("Environment:")
    print(_table([[k, v] for k, v in env.items()], headers=["Key", "Value"]))
    print()
    print("Dependency probes:")
    print(_table(rows, headers=["name", "ok", "version", "pip hint if missing"]))

    if notes:
        for n in notes:
            print(f"\nNOTE: {n}")

    if req_warnings:
        for w in req_warnings:
            _warn(w)

    if req_evals:
        print()
        print("Requirements:")
        req_rows: list[list[str]] = []
        for e in req_evals:
            req_rows.append(
                [
                    str(e["group"]),
                    str(e["policy"]),
                    "yes" if e["satisfied"] else "no",
                    ", ".join(e["missing"]) if e["missing"] else "-",
                ]
            )
        print(_table(req_rows, headers=["group", "policy", "ok", "missing"]))

    return rc


def cmd_pip(args: argparse.Namespace) -> int:
    report = optional_dependencies(include_pip_hints=True)

    missing_names = [name for name, item in report.items() if not item.get("available")]
    cmds: list[str] = []
    seen: set[str] = set()
    for name in missing_names:
        pip_hint = report.get(name, {}).get("pip")
        if isinstance(pip_hint, str) and pip_hint and pip_hint not in seen:
            seen.add(pip_hint)
            cmds.append(pip_hint)

    if args.json:
        _emit_json(
            {
                "schema": _JSON_SCHEMA_VERSION,
                "tool": "neural_pde_stl_strel",
                "command": "pip",
                "version": __version__,
                "missing": missing_names,
                "pip_install": cmds,
                "count": len(cmds),
            }
        )
        return 0

    if not cmds:
        print("All probed dependencies appear to be installed.")
        return 0

    print("Run the following to install missing dependencies:")
    print("\n".join(cmds))
    return 0


def cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


# Parser / dispatch


def _make_parser() -> argparse.ArgumentParser:
    examples = (
        "Examples:\n"
        "  python -m neural_pde_stl_strel\n"
        "  python -m neural_pde_stl_strel --about\n"
        "  python -m neural_pde_stl_strel --version\n"
        "  python -m neural_pde_stl_strel --brief\n"
        "  python -m neural_pde_stl_strel doctor\n"
        "  python -m neural_pde_stl_strel doctor --require physics stl\n"
        "  python -m neural_pde_stl_strel doctor --json\n"
    )

    p = argparse.ArgumentParser(
        prog="neural_pde_stl_strel",
        description=(
            "neural-pde-stl-strel CLI -- fast environment checks,\n"
            "installation hints, and a concise 'about' report.\n\n"
            f"{examples}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Convenience flags when no explicit subcommand is provided.
    alias = p.add_mutually_exclusive_group()
    alias.add_argument(
        "--about",
        dest="_alias_about",
        action="store_true",
        help="Show the default about() report explicitly",
    )
    alias.add_argument(
        "--version",
        dest="_alias_version",
        action="store_true",
        help="Print package version (alias for 'version')",
    )
    p.add_argument(
        "--brief",
        action="store_true",
        help="Print a one-line summary (alias for 'about --brief')",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON payload (alias for 'about --json')",
    )

    # No-subcommand defaults to `about`.
    p.set_defaults(_fn=cmd_about)

    sub = p.add_subparsers(dest="cmd")

    # about ----------------------------------------------------------------
    sp = sub.add_parser("about", help="Show package about() and dependency summary")
    sp.add_argument("--brief", action="store_true", help="Print a one-line summary")
    sp.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON payload (includes env + optional-deps)",
    )
    sp.set_defaults(_fn=cmd_about)

    # doctor ---------------------------------------------------------------
    sp = sub.add_parser("doctor", help="Inspect environment and validate dependency probes")
    sp.add_argument(
        "--refresh",
        action="store_true",
        help="Rescan environment (ignore cached probe)",
    )
    sp.add_argument(
        "--no-pip-hints",
        action="store_true",
        help="Hide pip install suggestions",
    )
    sp.add_argument(
        "--require",
        nargs="*",
        default=[],
        help=(
            "Require groups to be satisfied; space-separated, e.g. 'physics stl'.\n"
            "Defaults to 'core' when omitted. Override policy per-group via 'group:POLICY'.\n"
            "Groups: core, physics, stl, all. Policies: any or all.\n"
            "Unqualified core/all default to all; physics/stl default to any."
        ),
    )
    sp.add_argument(
        "--policy",
        choices=["any", "all"],
        default="any",
        help="Default requirement policy for groups",
    )
    sp.add_argument("--json", action="store_true", help="Emit JSON payload")
    sp.set_defaults(_fn=cmd_doctor)

    # pip ------------------------------------------------------------------
    sp = sub.add_parser(
        "pip",
        help="Print pip install commands for missing dependencies",
    )
    sp.add_argument("--json", action="store_true", help="Emit JSON payload")
    sp.set_defaults(_fn=cmd_pip)

    # version --------------------------------------------------------------
    sp = sub.add_parser("version", help="Print package version")
    sp.set_defaults(_fn=cmd_version)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if getattr(args, "brief", False) and getattr(args, "json", False):
        parser.error("--brief and --json are mutually exclusive")
    if getattr(args, "_alias_version", False) and (
        getattr(args, "brief", False) or getattr(args, "json", False)
    ):
        parser.error("--version cannot be combined with --brief or --json")

    if getattr(args, "_alias_version", False):
        return int(cmd_version(args))
    if getattr(args, "_alias_about", False):
        return int(cmd_about(args))

    fn = cast(Callable[[argparse.Namespace], int], getattr(args, "_fn"))
    return int(fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
