from __future__ import annotations

"""Lightweight version-comparison helpers for minimal environments.

The packaged CLI and optional-dependency guards should keep enforcing minimum
supported versions even in clean installs that do not happen to include the
third-party :mod:`packaging` distribution.

We therefore prefer :class:`packaging.version.Version` when available, but fall
back to a conservative parser that handles the common version forms seen in this
repository's dependency probes:

- plain release versions (``1.24``, ``6.0.3``)
- local build suffixes (``2.10.0+cpu``)
- common prereleases (``1.24.0rc1``, ``2.0b2``)
- post releases (``1.24.post1``)

If a version string is too unusual for the fallback parser, callers stay
permissive rather than failing spuriously.
"""

import re
from typing import Final

_PRE_RELEASE_PREFIXES: Final[tuple[str, ...]] = (
    "dev",
    "a",
    "alpha",
    "b",
    "beta",
    "c",
    "rc",
    "pre",
    "preview",
)
_POST_RELEASE_PREFIXES: Final[tuple[str, ...]] = ("post", "rev", "r")
_SUFFIX_TRIM_CHARS: Final[str] = ".-_"


def _leading_int(text: str) -> int:
    """Return the leading decimal integer from *text*, defaulting to ``0``."""

    match = re.match(r"(\d+)", text)
    return int(match.group(1)) if match is not None else 0


def _parse_suffix_fallback(suffix: str) -> tuple[int, int] | None:
    """Return a comparable ``(rank, number)`` tuple for a version suffix.

    Higher ranks indicate newer releases *for identical numeric release parts*:

    - prereleases (``dev``, ``a``, ``b``, ``rc``) rank below the final release
    - final releases rank in the middle
    - post releases rank above the final release

    ``None`` means the suffix is too unusual for the fallback parser.
    """

    cleaned = suffix.strip().lower()
    if not cleaned:
        return (4, 0)  # final release

    cleaned = cleaned.lstrip(_SUFFIX_TRIM_CHARS)
    if not cleaned:
        return (4, 0)

    pre_rank = {
        "dev": 0,
        "a": 1,
        "alpha": 1,
        "b": 2,
        "beta": 2,
        "c": 3,
        "rc": 3,
        "pre": 3,
        "preview": 3,
    }
    for prefix in _PRE_RELEASE_PREFIXES:
        if cleaned.startswith(prefix):
            return (pre_rank[prefix], _leading_int(cleaned[len(prefix) :]))

    for prefix in _POST_RELEASE_PREFIXES:
        if cleaned.startswith(prefix):
            return (5, _leading_int(cleaned[len(prefix) :]))

    return None


def _parse_version_fallback(version: str) -> tuple[int, tuple[int, ...], tuple[int, int]] | None:
    """Best-effort fallback parser for version comparison.

    Returns ``(epoch, release_parts, suffix_rank)`` on success, else ``None``.
    Local-version metadata (``+...``) is ignored for minimum-version checks.
    """

    raw = str(version).strip()
    if not raw:
        return None

    public = raw.split("+", 1)[0].strip()
    if not public:
        return None

    epoch = 0
    if "!" in public:
        epoch_part, public = public.split("!", 1)
        if not epoch_part.isdigit():
            return None
        epoch = int(epoch_part)

    match = re.match(r"^(?P<release>\d+(?:\.\d+)*)(?P<suffix>.*)$", public)
    if match is None:
        return None

    release_parts = tuple(int(part) for part in match.group("release").split("."))
    suffix_rank = _parse_suffix_fallback(match.group("suffix"))
    if suffix_rank is None:
        return None

    return epoch, release_parts, suffix_rank


def _compare_versions_fallback(found_version: str, minimum_version: str) -> int | None:
    """Compare two versions using the built-in fallback parser.

    Returns ``-1`` / ``0`` / ``1`` or ``None`` if comparison is not possible.
    """

    found = _parse_version_fallback(found_version)
    minimum = _parse_version_fallback(minimum_version)
    if found is None or minimum is None:
        return None

    found_epoch, found_release, found_suffix = found
    minimum_epoch, minimum_release, minimum_suffix = minimum

    if found_epoch != minimum_epoch:
        return -1 if found_epoch < minimum_epoch else 1

    width = max(len(found_release), len(minimum_release))
    padded_found = found_release + (0,) * (width - len(found_release))
    padded_minimum = minimum_release + (0,) * (width - len(minimum_release))
    if padded_found != padded_minimum:
        return -1 if padded_found < padded_minimum else 1

    if found_suffix != minimum_suffix:
        return -1 if found_suffix < minimum_suffix else 1

    return 0


def version_satisfies_minimum(found_version: str | None, minimum_version: str | None) -> bool:
    """Return whether *found_version* satisfies *minimum_version*.

    The helper prefers :mod:`packaging` when available, but retains minimum
    version enforcement via a small fallback parser when running in minimal
    environments that do not install ``packaging`` separately.
    """

    if not minimum_version or not found_version:
        return True

    try:
        from packaging.version import InvalidVersion, Version
    except Exception:
        cmp_result = _compare_versions_fallback(found_version, minimum_version)
        return True if cmp_result is None else cmp_result >= 0

    try:
        return Version(found_version) >= Version(minimum_version)
    except InvalidVersion:
        cmp_result = _compare_versions_fallback(found_version, minimum_version)
        return True if cmp_result is None else cmp_result >= 0
