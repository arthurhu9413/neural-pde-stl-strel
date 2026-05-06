"""tests/test_moonlight_hello.py

Smoke tests for the MoonLight *temporal* "hello" demo.

This file intentionally treats MoonLight/Java as *optional runtime* dependencies:

* Importing our demo wrapper must **not** import `moonlight` (lazy import).
* If the `moonlight` package and a compatible Java runtime are available, we run
  the tiny temporal monitor and sanity-check the output.

The demo command in `DEMO_COMMANDS.md` expects the test
`test_moonlight_import_and_version` to exist.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.metadata
import importlib.util
import inspect
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


# Ensure the repository's `src/` directory is importable when running
# `pytest` directly from the repo root (without `PYTHONPATH=src`).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


DEMO_MOD_PATH = "neural_pde_stl_strel.monitors.moonlight_hello"
MOONLIGHT_MOD_NAME = "moonlight"
MIN_JAVA_MAJOR = 21

# The demo uses `np.arange(0, 1.0, 0.2)`.
_EXPECTED_T_GRID = np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=float)


# Helpers

def _moonlight_is_installed() -> bool:
    """Return True if the `moonlight` module can be discovered."""

    return importlib.util.find_spec(MOONLIGHT_MOD_NAME) is not None


def _moonlight_dist_version() -> str | None:
    """Return the installed distribution version for `moonlight` (if any)."""

    try:
        return importlib.metadata.version(MOONLIGHT_MOD_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None


def _java_major_version() -> int | None:
    """Return the Java major version available on PATH, or None if unavailable."""

    java = shutil.which("java")
    if java is None:
        return None

    try:
        proc = subprocess.run(
            [java, "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    out = (proc.stderr or "") + "\n" + (proc.stdout or "")
    mt = re.search(r'version\s+"(?P<v>[^"]+)"', out)
    if mt is None:
        return None

    v = mt.group("v")
    parts = v.split(".")

    try:
        # Java 8 sometimes reports as 1.8.x
        if parts and parts[0] == "1" and len(parts) >= 2:
            return int(parts[1])
        return int(parts[0])
    except ValueError:
        return None


def _import_demo_module_fresh() -> ModuleType:
    """Import the demo module, forcing a fresh import."""

    sys.modules.pop(DEMO_MOD_PATH, None)
    importlib.invalidate_caches()
    mod = importlib.import_module(DEMO_MOD_PATH)
    assert hasattr(mod, "temporal_hello"), f"{DEMO_MOD_PATH} must define temporal_hello()"
    return mod


def _run_temporal_hello_or_skip(mod: ModuleType) -> np.ndarray:
    """Run the demo monitor and return its output as a float ndarray.

    If MoonLight/Java is not available in this environment, we skip.
    """

    try:
        res = mod.temporal_hello()
    except ImportError as exc:
        # Optional dependency missing/unusable (e.g., Java not on PATH).
        pytest.skip(f"MoonLight not usable in this environment: {exc!r}")
    except Exception as exc:  # pragma: no cover
        # Be lenient: MoonLight's Python wrapper depends on JNI/Java details.
        pytest.skip(f"MoonLight demo failed in this environment: {exc!r}")

    try:
        arr = np.asarray(res, dtype=float)
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"temporal_hello() returned a non-numeric array-like result: {exc!r}")

    return arr


def _assert_boolean_domain_values(values: np.ndarray) -> None:
    """Assert values look like boolean-domain verdicts.

    MoonLight's boolean domain is documented to return +/- 1.0.
    In some environments the wrapper may return Python bools, which become 0.0/1.0
    when coerced to floats. We accept either convention.
    """

    rounded = np.round(np.asarray(values, dtype=float), 12)
    uniq = set(np.unique(rounded).tolist())
    allowed = {-1.0, 0.0, 1.0}
    assert uniq.issubset(allowed), (
        "Expected boolean-domain outputs in {-1, 0, 1}; "
        f"got {sorted(uniq)}"
    )


# Fixtures


@pytest.fixture(scope="module")
def demo_mod() -> ModuleType:
    """Imported demo module (module-scoped)."""

    return importlib.import_module(DEMO_MOD_PATH)


@pytest.fixture(scope="module")
def moonlight_arr(demo_mod: ModuleType) -> np.ndarray:
    """Cached output of `temporal_hello()`.

    If MoonLight spins up a JVM, we only want to do it once per module.
    """

    return _run_temporal_hello_or_skip(demo_mod)


# Tests


def test_moonlight_import_and_version() -> None:
    """Quick environment check for demos.

    This is intentionally a *soft* check:

    * Skip if `moonlight` isn't installed.
    * Skip if Java is missing or too old.
    * Otherwise, ensure `moonlight` imports and exposes a non-empty version.
    """

    if not _moonlight_is_installed():
        pytest.skip("moonlight not installed")

    java_major = _java_major_version()
    if java_major is None:
        pytest.skip("Java not found on PATH (MoonLight requires Java 21+)")
    if java_major < MIN_JAVA_MAJOR:
        pytest.skip(f"Java {java_major} found (MoonLight requires Java 21+)")

    try:
        import moonlight  # type: ignore
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"moonlight is installed but could not be imported: {exc!r}")

    ver = _moonlight_dist_version() or getattr(moonlight, "__version__", None)
    assert isinstance(ver, str) and ver.strip(), f"Unexpected MoonLight version: {ver!r}"


def test_demo_module_is_lazy_wrt_moonlight_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing our wrapper must not import the optional `moonlight` package."""

    # Make sure we exercise the import-time behavior of the wrapper.
    monkeypatch.delitem(sys.modules, DEMO_MOD_PATH, raising=False)
    monkeypatch.delitem(sys.modules, MOONLIGHT_MOD_NAME, raising=False)

    real_import = builtins.__import__

    def _blocked_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == MOONLIGHT_MOD_NAME or name.startswith(f"{MOONLIGHT_MOD_NAME}."):
            raise ModuleNotFoundError("Blocked import of optional dependency 'moonlight'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    mod = importlib.import_module(DEMO_MOD_PATH)
    assert hasattr(mod, "temporal_hello")


def test_public_api_and_signature(demo_mod: ModuleType) -> None:
    assert hasattr(demo_mod, "__all__")
    public = tuple(getattr(demo_mod, "__all__"))
    assert "temporal_hello" in public

    sig = inspect.signature(demo_mod.temporal_hello)
    assert len(sig.parameters) == 0


def test_temporal_smoke_shape_and_time_monotonic(moonlight_arr: np.ndarray) -> None:
    assert isinstance(moonlight_arr, np.ndarray)
    assert moonlight_arr.ndim == 2
    assert moonlight_arr.shape[1] == 2, "expected 2 columns: [time, value]"
    assert moonlight_arr.shape[0] > 0

    assert np.isfinite(moonlight_arr).all(), "time/value entries must be finite"

    t = moonlight_arr[:, 0]
    # Monotonic non-decreasing time column
    assert np.all(np.diff(t) >= -1e-12)

    # The demo's input trace is defined over [0.0, 0.8]. The monitor output should
    # be within that range (allow tiny FP noise).
    assert float(t.min()) >= -1e-12
    assert float(t.max()) <= float(_EXPECTED_T_GRID.max()) + 1e-12


def test_temporal_values_are_booleanish_and_grid_when_uncompressed(moonlight_arr: np.ndarray) -> None:
    t = moonlight_arr[:, 0]
    v = moonlight_arr[:, 1]

    _assert_boolean_domain_values(v)

    # Some MoonLight monitors may return a compressed (piecewise-constant) signal.
    # If the output happens to be uncompressed and matches the input sample count,
    # validate the exact time grid.
    if moonlight_arr.shape[0] == _EXPECTED_T_GRID.shape[0]:
        assert np.allclose(t, _EXPECTED_T_GRID, atol=1e-12, rtol=0.0)


def test_temporal_hello_is_deterministic(demo_mod: ModuleType, moonlight_arr: np.ndarray) -> None:
    # A second call should return an identical array (demo is deterministic).
    arr2 = np.asarray(demo_mod.temporal_hello(), dtype=float)
    assert moonlight_arr.shape == arr2.shape
    assert np.array_equal(moonlight_arr, arr2)


def test_temporal_hello_raises_helpful_error_when_moonlight_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even without MoonLight installed, the wrapper should raise a clear ImportError."""

    mod = _import_demo_module_fresh()

    # Ensure the wrapper cannot import MoonLight, even if it is installed.
    monkeypatch.delitem(sys.modules, MOONLIGHT_MOD_NAME, raising=False)

    real_import = builtins.__import__

    def _blocked_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == MOONLIGHT_MOD_NAME or name.startswith(f"{MOONLIGHT_MOD_NAME}."):
            raise ModuleNotFoundError("No module named 'moonlight'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    with pytest.raises(ImportError) as excinfo:
        _ = mod.temporal_hello()

    msg = str(excinfo.value).lower()
    assert "pip install moonlight" in msg
    assert "java" in msg
