from __future__ import annotations

"""Reproducibility helpers.

This module centralizes the steps needed to make experiments *as reproducible as
practical* across Python, NumPy, and PyTorch.

Key points
- Deterministic results are not guaranteed across PyTorch releases, CUDA
  versions, or different hardware.
- Some CUDA determinism knobs (e.g., ``CUBLAS_WORKSPACE_CONFIG``) must be set
  *before* the CUDA context is initialized. We therefore set them as early as
  possible and never override values already provided by the user.
- PyTorch 2.9 introduced a new TF32 control surface via ``fp32_precision``.
  PyTorch explicitly warns against mixing the new API with the legacy
  ``allow_tf32`` flags, so this module uses one or the other depending on what
  the installed PyTorch build supports.
"""

import os
import random
import warnings
from typing import Any

__all__ = ["seed_everything", "seed_worker", "torch_generator"]

_UINT32_MOD = 2**32
_UINT64_MOD = 2**64
_CUBLAS_WORKSPACE_CONFIG_DEFAULT = ":4096:8"


def _as_int_seed(seed: int) -> int:
    """Return ``seed`` as a plain ``int``.

    Accepts int-like objects (e.g., ``numpy.int64``) but rejects ``bool``.
    """
    if isinstance(seed, bool):
        raise TypeError("seed must be an int, not bool")
    try:
        return int(seed)
    except Exception as e:  # pragma: no cover
        raise TypeError(f"seed must be int-like, got {type(seed).__name__}") from e


def _to_uint32(seed: int) -> int:
    """Map ``seed`` into the unsigned 32-bit range ``[0, 2**32 - 1]``."""
    return _as_int_seed(seed) % _UINT32_MOD


def _to_uint64(seed: int) -> int:
    """Map ``seed`` into the unsigned 64-bit range ``[0, 2**64 - 1]``."""
    return _as_int_seed(seed) % _UINT64_MOD


def _set_env_if_unset(name: str, value: str) -> bool:
    """Set environment variable ``name`` to ``value`` iff it is not already set.

    Returns
    -------
    bool
        ``True`` if the variable was set by this call, ``False`` if it already
        existed.
    """
    if name in os.environ:
        return False
    os.environ[name] = value
    return True


def _maybe_setattr(obj: Any, name: str, value: Any) -> bool:
    """Best-effort ``setattr`` that only runs if the attribute exists."""
    try:
        if not hasattr(obj, name):
            return False
        setattr(obj, name, value)
        return True
    except Exception:
        return False


def _torch_supports_fp32_precision_api(torch: Any) -> bool:
    """Return ``True`` iff the installed PyTorch exposes the TF32 fp32_precision API."""
    try:
        backends = torch.backends
        matmul = backends.cuda.matmul
        cudnn = backends.cudnn
    except Exception:
        return False

    return (
        hasattr(backends, "fp32_precision")
        and hasattr(matmul, "fp32_precision")
        and hasattr(cudnn, "fp32_precision")
    )


def _set_torch_fp32_precision(torch: Any, mode: str) -> None:
    """Best-effort set of PyTorch 2.9+ fp32 precision controls.

    Parameters
    mode:
        Either ``"ieee"`` (full FP32 internal math) or ``"tf32"`` (allow TF32
        internal math where supported).

    Notes
    -----
    This is intentionally best-effort: some submodules (e.g., cuDNN conv/rnn)
    may not exist in CPU-only builds or older releases.
    """
    backends = torch.backends

    # Global/default setting.
    _maybe_setattr(backends, "fp32_precision", mode)

    # cuBLAS matmul backend.
    try:
        _maybe_setattr(backends.cuda.matmul, "fp32_precision", mode)
    except Exception:
        pass

    # cuDNN backend + operator-level overrides (conv/rnn).
    try:
        cudnn = backends.cudnn
        _maybe_setattr(cudnn, "fp32_precision", mode)

        # These operator-level modules are new and may not exist everywhere.
        if hasattr(cudnn, "conv"):
            _maybe_setattr(cudnn.conv, "fp32_precision", mode)
        if hasattr(cudnn, "rnn"):
            _maybe_setattr(cudnn.rnn, "fp32_precision", mode)
    except Exception:
        pass


def seed_everything(
    seed: int = 0,
    *,
    deterministic: bool = True,
    warn_only: bool = True,
    set_pythonhashseed: bool = True,
    configure_cuda_env: bool = True,
    disable_tf32_when_deterministic: bool = True,
    enable_tf32_when_nondeterministic: bool = True,
    verbose: bool = False,
) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch for reproducibility.

    The function is safe to call in CPU-only environments; if NumPy and/or
    PyTorch are not installed, it will still seed what it can and then return.

    Parameters
    seed:
        Base seed used for Python and (if available) NumPy and PyTorch.
    deterministic:
        If ``True``, request deterministic algorithms and disable cuDNN
        benchmarking. If ``False``, prefer performance-oriented defaults.
    warn_only:
        Forwarded to ``torch.use_deterministic_algorithms(..., warn_only=...)``
        when available. If ``True``, PyTorch will warn instead of raising when a
        strictly deterministic implementation is unavailable.
    set_pythonhashseed:
        If ``True``, set ``PYTHONHASHSEED`` *if it is not already set*. Changing
        this after interpreter start will not retroactively re-hash existing
        objects, but it is still useful for child processes spawned later.
    configure_cuda_env:
        If ``True`` and determinism is requested, set CUDA/cuBLAS environment
        knobs (e.g., ``CUBLAS_WORKSPACE_CONFIG``) *only if not already set*.
    disable_tf32_when_deterministic:
        If ``True`` and determinism is requested, attempt to disable TF32 on
        Ampere+ GPUs for full FP32 internal precision.
    enable_tf32_when_nondeterministic:
        If ``True`` and determinism is *not* requested, attempt to enable TF32
        for speed on Ampere+ GPUs.
    verbose:
        If ``True``, emit ``warnings.warn`` diagnostics instead of failing hard.
    """
    seed_int = _as_int_seed(seed)
    seed_u32 = seed_int % _UINT32_MOD
    seed_u64 = seed_int % _UINT64_MOD

    # Environment variables (set as early as possible)
    if set_pythonhashseed:
        _set_env_if_unset("PYTHONHASHSEED", str(seed_u32))

    did_set_cublas = False
    if deterministic and configure_cuda_env:
        # Recommended by NVIDIA/cuBLAS for deterministic GEMMs (CUDA >= 10.2).
        did_set_cublas = _set_env_if_unset(
            "CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG_DEFAULT
        )

    # Python stdlib RNG
    random.seed(seed_int)

    # NumPy (optional)
    try:
        import numpy as np  # type: ignore

        # NumPy's legacy global RNG requires a uint32 seed.
        np.random.seed(seed_u32)
    except Exception:
        pass

    # PyTorch (optional)
    try:
        import torch  # type: ignore
    except Exception:
        return

    try:
        torch.manual_seed(seed_u64)
        # Explicitly seed all CUDA devices as well (safe if CUDA is unavailable).
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_u64)
    except Exception:
        if verbose:
            warnings.warn("Failed to set PyTorch RNG seeds.", RuntimeWarning)

    # Warn if we set cuBLAS determinism too late.
    if did_set_cublas:
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized() and verbose:
                warnings.warn(
                    "CUBLAS_WORKSPACE_CONFIG was set after CUDA initialization; "
                    "full cuBLAS determinism may not be guaranteed. Prefer setting "
                    "it in the environment before importing/using PyTorch.",
                    RuntimeWarning,
                )
        except Exception:
            pass

    # Determinism / performance knobs
    try:
        if deterministic:
            # Prefer official switch when available.
            try:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)  # type: ignore[call-arg]
            except TypeError:
                # Older PyTorch that lacks warn_only.
                torch.use_deterministic_algorithms(True)  # type: ignore[misc]
            except Exception:
                pass

            # cuDNN algorithm selection + determinism.
            try:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except Exception:
                pass

            # TF32 controls.
            if disable_tf32_when_deterministic:
                if _torch_supports_fp32_precision_api(torch):
                    _set_torch_fp32_precision(torch, "ieee")
                else:
                    _maybe_setattr(torch.backends.cuda.matmul, "allow_tf32", False)
                    _maybe_setattr(torch.backends.cudnn, "allow_tf32", False)

        else:
            # Performance-oriented path.
            try:
                torch.use_deterministic_algorithms(False)  # type: ignore[misc]
            except Exception:
                pass

            try:
                torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
            except Exception:
                pass

            if enable_tf32_when_nondeterministic:
                if _torch_supports_fp32_precision_api(torch):
                    _set_torch_fp32_precision(torch, "tf32")
                else:
                    _maybe_setattr(torch.backends.cuda.matmul, "allow_tf32", True)
                    _maybe_setattr(torch.backends.cudnn, "allow_tf32", True)

    except Exception:
        if verbose:
            warnings.warn(
                "Failed to configure PyTorch determinism/performance flags.",
                RuntimeWarning,
            )


def seed_worker(worker_id: int) -> None:  # pragma: no cover
    """Deterministically (re)seed NumPy & Python inside a PyTorch DataLoader worker.

    This mirrors the official PyTorch recipe, deriving a 32-bit seed from
    ``torch.initial_seed()`` so each worker has a different, reproducible seed
    when a ``torch.Generator`` is passed to the ``DataLoader``.
    """
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except Exception:
        return

    worker_seed = int(torch.initial_seed()) % _UINT32_MOD
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # Helpful for libraries that rely on PYTHONHASHSEED in subprocesses.
    _set_env_if_unset("PYTHONHASHSEED", str(worker_seed))


def torch_generator(seed: int, device: Any | None = None) -> Any:
    """Create a seeded :class:`torch.Generator` for deterministic DataLoader usage."""
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required for torch_generator") from e

    gen = torch.Generator(device=device) if device is not None else torch.Generator()
    gen.manual_seed(_to_uint64(seed))
    return gen
