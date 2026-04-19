from __future__ import annotations

"""Lightweight, robust CSV experiment logger.

This module provides :class:`CSVLogger`, a small utility for writing
append-only experiment metrics to a CSV file.

Design goals
- **Safe header handling**: you can (a) provide a header up-front, (b) adopt an
  on-disk header from an existing file, or (c) infer a header from the first
  mapping that is appended. When a file already exists, the logger adopts its
  header instead of clobbering it.
- **Concurrent-friendly**: an optional cross-platform lock implemented via a
  sidecar ``.lock`` file helps multiple processes append safely.
- **Practical ergonomics**: accepts sequences or mappings; can enforce row
  lengths; supports compact float formatting.
- **Small + dependency-free**: standard library only.

Example
-------
>>> log = CSVLogger("metrics.csv", header=["step", "loss"])
>>> log.append({"step": 1, "loss": 0.1234})
>>> log.append([2, 0.0567])  # sequences work too

"""

import csv
import numbers
import os
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, TextIO

__all__ = ["CSVLogger"]

class _FileLock:
    """A simple inter-process file lock using exclusive creation of ``.lock``.

    The lock is acquired by atomically creating a sidecar file
    ``<target>.lock`` (``O_CREAT | O_EXCL``). This is a pragmatic technique
    that works well on local filesystems.

    Notes
    -----
    - Like most lock-file schemes, this may be unreliable on some networked
      filesystems.
    - This lock is **not re-entrant**. Public methods in this module are
      structured to never acquire it twice.
    """

    def __init__(
        self,
        target: Path,
        *,
        timeout: float = 5.0,
        poll: float = 0.01,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        if poll <= 0:
            raise ValueError("poll must be > 0")

        self._lock_path = Path(f"{target}.lock")
        self._timeout = float(timeout)
        self._poll = float(poll)
        self._fd: int | None = None

    def acquire(self) -> None:
        """Acquire the lock, blocking up to ``timeout`` seconds."""
        if self._fd is not None:
            # Nested locking is a logic error in this module; fail loudly.
            raise RuntimeError(f"Lock already held: {self._lock_path}")

        deadline = time.monotonic() + self._timeout
        pid_bytes = f"pid={os.getpid()}\n".encode("ascii", errors="replace")

        while True:
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timeout acquiring lock: {self._lock_path}")
                time.sleep(self._poll)
                continue

            # We created the lock file. Best-effort write some debug info.
            try:
                os.write(fd, pid_bytes)
            except Exception:
                # Don't leave a stuck lock file if the metadata write fails.
                try:
                    os.close(fd)
                finally:
                    try:
                        os.unlink(self._lock_path)
                    except FileNotFoundError:
                        pass
                raise

            self._fd = fd
            return

    def release(self) -> None:
        """Release the lock."""
        if self._fd is None:
            return

        fd = self._fd
        self._fd = None
        try:
            os.close(fd)
        finally:
            try:
                os.unlink(self._lock_path)
            except FileNotFoundError:
                # Someone may have cleaned up for us; that's fine.
                pass

    def __enter__(self) -> _FileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class _Opts:
    __slots__ = (
        "delimiter",
        "encoding",
        "lineterminator",
        "float_precision",
        "none_as_empty",
        "strict_lengths",
    )

    def __init__(
        self,
        *,
        delimiter: str,
        encoding: str,
        lineterminator: str,
        float_precision: int | None,
        none_as_empty: bool,
        strict_lengths: bool,
    ) -> None:
        self.delimiter = delimiter
        self.encoding = encoding
        self.lineterminator = lineterminator
        self.float_precision = float_precision
        self.none_as_empty = none_as_empty
        self.strict_lengths = strict_lengths


def _validate_delimiter(delimiter: str) -> str:
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise TypeError("delimiter must be a single-character string")
    if delimiter in {"\n", "\r"}:
        raise ValueError("delimiter must not be a newline character")
    return delimiter


def _validate_columns(columns: Iterable[str], *, allow_duplicates: bool) -> list[str]:
    """Validate and return a concrete list of column names."""

    # A very common mistake is passing a single string.
    if isinstance(columns, (str, bytes, bytearray)):
        raise TypeError("columns must be an iterable of strings, not a single string")

    cols = list(columns)
    if any((not isinstance(c, str)) for c in cols):
        raise TypeError("column names must be strings")
    if any(c == "" for c in cols):
        raise ValueError("column names must be non-empty strings")
    if any(("\n" in c) or ("\r" in c) for c in cols):
        raise ValueError("column names must not contain newlines")
    if (not allow_duplicates) and (len(set(cols)) != len(cols)):
        raise ValueError("column names contain duplicates")
    return cols


def _validate_header(header: Iterable[str]) -> list[str]:
    """Validate a CSV header (must have unique, non-empty column names)."""

    return _validate_columns(header, allow_duplicates=False)


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class CSVLogger:
    """Append-only CSV logger for small/medium experiment logs.

    Parameters
    path:
        Destination CSV file.
    header:
        Optional header (column names).

        - If provided and ``overwrite`` is true (the default when ``header`` is
          provided), the file will be clobbered and the header written.
        - If provided and ``overwrite`` is false, an existing non-empty file is
          validated to have the same header.
        - If omitted and the file already exists, the first row is adopted as
          the header.
        - If omitted and the first appended row is a mapping, its keys become
          the header.
    overwrite:
        Whether to clobber the file when ``header`` is provided. Defaults to
        ``True`` when ``header`` is not ``None``.
    create_dirs:
        Create parent directories as needed.
    delimiter, encoding:
        CSV formatting options.
    lineterminator:
        Row terminator written by :mod:`csv`. The default (``"\n"``) keeps
        output consistent across platforms.
    float_precision:
        If given, numeric (non-integral) values are formatted with
        ``.{precision}g``.
    none_as_empty:
        If true (default), ``None`` is written as the empty string. If false,
        ``None`` is written as the literal string ``"None"``.
    strict_lengths:
        When a header is set, enforce exact row length for sequence rows. If
        false, rows are truncated or right-padded with empties.
    lock, lock_timeout:
        Enable/parameterize the optional sidecar lock.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        header: Iterable[str] | None = None,
        *,
        overwrite: bool | None = None,
        create_dirs: bool = True,
        delimiter: str = ",",
        encoding: str = "utf-8",
        lineterminator: str = "\n",
        float_precision: int | None = None,
        none_as_empty: bool = True,
        strict_lengths: bool = True,
        lock: bool = False,
        lock_timeout: float = 5.0,
    ) -> None:
        self.path = Path(path)
        if create_dirs:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        delimiter = _validate_delimiter(delimiter)
        if not isinstance(encoding, str) or encoding == "":
            raise TypeError("encoding must be a non-empty string")
        if not isinstance(lineterminator, str) or lineterminator == "":
            raise TypeError("lineterminator must be a non-empty string")
        if float_precision is not None:
            if not isinstance(float_precision, int):
                raise TypeError("float_precision must be an int or None")
            if float_precision <= 0:
                raise ValueError("float_precision must be > 0")
        if lock_timeout <= 0:
            raise ValueError("lock_timeout must be > 0")

        self._opt = _Opts(
            delimiter=delimiter,
            encoding=encoding,
            lineterminator=lineterminator,
            float_precision=float_precision,
            none_as_empty=bool(none_as_empty),
            strict_lengths=bool(strict_lengths),
        )

        self._lock: _FileLock | None = _FileLock(self.path, timeout=lock_timeout) if lock else None

        self._header: list[str] | None = _validate_header(header) if header is not None else None

        # If the user supplies a header, preserve prior behavior: default to
        # overwriting unless explicitly disabled.
        if self._header is not None and overwrite is None:
            overwrite = True

        with self._lock_ctx():
            if self._header is not None:
                if overwrite:
                    self._write_header(clobber=True)
                else:
                    # Validate an existing non-empty file's header.
                    disk_header = self._read_existing_header()
                    if disk_header is not None and disk_header != self._header:
                        raise ValueError(
                            "Existing CSV header does not match requested header.\n"
                            f"  Existing: {disk_header}\n  New:      {self._header}\n"
                            "Pass overwrite=True to replace the file or adjust the header."
                        )
                    # Ensure the header exists on disk if the file is missing/empty.
                    try:
                        file_is_empty = (not self.path.exists()) or (self.path.stat().st_size == 0)
                    except FileNotFoundError:
                        file_is_empty = True
                    if file_is_empty:
                        self._write_header(clobber=True)
            else:
                # No header provided: adopt an on-disk header if present.
                self._header = self._read_existing_header()

    @property
    def header(self) -> tuple[str, ...] | None:
        """Current header as an immutable tuple, or ``None`` if unset."""
        return tuple(self._header) if self._header is not None else None

    def append(self, row: Iterable[Any] | Mapping[str, Any]) -> None:
        """Append a single row."""
        with self._lock_ctx():
            self._ensure_header_for_row(row)
            values = self._coerce_row(row)

            with self._open("a") as f:
                w = self._writer(f)
                self._write_header_if_empty_file(f, w)
                w.writerow(values)

    def append_many(self, rows: Iterable[Iterable[Any] | Mapping[str, Any]]) -> None:
        """Append many rows efficiently."""
        rows_iter = iter(rows)
        try:
            first = next(rows_iter)
        except StopIteration:
            return

        with self._lock_ctx():
            self._ensure_header_for_row(first)

            with self._open("a") as f:
                w = self._writer(f)
                self._write_header_if_empty_file(f, w)

                w.writerow(self._coerce_row(first))
                for row in rows_iter:
                    # If the first row could not establish a header (i.e., a
                    # headerless sequence log), we do not allow later mapping
                    # rows because there is nowhere sensible to put their keys.
                    if isinstance(row, Mapping) and self._header is None:
                        raise ValueError(
                            "Cannot append mapping rows without a header. "
                            "Provide header=... or make the first row a mapping."
                        )
                    w.writerow(self._coerce_row(row))

    def extend_header(self, new_columns: Iterable[str]) -> None:
        """Extend the header by adding any columns from ``new_columns``.

        Existing rows are padded with empty strings for the new columns.
        """
        additions_all = _unique_preserve_order(_validate_columns(new_columns, allow_duplicates=True))
        if not additions_all:
            return

        with self._lock_ctx():
            # Adopt an on-disk header if we don't have one yet.
            if self._header is None:
                self._header = self._read_existing_header()

            if self._header is None:
                # No header exists anywhere. It's only safe to create one if the
                # file is missing or empty.
                try:
                    file_is_nonempty = self.path.exists() and (self.path.stat().st_size > 0)
                except FileNotFoundError:
                    file_is_nonempty = False
                if file_is_nonempty:
                    raise ValueError(
                        "Cannot create a header for a non-empty CSV with no known header. "
                        "Create a new file or provide header=... up-front."
                    )
                self._header = additions_all
                self._write_header(clobber=True)
                return

            additions = [c for c in additions_all if c not in self._header]
            if not additions:
                return

            old_header = list(self._header)
            old_len = len(old_header)

            # Read the existing file (small logs typical for experiments).
            rows: list[list[str]] = []
            try:
                file_is_nonempty = self.path.exists() and (self.path.stat().st_size > 0)
            except FileNotFoundError:
                file_is_nonempty = False
            if file_is_nonempty:
                with self._open("r") as f:
                    reader = list(csv.reader(f, delimiter=self._opt.delimiter))
                if reader:
                    # If the on-disk header doesn't match what we think it is,
                    # fail loudly rather than silently corrupting alignment.
                    if reader[0] != old_header:
                        raise ValueError(
                            "On-disk CSV header does not match in-memory header.\n"
                            f"  Disk: {reader[0]}\n  Mem:  {old_header}"
                        )
                    rows = reader[1:]

            # Rewrite file atomically with an extended header.
            new_header = old_header + additions
            pad = [""] * len(additions)

            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            try:
                with tmp.open("w", newline="", encoding=self._opt.encoding) as f:
                    w = self._writer(f)
                    w.writerow(new_header)
                    for r in rows:
                        if len(r) < old_len:
                            r = list(r) + [""] * (old_len - len(r))
                        elif len(r) > old_len:
                            raise ValueError(
                                "Encountered a row longer than the existing header while extending. "
                                f"Row length={len(r)}, header length={old_len}."
                            )
                        w.writerow(list(r) + pad)
                os.replace(tmp, self.path)
                self._header = new_header
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    # Best-effort cleanup only.
                    pass

    def _lock_ctx(self):
        return self._lock if self._lock is not None else nullcontext()

    def _open(self, mode: str) -> TextIO:
        return self.path.open(mode, newline="", encoding=self._opt.encoding)

    def _writer(self, f: TextIO) -> Any:
        # The concrete writer type is ``_csv.writer`` (an internal C extension
        # type), so we annotate as ``Any``.
        return csv.writer(
            f,
            delimiter=self._opt.delimiter,
            lineterminator=self._opt.lineterminator,
        )

    def _read_existing_header(self) -> list[str] | None:
        """Return the first CSV row if the file exists and is non-empty."""
        try:
            if not self.path.exists() or self.path.stat().st_size == 0:
                return None
            with self._open("r") as f:
                r = csv.reader(f, delimiter=self._opt.delimiter)
                first = next(r, None)
            if not first:
                return None
            return _validate_header(first)
        except FileNotFoundError:
            return None

    def _write_header(self, *, clobber: bool) -> None:
        if self._header is None:
            raise RuntimeError("Cannot write header: header is not set")
        mode = "w" if clobber else "a"
        with self._open(mode) as f:
            self._writer(f).writerow(self._header)

    def _write_header_if_empty_file(self, f: TextIO, w: Any) -> None:
        """Write the header row if (and only if) the opened file is empty."""
        if self._header is None:
            return
        # In text append mode, the stream position is at EOF.
        if f.tell() == 0:
            w.writerow(self._header)

    def _ensure_header_for_row(self, row: Iterable[Any] | Mapping[str, Any]) -> None:
        """Ensure ``self._header`` is set when possible/necessary."""
        if self._header is None:
            self._header = self._read_existing_header()

        if self._header is None and isinstance(row, Mapping):
            self._header = _validate_header(row.keys())

    def _format_value(self, v: Any) -> Any:
        """Normalize a value for CSV writing (precision + None handling)."""
        if v is None:
            return "" if self._opt.none_as_empty else "None"

        if (
            self._opt.float_precision is not None
            and isinstance(v, numbers.Real)
            and not isinstance(v, numbers.Integral)
        ):
            # Cast to float to ensure consistent formatting across numpy/Decimal reals.
            return format(float(v), f".{self._opt.float_precision}g")
        return v

    def _coerce_row(self, row: Iterable[Any] | Mapping[str, Any]) -> list[Any]:
        if isinstance(row, Mapping):
            return self._row_from_mapping(row)
        if isinstance(row, (str, bytes, bytearray)):
            raise TypeError("row must be a sequence of values, not a string")
        if not isinstance(row, Iterable):
            raise TypeError(f"row must be iterable, got {type(row).__name__}")
        return self._row_from_sequence(row)

    def _row_from_sequence(self, row: Iterable[Any]) -> list[Any]:
        vals = [self._format_value(v) for v in row]

        if self._header is not None:
            if self._opt.strict_lengths and len(vals) != len(self._header):
                raise ValueError(f"Row length {len(vals)} != header length {len(self._header)}")
            if not self._opt.strict_lengths and len(vals) < len(self._header):
                vals = vals + [""] * (len(self._header) - len(vals))
            elif not self._opt.strict_lengths and len(vals) > len(self._header):
                vals = vals[: len(self._header)]

        return vals

    def _row_from_mapping(self, m: Mapping[str, Any]) -> list[Any]:
        if self._header is None:
            raise RuntimeError("Internal error: header must be set before formatting mapping rows")

        extra = [k for k in m.keys() if k not in self._header]
        if extra:
            raise KeyError(
                f"Mapping contains keys not in header: {extra}. "
                "Call extend_header([...]) to add new columns."
            )

        return [self._format_value(m.get(k, "")) for k in self._header]
