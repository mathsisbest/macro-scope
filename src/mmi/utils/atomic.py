"""Atomic file writes: temp file in the same dir + fsync + ``os.replace()``.

A crash mid-write can never leave a partially-written file at the destination:
readers either observe the previous content or the fully-written new content, and
the original file is untouched on any failure.  Any temp file is cleaned up.

Two entry points:

* :func:`atomic_write` — data is produced in-process (text or bytes).
* :func:`atomic_replace` — content is produced by another tool (e.g. DuckDB's
  ``COPY``) into a caller-created temp file; this fsyncs and renames it.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as fh:
        os.fsync(fh.fileno())


def _fsync_dir(directory: Path) -> None:
    """Best-effort durability for the rename itself (dir entries live in the dir)."""
    with contextlib.suppress(OSError):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write(
    path: str | os.PathLike[str],
    data: str | bytes,
    mode: str = "w",
    *,
    encoding: str | None = "utf-8",
    fsync: bool = True,
    create_dirs: bool = False,
) -> Path:
    """Atomically write ``data`` to ``path`` via a same-directory temp file.

    Writes to a unique temp file in ``path``'s directory, flushes + fsyncs it,
    then ``os.replace()``s it into place.  A crash mid-write leaves the original
    file intact and never a partial ``path``; on any failure the temp file is
    removed and the exception propagates.

    ``mode``: ``"w"`` for text or ``"wb"`` for bytes; ``encoding`` is ignored for
    binary modes.  ``create_dirs``: if True, missing parent directories are
    created; otherwise a missing target dir raises ``FileNotFoundError``.
    """
    dest = Path(path)
    directory = dest.parent
    if create_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    elif not directory.is_dir():
        raise FileNotFoundError(
            f"cannot atomically write {dest}: parent directory {directory} does not "
            "exist (pass create_dirs=True to create it)"
        )
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=f".{dest.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, mode, encoding=None if "b" in mode else encoding) as fh:
            fh.write(data)
            if fsync:
                fh.flush()
                os.fsync(fh.fileno())
        os.replace(tmp_path, dest)
        if fsync:
            _fsync_dir(directory)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return dest


def atomic_replace(
    tmp_path: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    fsync: bool = True,
) -> Path:
    """Atomically move an already-written temp file to its final destination.

    For content produced by another tool (e.g. DuckDB's ``COPY``) rather than
    in-process: fsyncs the temp file, then ``os.replace()``s it into place.  On
    any failure the temp file is removed and the destination is untouched.
    """
    tmp = Path(tmp_path)
    target = Path(dest)
    try:
        if fsync:
            _fsync_file(tmp)
        os.replace(tmp, target)
        if fsync:
            _fsync_dir(target.parent)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return target
