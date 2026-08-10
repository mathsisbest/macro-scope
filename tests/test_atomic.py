"""Atomic file-write utility: crash-safe replace semantics + cleanup guarantees.

Coverage for src/mmi/utils/atomic.py — every guarantee in the contract:
* the destination only ever observes old or fully-new content (never partial);
* a failure at any stage cleans up the temp file and preserves the original;
* a missing target dir is a documented policy (raise vs create).
"""

import os
from pathlib import Path

import pytest

import mmi.utils.atomic as atomic
from mmi.utils.atomic import atomic_replace, atomic_write


def _temp_files(directory: Path) -> list[Path]:
    """Temp files are hidden dotfiles with a ``.tmp`` suffix in the target dir."""
    return sorted(directory.glob(".*.tmp"))


def test_atomic_write_text_round_trip(tmp_path):
    dest = tmp_path / "out.txt"
    result = atomic_write(dest, "hello café — Σ")
    assert result == dest
    assert dest.read_text(encoding="utf-8") == "hello café — Σ"
    assert _temp_files(tmp_path) == []


def test_atomic_write_bytes_mode(tmp_path):
    dest = tmp_path / "blob.bin"
    atomic_write(dest, b"\x00\x01\xff\xfe", mode="wb")
    assert dest.read_bytes() == b"\x00\x01\xff\xfe"


def test_atomic_write_overwrites_existing(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("old", encoding="utf-8")
    atomic_write(dest, "new")
    assert dest.read_text(encoding="utf-8") == "new"


def test_atomic_write_missing_parent_dir_raises_by_default(tmp_path):
    dest = tmp_path / "missing" / "out.txt"
    with pytest.raises(FileNotFoundError):
        atomic_write(dest, "x")
    assert not dest.exists()
    assert _temp_files(tmp_path) == []


def test_atomic_write_create_dirs_creates_missing_parent(tmp_path):
    dest = tmp_path / "a" / "b" / "out.txt"
    atomic_write(dest, "x", create_dirs=True)
    assert dest.read_text(encoding="utf-8") == "x"
    assert _temp_files(tmp_path / "a" / "b") == []


def test_atomic_write_failed_replace_preserves_original_and_cleans_temp(tmp_path, monkeypatch):
    dest = tmp_path / "out.txt"
    original = "precious original"
    dest.write_text(original, encoding="utf-8")

    def failing_replace(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated rename failure"):
        atomic_write(dest, "new content")
    assert dest.read_text(encoding="utf-8") == original, "original must survive a failed commit"
    assert _temp_files(tmp_path) == [], "temp file must be cleaned up on failure"


def test_atomic_write_failure_mid_write_preserves_original_and_cleans_temp(tmp_path, monkeypatch):
    """Failure while streaming the payload (fsync raising = post-write, pre-commit crash)."""
    dest = tmp_path / "out.txt"
    original = "precious original"
    dest.write_text(original, encoding="utf-8")

    def failing_fsync(fd):
        raise OSError("simulated disk error")

    monkeypatch.setattr(atomic.os, "fsync", failing_fsync)

    with pytest.raises(OSError, match="simulated disk error"):
        atomic_write(dest, "new content")
    assert dest.read_text(encoding="utf-8") == original, "original must survive a mid-write crash"
    assert _temp_files(tmp_path) == [], "temp file must be cleaned up on failure"


def test_atomic_write_fsync_is_invoked(tmp_path, monkeypatch):
    """fsync runs on both the temp file and (best-effort) the parent directory."""
    fsynced: list[str] = []
    real_fsync = os.fsync

    def spy_fsync(fd):
        fsynced.append("fsync")
        return real_fsync(fd)

    monkeypatch.setattr(atomic.os, "fsync", spy_fsync)
    atomic_write(tmp_path / "out.txt", "x")
    assert fsynced, "fsync must be called before the rename commits"


def test_atomic_replace_moves_temp_into_place(tmp_path):
    tmp = tmp_path / "staged.bin"
    dest = tmp_path / "final.bin"
    tmp.write_bytes(b"staged bytes")
    result = atomic_replace(tmp, dest)
    assert result == dest
    assert dest.read_bytes() == b"staged bytes"
    assert not tmp.exists()


def test_atomic_replace_overwrites_existing_destination(tmp_path):
    tmp = tmp_path / "staged.bin"
    dest = tmp_path / "final.bin"
    dest.write_bytes(b"old bytes")
    tmp.write_bytes(b"new bytes")
    atomic_replace(tmp, dest)
    assert dest.read_bytes() == b"new bytes"


def test_atomic_replace_failure_preserves_destination_and_cleans_temp(tmp_path, monkeypatch):
    tmp = tmp_path / "staged.bin"
    dest = tmp_path / "final.bin"
    dest.write_bytes(b"old bytes")
    tmp.write_bytes(b"new bytes")

    def failing_replace(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated rename failure"):
        atomic_replace(tmp, dest)
    assert dest.read_bytes() == b"old bytes", "destination must be untouched on failure"
    assert not tmp.exists(), "temp file must be cleaned up on failure"
