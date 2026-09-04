#!/usr/bin/env python3
"""Validate Python release archives without extracting them.

The checks in this module run before the release pipeline unpacks or installs
an artifact.  They reject archive member names that could escape an extraction
root, non-regular members, corrupt ZIP payloads, and wheels whose ``RECORD``
does not provide an exact SHA-256-and-size closure over the wheel files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import stat
import sys
import tarfile
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Never


class ArchiveValidationError(ValueError):
    """Raised when a release archive is unsafe, corrupt, or inconsistent."""


def _canonical_member_name(value: str, *, directory: bool = False) -> str | None:
    """Return a canonical extraction key, or ``None`` for an unsafe name.

    A single trailing slash is conventional for an explicit directory entry.
    It is removed from the returned key so that ``path`` and ``path/`` cannot
    coexist and alias after extraction. All other normalization is rejected
    instead of silently applied.
    """

    has_control_character = any(
        ord(character) < 32 or ord(character) == 127 for character in value
    )
    if not value or "\\" in value or has_control_character:
        return None
    if value.endswith("/"):
        if not directory:
            return None
        value = value[:-1]
    if not value or value == ".":
        return None

    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        return None
    canonical = path.as_posix()
    if canonical != value:
        return None
    return canonical


def safe_member_name(value: str, *, directory: bool = False) -> bool:
    """Return whether *value* is a canonical, safe archive-relative name."""

    return _canonical_member_name(value, directory=directory) is not None


def _validated_member_keys(
    archive_path: Path,
    *,
    kind: str,
    entries: Sequence[tuple[str, bool]],
) -> list[str]:
    """Validate names and return extraction keys with no overwrite conflicts."""

    keys: list[str] = []
    for name, directory in entries:
        key = _canonical_member_name(name, directory=directory)
        if key is None:
            _fail(archive_path, f"unsafe {kind} member path")
        keys.append(key)
    if len(keys) != len(set(keys)):
        _fail(archive_path, f"duplicate {kind} member or extraction-path alias")

    files = {key for key, (_, directory) in zip(keys, entries, strict=True) if not directory}
    for key in keys:
        parts = key.split("/")
        for length in range(1, len(parts)):
            ancestor = "/".join(parts[:length])
            if ancestor in files:
                _fail(
                    archive_path,
                    f"{kind} regular-file ancestor conflict: {ancestor!r} and {key!r}",
                )
    return keys


def _fail(archive: Path, message: str) -> Never:
    raise ArchiveValidationError(f"unsafe or invalid archive {archive}: {message}")


def _read_zip_member(archive_path: Path, archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except (OSError, RuntimeError, zipfile.BadZipFile, zlib.error) as error:
        _fail(archive_path, f"cannot read ZIP member {name!r}: {error}")


def _verify_record(
    archive_path: Path,
    archive: zipfile.ZipFile,
    infos: Sequence[zipfile.ZipInfo],
) -> None:
    files = {info.filename: info for info in infos if not info.is_dir()}
    records = [name for name in files if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        _fail(archive_path, f"expected one RECORD file, found {len(records)}")
    record = records[0]
    try:
        record_text = _read_zip_member(archive_path, archive, record).decode("utf-8")
        rows = list(csv.reader(io.StringIO(record_text)))
    except (UnicodeDecodeError, csv.Error) as error:
        _fail(archive_path, f"cannot parse RECORD: {error}")
    if any(len(row) != 3 for row in rows):
        _fail(archive_path, "every RECORD row must have exactly three fields")

    recorded_names = [row[0] for row in rows]
    recorded_keys = [_canonical_member_name(name) for name in recorded_names]
    if any(key is None for key in recorded_keys):
        _fail(archive_path, "unsafe RECORD path")
    if len(recorded_keys) != len(set(recorded_keys)):
        _fail(archive_path, "RECORD contains duplicate paths or extraction-path aliases")
    if set(recorded_keys) != set(files):
        _fail(archive_path, "RECORD membership does not exactly match wheel files")

    for name, digest, declared_size in rows:
        if name == record:
            if digest or declared_size:
                _fail(archive_path, "RECORD must leave its own hash and size empty")
            continue
        if not digest.startswith("sha256="):
            _fail(archive_path, f"non-sha256 or missing RECORD digest for {name!r}")

        data = _read_zip_member(archive_path, archive, name)
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        if digest.removeprefix("sha256=") != encoded:
            _fail(archive_path, f"RECORD digest mismatch for {name!r}")
        try:
            size = int(declared_size)
        except ValueError:
            _fail(archive_path, f"invalid RECORD size for {name!r}")
        if size != len(data) or files[name].file_size != len(data):
            _fail(archive_path, f"RECORD size mismatch for {name!r}")


def validate_wheel(path: str | Path) -> None:
    """Validate one wheel as a safe ZIP with an exact, verified RECORD."""

    archive_path = Path(path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            _validated_member_keys(
                archive_path,
                kind="ZIP",
                entries=[(info.filename, info.is_dir()) for info in infos],
            )
            for info in infos:
                kind = stat.S_IFMT(info.external_attr >> 16)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    _fail(archive_path, f"non-regular ZIP member: {info.filename!r}")
            corrupt = archive.testzip()
            if corrupt is not None:
                _fail(archive_path, f"CRC failure in {corrupt!r}")
            _verify_record(archive_path, archive, infos)
    except ArchiveValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zlib.error) as error:
        _fail(archive_path, f"cannot read ZIP archive: {error}")


def validate_sdist(path: str | Path) -> None:
    """Validate one gzip-compressed sdist tar without extracting it."""

    archive_path = Path(path)
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            _validated_member_keys(
                archive_path,
                kind="tar",
                entries=[(member.name, member.isdir()) for member in members],
            )
            if not all(member.isfile() or member.isdir() for member in members):
                _fail(archive_path, "tar contains a non-regular member")
    except ArchiveValidationError:
        raise
    except (OSError, tarfile.TarError) as error:
        _fail(archive_path, f"cannot read tar archive: {error}")


def validate_archives(*, wheels: Sequence[str | Path], sdists: Sequence[str | Path]) -> None:
    """Validate all supplied wheel and sdist paths in order."""

    for wheel in wheels:
        validate_wheel(wheel)
    for sdist in sdists:
        validate_sdist(sdist)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", action="append", required=True, type=Path)
    parser.add_argument("--sdist", action="append", required=True, type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the archive validator CLI."""

    options = _parser().parse_args(arguments)
    try:
        validate_archives(wheels=options.wheel, sdists=options.sdist)
    except ArchiveValidationError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
