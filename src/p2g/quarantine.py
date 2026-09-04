"""Recoverable, hash-inventoried quarantine for incomplete stage directories."""

from __future__ import annotations

import contextlib
import re
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json, write_new_bytes
from p2g.errors import ContractError
from p2g.schema import validate_payload

_ATTEMPT = re.compile(r"^(?P<ordinal>[0-9]{2})-(?P<stage>[a-z]+)-(?P<attempt>[0-9]{6})$")


def _below(path: Path, root: Path, *, label: str) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError(f"{label} escaped its declared root") from exc
    if not relative.parts:
        raise ContractError(f"{label} cannot be the declared root")
    return relative.as_posix()


def _inventory(root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    directories: list[str] = []
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError(f"quarantine source contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(relative)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"quarantine source contains a special file: {relative}")
        before = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        digest = sha256_file(path)
        after_stat = path.stat()
        after = (
            after_stat.st_dev,
            after_stat.st_ino,
            after_stat.st_size,
            after_stat.st_mtime_ns,
        )
        if before != after:
            raise ContractError(f"quarantine source changed during inventory: {relative}")
        files.append(
            {
                "path": relative,
                "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
                "bytes": metadata.st_size,
                "sha256": digest,
            }
        )
    return directories, files


def _next_attempt(root: Path, *, ordinal: int, stage: str) -> tuple[int, Path]:
    attempts: list[int] = []
    prefix = f"{ordinal:02d}-{stage}-"
    for entry in root.iterdir():
        if entry.name.startswith(".pending-"):
            raise ContractError(
                f"quarantine contains an interrupted pending transaction: {entry.name}"
            )
        match = _ATTEMPT.fullmatch(entry.name)
        if match is None or not entry.is_dir() or entry.is_symlink():
            raise ContractError("quarantine directory contains an unsafe entry")
        if entry.name.startswith(prefix):
            attempts.append(int(match.group("attempt")))
    attempts.sort()
    if attempts != list(range(1, len(attempts) + 1)):
        raise ContractError("quarantine attempt sequence is not contiguous")
    attempt = len(attempts) + 1
    return attempt, root / f"{prefix}{attempt:06d}"


def quarantine_stage_directory(
    source: Path,
    *,
    workspace_root: Path,
    quarantine_root: Path,
    ordinal: int,
    stage: str,
    reason: str,
) -> Path:
    """Atomically remove a partial stage path while retaining every byte."""

    workspace = workspace_root.resolve()
    candidate = source.resolve()
    if source.is_symlink() or not source.is_dir():
        raise ContractError("quarantine source must be a regular non-symlink directory")
    original_relative = _below(candidate, workspace, label="quarantine source")
    if quarantine_root.is_symlink():
        raise ContractError("quarantine root must not be a symlink")
    root = quarantine_root.resolve()
    if root == workspace or workspace not in root.parents:
        raise ContractError("quarantine root must be below the workspace")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ContractError("quarantine root must be a regular non-symlink directory")
    if type(ordinal) is not int or ordinal < 0 or not stage.isalpha() or not reason:
        raise ContractError("quarantine stage metadata is invalid")

    directories, files = _inventory(candidate)
    attempt, destination = _next_attempt(root, ordinal=ordinal, stage=stage)
    staging = Path(
        tempfile.mkdtemp(prefix=f".pending-{ordinal:02d}-{stage}-", dir=root)
    )
    payload = staging / "payload"
    try:
        candidate.rename(payload)
    except OSError as exc:
        with contextlib.suppress(OSError):
            staging.rmdir()
        raise ContractError(f"cannot atomically quarantine stage directory: {exc}") from exc

    receipt: dict[str, Any] = {
        "schema_version": "p2g.stage_quarantine.v1",
        "status": "QUARANTINED",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "stage": stage,
        "ordinal": ordinal,
        "attempt": attempt,
        "reason": reason,
        "original_path": original_relative,
        "payload_path": "payload",
        "directories": directories,
        "files": files,
        "summary": {
            "directory_count": len(directories),
            "file_count": len(files),
            "total_bytes": sum(item["bytes"] for item in files),
        },
        "claim_boundary": (
            "Quarantine preserves an incomplete or invalidated attempt for audit; "
            "it is not a completed stage and may not be consumed downstream."
        ),
    }
    receipt["logical_sha256"] = sha256_json(receipt)
    validate_payload("stage_quarantine", receipt)
    write_new_bytes(staging / "quarantine.json", canonical_json_bytes(receipt))
    try:
        staging.rename(destination)
    except OSError as exc:
        raise ContractError(
            f"quarantine payload was preserved at {staging}, but closure failed: {exc}"
        ) from exc
    return destination / "quarantine.json"


__all__ = ["quarantine_stage_directory"]
