#!/usr/bin/env python3
"""Verify that a clean Git snapshot contains source rather than run artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

MAX_TRACKED_FILE_BYTES = 1024 * 1024
FORBIDDEN_ARTIFACT_SUFFIXES = frozenset(
    {
        ".avi",
        ".bin",
        ".ckpt",
        ".mkv",
        ".mov",
        ".mp4",
        ".npy",
        ".npz",
        ".ply",
        ".pt",
        ".pth",
        ".safetensors",
    }
)


class SourceTreeError(RuntimeError):
    """Raised when the checked-out release tree violates a source boundary."""


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise SourceTreeError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and "\\" not in value and not path.is_absolute() and ".." not in path.parts


def _tracked_modes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in _git(root, "ls-files", "-s", "-z").split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3:
            raise SourceTreeError("git index emitted an unrecognized entry")
        try:
            path = raw_path.decode("utf-8")
            mode = fields[0].decode("ascii")
            stage = fields[2].decode("ascii")
        except UnicodeDecodeError as exc:
            raise SourceTreeError("tracked paths and index metadata must be UTF-8/ASCII") from exc
        if stage != "0" or path in result:
            raise SourceTreeError(f"tracked path has unresolved index stages: {path!r}")
        if not _safe_relative(path):
            raise SourceTreeError(f"tracked path is not repository-relative: {path!r}")
        result[path] = mode
    return result


def _require_clean_head(root: Path) -> None:
    _git(root, "rev-parse", "--verify", "HEAD")
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=normal",
        "--ignore-submodules=none",
    )
    if status:
        raise SourceTreeError("verification requires a clean committed HEAD")


def _credential_labels(payload: bytes) -> list[str]:
    patterns = (
        ("aws_access_key", re.compile((b"AK" + b"IA") + rb"[0-9A-Z]{16}")),
        (
            "private_key",
            re.compile(b"-----BEGIN " + rb"(?:RSA |OPENSSH |EC )?" + b"PRIVATE KEY-----"),
        ),
        ("github_token", re.compile((b"gh" + rb"[pousr]_") + rb"[A-Za-z0-9_]{20,}")),
        ("api_key", re.compile((b"s" + b"k-") + rb"[A-Za-z0-9]{20,}")),
        ("huggingface_token", re.compile((b"hf" + b"_") + rb"[A-Za-z0-9]{20,}")),
        ("url_credentials", re.compile(rb"https?://[^\s/@:]+:[^\s/@]+@")),
    )
    return [label for label, pattern in patterns if pattern.search(payload)]


def verify_repository(root: Path) -> dict[str, object]:
    root = root.resolve()
    if not (root / ".git").exists():
        raise SourceTreeError("verification root must be a Git worktree")
    _require_clean_head(root)
    tracked = _tracked_modes(root)
    if not tracked:
        raise SourceTreeError("release tree contains no tracked files")

    findings: list[tuple[str, str]] = []
    scanned_bytes = 0
    for relative, mode in sorted(tracked.items()):
        if mode in {"120000", "160000"} or not mode.startswith("100"):
            raise SourceTreeError(f"unsupported tracked Git mode {mode} for {relative!r}")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise SourceTreeError(f"tracked member is not a regular file: {relative!r}")
        payload = path.read_bytes()
        scanned_bytes += len(payload)
        if len(payload) > MAX_TRACKED_FILE_BYTES:
            raise SourceTreeError(
                f"tracked member exceeds {MAX_TRACKED_FILE_BYTES} bytes: {relative!r}"
            )
        if PurePosixPath(relative).suffix.casefold() in FORBIDDEN_ARTIFACT_SUFFIXES:
            raise SourceTreeError(f"tracked training/data artifact is forbidden: {relative!r}")
        findings.extend((relative, label) for label in _credential_labels(payload))

    if findings:
        rendered = ", ".join(f"{path}:{label}" for path, label in findings)
        raise SourceTreeError(f"tracked source contains credential-shaped material: {rendered}")

    return {
        "credential_pattern_matches": 0,
        "forbidden_artifact_count": 0,
        "maximum_tracked_file_bytes": MAX_TRACKED_FILE_BYTES,
        "result": "PASS",
        "scanned_bytes": scanned_bytes,
        "schema_version": "p2g.source_hygiene.v1",
        "symlink_or_submodule_count": 0,
        "tracked_file_count": len(tracked),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    try:
        result = verify_repository(arguments.root)
    except SourceTreeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
