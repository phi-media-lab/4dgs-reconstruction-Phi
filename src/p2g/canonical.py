from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from p2g.errors import ContractError, OutputExistsError


def _normalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return _normalize(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("canonical JSON rejects NaN and infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in cast(Mapping[object, object], value).items():
            if not isinstance(key, str):
                raise ContractError(f"canonical JSON requires string keys, got {type(key)!r}")
            normalized[key] = _normalize(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in cast(Sequence[object], value)]
    raise ContractError(f"unsupported canonical JSON value: {type(value)!r}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using the project's stable JSON representation."""

    normalized = _normalize(value)
    encoded = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (encoded + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new_bytes(path: Path, data: bytes) -> None:
    """Publish a complete file without overwriting an existing artifact."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise OutputExistsError(f"refusing to overwrite evidence path: {path}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise OutputExistsError(f"refusing to overwrite evidence path: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_new_json(path: Path, value: Any) -> None:
    write_new_bytes(path, canonical_json_bytes(value))


def read_json(path: Path) -> Any:
    with path.open("rb") as stream:
        return json.load(stream)


def content_id(namespace: str, value: Any, *, prefix: str | None = None) -> str:
    if not namespace:
        raise ContractError("content ID namespace must not be empty")
    digest = sha256_bytes(namespace.encode("utf-8") + b"\0" + canonical_json_bytes(value))
    return f"{prefix}_{digest[:32]}" if prefix else digest[:32]
