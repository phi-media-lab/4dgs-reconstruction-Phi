"""Build the public NumPy tensor cache from an admitted observation manifest.

Preparation is deliberately a format conversion, not an implicit calibration
step.  Images must already satisfy the public undistorted-pinhole contract.
The source manifest remains authoritative for camera, time, role, and image
identity; this cache changes only the I/O representation.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from p2g.audit import audit_observation_manifest
from p2g.canonical import sha256_file, write_new_bytes, write_new_json
from p2g.errors import ContractError, OutputExistsError
from p2g.schema import validate_payload

CACHE_MANIFEST_NAME = "tensor_cache.json"
CACHE_SCHEMA = "p2g.tensor_cache.v1"
INCOMPLETE_MARKER = ".p2g-incomplete"

Progress = Callable[[str], None]


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read observation manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("observation manifest must be a JSON object")
    return cast(dict[str, Any], value)


def _safe_image(root: Path, value: object, *, observation_id: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"observation {observation_id} has an invalid image path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"observation {observation_id} image escapes the image root")
    resolved_root = root.resolve()
    candidate = resolved_root / relative
    if candidate.is_symlink():
        raise ContractError(f"observation {observation_id} image is a symlink")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ContractError(f"observation {observation_id} image escapes the image root")
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError(f"observation {observation_id} image is not a regular file")
    return resolved


def _matrix(
    value: object,
    *,
    rows: int,
    columns: int,
    label: str,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must contain numbers") from exc
    if matrix.shape != (rows, columns) or not np.isfinite(matrix).all():
        raise ContractError(f"{label} must be a finite [{rows},{columns}] matrix")
    return matrix


def _decode_rgb8(
    path: Path,
    *,
    observation_id: str,
    expected_sha256: str,
    height: int,
    width: int,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    if sha256_file(path) != expected_sha256:
        raise ContractError(f"observation image changed after audit: {observation_id}")
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                raise ContractError(
                    f"observation {observation_id} must decode as RGB8, "
                    f"got Pillow mode {image.mode!r}"
                )
            decoded = np.array(image, dtype=np.uint8, copy=True)
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"cannot decode observation image {observation_id}: {exc}") from exc
    if decoded.shape != (height, width, 3):
        raise ContractError(f"observation {observation_id} decoded dimensions changed")
    return decoded


def _validate_public_pixel_contract(observation: Mapping[str, Any]) -> None:
    observation_id = cast(str, observation["observation_id"])
    camera = cast(Mapping[str, Any], observation["camera"])
    image = cast(Mapping[str, Any], observation["image"])
    encoding = cast(Mapping[str, Any], image["encoding"])
    if (
        camera["model"] != "pinhole"
        or camera["pixel_domain"] != "undistorted"
        or camera["distortion"] != []
    ):
        raise ContractError(
            f"observation {observation_id} is not an offline-undistorted pinhole image"
        )
    if (
        encoding["container"] not in {"png", "jpeg"}
        or
        encoding["channel_order"] != "RGB"
        or encoding["bit_depth"] != 8
        or encoding["stored_range"] != "full"
    ):
        raise ContractError(f"observation {observation_id} is outside the RGB8 cache subset")
    accepted_profiles = {
        "srgb_encoded": {"srgb_eotf_v1", "srgb_reference_assumption_v1"},
        "linear_rgb": {"linear_passthrough_v1"},
    }
    color_space = cast(str, image["color_space"])
    profile = cast(str, encoding["canonical_decode_profile"])
    if profile not in accepted_profiles.get(color_space, set()):
        raise ContractError(f"observation {observation_id} has no public decode contract")


def _flush_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _flush_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _array_record(path: Path, array: np.memmap[Any, Any]) -> dict[str, Any]:
    array.flush()
    _flush_file(path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "order": "C",
    }


def build_tensor_cache(
    output: Path,
    *,
    observation_manifest: Path,
    image_root: Path | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Create one append-only ``p2g.tensor_cache.v1`` directory.

    The output directory is claimed before any large write.  A manifest is
    published only after every array is flushed, hashed, and schema-validated.
    If preparation is interrupted, the directory remains visibly incomplete
    and is never resumed or overwritten implicitly.
    """

    manifest_path = observation_manifest.expanduser().resolve()
    root = (manifest_path.parent if image_root is None else image_root).expanduser().resolve()
    destination = output.expanduser().resolve()
    if os.path.lexists(destination):
        raise OutputExistsError(f"refusing to overwrite tensor cache: {destination}")

    manifest = _read_manifest(manifest_path)
    validate_payload("observation", manifest)
    for raw_observation in cast(list[dict[str, Any]], manifest["observations"]):
        _validate_public_pixel_contract(raw_observation)
    report = audit_observation_manifest(manifest, base_dir=root, verify_files=True)
    if report.status != "PASS":
        failures = [
            check.name for check in report.checks if check.required and check.status == "FAIL"
        ]
        raise ContractError(f"observation manifest audit failed: {failures}")

    raw_observations = cast(list[dict[str, Any]], manifest["observations"])
    observations: dict[tuple[int, str], dict[str, Any]] = {}
    for observation in raw_observations:
        key = (int(observation["frame_id"]), cast(str, observation["camera_id"]))
        if key in observations:
            raise ContractError(f"duplicate camera/frame observation: {key}")
        observations[key] = observation

    camera_ids = tuple(sorted({camera_id for _, camera_id in observations}))
    frame_ids = tuple(sorted({frame_id for frame_id, _ in observations}))
    expected = {(frame_id, camera_id) for frame_id in frame_ids for camera_id in camera_ids}
    if set(observations) != expected:
        raise ContractError("tensor cache requires a complete camera/frame grid")
    dimensions = {
        (int(item["image"]["height"]), int(item["image"]["width"]))
        for item in observations.values()
    }
    if len(dimensions) != 1:
        raise ContractError("tensor cache requires one RGB resolution across the scene")
    height, width = next(iter(dimensions))
    if height <= 0 or width <= 0:
        raise ContractError("tensor cache image dimensions must be positive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise OutputExistsError(f"refusing to overwrite tensor cache: {destination}") from exc
    marker = destination / INCOMPLETE_MARKER
    write_new_bytes(marker, b"p2g.tensor_cache.v1 preparation incomplete\n")

    rgb_path = destination / "rgb.npy"
    intrinsic_path = destination / "intrinsic.npy"
    extrinsic_path = destination / "world_to_camera.npy"
    time_path = destination / "timestamp_seconds.npy"
    axes = (len(frame_ids), len(camera_ids))
    rgb = np.lib.format.open_memmap(
        rgb_path,
        mode="w+",
        dtype=np.uint8,
        shape=(*axes, height, width, 3),
    )
    intrinsic = np.lib.format.open_memmap(
        intrinsic_path,
        mode="w+",
        dtype=np.float32,
        shape=(*axes, 3, 3),
    )
    world_to_camera = np.lib.format.open_memmap(
        extrinsic_path,
        mode="w+",
        dtype=np.float32,
        shape=(*axes, 4, 4),
    )
    timestamp_seconds = np.lib.format.open_memmap(
        time_path,
        mode="w+",
        dtype=np.float64,
        shape=axes,
    )

    total = len(expected)
    completed = 0
    for frame_index, frame_id in enumerate(frame_ids):
        for camera_index, camera_id in enumerate(camera_ids):
            observation = observations[(frame_id, camera_id)]
            observation_id = cast(str, observation["observation_id"])
            image = cast(dict[str, Any], observation["image"])
            camera = cast(dict[str, Any], observation["camera"])
            image_path = _safe_image(root, image["path"], observation_id=observation_id)
            rgb[frame_index, camera_index] = _decode_rgb8(
                image_path,
                observation_id=observation_id,
                expected_sha256=cast(str, image["sha256"]),
                height=height,
                width=width,
            )
            intrinsic[frame_index, camera_index] = _matrix(
                camera["intrinsic"],
                rows=3,
                columns=3,
                label=f"observation {observation_id} intrinsic",
            )
            world_to_camera[frame_index, camera_index] = _matrix(
                camera["world_to_camera"],
                rows=4,
                columns=4,
                label=f"observation {observation_id} world_to_camera",
            )
            timestamp = observation["timestamp_seconds"]
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise ContractError(f"observation {observation_id} timestamp must be numeric")
            timestamp_value = float(timestamp)
            if not math.isfinite(timestamp_value):
                raise ContractError(f"observation {observation_id} timestamp must be finite")
            timestamp_seconds[frame_index, camera_index] = timestamp_value
            completed += 1
            if progress is not None:
                progress(f"prepared {completed}/{total} observations")

    arrays = {
        "rgb": _array_record(rgb_path, rgb),
        "intrinsic": _array_record(intrinsic_path, intrinsic),
        "world_to_camera": _array_record(extrinsic_path, world_to_camera),
        "timestamp_seconds": _array_record(time_path, timestamp_seconds),
    }
    payload: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA,
        "observation_manifest_sha256": sha256_file(manifest_path),
        "camera_ids": list(camera_ids),
        "frame_ids": list(frame_ids),
        "arrays": arrays,
    }
    validate_payload("tensor_cache", payload)
    marker.unlink()
    _flush_directory(destination)
    write_new_json(destination / CACHE_MANIFEST_NAME, payload)
    _flush_directory(destination)
    _flush_directory(destination.parent)
    return payload


__all__ = ["CACHE_MANIFEST_NAME", "CACHE_SCHEMA", "build_tensor_cache"]
