"""Import an authorized local Charge v1.0 camera task into the public manifest."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlsplit

from p2g.audit import audit_observation_manifest
from p2g.canonical import sha256_file, sha256_json, write_new_json
from p2g.errors import ContractError
from p2g.image_probe import ImageProbe, probe_image
from p2g.schema import validate_payload

Progress = Callable[[str], None]

CHARGE_LICENSE = "CC-BY-4.0"
CHARGE_FPS = 96.0
CHARGE_CONVERSION = {
    "algorithm": "charge_v1_blender_c2w_to_p2g_opencv_w2c_v1",
    "source_extrinsic": "camera_to_world",
    "source_camera_axes": "blender_x_right_y_up_z_backward",
    "axis_change": [1.0, -1.0, -1.0, 1.0],
    "target_extrinsic": "world_to_camera",
    "target_camera_axes": "opencv_x_right_y_down_z_forward",
    "frame_mapping": "sorted_common_source_frame_ids_to_zero_based_contiguous_ids",
    "timestamp_mapping": "(source_frame_id-first_source_frame_id)/fps",
}
CHARGE_CONVERSION_SHA256 = sha256_json(CHARGE_CONVERSION)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CHARGE_SCENE = re.compile(r"^Charge-[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_source_repository(value: str) -> None:
    parsed = urlsplit(value)
    path = PurePosixPath(parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or parsed.query
        or parsed.fragment
        or parsed.path.endswith("/")
        or len(path.parts) != 4
        or path.parts[1:3] != ("datasets", "charge-benchmark")
        or not _CHARGE_SCENE.fullmatch(path.parts[3])
    ):
        raise ContractError(
            "source_repository must be the canonical HTTPS URL of an official "
            "charge-benchmark/Charge-* dataset repository"
        )


def _matrix(
    value: object,
    *,
    rows: int,
    columns: int,
    label: str,
) -> list[list[float]]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be a finite {rows}x{columns} matrix")
    raw_matrix = cast(list[object], value)
    if len(raw_matrix) != rows:
        raise ContractError(f"{label} must be a finite {rows}x{columns} matrix")
    result: list[list[float]] = []
    for raw_row in raw_matrix:
        if not isinstance(raw_row, list):
            raise ContractError(f"{label} must be a finite {rows}x{columns} matrix")
        raw_items = cast(list[object], raw_row)
        if len(raw_items) != columns:
            raise ContractError(f"{label} must be a finite {rows}x{columns} matrix")
        row: list[float] = []
        for raw_item in raw_items:
            if (
                not isinstance(raw_item, (int, float))
                or isinstance(raw_item, bool)
                or not math.isfinite(raw_item)
            ):
                raise ContractError(f"{label} must be a finite {rows}x{columns} matrix")
            row.append(float(raw_item))
        result.append(row)
    return result


def _determinant3(matrix: list[list[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _validate_rigid(matrix: list[list[float]], *, label: str) -> None:
    final_row_error = max(
        abs(value - expected)
        for value, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0), strict=True)
    )
    rotation = [row[:3] for row in matrix[:3]]
    orthogonality_error = max(
        abs(
            sum(rotation[left][axis] * rotation[right][axis] for axis in range(3))
            - (1.0 if left == right else 0.0)
        )
        for left in range(3)
        for right in range(3)
    )
    determinant = _determinant3(rotation)
    if (
        final_row_error > 1e-6
        or orthogonality_error > 1e-3
        or abs(determinant - 1.0) > 1e-3
    ):
        raise ContractError(f"{label} is not a right-handed rigid transform")


def charge_c2w_blender_to_w2c_opencv(value: object) -> list[list[float]]:
    """Convert a Charge Blender camera object transform into the public convention."""

    source = _matrix(value, rows=4, columns=4, label="Charge transformation_matrix")
    _validate_rigid(source, label="Charge transformation_matrix")
    signs = (1.0, -1.0, -1.0)
    camera_to_world_rotation = [
        [source[row][column] * signs[column] for column in range(3)]
        for row in range(3)
    ]
    translation = [source[row][3] for row in range(3)]
    world_to_camera_rotation = [
        [camera_to_world_rotation[column][row] for column in range(3)]
        for row in range(3)
    ]
    inverse_translation = [
        -sum(
            world_to_camera_rotation[row][axis] * translation[axis]
            for axis in range(3)
        )
        for row in range(3)
    ]
    converted = [
        [*world_to_camera_rotation[0], inverse_translation[0]],
        [*world_to_camera_rotation[1], inverse_translation[1]],
        [*world_to_camera_rotation[2], inverse_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    _validate_rigid(converted, label="converted world_to_camera")
    return converted


def _load_transforms(path: Path, *, label: str) -> dict[str, list[dict[str, Any]]]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ContractError(f"{label} must be a non-empty camera dictionary")
    result: dict[str, list[dict[str, Any]]] = {}
    for raw_camera_id, raw_records in cast(dict[object, object], value).items():
        if (
            not isinstance(raw_camera_id, str)
            or not _IDENTIFIER.fullmatch(raw_camera_id)
            or len(raw_camera_id) > 96
        ):
            raise ContractError(f"{label} contains an invalid camera identifier")
        if not isinstance(raw_records, list) or not raw_records:
            raise ContractError(f"{label} camera {raw_camera_id} has no frame records")
        records: list[dict[str, Any]] = []
        for raw_record in cast(list[object], raw_records):
            if not isinstance(raw_record, dict):
                raise ContractError(f"{label} camera {raw_camera_id} has a non-object record")
            records.append(cast(dict[str, Any], raw_record))
        result[raw_camera_id] = records
    return result


def _source_member(root: Path, value: object, *, label: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} must be a non-empty POSIX relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ContractError(f"{label} escapes the Charge task root")
    unresolved = root.joinpath(*relative.parts)
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file() or unresolved.is_symlink():
        raise ContractError(f"{label} is not a regular file inside the Charge task root")
    return relative.as_posix(), resolved


def _relative_input(root: Path, value: Path, *, label: str) -> tuple[str, Path]:
    if value.is_absolute():
        raise ContractError(f"{label} must be relative to the Charge task root")
    return _source_member(root, value.as_posix(), label=label)


def _frame_id(record: dict[str, Any], *, label: str) -> int:
    value = record.get("frame_id")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{label} frame_id must be a non-negative integer")
    return value


def _ordered_records(
    cameras: dict[str, list[dict[str, Any]]], *, label: str
) -> tuple[tuple[int, ...], dict[str, list[dict[str, Any]]]]:
    ordered: dict[str, list[dict[str, Any]]] = {}
    expected: tuple[int, ...] | None = None
    for camera_id in sorted(cameras):
        records = sorted(
            cameras[camera_id],
            key=lambda item: _frame_id(item, label=f"{label}/{camera_id}"),
        )
        frame_ids = tuple(
            _frame_id(item, label=f"{label}/{camera_id}") for item in records
        )
        if len(frame_ids) != len(set(frame_ids)):
            raise ContractError(f"{label} camera {camera_id} repeats a frame_id")
        if expected is None:
            expected = frame_ids
        elif frame_ids != expected:
            raise ContractError(f"{label} cameras do not share one exact frame set")
        ordered[camera_id] = records
    assert expected is not None
    if expected != tuple(range(expected[0], expected[-1] + 1)):
        raise ContractError(f"{label} source frame identifiers must be contiguous")
    return expected, ordered


def _static_camera(
    camera_id: str,
    records: list[dict[str, Any]],
) -> tuple[list[list[float]], list[list[float]], int, int]:
    first = records[0]
    intrinsic = _matrix(first.get("K"), rows=3, columns=3, label=f"{camera_id} K")
    source_pose = _matrix(
        first.get("transformation_matrix"),
        rows=4,
        columns=4,
        label=f"{camera_id} transformation_matrix",
    )
    width = first.get("resolution_x")
    height = first.get("resolution_y")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height <= 0
    ):
        raise ContractError(f"{camera_id} has invalid image dimensions")
    for record in records:
        if record.get("type") != "PERSP":
            raise ContractError(f"{camera_id} must use a Charge PERSP camera")
        aspect = record.get("pixel_aspect_ratio")
        if (
            not isinstance(aspect, (int, float))
            or isinstance(aspect, bool)
            or not math.isfinite(aspect)
            or abs(float(aspect) - 1.0) > 1e-9
        ):
            raise ContractError(f"{camera_id} must use square pixels")
        if record.get("resolution_x") != width or record.get("resolution_y") != height:
            raise ContractError(f"{camera_id} image dimensions vary over time")
        if _matrix(record.get("K"), rows=3, columns=3, label=f"{camera_id} K") != intrinsic:
            raise ContractError(f"{camera_id} intrinsics vary over time")
        if (
            _matrix(
                record.get("transformation_matrix"),
                rows=4,
                columns=4,
                label=f"{camera_id} transformation_matrix",
            )
            != source_pose
        ):
            raise ContractError(f"{camera_id} pose varies over time")
    return intrinsic, charge_c2w_blender_to_w2c_opencv(source_pose), width, height


def _encoding(probe: ImageProbe) -> dict[str, Any]:
    if (
        probe.container != "png"
        or probe.channel_order != "RGB"
        or probe.bit_depth != 8
        or probe.stored_range != "full"
    ):
        raise ContractError("Charge RGB input must be a full-range RGB8 PNG")
    declared_srgb = (
        probe.declared_transfer == "IEC 61966-2-1 sRGB"
        and probe.declared_primaries == "IEC 61966-2-1 sRGB"
    )
    return {
        "container": probe.container,
        "channel_order": probe.channel_order,
        "bit_depth": probe.bit_depth,
        "stored_range": probe.stored_range,
        "declared_transfer": probe.declared_transfer,
        "declared_primaries": probe.declared_primaries,
        "declared_matrix": probe.declared_matrix,
        "canonical_decode_profile": (
            "srgb_eotf_v1" if declared_srgb else "srgb_reference_assumption_v1"
        ),
    }


def import_charge_manifest(
    task_root: Path,
    *,
    train_transforms: Path,
    test_transforms: Path,
    output: Path,
    dataset_id: str,
    source_repository: str,
    source_revision: str,
    sealed_camera_count: int,
    fps: float = CHARGE_FPS,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Hash and convert a local fixed-rig Charge task without copying its pixels."""

    root = task_root.expanduser().resolve()
    if not root.is_dir():
        raise ContractError("Charge task root is not a directory")
    if not _IDENTIFIER.fullmatch(dataset_id) or len(dataset_id) > 256:
        raise ContractError("dataset_id is outside the public identifier contract")
    _validate_source_repository(source_repository)
    if not _REVISION.fullmatch(source_revision):
        raise ContractError("source_revision must be one lowercase 40-character Git revision")
    if isinstance(sealed_camera_count, bool) or sealed_camera_count < 0:
        raise ContractError("sealed_camera_count must be a non-negative integer")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ContractError("fps must be a positive finite number")

    train_relative, train_path = _relative_input(
        root, train_transforms, label="train_transforms"
    )
    test_relative, test_path = _relative_input(
        root, test_transforms, label="test_transforms"
    )
    train_frame_ids, train_cameras = _ordered_records(
        _load_transforms(train_path, label="train transforms"), label="train transforms"
    )
    test_frame_ids, test_cameras = _ordered_records(
        _load_transforms(test_path, label="test transforms"), label="test transforms"
    )
    if train_frame_ids != test_frame_ids:
        raise ContractError("Charge train and test cameras do not share one exact frame set")
    if set(train_cameras) & set(test_cameras):
        raise ContractError("Charge train and test camera identifiers overlap")
    if sealed_camera_count >= len(test_cameras):
        raise ContractError("sealed_camera_count must leave at least one diagnostic camera")

    test_ids = sorted(test_cameras)
    sealed_ids: set[str] = set()
    if sealed_camera_count:
        sealed_ids = set(test_ids[len(test_ids) - sealed_camera_count :])
    roles = {
        **dict.fromkeys(train_cameras, "train"),
        **{
            camera_id: "sealed" if camera_id in sealed_ids else "diagnostic"
            for camera_id in test_ids
        },
    }
    cameras = {**train_cameras, **test_cameras}
    camera_models = {
        camera_id: _static_camera(camera_id, records)
        for camera_id, records in cameras.items()
    }
    observations: list[dict[str, Any]] = []
    source_files = [
        {
            "bytes": train_path.stat().st_size,
            "path": train_relative,
            "sha256": sha256_file(train_path),
        },
        {
            "bytes": test_path.stat().st_size,
            "path": test_relative,
            "sha256": sha256_file(test_path),
        },
    ]
    seen_paths: set[str] = set()
    total = len(cameras) * len(train_frame_ids)
    completed = 0
    source_to_canonical = {
        source_frame_id: canonical_frame_id
        for canonical_frame_id, source_frame_id in enumerate(train_frame_ids)
    }
    for camera_id in sorted(cameras):
        intrinsic, world_to_camera, width, height = camera_models[camera_id]
        for record in cameras[camera_id]:
            source_frame_id = _frame_id(record, label=f"{camera_id} record")
            frame_id = source_to_canonical[source_frame_id]
            relative, image_path = _source_member(
                root,
                record.get("image_path"),
                label=f"{camera_id}/{source_frame_id} image_path",
            )
            if relative in seen_paths:
                raise ContractError(f"Charge image path is repeated: {relative}")
            seen_paths.add(relative)
            probe = probe_image(image_path)
            if (probe.width, probe.height) != (width, height):
                raise ContractError(
                    f"Charge image dimensions disagree with camera JSON: {relative}"
                )
            digest = sha256_file(image_path)
            source_files.append(
                {"bytes": image_path.stat().st_size, "path": relative, "sha256": digest}
            )
            observations.append(
                {
                    "observation_id": f"charge_{camera_id}_{frame_id:06d}",
                    "camera_id": camera_id,
                    "frame_id": frame_id,
                    "timestamp_seconds": (source_frame_id - train_frame_ids[0]) / float(fps),
                    "role": roles[camera_id],
                    "image": {
                        "path": relative,
                        "sha256": digest,
                        "width": width,
                        "height": height,
                        "color_space": "srgb_encoded",
                        "encoding": _encoding(probe),
                    },
                    "camera": {
                        "model": "pinhole",
                        "pixel_domain": "undistorted",
                        "intrinsic": intrinsic,
                        "world_to_camera": world_to_camera,
                        "distortion": [],
                    },
                }
            )
            completed += 1
            if progress is not None and (completed == total or completed % 100 == 0):
                progress(f"hashed {completed}/{total} Charge RGB observations")

    source_inventory = {
        "schema_version": "p2g.charge_source_selection.v1",
        "source_repository": source_repository,
        "source_revision": source_revision,
        "train_transforms": train_relative,
        "test_transforms": test_relative,
        "fps": float(fps),
        "sealed_camera_count": sealed_camera_count,
        "conversion_sha256": CHARGE_CONVERSION_SHA256,
        "files": sorted(source_files, key=lambda item: cast(str, item["path"])),
    }
    manifest: dict[str, Any] = {
        "schema_version": "p2g.observation_manifest.v2",
        "dataset_id": dataset_id,
        "source": {
            "description": (
                f"Charge v1.0 RGB selection from {source_repository} at "
                f"{source_revision}; source frames {train_frame_ids[0]}..{train_frame_ids[-1]}"
            ),
            "license": CHARGE_LICENSE,
            "license_status": "declared",
            "root_sha256": sha256_json(source_inventory),
        },
        "coordinate_conventions": {
            "handedness": "right",
            "extrinsic": "world_to_camera",
            "pixel_center": "half_pixel",
            "time_unit": "seconds",
            "photometric_space": "linear_rgb",
        },
        "sync": {
            "variant": "charge_exact_blender_frame_v1",
            "tolerance_seconds": 0.0,
            "per_camera_offset_seconds": dict.fromkeys(sorted(cameras), 0.0),
        },
        "transforms": [
            {
                "name": cast(str, CHARGE_CONVERSION["algorithm"]),
                "config_sha256": CHARGE_CONVERSION_SHA256,
            }
        ],
        "observations": observations,
    }
    validate_payload("observation", manifest)
    report = audit_observation_manifest(manifest, base_dir=root, verify_files=False)
    if report.status != "PASS":
        failures = [
            check.name
            for check in report.checks
            if check.required and check.status == "FAIL"
        ]
        raise ContractError(f"generated Charge manifest failed semantic audit: {failures}")
    destination = output.expanduser().resolve()
    write_new_json(destination, manifest)
    role_counts = Counter(cast(str, item["role"]) for item in observations)
    return {
        "status": "PASS",
        "dataset_id": dataset_id,
        "manifest_sha256": sha256_file(destination),
        "source_root_sha256": manifest["source"]["root_sha256"],
        "conversion": cast(str, CHARGE_CONVERSION["algorithm"]),
        "fps": float(fps),
        "source_frame_range_inclusive": [train_frame_ids[0], train_frame_ids[-1]],
        "frame_count": len(train_frame_ids),
        "camera_counts": {
            "train": len(train_cameras),
            "diagnostic": len(test_cameras) - sealed_camera_count,
            "sealed": sealed_camera_count,
        },
        "observation_counts": dict(sorted(role_counts.items())),
    }


__all__ = [
    "CHARGE_CONVERSION",
    "CHARGE_CONVERSION_SHA256",
    "CHARGE_FPS",
    "CHARGE_LICENSE",
    "charge_c2w_blender_to_w2c_opencv",
    "import_charge_manifest",
]
