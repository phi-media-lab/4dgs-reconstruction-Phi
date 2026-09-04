"""Semantic audits for public Pixel4DGS artifact contracts.

The public package accepts only the v2 observation manifest.  Schema validation
is necessary but deliberately not sufficient: this module also checks path
containment, image identity and headers, camera geometry, synchronization, and
role isolation before pixels can enter the training chain.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from p2g.canonical import sha256_file, sha256_json
from p2g.errors import ContractError
from p2g.image_probe import probe_image
from p2g.schema import validate_payload


@dataclass(frozen=True)
class AuditCheck:
    """One machine-readable audit assertion."""

    name: str
    status: str
    required: bool
    detail: Any = None


def _empty_audit_checks() -> list[AuditCheck]:
    return []


@dataclass
class AuditReport:
    """A collection of checks whose required failures determine the verdict."""

    subject: str
    checks: list[AuditCheck] = field(default_factory=_empty_audit_checks)
    created_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    schema_version: str = "p2g.audit.v1"

    @property
    def status(self) -> str:
        """Return ``FAIL`` exactly when a required check failed."""

        if any(check.required and check.status == "FAIL" for check in self.checks):
            return "FAIL"
        return "PASS"

    def add(
        self,
        name: str,
        passed: bool,
        *,
        detail: Any = None,
        required: bool = True,
    ) -> None:
        """Append one explicit check result."""

        self.checks.append(
            AuditCheck(
                name=name,
                status="PASS" if passed else "FAIL",
                required=required,
                detail=detail,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report without hiding optional failures."""

        return {
            "schema_version": self.schema_version,
            "subject": self.subject,
            "created_utc": self.created_utc,
            "status": self.status,
            "checks": [asdict(check) for check in self.checks],
        }


def _finite_matrix(value: Any, rows: int, columns: int) -> bool:
    if not isinstance(value, list):
        return False
    matrix = cast(list[object], value)
    if len(matrix) != rows:
        return False
    for untyped_row in matrix:
        if not isinstance(untyped_row, list):
            return False
        row = cast(list[object], untyped_row)
        if len(row) != columns:
            return False
        if not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            for item in row
        ):
            return False
    return True


def _determinant3(matrix: list[list[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _rotation_error(matrix: list[list[float]]) -> tuple[float, float]:
    rotation = [row[:3] for row in matrix[:3]]
    maximum_error = 0.0
    for row_a in range(3):
        for row_b in range(3):
            dot = sum(
                rotation[row_a][axis] * rotation[row_b][axis] for axis in range(3)
            )
            expected = 1.0 if row_a == row_b else 0.0
            maximum_error = max(maximum_error, abs(dot - expected))
    return maximum_error, _determinant3(rotation)


def _rigid_inverse(matrix: list[list[float]]) -> list[list[float]]:
    rotation = [row[:3] for row in matrix[:3]]
    translation = [row[3] for row in matrix[:3]]
    rotation_transpose = [
        [rotation[column][row] for column in range(3)] for row in range(3)
    ]
    inverse_translation = [
        -sum(rotation_transpose[row][axis] * translation[axis] for axis in range(3))
        for row in range(3)
    ]
    return [
        [*rotation_transpose[0], inverse_translation[0]],
        [*rotation_transpose[1], inverse_translation[1]],
        [*rotation_transpose[2], inverse_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matrix_identity_error(left: list[list[float]], right: list[list[float]]) -> float:
    maximum_error = 0.0
    for row in range(4):
        for column in range(4):
            value = sum(left[row][axis] * right[axis][column] for axis in range(4))
            expected = 1.0 if row == column else 0.0
            maximum_error = max(maximum_error, abs(value - expected))
    return maximum_error


def _distort_radtan(x: float, y: float, distortion: list[float]) -> tuple[float, float]:
    k1, k2, p1, p2 = distortion[:4]
    k3 = distortion[4] if len(distortion) == 5 else 0.0
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    xy2 = 2.0 * x * y
    return (
        x * radial + p1 * xy2 + p2 * (radius2 + 2.0 * x * x),
        y * radial + p1 * (radius2 + 2.0 * y * y) + p2 * xy2,
    )


def _undistort_radtan(
    distorted_x: float,
    distorted_y: float,
    distortion: list[float],
) -> tuple[float, float]:
    x, y = distorted_x, distorted_y
    for _ in range(40):
        projected_x, projected_y = _distort_radtan(x, y, distortion)
        residual_x = distorted_x - projected_x
        residual_y = distorted_y - projected_y
        x += residual_x
        y += residual_y
        if max(abs(residual_x), abs(residual_y)) <= 1e-13:
            break
        if not math.isfinite(x) or not math.isfinite(y):
            raise ContractError("OpenCV distortion inversion diverged")
    return x, y


def _camera_roundtrip_error(camera: dict[str, Any], width: int, height: int) -> float:
    intrinsic = cast(list[list[float]], camera["intrinsic"])
    fx = intrinsic[0][0]
    skew = intrinsic[0][1]
    cx = intrinsic[0][2]
    fy = intrinsic[1][1]
    cy = intrinsic[1][2]
    distortion = cast(list[float], camera["distortion"])
    pixels = (
        (0.5, 0.5),
        (width - 0.5, 0.5),
        (0.5, height - 0.5),
        (width - 0.5, height - 0.5),
        (cx, cy),
    )
    maximum_error = 0.0
    for pixel_x, pixel_y in pixels:
        distorted_y = (pixel_y - cy) / fy
        distorted_x = (pixel_x - cx - skew * distorted_y) / fx
        if camera["model"] == "opencv_radtan":
            ray_x, ray_y = _undistort_radtan(distorted_x, distorted_y, distortion)
            projected_x, projected_y = _distort_radtan(ray_x, ray_y, distortion)
        else:
            projected_x, projected_y = distorted_x, distorted_y
        roundtrip_x = fx * projected_x + skew * projected_y + cx
        roundtrip_y = fy * projected_y + cy
        maximum_error = max(
            maximum_error,
            abs(roundtrip_x - pixel_x),
            abs(roundtrip_y - pixel_y),
        )
    return maximum_error


def _safe_image_path(base_dir: Path, relative: str) -> tuple[bool, Path]:
    candidate = Path(relative)
    if candidate.is_absolute() or "\\" in relative:
        return False, candidate
    root = base_dir.resolve()
    resolved = (root / candidate).resolve()
    return resolved.is_relative_to(root), resolved


def _audit_paths_and_files(
    report: AuditReport,
    observations: list[dict[str, Any]],
    *,
    base_dir: Path,
    verify_files: bool,
) -> None:
    path_failures: list[dict[str, Any]] = []
    file_failures: list[dict[str, Any]] = []
    header_failures: list[dict[str, Any]] = []
    files_verified = 0

    for item in observations:
        identifier = item["observation_id"]
        relative = item["image"]["path"]
        is_safe, image_path = _safe_image_path(base_dir, relative)
        if not is_safe:
            path_failures.append({"observation_id": identifier, "path": relative})
            continue
        if not verify_files:
            continue
        if not image_path.is_file():
            file_failures.append({"observation_id": identifier, "reason": "missing"})
            continue
        actual = sha256_file(image_path)
        expected = item["image"]["sha256"]
        if actual != expected:
            file_failures.append(
                {
                    "observation_id": identifier,
                    "reason": "sha256_mismatch",
                    "expected": expected,
                    "actual": actual,
                }
            )
            continue
        files_verified += 1
        try:
            probe = probe_image(image_path)
        except ContractError as exc:
            header_failures.append({"observation_id": identifier, "reason": str(exc)})
            continue

        encoding = item["image"]["encoding"]
        comparisons = {
            "container": (encoding["container"], probe.container),
            "channel_order": (encoding["channel_order"], probe.channel_order),
            "bit_depth": (encoding["bit_depth"], probe.bit_depth),
            "stored_range": (encoding["stored_range"], probe.stored_range),
            "declared_transfer": (
                encoding["declared_transfer"],
                probe.declared_transfer,
            ),
            "declared_primaries": (
                encoding["declared_primaries"],
                probe.declared_primaries,
            ),
            "declared_matrix": (encoding["declared_matrix"], probe.declared_matrix),
            "width": (item["image"]["width"], probe.width),
            "height": (item["image"]["height"], probe.height),
        }
        mismatches = {
            name: {"expected": expected_value, "actual": actual_value}
            for name, (expected_value, actual_value) in comparisons.items()
            if expected_value != actual_value
        }
        if mismatches:
            header_failures.append(
                {
                    "observation_id": identifier,
                    "reason": "header_mismatch",
                    "mismatches": mismatches,
                }
            )

    report.add("relative_paths_within_root", not path_failures, detail=path_failures)
    report.add(
        "image_files_and_hashes",
        not file_failures,
        detail={
            "requested": verify_files,
            "verified": files_verified,
            "failures": file_failures,
        },
        required=verify_files,
    )
    report.add(
        "image_headers_match_manifest",
        not header_failures,
        detail={
            "requested": verify_files,
            "probed": files_verified - len(header_failures),
            "failures": header_failures,
        },
        required=verify_files,
    )


def _audit_cameras(report: AuditReport, observations: list[dict[str, Any]]) -> None:
    intrinsic_failures: list[str] = []
    extrinsic_failures: list[dict[str, Any]] = []
    inverse_failures: list[dict[str, Any]] = []
    camera_image_failures: list[dict[str, Any]] = []
    roundtrip_failures: list[dict[str, Any]] = []

    for item in observations:
        identifier = item["observation_id"]
        intrinsic_value = item["camera"]["intrinsic"]
        extrinsic_value = item["camera"]["world_to_camera"]
        if not _finite_matrix(intrinsic_value, 3, 3):
            intrinsic_failures.append(identifier)
        else:
            intrinsic = cast(list[list[float]], intrinsic_value)
            focal_ok = intrinsic[0][0] > 0 and intrinsic[1][1] > 0
            canonical_form_ok = (
                max(
                    abs(intrinsic[1][0]),
                    abs(intrinsic[2][0]),
                    abs(intrinsic[2][1]),
                    abs(intrinsic[2][2] - 1.0),
                )
                <= 1e-6
            )
            if not focal_ok or not canonical_form_ok:
                intrinsic_failures.append(identifier)
            else:
                width = item["image"]["width"]
                height = item["image"]["height"]
                principal_point_ok = (
                    -0.5 <= intrinsic[0][2] <= width + 0.5
                    and -0.5 <= intrinsic[1][2] <= height + 0.5
                )
                if not principal_point_ok:
                    camera_image_failures.append(
                        {
                            "observation_id": identifier,
                            "reason": "principal_point_outside_image",
                            "principal_point": [intrinsic[0][2], intrinsic[1][2]],
                            "image_size": [width, height],
                        }
                    )
                try:
                    roundtrip_error = _camera_roundtrip_error(
                        item["camera"], width, height
                    )
                except (ContractError, ZeroDivisionError, OverflowError) as exc:
                    roundtrip_failures.append(
                        {"observation_id": identifier, "reason": str(exc)}
                    )
                else:
                    if roundtrip_error > 1e-6:
                        roundtrip_failures.append(
                            {
                                "observation_id": identifier,
                                "maximum_pixel_error": roundtrip_error,
                            }
                        )

        if not _finite_matrix(extrinsic_value, 4, 4):
            extrinsic_failures.append(
                {"observation_id": identifier, "reason": "non_finite"}
            )
            continue
        extrinsic = cast(list[list[float]], extrinsic_value)
        final_row_error = max(
            abs(actual - expected)
            for actual, expected in zip(extrinsic[3], [0.0, 0.0, 0.0, 1.0], strict=True)
        )
        rotation_error, determinant = _rotation_error(extrinsic)
        if (
            final_row_error > 1e-6
            or rotation_error > 1e-3
            or abs(determinant - 1.0) > 1e-3
        ):
            extrinsic_failures.append(
                {
                    "observation_id": identifier,
                    "final_row_error": final_row_error,
                    "rotation_error": rotation_error,
                    "rotation_determinant": determinant,
                }
            )
            continue
        camera_to_world = _rigid_inverse(extrinsic)
        inverse_error = max(
            _matrix_identity_error(extrinsic, camera_to_world),
            _matrix_identity_error(camera_to_world, extrinsic),
        )
        if inverse_error > 1e-6:
            inverse_failures.append(
                {
                    "observation_id": identifier,
                    "maximum_identity_error": inverse_error,
                }
            )

    report.add("intrinsic_contract", not intrinsic_failures, detail=intrinsic_failures)
    report.add(
        "extrinsic_rigid_contract", not extrinsic_failures, detail=extrinsic_failures
    )
    report.add(
        "extrinsic_inverse_roundtrip", not inverse_failures, detail=inverse_failures
    )
    report.add(
        "camera_image_domain_contract",
        not camera_image_failures,
        detail=camera_image_failures,
    )
    report.add(
        "ray_project_roundtrip", not roundtrip_failures, detail=roundtrip_failures
    )


def _audit_calibration_and_photometry(
    report: AuditReport, observations: list[dict[str, Any]]
) -> None:
    calibration_by_camera: dict[str, set[str]] = defaultdict(set)
    for item in observations:
        calibration_by_camera[item["camera_id"]].add(sha256_json(item["camera"]))
    varying_calibration = {
        camera_id: sorted(hashes)
        for camera_id, hashes in sorted(calibration_by_camera.items())
        if len(hashes) != 1
    }
    report.add(
        "static_calibration_per_camera",
        not varying_calibration,
        detail=varying_calibration,
    )

    valid_decode_profiles = {
        "srgb_encoded": {"srgb_eotf_v1", "srgb_reference_assumption_v1"},
        "linear_rgb": {"linear_passthrough_v1"},
        "unknown": {"unspecified_v1"},
    }
    photometric_failures: list[dict[str, Any]] = []
    for item in observations:
        color_space = item["image"]["color_space"]
        profile = item["image"]["encoding"]["canonical_decode_profile"]
        if profile not in valid_decode_profiles[color_space]:
            photometric_failures.append(
                {
                    "observation_id": item["observation_id"],
                    "color_space": color_space,
                    "canonical_decode_profile": profile,
                }
            )
    report.add(
        "photometric_decode_contract",
        not photometric_failures,
        detail=photometric_failures,
    )


def _audit_time_and_roles(
    report: AuditReport,
    manifest: dict[str, Any],
    observations: list[dict[str, Any]],
) -> None:
    per_camera: dict[str, list[tuple[int, float]]] = defaultdict(list)
    per_frame: dict[int, list[float]] = defaultdict(list)
    for item in observations:
        per_camera[item["camera_id"]].append(
            (item["frame_id"], item["timestamp_seconds"])
        )
        per_frame[item["frame_id"]].append(item["timestamp_seconds"])

    timestamp_failures = [
        item["observation_id"]
        for item in observations
        if not math.isfinite(item["timestamp_seconds"])
    ]
    report.add("finite_timestamps", not timestamp_failures, detail=timestamp_failures)

    monotonic_failures: list[dict[str, Any]] = []
    for camera_id, records in per_camera.items():
        for previous, current in pairwise(sorted(records)):
            if current[0] <= previous[0] or current[1] <= previous[1]:
                monotonic_failures.append(
                    {"camera_id": camera_id, "previous": previous, "current": current}
                )
    report.add(
        "per_camera_time_monotonic",
        not monotonic_failures,
        detail=monotonic_failures,
    )

    tolerance = manifest["sync"]["tolerance_seconds"]
    sync_failures = [
        {
            "frame_id": frame_id,
            "spread_seconds": max(timestamps) - min(timestamps),
            "tolerance_seconds": tolerance,
        }
        for frame_id, timestamps in sorted(per_frame.items())
        if len(timestamps) > 1 and max(timestamps) - min(timestamps) > tolerance
    ]
    report.add("frame_sync_tolerance", not sync_failures, detail=sync_failures)

    camera_frame_sets = {
        camera_id: {frame_id for frame_id, _ in records}
        for camera_id, records in per_camera.items()
    }
    union_frames: set[int] = set()
    for frame_ids in camera_frame_sets.values():
        union_frames.update(frame_ids)
    incomplete_cameras = {
        camera_id: sorted(union_frames - frame_ids)
        for camera_id, frame_ids in sorted(camera_frame_sets.items())
        if frame_ids != union_frames
    }
    report.add(
        "rectangular_camera_frame_grid",
        not incomplete_cameras,
        detail={
            "camera_count": len(camera_frame_sets),
            "frame_count": len(union_frames),
            "missing_frames": incomplete_cameras,
        },
    )

    offsets = manifest["sync"].get("per_camera_offset_seconds")
    offset_failures: dict[str, Any] = {}
    if offsets is not None:
        camera_ids = set(per_camera)
        offset_ids = set(offsets)
        nonfinite = sorted(
            camera_id
            for camera_id, value in offsets.items()
            if not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        )
        if camera_ids != offset_ids:
            offset_failures["missing"] = sorted(camera_ids - offset_ids)
            offset_failures["unexpected"] = sorted(offset_ids - camera_ids)
        if nonfinite:
            offset_failures["nonfinite"] = nonfinite
    report.add(
        "sync_offsets_complete",
        not offset_failures,
        detail={"declared": offsets is not None, **offset_failures},
    )

    by_hash: dict[str, set[str]] = defaultdict(set)
    for item in observations:
        by_hash[item["image"]["sha256"]].add(item["role"])
    cross_role_hashes = [
        {"sha256": digest, "roles": sorted(roles)}
        for digest, roles in sorted(by_hash.items())
        if len(roles) > 1
    ]
    report.add(
        "no_identical_image_across_roles",
        not cross_role_hashes,
        detail=cross_role_hashes,
    )


def audit_observation_manifest(
    manifest: dict[str, Any],
    *,
    base_dir: Path,
    verify_files: bool = False,
) -> AuditReport:
    """Audit one public v2 observation manifest and its optional image files."""

    report = AuditReport(subject=str(manifest.get("dataset_id", "<unknown>")))
    try:
        validate_payload("observation", manifest)
    except ContractError as exc:
        report.add("schema", False, detail=str(exc))
        return report
    report.add("schema", True, detail=manifest["schema_version"])

    observations = cast(list[dict[str, Any]], manifest["observations"])
    observation_ids = [item["observation_id"] for item in observations]
    report.add(
        "unique_observation_ids",
        len(observation_ids) == len(set(observation_ids)),
        detail={"count": len(observation_ids), "unique": len(set(observation_ids))},
    )

    camera_frames = [(item["camera_id"], item["frame_id"]) for item in observations]
    report.add(
        "unique_camera_frame",
        len(camera_frames) == len(set(camera_frames)),
        detail={"count": len(camera_frames), "unique": len(set(camera_frames))},
    )

    role_counts = Counter(item["role"] for item in observations)
    report.add(
        "train_role_present",
        role_counts["train"] > 0,
        detail=dict(sorted(role_counts.items())),
    )

    _audit_paths_and_files(
        report,
        observations,
        base_dir=base_dir,
        verify_files=verify_files,
    )
    _audit_cameras(report, observations)
    _audit_calibration_and_photometry(report, observations)
    _audit_time_and_roles(report, manifest, observations)
    return report
