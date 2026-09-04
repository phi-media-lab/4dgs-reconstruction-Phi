from __future__ import annotations

import copy
import hashlib
import struct
import zlib
from pathlib import Path
from typing import Any

from p2g.audit import AuditReport, audit_observation_manifest


def _png_chunk(name: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(name)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", crc)


def _rgb_png(width: int, height: int, marker: str) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"tEXt", f"observation\0{marker}".encode())
        + _png_chunk(b"IEND", b"")
    )


def _observation(
    camera_id: str,
    frame_id: int,
    timestamp: float,
    role: str,
    digest_digit: str,
    tx: float,
) -> dict[str, Any]:
    return {
        "observation_id": f"obs_{camera_id}_{frame_id:06d}",
        "camera_id": camera_id,
        "frame_id": frame_id,
        "timestamp_seconds": timestamp,
        "role": role,
        "image": {
            "path": f"images/{camera_id}/{frame_id:06d}.png",
            "sha256": digest_digit * 64,
            "width": 640,
            "height": 480,
            "color_space": "srgb_encoded",
            "encoding": {
                "container": "png",
                "channel_order": "RGB",
                "bit_depth": 8,
                "stored_range": "full",
                "declared_transfer": None,
                "declared_primaries": None,
                "declared_matrix": None,
                "canonical_decode_profile": "srgb_reference_assumption_v1",
            },
        },
        "camera": {
            "model": "pinhole",
            "pixel_domain": "undistorted",
            "intrinsic": [
                [1000.0, 0.0, 320.0],
                [0.0, 1000.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "world_to_camera": [
                [1.0, 0.0, 0.0, tx],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "distortion": [],
        },
    }


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": "p2g.observation_manifest.v2",
        "dataset_id": "public_audit_fixture",
        "source": {
            "description": "project-owned synthetic contract fixture",
            "license": "CC0-1.0",
            "license_status": "declared",
            "root_sha256": "f" * 64,
        },
        "coordinate_conventions": {
            "handedness": "right",
            "extrinsic": "world_to_camera",
            "pixel_center": "half_pixel",
            "time_unit": "seconds",
            "photometric_space": "linear_rgb",
        },
        "sync": {
            "variant": "synthetic_exact_v1",
            "tolerance_seconds": 0.001,
            "per_camera_offset_seconds": {"cam000": 0.0, "cam001": 0.0},
        },
        "transforms": [],
        "observations": [
            _observation("cam000", 0, 0.0, "train", "0", 0.0),
            _observation("cam001", 0, 0.0005, "sealed", "1", -0.1),
            _observation("cam000", 1, 1.0, "train", "2", 0.0),
            _observation("cam001", 1, 1.0005, "sealed", "3", -0.1),
        ],
    }


def _check(report: AuditReport, name: str) -> str:
    return next(check.status for check in report.checks if check.name == name)


def test_valid_v2_manifest_passes_semantic_audit(tmp_path: Path) -> None:
    report = audit_observation_manifest(_manifest(), base_dir=tmp_path)

    assert report.status == "PASS", report.to_dict()
    assert _check(report, "schema") == "PASS"
    assert _check(report, "image_files_and_hashes") == "PASS"
    assert (
        next(
            check.required
            for check in report.checks
            if check.name == "image_files_and_hashes"
        )
        is False
    )


def test_legacy_manifest_is_rejected_at_single_public_schema_boundary(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest["schema_version"] = "p2g.observation_manifest.v1"

    report = audit_observation_manifest(manifest, base_dir=tmp_path)

    assert report.status == "FAIL"
    assert [check.name for check in report.checks] == ["schema"]
    assert "p2g.observation_manifest.v2" in str(report.checks[0].detail)


def test_duplicate_camera_frame_and_cross_role_image_are_rejected(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    duplicate = copy.deepcopy(manifest["observations"][0])
    duplicate["observation_id"] = "obs_duplicate_000000"
    duplicate["role"] = "sealed"
    manifest["observations"].append(duplicate)

    report = audit_observation_manifest(manifest, base_dir=tmp_path)

    assert report.status == "FAIL"
    assert _check(report, "unique_camera_frame") == "FAIL"
    assert _check(report, "no_identical_image_across_roles") == "FAIL"


def test_parent_and_backslash_path_escapes_are_rejected(tmp_path: Path) -> None:
    for unsafe in ("../sealed.png", "images\\sealed.png"):
        manifest = _manifest()
        manifest["observations"][0]["image"]["path"] = unsafe

        report = audit_observation_manifest(manifest, base_dir=tmp_path)

        assert report.status == "FAIL"
        assert _check(report, "relative_paths_within_root") == "FAIL"


def test_symlink_path_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "images").symlink_to(outside, target_is_directory=True)
    manifest = _manifest()
    manifest["observations"][0]["image"]["path"] = "images/escaped.png"

    report = audit_observation_manifest(manifest, base_dir=root)

    assert report.status == "FAIL"
    assert _check(report, "relative_paths_within_root") == "FAIL"


def test_hash_closed_png_headers_pass(tmp_path: Path) -> None:
    manifest = _manifest()
    for item in manifest["observations"]:
        image_path = tmp_path / item["image"]["path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _rgb_png(
            item["image"]["width"],
            item["image"]["height"],
            item["observation_id"],
        )
        image_path.write_bytes(payload)
        item["image"]["sha256"] = hashlib.sha256(payload).hexdigest()

    report = audit_observation_manifest(manifest, base_dir=tmp_path, verify_files=True)

    assert report.status == "PASS", report.to_dict()
    assert _check(report, "image_files_and_hashes") == "PASS"
    assert _check(report, "image_headers_match_manifest") == "PASS"


def test_tampered_image_fails_before_header_admission(tmp_path: Path) -> None:
    manifest = _manifest()
    for item in manifest["observations"]:
        image_path = tmp_path / item["image"]["path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _rgb_png(640, 480, item["observation_id"])
        image_path.write_bytes(payload)
        item["image"]["sha256"] = hashlib.sha256(payload).hexdigest()
    first_path = tmp_path / manifest["observations"][0]["image"]["path"]
    first_path.write_bytes(first_path.read_bytes() + b"tampered")

    report = audit_observation_manifest(manifest, base_dir=tmp_path, verify_files=True)

    assert report.status == "FAIL"
    assert _check(report, "image_files_and_hashes") == "FAIL"


def test_non_rigid_extrinsic_and_photometric_mismatch_are_rejected(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest["observations"][0]["camera"]["world_to_camera"][0][0] = 2.0
    manifest["observations"][1]["image"]["encoding"]["canonical_decode_profile"] = (
        "linear_passthrough_v1"
    )

    report = audit_observation_manifest(manifest, base_dir=tmp_path)

    assert report.status == "FAIL"
    assert _check(report, "extrinsic_rigid_contract") == "FAIL"
    assert _check(report, "photometric_decode_contract") == "FAIL"


def test_sync_violation_and_incomplete_offset_map_are_rejected(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["observations"][1]["timestamp_seconds"] = 0.1
    del manifest["sync"]["per_camera_offset_seconds"]["cam001"]

    report = audit_observation_manifest(manifest, base_dir=tmp_path)

    assert report.status == "FAIL"
    assert _check(report, "frame_sync_tolerance") == "FAIL"
    assert _check(report, "sync_offsets_complete") == "FAIL"


def test_optional_failure_does_not_override_required_verdict() -> None:
    report = AuditReport(subject="fixture")
    report.add("advisory", False, required=False)
    report.add("contract", True)

    assert report.status == "PASS"
    assert report.to_dict()["checks"][0]["status"] == "FAIL"
