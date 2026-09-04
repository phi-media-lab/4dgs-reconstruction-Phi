from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json, write_new_json
from p2g.errors import OutputExistsError
from p2g.training import roma_point_sequence


class _FakeSampler:
    @property
    def identity(self) -> dict[str, Any]:
        return {"name": "fake-deterministic-sampler", "revision": "v1"}


def _fake_frame_builder(
    output: Path,
    *,
    frame_id: int,
    num_points_per_frame: int,
    nearest_cameras: int,
    seed: int,
    world_bound: float,
    sampler: Any,
    observation_manifest: Path,
    **_: Any,
) -> dict[str, Any]:
    output.mkdir()
    ply_path = output / f"f{frame_id:06d}.ply"
    provenance_path = output / "provenance.safetensors"
    ply_path.write_bytes(f"point-frame-{frame_id}\n".encode())
    provenance_path.write_bytes(f"provenance-frame-{frame_id}\n".encode())
    role_admission_unsigned = {
        "schema": "p2g.observation_role_admission.v1",
        "role": "train",
        "observation_manifest_sha256": sha256_file(observation_manifest),
        "frame_id": frame_id,
        "frame_timestamp_operator": (
            "arithmetic_mean_of_train_observation_timestamps_v1"
        ),
        "cache_camera_count": 2,
        "admitted_camera_ids": ["left", "right"],
        "admitted_observation_ids": [
            f"obs_left_{frame_id:06d}",
            f"obs_right_{frame_id:06d}",
        ],
        "excluded_camera_ids_by_role": {
            "diagnostic": [],
            "sealed": [],
            "free_view": [],
        },
    }
    receipt = {
        "schema": "p2g.roma_point_proposals.v1",
        "status": "COMPLETE",
        "frame": {
            "frame_id": frame_id,
            "role": "train",
            "camera_ids": ["left", "right"],
        },
        "provider": sampler.identity,
        "policy": {
            "num_points_per_frame_requested": num_points_per_frame,
            "nearest_cameras": nearest_cameras,
            "global_seed": seed,
            "world_coordinate_absolute_bound": world_bound,
        },
        "source": {
            "frame_payload_sha256": {"rgb": f"rgb-{frame_id}"},
            "role_admission": {
                **role_admission_unsigned,
                "logical_sha256": sha256_json(role_admission_unsigned),
            },
        },
        "aggregate": {"sampled_count": 10, "admitted_count": 8 + frame_id},
        "artifacts": {
            "point_ply": {
                "path": ply_path.name,
                "sha256": sha256_file(ply_path),
            },
            "provenance": {
                "path": provenance_path.name,
                "sha256": sha256_file(provenance_path),
                "canonical_tensor_sha256": f"{frame_id:064x}",
                "canonical_digest_schema": ("p2g.roma_point_provenance_canonical_digest.v1"),
            },
        },
    }
    write_new_json(output / "receipt.json", receipt)
    return receipt


def _write_observation_manifest(path: Path) -> Path:
    observations: list[dict[str, Any]] = []
    for frame_id in (0, 1):
        for camera_index, camera_id in enumerate(("left", "right")):
            observations.append(
                {
                    "observation_id": f"obs_{camera_id}_{frame_id:06d}",
                    "camera_id": camera_id,
                    "frame_id": frame_id,
                    "timestamp_seconds": frame_id * 0.1,
                    "role": "train",
                    "image": {
                        "path": f"images/{camera_id}/{frame_id:06d}.png",
                        "sha256": f"{frame_id * 2 + camera_index + 1:x}" * 64,
                        "width": 4,
                        "height": 3,
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
                            [10.0, 0.0, 2.0],
                            [0.0, 10.0, 1.5],
                            [0.0, 0.0, 1.0],
                        ],
                        "world_to_camera": [
                            [1.0, 0.0, 0.0, 0.5 - camera_index],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "distortion": [],
                    },
                }
            )
    manifest = {
        "schema_version": "p2g.observation_manifest.v2",
        "dataset_id": "public_sequence_fixture",
        "source": {
            "description": "project-owned synthetic sequence fixture",
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
            "tolerance_seconds": 0.0,
            "per_camera_offset_seconds": {"left": 0.0, "right": 0.0},
        },
        "transforms": [],
        "observations": observations,
    }
    path.write_bytes(canonical_json_bytes(manifest))
    return path


def test_sequence_publishes_and_resumes_verified_frame_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        roma_point_sequence,
        "RomaIndoorPairSampler",
        lambda **_: _FakeSampler(),
    )
    monkeypatch.setattr(
        roma_point_sequence,
        "build_roma_point_proposals",
        _fake_frame_builder,
    )
    output = tmp_path / "sequence"
    missing = tmp_path / "unused"
    observation_manifest = _write_observation_manifest(tmp_path / "observations.json")

    first = roma_point_sequence.build_roma_point_sequence(
        output,
        memmap_root=missing,
        observation_manifest=observation_manifest,
        roma_weight=missing,
        dino_weight=missing,
        environment_lock=missing,
        frame_ids=(0, 1),
        num_points_per_frame=20,
        nearest_cameras=1,
        seed=7,
        world_bound=50.0,
    )

    assert first["status"] == "COMPLETE"
    assert first["admitted_observation_role"] == "train"
    assert first["observation_manifest_sha256"] == sha256_file(observation_manifest)
    assert all(len(row["role_admission_sha256"]) == 64 for row in first["frames"])
    assert first["aggregate"] == {
        "sampled_count": 20,
        "admitted_count": 17,
        "admitted_fraction": 0.85,
    }
    for frame_id in (0, 1):
        shard = output / "frames" / f"f{frame_id:06d}" / f"f{frame_id:06d}.ply"
        published = output / "points" / f"f{frame_id:06d}.ply"
        assert shard.read_bytes() == published.read_bytes()
        assert shard.stat().st_ino == published.stat().st_ino

    with pytest.raises(OutputExistsError):
        roma_point_sequence.build_roma_point_sequence(
            output,
            memmap_root=missing,
            observation_manifest=observation_manifest,
            roma_weight=missing,
            dino_weight=missing,
            environment_lock=missing,
            frame_ids=(0, 1),
            num_points_per_frame=20,
            nearest_cameras=1,
            seed=7,
            world_bound=50.0,
        )

    (output / "collection.json").unlink()
    resumed = roma_point_sequence.build_roma_point_sequence(
        output,
        memmap_root=missing,
        observation_manifest=observation_manifest,
        roma_weight=missing,
        dino_weight=missing,
        environment_lock=missing,
        frame_ids=(0, 1),
        num_points_per_frame=20,
        nearest_cameras=1,
        seed=7,
        world_bound=50.0,
    )

    assert resumed["aggregate"] == first["aggregate"]
    assert (output / "collection.json").is_file()
