from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import load_file as load_safetensors

from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json
from p2g.errors import ContractError, OutputExistsError
from p2g.training import roma_point_provider as provider


def _write_cache(
    root: Path,
    *,
    height: int = 10,
    width: int = 10,
    camera_ids: tuple[str, ...] = ("left", "right"),
    camera_timestamps: tuple[float, ...] | None = None,
    observation_manifest_sha256: str = "1" * 64,
) -> Path:
    root.mkdir()
    frame_ids = [3]
    camera_count = len(camera_ids)
    rgb = np.zeros((1, camera_count, height, width, 3), dtype=np.uint8)
    rgb[0, 0, :, :, 0] = 200
    rgb[0, 1, :, :, 1] = 180
    if camera_count > 2:
        rgb[0, 2:, :, :, 2] = 160
    intrinsic = np.tile(
        np.asarray(
            (
                (10.0, 0.0, width / 2.0),
                (0.0, 10.0, height / 2.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        ),
        (1, camera_count, 1, 1),
    )
    world_to_camera = np.tile(
        np.eye(4, dtype=np.float32), (1, camera_count, 1, 1)
    )
    for camera_index in range(camera_count):
        world_to_camera[0, camera_index, 0, 3] = 0.5 - camera_index
    timestamp_values = camera_timestamps or (0.25,) * camera_count
    if len(timestamp_values) != camera_count:
        raise ValueError("camera timestamp fixture does not match its camera axis")
    timestamp = np.asarray((timestamp_values,), dtype=np.float64)
    arrays = {
        "rgb": rgb,
        "intrinsic": intrinsic,
        "world_to_camera": world_to_camera,
        "timestamp_seconds": timestamp,
    }
    records: dict[str, Any] = {}
    for name, value in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        records[name] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "dtype": value.dtype.name,
            "shape": list(value.shape),
            "order": "C",
        }
    manifest = {
        "schema_version": "p2g.tensor_cache.v1",
        "observation_manifest_sha256": observation_manifest_sha256,
        "camera_ids": list(camera_ids),
        "frame_ids": frame_ids,
        "arrays": records,
    }
    (root / "tensor_cache.json").write_bytes(canonical_json_bytes(manifest))
    return root


def _write_observation_manifest(
    path: Path,
    *,
    camera_roles: dict[str, str],
    camera_timestamps: dict[str, float] | None = None,
    height: int = 10,
    width: int = 10,
) -> Path:
    camera_ids = tuple(sorted(camera_roles))
    timestamps = camera_timestamps or {camera_id: 0.25 for camera_id in camera_ids}
    observations: list[dict[str, Any]] = []
    for camera_index, camera_id in enumerate(camera_ids):
        observations.append(
            {
                "observation_id": f"obs_{camera_id}_000003",
                "camera_id": camera_id,
                "frame_id": 3,
                "timestamp_seconds": timestamps[camera_id],
                "role": camera_roles[camera_id],
                "image": {
                    "path": f"images/{camera_id}/000003.png",
                    "sha256": f"{camera_index + 1:x}" * 64,
                    "width": width,
                    "height": height,
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
                        [10.0, 0.0, width / 2.0],
                        [0.0, 10.0, height / 2.0],
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
        "dataset_id": "public_roma_role_fixture",
        "source": {
            "description": "project-owned synthetic role-isolation fixture",
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
            "tolerance_seconds": max(timestamps.values()) - min(timestamps.values()),
            "per_camera_offset_seconds": {camera_id: 0.0 for camera_id in camera_ids},
        },
        "transforms": [],
        "observations": observations,
    }
    path.write_bytes(canonical_json_bytes(manifest))
    return path


def _sample_payload(source_is_left: bool, count: int) -> dict[str, Any]:
    source_x, target_x = (6.0, 4.0) if source_is_left else (4.0, 6.0)
    source_normalized = np.asarray((source_x / 5.0 - 1.0, 0.0), dtype=np.float64)
    target_normalized = np.asarray((target_x / 5.0 - 1.0, 0.0), dtype=np.float64)
    matches = np.tile(np.concatenate((source_normalized, target_normalized)), (count, 1))
    return {
        "matches_normalized": matches,
        "raw_certainty": np.linspace(0.6, 0.9, count, dtype=np.float64),
        "selection_score": np.ones(count, dtype=np.float64),
        "dense_source_xy": np.column_stack(
            (np.arange(count, dtype=np.int64), np.zeros(count, dtype=np.int64))
        ),
        "dense_source_valid": np.ones(count, dtype=np.bool_),
        "runtime": {"provider": "synthetic"},
    }


class _SyntheticSampler:
    @property
    def identity(self) -> dict[str, Any]:
        return {"name": "project_owned_synthetic_pair_sampler", "revision": "v1"}

    def sample_pair(
        self,
        source_rgb: np.ndarray[Any, Any],
        target_rgb: np.ndarray[Any, Any],
        *,
        count: int,
        seed: int,
    ) -> dict[str, Any]:
        del target_rgb, seed
        return _sample_payload(bool(source_rgb[0, 0, 0] == 200), count)


def _write_minimal_lock(path: Path, registry: dict[str, Any]) -> None:
    upstream = registry["provider"]
    runtime = registry["runtime"]
    revision = upstream["revision"]
    path.write_text(
        "\n".join(
            (
                "version = 1",
                "revision = 3",
                'requires-python = "==3.12.*"',
                "",
                "[[package]]",
                'name = "romatch"',
                f'version = "{upstream["distribution_version"]}"',
                (f'source = {{ git = "{upstream["repository"]}?rev={revision}#{revision}" }}'),
                "",
                "[[package]]",
                'name = "torch"',
                f'version = "{runtime["torch"]}"',
                f'source = {{ registry = "{runtime["torch_index"]}" }}',
                "",
                "[[package]]",
                'name = "torchvision"',
                f'version = "{runtime["torchvision"]}"',
                f'source = {{ registry = "{runtime["torch_index"]}" }}',
                "",
            )
        ),
        encoding="utf-8",
    )


def test_provider_import_does_not_eagerly_import_torch() -> None:
    source_root = Path(__file__).parents[1] / "src"
    program = (
        "import json,sys;"
        "sys.meta_path[:]=[finder for finder in sys.meta_path "
        "if getattr(finder,'__module__','')!='_pixel4dgs_editable'];"
        f"sys.path.insert(0,{str(source_root)!r});"
        "import p2g.training.roma_point_provider;"
        'print(json.dumps({"torch_loaded":"torch" in sys.modules}))'
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == {"torch_loaded": False}


def test_registry_pins_code_license_runtime_and_external_weights() -> None:
    registry, digest = provider.load_roma_provider_registry()

    assert len(digest) == 64
    assert registry["provider"] == {
        "distribution": "romatch",
        "distribution_version": "0.1.2",
        "factory": "romatch.roma_indoor",
        "repository": "https://github.com/Parskatt/RoMa.git",
        "revision": "77f8d68803526dcddfd9b7a46bc76125bdc25f15",
        "license": {
            "spdx": "MIT",
            "copyright": "Copyright (c) 2023 Johan Edstedt",
            "url": (
                "https://github.com/Parskatt/RoMa/blob/"
                "77f8d68803526dcddfd9b7a46bc76125bdc25f15/LICENSE"
            ),
            "sha256": "6f02ad61c3fd4509343d619b55501354bcebfd67ad5460c853d02c7ab46ff4bd",
        },
    }
    assert registry["weights"]["roma_indoor"]["license"]["spdx"] == "NOASSERTION"
    assert registry["weights"]["roma_indoor"]["license"]["status"] == "review_required"
    assert registry["weights"]["dinov2_vitl14"]["license"]["spdx"] == "Apache-2.0"
    for record in registry["weights"].values():
        assert record["redistribution"] == "external_only_not_bundled"
        assert record["url"].startswith("https://")
        assert len(record["sha256"]) == 64
    assert registry["policy"]["automatic_download"] is False
    assert registry["policy"]["bundle_weights"] is False


def test_provider_source_has_no_downloader_or_private_dataset_adapter() -> None:
    source = Path(provider.__file__).read_text(encoding="utf-8")
    lowered = source.casefold()

    for forbidden in (
        "torch.hub",
        "urlretrieve",
        "requests.get",
        "hair_train",
        "freetime",
        "/mnt/",
        "/home/",
    ):
        assert forbidden not in lowered


def test_tensor_cache_frame_is_hash_closed_and_resolution_independent(tmp_path: Path) -> None:
    cache = _write_cache(tmp_path / "cache", height=17, width=31)

    frame = provider.load_tensor_cache_frame(cache, frame_id=3)

    assert frame.rgb.shape == (2, 17, 31, 3)
    assert frame.camera_ids == ("left", "right")
    assert frame.timestamp_seconds == 0.25
    assert frame.source_receipt["schema"] == "p2g.tensor_cache.v1"
    assert len(frame.source_receipt["frame_payload_sha256"]["rgb"]) == 64

    rgb_path = cache / "rgb.npy"
    content = bytearray(rgb_path.read_bytes())
    content[-1] ^= 1
    rgb_path.write_bytes(content)
    with pytest.raises(ContractError, match="SHA-256"):
        provider.load_tensor_cache_frame(cache, frame_id=3)


def test_pair_graph_seed_and_coordinate_conventions_are_deterministic(tmp_path: Path) -> None:
    frame = provider.load_tensor_cache_frame(_write_cache(tmp_path / "cache"), frame_id=3)
    graph = provider.nearest_camera_graph(frame.world_to_camera, neighbors=1)
    assert graph.tolist() == [[1], [0]]

    first = provider.pair_seed(7, frame_id=3, source_camera=0, target_camera=1, neighbor_rank=0)
    assert first == provider.pair_seed(
        7, frame_id=3, source_camera=0, target_camera=1, neighbor_rank=0
    )
    assert first != provider.pair_seed(
        7, frame_id=3, source_camera=1, target_camera=0, neighbor_rank=0
    )

    cells = np.asarray(((0, 0), (431, 700), (863, 863)), dtype=np.float64)
    normalized = np.column_stack(
        (
            2.0 * (cells[:, 0] + 0.5) / 864.0 - 1.0,
            2.0 * (cells[:, 1] + 0.5) / 864.0 - 1.0,
        )
    )
    recovered, valid = provider.recover_dense_source_xy(
        normalized, dense_height=864, dense_width=864
    )
    assert recovered.tolist() == cells.astype(int).tolist()
    assert valid.tolist() == [True, True, True]


def test_pair_provenance_admits_geometry_without_hidden_quality_filter(tmp_path: Path) -> None:
    frame = provider.load_tensor_cache_frame(_write_cache(tmp_path / "cache"), frame_id=3)
    sampled = _sample_payload(True, 2)
    sampled["raw_certainty"] = np.asarray((0.9, 0.001), dtype=np.float64)

    planes = provider.assemble_pair_provenance(
        frame,
        source_camera=0,
        target_camera=1,
        neighbor_rank=0,
        pair_ordinal=0,
        sampled=sampled,
    )

    assert planes["admitted"].tolist() == [True, True]
    np.testing.assert_allclose(planes["xyz"][:, 2], (5.0, 5.0), atol=1.0e-5)
    assert planes["source_camera_z"].tolist() == pytest.approx([5.0, 5.0])
    assert planes["target_camera_z"].tolist() == pytest.approx([5.0, 5.0])
    assert planes["color_camera"].tolist() == [0, 0]
    assert planes["rgb"][:, 0].tolist() == [200, 200]
    assert planes["raw_certainty"].tolist() == pytest.approx([0.9, 0.001])


def test_sample_canonicalization_recovers_raw_certainty_and_stable_order() -> None:
    # RoMa 0.1.2 returns certainty with a retained singleton batch dimension.
    certainty = np.arange(16, dtype=np.float64).reshape(1, 4, 4)
    centers = np.asarray(((2, 1), (0, 3), (1, 0)), dtype=np.float64)
    normalized = np.column_stack(
        (
            2.0 * (centers[:, 0] + 0.5) / 4.0 - 1.0,
            2.0 * (centers[:, 1] + 0.5) / 4.0 - 1.0,
        )
    )
    matches = np.column_stack((normalized, normalized))

    result = provider._canonicalize_sampled_pair(
        matches,
        np.asarray((0.2, 0.3, 0.4)),
        certainty,
        count=3,
    )

    assert result["dense_source_xy"].tolist() == [[1, 0], [2, 1], [0, 3]]
    assert result["raw_certainty"].tolist() == [1.0, 6.0, 12.0]
    assert result["dense_shape"] == [4, 4]


def test_sample_canonicalization_rejects_multiple_certainty_batches() -> None:
    matches = np.zeros((3, 4), dtype=np.float64)
    certainty = np.zeros((2, 4, 4), dtype=np.float64)

    with pytest.raises(ContractError, match="invalid shape or coordinate"):
        provider._canonicalize_sampled_pair(
            matches,
            np.ones(3, dtype=np.float64),
            certainty,
            count=3,
        )


def test_environment_lock_and_registered_file_are_content_bound(tmp_path: Path) -> None:
    registry, _ = provider.load_roma_provider_registry()
    lock = tmp_path / "uv.lock"
    _write_minimal_lock(lock, registry)

    identity = provider._validate_environment_lock(lock, registry)

    assert identity["requires_python"] == "==3.12.*"
    assert identity["sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    lock.write_text(lock.read_text().replace("0.1.2", "0.1.1"), encoding="utf-8")
    with pytest.raises(ContractError, match="romatch source identity"):
        provider._validate_environment_lock(lock, registry)

    payload = tmp_path / "weight.bin"
    payload.write_bytes(b"project-owned-test-bytes")
    record = {
        "bytes": payload.stat().st_size,
        "sha256": sha256_file(payload),
    }
    assert provider._verify_registered_file(payload, record, label="fixture") == payload.resolve()
    payload.write_bytes(b"changed")
    with pytest.raises(ContractError, match=r"size differs|SHA-256 differs"):
        provider._verify_registered_file(payload, record, label="fixture")


def test_builder_publishes_replayable_ply_and_provenance(tmp_path: Path) -> None:
    manifest = _write_observation_manifest(
        tmp_path / "observations.json",
        camera_roles={"left": "train", "right": "train"},
    )
    cache = _write_cache(
        tmp_path / "cache",
        observation_manifest_sha256=sha256_file(manifest),
    )
    output = tmp_path / "proposals"
    missing = tmp_path / "unused"

    receipt = provider.build_roma_point_proposals(
        output,
        memmap_root=cache,
        observation_manifest=manifest,
        frame_id=3,
        roma_weight=missing,
        dino_weight=missing,
        environment_lock=missing,
        num_points_per_frame=4,
        nearest_cameras=1,
        seed=19,
        sampler=_SyntheticSampler(),
    )

    assert receipt["status"] == "COMPLETE"
    assert receipt["aggregate"]["sampled_count"] == 4
    assert receipt["aggregate"]["admitted_count"] == 4
    assert receipt["provider"]["name"] == "project_owned_synthetic_pair_sampler"
    assert receipt["frame"]["role"] == "train"
    assert receipt["frame"]["camera_ids"] == ["left", "right"]
    admission = receipt["source"]["role_admission"]
    assert admission["role"] == "train"
    assert admission["admitted_camera_ids"] == ["left", "right"]
    assert admission["excluded_camera_ids_by_role"] == {
        "diagnostic": [],
        "sealed": [],
        "free_view": [],
    }
    assert receipt["policy"]["color"] == "minimum_camera_index_bilinear_rgb8_pixel_centers_v1"
    assert sha256_file(output / "f000003.ply") == receipt["artifacts"]["point_ply"]["sha256"]
    assert (
        sha256_file(output / "provenance.safetensors")
        == receipt["artifacts"]["provenance"]["sha256"]
    )
    planes = load_safetensors(output / "provenance.safetensors")
    assert planes["xyz"].shape == (4, 3)
    assert planes["admitted"].tolist() == [True, True, True, True]
    assert planes["rgb"][:, 0].tolist() == [200, 200, 200, 200]
    assert (
        provider.canonical_provenance_sha256(planes, frame_id=3)
        == receipt["artifacts"]["provenance"]["canonical_tensor_sha256"]
    )
    assert str(tmp_path) not in json.dumps(receipt, sort_keys=True)
    with pytest.raises(OutputExistsError):
        provider.build_roma_point_proposals(
            output,
            memmap_root=cache,
            observation_manifest=manifest,
            frame_id=3,
            roma_weight=missing,
            dino_weight=missing,
            environment_lock=missing,
            num_points_per_frame=4,
            nearest_cameras=1,
            sampler=_SyntheticSampler(),
        )


def test_builder_excludes_non_train_cameras_and_rejects_manifest_substitution(
    tmp_path: Path,
) -> None:
    manifest = _write_observation_manifest(
        tmp_path / "observations.json",
        camera_roles={"left": "train", "right": "train", "sealed": "sealed"},
        camera_timestamps={"left": 0.249, "right": 0.251, "sealed": 0.252},
    )
    cache = _write_cache(
        tmp_path / "cache",
        camera_ids=("left", "right", "sealed"),
        camera_timestamps=(0.249, 0.251, 0.252),
        observation_manifest_sha256=sha256_file(manifest),
    )
    missing = tmp_path / "unused"
    receipt = provider.build_roma_point_proposals(
        tmp_path / "proposals",
        memmap_root=cache,
        observation_manifest=manifest,
        frame_id=3,
        roma_weight=missing,
        dino_weight=missing,
        environment_lock=missing,
        num_points_per_frame=4,
        nearest_cameras=1,
        sampler=_SyntheticSampler(),
    )

    assert receipt["frame"]["camera_ids"] == ["left", "right"]
    assert receipt["frame"]["timestamp_seconds"] == pytest.approx(0.25)
    assert receipt["source"]["cache_camera_count"] == 3
    admission = receipt["source"]["role_admission"]
    assert admission["excluded_camera_ids_by_role"]["sealed"] == ["sealed"]
    assert admission["cache_camera_count"] == 3
    unsigned = {key: value for key, value in admission.items() if key != "logical_sha256"}
    assert admission["logical_sha256"] == sha256_json(unsigned)

    substituted = _write_observation_manifest(
        tmp_path / "substituted.json",
        camera_roles={"left": "train", "right": "train", "sealed": "diagnostic"},
        camera_timestamps={"left": 0.249, "right": 0.251, "sealed": 0.252},
    )
    with pytest.raises(ContractError, match="different observation manifest"):
        provider.build_roma_point_proposals(
            tmp_path / "rejected",
            memmap_root=cache,
            observation_manifest=substituted,
            frame_id=3,
            roma_weight=missing,
            dino_weight=missing,
            environment_lock=missing,
            num_points_per_frame=4,
            nearest_cameras=1,
            sampler=_SyntheticSampler(),
        )
    assert not (tmp_path / "rejected").exists()
