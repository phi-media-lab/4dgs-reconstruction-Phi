from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from p2g.canonical import sha256_file
from p2g.errors import ContractError
from p2g.training.config import DataConfig, TensorMemmapConfig
from p2g.training.dataset import (
    CACHE_MANIFEST_NAME,
    PreparedScene,
    SceneSampler,
)

ROOT = Path(__file__).parents[1]
WIDTH = 8
HEIGHT = 6
CAMERAS = ("cam-a", "cam-b")
FRAMES = (0, 1, 2)
ROLES = {0: "train", 1: "diagnostic", 2: "sealed"}


def _camera(camera_id: str) -> dict[str, Any]:
    translation = 0.0 if camera_id == "cam-a" else -2.0
    return {
        "model": "pinhole",
        "pixel_domain": "undistorted",
        "intrinsic": [
            [8.0, 0.0, 4.0],
            [0.0, 8.0, 3.0],
            [0.0, 0.0, 1.0],
        ],
        "world_to_camera": [
            [1.0, 0.0, 0.0, translation],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "distortion": [],
    }


def _write_scene(root: Path) -> tuple[Path, dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for frame_id in FRAMES:
        for camera_index, camera_id in enumerate(CAMERAS):
            value = 24 + frame_id * 48 + camera_index * 12
            relative = Path("images") / camera_id / f"{frame_id:06d}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)
            Image.fromarray(pixels, mode="RGB").save(path, format="PNG")
            observations.append(
                {
                    "observation_id": f"obs_{camera_id}_{frame_id:06d}",
                    "camera_id": camera_id,
                    "frame_id": frame_id,
                    "timestamp_seconds": frame_id + camera_index * 0.0005,
                    "role": ROLES[frame_id],
                    "image": {
                        "path": relative.as_posix(),
                        "sha256": sha256_file(path),
                        "width": WIDTH,
                        "height": HEIGHT,
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
                    "camera": _camera(camera_id),
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": "p2g.observation_manifest.v2",
        "dataset_id": "public_dataset_fixture",
        "source": {
            "description": "project-owned synthetic RGB fixture",
            "license": "CC0-1.0",
            "license_status": "declared",
            "root_sha256": "a" * 64,
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
            "per_camera_offset_seconds": {"cam-a": 0.0, "cam-b": 0.0005},
        },
        "transforms": [],
        "observations": observations,
    }
    manifest_path = root / "observations.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, manifest


def _data_config(root: Path, manifest: Path, **changes: Any) -> DataConfig:
    values: dict[str, Any] = {
        "manifest": manifest.resolve(),
        "image_root": root.resolve(),
        "downscale": 2,
        "image_cache_size": 2,
    }
    values.update(changes)
    return DataConfig(**values)


def _write_tensor_cache(
    root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> Path:
    cache_root = root / "tensor-cache"
    cache_root.mkdir()
    camera_to_index = {value: index for index, value in enumerate(CAMERAS)}
    frame_to_index = {value: index for index, value in enumerate(FRAMES)}
    rgb = np.empty((len(FRAMES), len(CAMERAS), HEIGHT, WIDTH, 3), dtype=np.uint8)
    intrinsic = np.empty((len(FRAMES), len(CAMERAS), 3, 3), dtype=np.float32)
    world_to_camera = np.empty((len(FRAMES), len(CAMERAS), 4, 4), dtype=np.float32)
    timestamp_seconds = np.empty((len(FRAMES), len(CAMERAS)), dtype=np.float64)
    for observation in manifest["observations"]:
        coordinates = (
            frame_to_index[observation["frame_id"]],
            camera_to_index[observation["camera_id"]],
        )
        with Image.open(root / observation["image"]["path"]) as image:
            rgb[coordinates] = np.asarray(image)
        intrinsic[coordinates] = np.asarray(observation["camera"]["intrinsic"])
        world_to_camera[coordinates] = np.asarray(observation["camera"]["world_to_camera"])
        timestamp_seconds[coordinates] = observation["timestamp_seconds"]

    arrays = {
        "rgb": rgb,
        "intrinsic": intrinsic,
        "world_to_camera": world_to_camera,
        "timestamp_seconds": timestamp_seconds,
    }
    records: dict[str, Any] = {}
    for name, array in arrays.items():
        path = cache_root / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        records[name] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "dtype": array.dtype.name,
            "shape": list(array.shape),
            "order": "C",
        }
    metadata = {
        "schema_version": "p2g.tensor_cache.v1",
        "observation_manifest_sha256": sha256_file(manifest_path),
        "camera_ids": list(CAMERAS),
        "frame_ids": list(FRAMES),
        "arrays": records,
    }
    (cache_root / CACHE_MANIFEST_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cache_root


def _tensor_config(root: Path, *, verify: bool = True) -> TensorMemmapConfig:
    return TensorMemmapConfig(
        root=root.resolve(),
        camera_ids=CAMERAS,
        frame_ids=FRAMES,
        verify_transport_sha256=verify,
    )


def test_raw_scene_is_audited_partitioned_and_photometrically_decoded(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_scene(tmp_path)
    scene = PreparedScene.load(_data_config(tmp_path, manifest_path))

    assert scene.audit_report.status == "PASS"
    assert len(scene.train_indices) == 2
    assert len(scene.diagnostic_indices) == 2
    assert scene.eval_indices == scene.diagnostic_indices
    assert len(scene.sealed_indices) == 2
    assert scene.free_view_indices == ()
    assert scene.excluded_indices == ()
    assert scene.train_frame_groups() == ((0, 1),)
    assert scene.camera_extent() == pytest.approx(1.0)

    batch = scene.load_batch(scene.train_indices[0])
    encoded = torch.tensor(24.0 / 255.0)
    expected = torch.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )
    assert batch.role == "train"
    assert tuple(batch.rgb.shape) == (HEIGHT // 2, WIDTH // 2, 3)
    assert torch.allclose(batch.rgb, torch.full_like(batch.rgb, expected), atol=1.0e-7)
    assert tuple(batch.intrinsic.shape) == (1, 3, 3)
    assert torch.equal(
        batch.intrinsic[0],
        torch.tensor([[4.0, 0.0, 2.0], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]),
    )


def test_sealed_access_is_explicit_even_after_the_batch_is_cached(tmp_path: Path) -> None:
    manifest_path, _ = _write_scene(tmp_path)
    scene = PreparedScene.load(_data_config(tmp_path, manifest_path))
    index = scene.sealed_indices[0]

    with pytest.raises(ContractError, match="routine access cannot load a sealed"):
        scene.load_batch(index)
    sealed = scene.load_batch(index, access="sealed")
    assert sealed.role == "sealed"
    with pytest.raises(ContractError, match="routine access cannot load a sealed"):
        scene.load_batch(index)


def test_selection_limits_remain_explicit_in_excluded_inventory(tmp_path: Path) -> None:
    manifest_path, _ = _write_scene(tmp_path)
    scene = PreparedScene.load(
        _data_config(
            tmp_path,
            manifest_path,
            max_train_observations=1,
            max_eval_observations=1,
        )
    )

    assert len(scene.train_indices) == 1
    assert len(scene.diagnostic_indices) == 1
    assert len(scene.excluded_indices) == 2
    assert set(scene.excluded_indices).isdisjoint(scene.train_indices)
    assert set(scene.excluded_indices).isdisjoint(scene.diagnostic_indices)


def test_manifest_file_tamper_fails_before_scene_construction(tmp_path: Path) -> None:
    manifest_path, manifest = _write_scene(tmp_path)
    image_path = tmp_path / manifest["observations"][0]["image"]["path"]
    image_path.write_bytes(image_path.read_bytes() + b"tamper")

    with pytest.raises(ContractError, match="image_files_and_hashes"):
        PreparedScene.load(_data_config(tmp_path, manifest_path))


def test_post_audit_image_tamper_is_detected_on_first_decode(tmp_path: Path) -> None:
    manifest_path, _ = _write_scene(tmp_path)
    scene = PreparedScene.load(_data_config(tmp_path, manifest_path))
    observation = scene.observations[scene.train_indices[0]]
    with Image.open(observation.image_path) as image:
        original = np.asarray(image).copy()
    Image.fromarray(255 - original, mode="RGB").save(observation.image_path)

    with pytest.raises(ContractError, match="changed after audit"):
        scene.load_batch(scene.train_indices[0])


def test_train_time_distortion_is_rejected_by_the_narrow_public_contract(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_scene(tmp_path)
    for observation in manifest["observations"]:
        if observation["camera_id"] == "cam-a":
            observation["camera"]["model"] = "opencv_radtan"
            observation["camera"]["pixel_domain"] = "distorted"
            observation["camera"]["distortion"] = [0.0, 0.0, 0.0, 0.0]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ContractError, match="offline-undistorted pinhole"):
        PreparedScene.load(_data_config(tmp_path, manifest_path))


def test_tensor_cache_matches_raw_pixels_and_geometry(tmp_path: Path) -> None:
    manifest_path, manifest = _write_scene(tmp_path)
    cache_root = _write_tensor_cache(tmp_path, manifest_path, manifest)
    raw = PreparedScene.load(_data_config(tmp_path, manifest_path))
    cached = PreparedScene.load(
        _data_config(
            tmp_path,
            manifest_path,
            tensor_memmap=_tensor_config(cache_root),
        )
    )

    for index in (*cached.train_indices, *cached.diagnostic_indices):
        raw_batch = raw.load_batch(index)
        cached_batch = cached.load_batch(index)
        assert torch.equal(cached_batch.rgb, raw_batch.rgb)
        assert torch.equal(cached_batch.intrinsic, raw_batch.intrinsic)
        assert torch.equal(cached_batch.world_to_camera, raw_batch.world_to_camera)
        assert torch.equal(cached_batch.timestamp, raw_batch.timestamp)


def test_tensor_cache_hash_and_coordinate_substitution_are_rejected(tmp_path: Path) -> None:
    manifest_path, manifest = _write_scene(tmp_path)
    cache_root = _write_tensor_cache(tmp_path, manifest_path, manifest)
    metadata_path = cache_root / CACHE_MANIFEST_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["arrays"]["rgb"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ContractError, match="rgb SHA-256 mismatch"):
        PreparedScene.load(
            _data_config(
                tmp_path,
                manifest_path,
                tensor_memmap=_tensor_config(cache_root),
            )
        )

    cache_root = tmp_path / "second-cache"
    source_cache = _write_tensor_cache(tmp_path / "second", *_write_scene(tmp_path / "second"))
    source_cache.rename(cache_root)
    second_manifest = tmp_path / "second" / "observations.json"
    intrinsic_path = cache_root / "intrinsic.npy"
    intrinsic = np.load(intrinsic_path)
    intrinsic[0, 0, 0, 0] += 1.0
    np.save(intrinsic_path, intrinsic, allow_pickle=False)
    with pytest.raises(ContractError, match="intrinsic differs"):
        PreparedScene.load(
            _data_config(
                tmp_path / "second",
                second_manifest,
                tensor_memmap=_tensor_config(cache_root, verify=False),
            )
        )


@pytest.mark.parametrize("policy", ["shuffled_epoch", "frame_camera_with_replacement"])
def test_sampler_resume_is_exact_after_json_roundtrip(policy: str) -> None:
    groups = ((0, 1), (2, 3), (4, 5))
    options: dict[str, Any] = {
        "seed": 17,
        "policy": cast(Any, policy),
    }
    if policy == "frame_camera_with_replacement":
        options["frame_groups"] = groups
    sampler = SceneSampler(range(6), **options)
    for _ in range(11):
        sampler.next_index()
    state = json.loads(json.dumps(sampler.state_dict()))
    expected = [sampler.next_index() for _ in range(40)]

    resumed = SceneSampler(range(6), **options)
    resumed.load_state_dict(state)

    assert [resumed.next_index() for _ in range(40)] == expected


def test_sampler_rejects_non_partitioning_groups_and_malformed_state() -> None:
    with pytest.raises(ContractError, match="partition every sampler index"):
        SceneSampler(
            range(4),
            seed=1,
            policy="frame_camera_with_replacement",
            frame_groups=((0, 1), (1, 2, 3)),
        )
    sampler = SceneSampler(range(4), seed=1)
    state = copy.deepcopy(sampler.state_dict())
    state["random_state"]["words"][0] = "not-an-integer"
    with pytest.raises(ContractError, match="words must be integers"):
        sampler.load_state_dict(state)


def test_public_dataset_source_has_no_reference_adapter_identity() -> None:
    source = (ROOT / "src/p2g/training/dataset.py").read_text(encoding="utf-8").casefold()
    forbidden = ("tensor" + "dict", "free" + "time", "ft" + "gs")

    assert not any(token in source for token in forbidden)
