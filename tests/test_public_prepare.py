from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from p2g.canonical import sha256_file
from p2g.errors import ContractError, OutputExistsError
from p2g.schema import validate_payload
from p2g.training.prepare import INCOMPLETE_MARKER, build_tensor_cache

ROOT = Path(__file__).parents[1]
CAMERAS = ("cam-a", "cam-b")
FRAMES = (0, 1)
HEIGHT = 4
WIDTH = 6


def _camera(camera_id: str) -> dict[str, Any]:
    translation = 0.0 if camera_id == "cam-a" else -1.0
    return {
        "model": "pinhole",
        "pixel_domain": "undistorted",
        "intrinsic": [
            [6.0, 0.0, 3.0],
            [0.0, 6.0, 2.0],
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


def _write_scene(root: Path) -> tuple[Path, dict[tuple[int, str], int]]:
    observations: list[dict[str, Any]] = []
    values: dict[tuple[int, str], int] = {}
    for frame_id in FRAMES:
        for camera_index, camera_id in enumerate(CAMERAS):
            value = 17 + frame_id * 53 + camera_index * 11
            values[(frame_id, camera_id)] = value
            relative = Path("images") / camera_id / f"{frame_id:06d}.png"
            image_path = root / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8),
                mode="RGB",
            ).save(image_path, format="PNG")
            observations.append(
                {
                    "observation_id": f"obs_{camera_id}_{frame_id:06d}",
                    "camera_id": camera_id,
                    "frame_id": frame_id,
                    "timestamp_seconds": frame_id + camera_index * 0.00025,
                    "role": "train" if frame_id == 0 else "diagnostic",
                    "image": {
                        "path": relative.as_posix(),
                        "sha256": sha256_file(image_path),
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
    manifest = {
        "schema_version": "p2g.observation_manifest.v2",
        "dataset_id": "public_prepare_fixture",
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
            "per_camera_offset_seconds": {"cam-a": 0.0, "cam-b": 0.00025},
        },
        "transforms": [],
        "observations": observations,
    }
    manifest_path = root / "observations.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, values


def test_prepare_builds_deterministic_hash_bound_cache(tmp_path: Path) -> None:
    manifest_path, values = _write_scene(tmp_path)
    progress: list[str] = []
    first = tmp_path / "cache-a"
    second = tmp_path / "cache-b"

    receipt = build_tensor_cache(
        first,
        observation_manifest=manifest_path,
        progress=progress.append,
    )
    second_receipt = build_tensor_cache(second, observation_manifest=manifest_path)

    assert receipt == second_receipt
    validate_payload("tensor_cache", receipt)
    assert receipt["observation_manifest_sha256"] == sha256_file(manifest_path)
    assert receipt["camera_ids"] == list(CAMERAS)
    assert receipt["frame_ids"] == list(FRAMES)
    assert progress[-1] == "prepared 4/4 observations"
    assert {path.name for path in first.iterdir()} == {
        "rgb.npy",
        "intrinsic.npy",
        "world_to_camera.npy",
        "timestamp_seconds.npy",
        "tensor_cache.json",
    }
    assert not (first / INCOMPLETE_MARKER).exists()
    assert str(tmp_path) not in (first / "tensor_cache.json").read_text(encoding="utf-8")

    for name, record in receipt["arrays"].items():
        assert record["sha256"] == sha256_file(first / record["path"])
        assert (first / record["path"]).read_bytes() == (
            second / second_receipt["arrays"][name]["path"]
        ).read_bytes()

    rgb = np.load(first / "rgb.npy", mmap_mode="r")
    assert rgb.shape == (2, 2, HEIGHT, WIDTH, 3)
    for frame_index, frame_id in enumerate(FRAMES):
        for camera_index, camera_id in enumerate(CAMERAS):
            assert np.all(rgb[frame_index, camera_index] == values[(frame_id, camera_id)])
    assert np.load(first / "intrinsic.npy", mmap_mode="r").dtype == np.float32
    assert np.load(first / "world_to_camera.npy", mmap_mode="r").dtype == np.float32
    assert np.load(first / "timestamp_seconds.npy", mmap_mode="r").dtype == np.float64


def test_prepare_fails_before_claiming_output_for_invalid_source(tmp_path: Path) -> None:
    manifest_path, _ = _write_scene(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["observations"][0]["camera"]["pixel_domain"] = "distorted"
    payload["observations"][0]["camera"]["model"] = "opencv_radtan"
    payload["observations"][0]["camera"]["distortion"] = [0.0, 0.0, 0.0, 0.0]
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "invalid-cache"

    with pytest.raises(ContractError, match="offline-undistorted pinhole"):
        build_tensor_cache(output, observation_manifest=manifest_path)
    assert not output.exists()


def test_prepare_never_overwrites_an_output(tmp_path: Path) -> None:
    manifest_path, _ = _write_scene(tmp_path)
    output = tmp_path / "cache"
    build_tensor_cache(output, observation_manifest=manifest_path)

    with pytest.raises(OutputExistsError, match="refusing to overwrite"):
        build_tensor_cache(output, observation_manifest=manifest_path)


def test_prepare_tool_help_is_lazy_and_source_is_dataset_neutral() -> None:
    tool = ROOT / "tools/prepare_observation_cache.py"
    source = tool.read_text(encoding="utf-8").casefold()
    for forbidden in ("cv2", "sync.json", "freetime", "/home/", "/mnt/"):
        assert forbidden not in source

    completed = subprocess.run(
        [sys.executable, "-S", str(tool), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert "p2g.tensor_cache.v1" in completed.stdout
    assert "torch" not in completed.stdout.casefold()
