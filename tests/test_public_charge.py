from __future__ import annotations

import copy
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from p2g.audit import audit_observation_manifest
from p2g.charge import (
    CHARGE_CONVERSION_SHA256,
    charge_c2w_blender_to_w2c_opencv,
    import_charge_manifest,
)
from p2g.cli import app
from p2g.errors import ContractError, OutputExistsError

REVISION = "3a2b0a91af66c02bf7444a8a2d6cef48b91bbf0c"
REPOSITORY = "https://huggingface.co/datasets/charge-benchmark/Charge-test"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _png(width: int, height: int, value: int) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = bytes((value, (value + 17) % 256, (value + 31) % 256)) * width
    scanlines = b"".join(b"\0" + row for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(scanlines))
        + _chunk(b"IEND", b"")
    )


def _record(camera_id: str, frame_id: int, *, tx: float, width: int, height: int) -> dict[str, Any]:
    return {
        "fov": 0.9,
        "f": 25.0,
        "type": "PERSP",
        "pixel_aspect_ratio": 1.0,
        "resolution_x": width,
        "resolution_y": height,
        "K": [
            [12.0, 0.0, (width - 1) / 2.0],
            [0.0, 12.0, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        "transformation_matrix": [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "frame_id": frame_id,
        "image_path": f"{camera_id}/frame_{frame_id:04d}.png",
    }


def _write_charge_task(root: Path) -> tuple[Path, Path]:
    width, height = 16, 12
    train: dict[str, list[dict[str, Any]]] = {}
    test: dict[str, list[dict[str, Any]]] = {}
    for camera_index, camera_id in enumerate(("Train_00", "Train_01")):
        train[camera_id] = [
            _record(camera_id, frame_id, tx=camera_index * 0.2, width=width, height=height)
            for frame_id in (416, 417)
        ]
    for camera_index, camera_id in enumerate(("Test_00", "Test_01")):
        test[camera_id] = [
            _record(
                camera_id,
                frame_id,
                tx=(camera_index + 0.5) * 0.2,
                width=width,
                height=height,
            )
            for frame_id in (416, 417)
        ]
    for camera_index, (_camera_id, records) in enumerate({**train, **test}.items()):
        for record in records:
            image = root / record["image_path"]
            image.parent.mkdir(parents=True, exist_ok=True)
            value = 20 + camera_index * 30 + record["frame_id"] - 416
            image.write_bytes(_png(width, height, value))
    train_path = root / "transforms_train.json"
    test_path = root / "transforms_test.json"
    train_path.write_text(json.dumps(train, sort_keys=True) + "\n", encoding="utf-8")
    test_path.write_text(json.dumps(test, sort_keys=True) + "\n", encoding="utf-8")
    return train_path, test_path


def _import(root: Path, output: Path) -> dict[str, Any]:
    return import_charge_manifest(
        root,
        train_transforms=Path("transforms_train.json"),
        test_transforms=Path("transforms_test.json"),
        output=output,
        dataset_id="charge_test_scene_dense",
        source_repository=REPOSITORY,
        source_revision=REVISION,
        sealed_camera_count=1,
    )


def test_charge_coordinate_conversion_is_explicit_and_right_handed() -> None:
    converted = charge_c2w_blender_to_w2c_opencv(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert converted == [
        [1.0, 0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, 2.0],
        [0.0, 0.0, -1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert len(CHARGE_CONVERSION_SHA256) == 64


def test_charge_import_hashes_pixels_reindexes_time_and_splits_test_cameras(
    tmp_path: Path,
) -> None:
    root = tmp_path / "charge"
    root.mkdir()
    _write_charge_task(root)
    output = tmp_path / "observation-manifest.json"
    progress: list[str] = []

    summary = import_charge_manifest(
        root,
        train_transforms=Path("transforms_train.json"),
        test_transforms=Path("transforms_test.json"),
        output=output,
        dataset_id="charge_test_scene_dense",
        source_repository=REPOSITORY,
        source_revision=REVISION,
        sealed_camera_count=1,
        progress=progress.append,
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))

    assert summary["status"] == "PASS"
    assert summary["frame_count"] == 2
    assert summary["source_frame_range_inclusive"] == [416, 417]
    assert summary["camera_counts"] == {"train": 2, "diagnostic": 1, "sealed": 1}
    assert summary["observation_counts"] == {"diagnostic": 2, "sealed": 2, "train": 4}
    assert progress == ["hashed 8/8 Charge RGB observations"]
    assert manifest["source"]["license"] == "CC-BY-4.0"
    assert manifest["transforms"] == [
        {
            "name": "charge_v1_blender_c2w_to_p2g_opencv_w2c_v1",
            "config_sha256": CHARGE_CONVERSION_SHA256,
        }
    ]
    assert sorted({item["frame_id"] for item in manifest["observations"]}) == [0, 1]
    assert sorted({item["timestamp_seconds"] for item in manifest["observations"]}) == [
        0.0,
        1.0 / 96.0,
    ]
    roles = {
        item["camera_id"]: item["role"]
        for item in manifest["observations"]
    }
    assert roles == {
        "Test_00": "diagnostic",
        "Test_01": "sealed",
        "Train_00": "train",
        "Train_01": "train",
    }
    report = audit_observation_manifest(manifest, base_dir=root, verify_files=True)
    assert report.status == "PASS", report.to_dict()


def test_charge_import_is_location_independent_and_never_overwrites(tmp_path: Path) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    for root in roots:
        root.mkdir()
        _write_charge_task(root)
    first = _import(roots[0], outputs[0])
    second = _import(roots[1], outputs[1])

    assert first == second
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    with pytest.raises(OutputExistsError, match="refusing to overwrite"):
        _import(roots[0], outputs[0])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("moving_camera", "pose varies over time"),
        ("mismatched_frames", "do not share one exact frame set"),
        ("non_perspective", "must use a Charge PERSP camera"),
    ],
)
def test_charge_import_rejects_unsupported_camera_inputs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    root = tmp_path / mutation
    root.mkdir()
    _, test_path = _write_charge_task(root)
    test = json.loads(test_path.read_text(encoding="utf-8"))
    if mutation == "moving_camera":
        test["Test_00"][1]["transformation_matrix"][0][3] += 0.1
    elif mutation == "mismatched_frames":
        test["Test_00"][1]["frame_id"] = 418
    else:
        test["Test_00"][0]["type"] = "ORTHO"
    test_path.write_text(json.dumps(test, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ContractError, match=message):
        _import(root, tmp_path / f"{mutation}.json")


def test_charge_cli_dispatches_lazily_and_reports_the_import(tmp_path: Path) -> None:
    root = tmp_path / "charge"
    root.mkdir()
    _write_charge_task(root)
    output = tmp_path / "manifest.json"

    result = CliRunner().invoke(
        app,
        [
            "data",
            "import-charge",
            str(root),
            "--output",
            str(output),
            "--dataset-id",
            "charge_test_scene_dense",
            "--source-repository",
            REPOSITORY,
            "--source-revision",
            REVISION,
            "--sealed-camera-count",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "PASS"
    assert output.is_file()


def test_charge_import_rejects_unpinned_source_identity(tmp_path: Path) -> None:
    root = tmp_path / "charge"
    root.mkdir()
    _write_charge_task(root)

    with pytest.raises(ContractError, match="40-character Git revision"):
        import_charge_manifest(
            root,
            train_transforms=Path("transforms_train.json"),
            test_transforms=Path("transforms_test.json"),
            output=tmp_path / "manifest.json",
            dataset_id="charge_test_scene_dense",
            source_repository=REPOSITORY,
            source_revision="main",
            sealed_camera_count=1,
        )


@pytest.mark.parametrize(
    "repository",
    [
        "https://example.com/datasets/charge-benchmark/Charge-test",
        "https://huggingface.co/charge-benchmark/Charge-test",
        "https://huggingface.co/datasets/charge-benchmark/not-charge",
        "https://huggingface.co/datasets/charge-benchmark/Charge-test/",
        "https://huggingface.co/datasets/charge-benchmark/Charge-test?download=true",
    ],
)
def test_charge_import_rejects_noncanonical_source_repository(
    tmp_path: Path, repository: str
) -> None:
    root = tmp_path / "charge"
    root.mkdir()
    _write_charge_task(root)

    with pytest.raises(ContractError, match="official charge-benchmark"):
        import_charge_manifest(
            root,
            train_transforms=Path("transforms_train.json"),
            test_transforms=Path("transforms_test.json"),
            output=tmp_path / "manifest.json",
            dataset_id="charge_test_scene_dense",
            source_repository=repository,
            source_revision=REVISION,
            sealed_camera_count=1,
        )


def test_charge_import_does_not_mutate_source_metadata(tmp_path: Path) -> None:
    root = tmp_path / "charge"
    root.mkdir()
    train_path, test_path = _write_charge_task(root)
    before = [
        copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
        for path in (train_path, test_path)
    ]

    _import(root, tmp_path / "manifest.json")

    after = [json.loads(path.read_text(encoding="utf-8")) for path in (train_path, test_path)]
    assert after == before
