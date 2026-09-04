from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

import p2g.selfcap as selfcap
from p2g.audit import audit_observation_manifest
from p2g.cli import app
from p2g.errors import ContractError


class _Node:
    def __init__(self, value: Any) -> None:
        self.value = value

    def size(self) -> int:
        return len(self.value)

    def at(self, index: int) -> _Node:
        return _Node(self.value[index])

    def string(self) -> str:
        return str(self.value)

    def mat(self) -> np.ndarray[Any, Any]:
        return np.asarray(self.value, dtype=np.float64)

    def real(self) -> float:
        return float(self.value)


class _FileStorage:
    def __init__(self, path: str, _mode: int, cameras: tuple[str, ...]) -> None:
        self.path = Path(path)
        self.cameras = cameras

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        return None

    def getNode(self, name: str) -> _Node:
        if name == "names":
            return _Node(list(self.cameras))
        camera_id = name.rsplit("_", 1)[-1]
        camera_index = self.cameras.index(camera_id)
        if name.startswith("K_"):
            return _Node([[4.0, 0.0, 3.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]])
        if name.startswith("D_"):
            return _Node([[0.0], [0.0], [0.0], [0.0], [0.0]])
        if name.startswith("Rot_"):
            return _Node(np.eye(3))
        if name.startswith("T_"):
            return _Node([[camera_index * 0.1], [0.0], [2.0]])
        if name.startswith("H_"):
            return _Node(6.0)
        if name.startswith("W_"):
            return _Node(8.0)
        raise AssertionError(name)


class _Capture:
    def __init__(self, path: str, backend: _FakeCv2) -> None:
        self.camera_id = Path(path).stem
        self.backend = backend
        self.position = 0

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        return None

    def set(self, property_id: int, value: float) -> bool:
        assert property_id == self.backend.CAP_PROP_POS_FRAMES
        self.position = round(value)
        return True

    def get(self, property_id: int) -> float:
        values = {
            self.backend.CAP_PROP_FRAME_WIDTH: 8.0,
            self.backend.CAP_PROP_FRAME_HEIGHT: 6.0,
            self.backend.CAP_PROP_FRAME_COUNT: 12.0,
            self.backend.CAP_PROP_FPS: 2.0,
            self.backend.CAP_PROP_POS_FRAMES: float(self.position),
        }
        return values[property_id]

    def read(self) -> tuple[bool, np.ndarray[Any, np.dtype[np.uint8]]]:
        camera_index = self.backend.cameras.index(self.camera_id)
        frame = np.empty((6, 8, 3), dtype=np.uint8)
        frame[..., 0] = 10 + self.position
        frame[..., 1] = 20 + camera_index
        frame[..., 2] = 30 + self.position
        self.position += 1
        return True, frame


class _FakeCv2:
    __version__ = "test-opencv-1"
    FILE_STORAGE_READ = 0
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FRAME_COUNT = 4
    CAP_PROP_FPS = 5
    CV_32FC1 = 6
    INTER_LINEAR = 7
    BORDER_CONSTANT = 8
    INTER_AREA = 9

    def __init__(self) -> None:
        self.cameras = ("0000", "0001", "0002", "0003")

    def getBuildInformation(self) -> str:
        return "deterministic fake OpenCV build"

    def FileStorage(self, path: str, mode: int) -> _FileStorage:
        return _FileStorage(path, mode, self.cameras)

    def VideoCapture(self, path: str) -> _Capture:
        return _Capture(path, self)

    def getOptimalNewCameraMatrix(
        self,
        intrinsic: np.ndarray[Any, Any],
        _distortion: np.ndarray[Any, Any],
        _source_size: tuple[int, int],
        _alpha: int,
        _output_size: tuple[int, int],
    ) -> tuple[np.ndarray[Any, Any], tuple[int, int, int, int]]:
        return intrinsic.copy(), (0, 0, 8, 6)

    def initUndistortRectifyMap(self, *_args: Any) -> tuple[None, None]:
        return None, None

    def remap(
        self, image: np.ndarray[Any, Any], *_args: Any, **_kwargs: Any
    ) -> np.ndarray[Any, Any]:
        return image

    def resize(
        self,
        image: np.ndarray[Any, Any],
        dimensions: tuple[int, int],
        *,
        interpolation: int,
    ) -> np.ndarray[Any, Any]:
        assert dimensions == (8, 6)
        assert interpolation == self.INTER_AREA
        return image

    def setNumThreads(self, threads: int) -> None:
        assert threads == 1


def _source(root: Path) -> Path:
    optimized = root / "optimized"
    videos = root / "videos"
    optimized.mkdir(parents=True)
    videos.mkdir()
    cameras = ("0000", "0001", "0002", "0003")
    (optimized / "intri.yml").write_text("test intrinsics\n", encoding="utf-8")
    (optimized / "extri.yml").write_text("test extrinsics\n", encoding="utf-8")
    (optimized / "sync.json").write_text(
        json.dumps({camera: index * 0.25 for index, camera in enumerate(cameras)}) + "\n",
        encoding="utf-8",
    )
    for camera in cameras:
        (videos / f"{camera}.mp4").write_bytes(f"video {camera}\n".encode())
    return root


def _import(root: Path, output: Path) -> dict[str, Any]:
    return selfcap.import_selfcap(
        root,
        output=output,
        dataset_id="selfcap_test",
        source_start_frame=2,
        frame_count=2,
        fps=2.0,
        scale=1.0,
        diagnostic_camera="0002",
        sealed_camera="0003",
        workers=1,
    )


@pytest.fixture(autouse=True)
def _fake_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCv2()
    monkeypatch.setattr(selfcap, "_opencv", lambda: fake)


def test_selfcap_import_materializes_public_rgb_manifest_and_roles(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    output = tmp_path / "output"
    progress: list[str] = []

    receipt = selfcap.import_selfcap(
        source,
        output=output,
        dataset_id="selfcap_test",
        source_start_frame=2,
        frame_count=2,
        fps=2.0,
        scale=1.0,
        diagnostic_camera="0002",
        sealed_camera="0003",
        workers=1,
        progress=progress.append,
    )
    manifest = json.loads((output / "observation_manifest.json").read_text(encoding="utf-8"))

    assert receipt["status"] == "PASS"
    assert receipt["camera_count"] == 4
    assert receipt["frame_count"] == 2
    assert receipt["source_frame_range_inclusive"] == [2, 3]
    assert receipt["role_counts"] == {"diagnostic": 2, "sealed": 2, "train": 4}
    assert progress == [
        "materialized 1/4 SelfCap cameras",
        "materialized 2/4 SelfCap cameras",
        "materialized 3/4 SelfCap cameras",
        "materialized 4/4 SelfCap cameras",
    ]
    assert manifest["source"]["license"] == selfcap.SELFCAP_LICENSE
    assert manifest["source"]["root_sha256"] == receipt["source_root_sha256"]
    assert manifest["transforms"][0]["name"] == selfcap.SELFCAP_CONVERSION["algorithm"]
    roles = {item["camera_id"]: item["role"] for item in manifest["observations"]}
    assert roles == {
        "0000": "train",
        "0001": "train",
        "0002": "diagnostic",
        "0003": "sealed",
    }
    assert [item["frame_id"] for item in manifest["observations"][:4]] == [0, 0, 0, 0]
    assert all(not Path(item["image"]["path"]).is_absolute() for item in manifest["observations"])
    assert all(item["image"]["encoding"]["container"] == "png" for item in manifest["observations"])
    audit = audit_observation_manifest(manifest, base_dir=output, verify_files=True)
    assert audit.status == "PASS", audit.to_dict()

    with Image.open(output / "rgb/0001/000000.png") as image:
        pixel = np.asarray(image)[0, 0].tolist()
    # Camera 0001 samples source frame 2.5: BGR [12.5, 21, 32.5] is rounded
    # half-up and then written as RGB.
    assert pixel == [33, 21, 13]


def test_selfcap_import_is_location_independent_and_exactly_resumable(tmp_path: Path) -> None:
    roots = [_source(tmp_path / "source-a"), _source(tmp_path / "source-b")]
    outputs = [tmp_path / "output-a", tmp_path / "output-b"]
    first = _import(roots[0], outputs[0])
    second = _import(roots[1], outputs[1])

    assert first == second
    assert (outputs[0] / "observation_manifest.json").read_bytes() == (
        outputs[1] / "observation_manifest.json"
    ).read_bytes()
    assert _import(roots[0], outputs[0]) == first


def test_selfcap_import_rejects_changed_output_and_role_overlap(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    output = tmp_path / "output"
    _import(source, output)
    image = output / "rgb/0000/000000.png"
    image.write_bytes(image.read_bytes() + b"changed")

    with pytest.raises(ContractError, match="no longer verifies"):
        _import(source, output)
    with pytest.raises(ContractError, match="distinct numeric IDs"):
        selfcap.import_selfcap(
            source,
            output=tmp_path / "overlap",
            dataset_id="selfcap_test",
            source_start_frame=2,
            frame_count=2,
            fps=2.0,
            scale=1.0,
            diagnostic_camera="0002",
            sealed_camera="0002",
        )


def test_selfcap_cli_dispatches_and_reports_terminal_receipt(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    output = tmp_path / "output"
    result = CliRunner().invoke(
        app,
        [
            "data",
            "import-selfcap",
            str(source),
            "--output",
            str(output),
            "--dataset-id",
            "selfcap_test",
            "--source-start-frame",
            "2",
            "--frame-count",
            "2",
            "--fps",
            "2",
            "--scale",
            "1",
            "--diagnostic-camera",
            "0002",
            "--sealed-camera",
            "0003",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "PASS"
    assert (output / "observation_manifest.json").is_file()
