"""Materialize synchronized SelfCap videos as public RGB observations.

The importer is intentionally a pixel-domain conversion, not a training
shortcut.  It reads the published EasyMocap calibration and synchronization
files, samples every camera at the same target times, undistorts into one
common valid crop, resizes once, quantizes once, and writes ordinary RGB8 PNG
files.  The resulting observation manifest is location independent and can be
consumed by :mod:`p2g.training.prepare` without a specialized TensorDict cache.

Completed camera directories are immutable, independently hash-closed units.
An interrupted import can therefore resume at the next camera while the final
manifest and receipt are published only after all images pass the public audit.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import io
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from p2g.audit import audit_observation_manifest
from p2g.canonical import (
    sha256_bytes,
    sha256_file,
    sha256_json,
    write_new_bytes,
    write_new_json,
)
from p2g.errors import ContractError, OutputExistsError
from p2g.image_probe import probe_image
from p2g.schema import validate_payload

Progress = Callable[[str], None]

SELFCAP_ADAPTER = "selfcap_video_rgb_v1"
SELFCAP_LICENSE = "LicenseRef-SelfCap-Research-NonCommercial"
SELFCAP_REPOSITORY = "https://huggingface.co/datasets/zju3dv/SelfCap-Dataset"
SELFCAP_CONVERSION = {
    "algorithm": "selfcap_fractional_rgb_undistort_common_roi_v1",
    "source_time": "source_start_frame + target_frame + fps * sync_seconds",
    "fractional_sample": "linear interpolation in decoded source BGR float32",
    "undistortion": "OpenCV initUndistortRectifyMap/remap INTER_LINEAR",
    "crop": "intersection of getOptimalNewCameraMatrix(alpha=0) valid ROIs",
    "resize": "OpenCV INTER_AREA",
    "quantization": "clip [0,255], round-half-up, uint8, BGR-to-RGB",
    "output": "lossless RGB8 PNG, one file per camera and target frame",
}

_CAMERA_RECEIPT = "p2g.selfcap_camera_rgb.v1"
_IMPORT_RECEIPT = "p2g.selfcap_import.v1"
_REQUEST = "p2g.selfcap_import_request.v1"


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """All source and output facts needed to materialize one camera."""

    camera_id: str
    video_path: str
    video_bytes: int
    video_sha256: str
    sync_seconds: float
    intrinsic: tuple[tuple[float, ...], ...]
    distortion: tuple[float, ...]
    output_intrinsic: tuple[tuple[float, ...], ...]
    world_to_camera: tuple[tuple[float, ...], ...]
    source_width: int
    source_height: int
    source_fps: float
    source_frame_count: int
    source_start_frame: int
    output_frame_count: int
    scale: float
    roi: tuple[int, int, int, int]
    output_width: int
    output_height: int
    runtime_sha256: str


def _opencv() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise ContractError(
            "SelfCap import requires opencv-python-headless; install pixel4dgs[selfcap]"
        ) from exc


def _pillow_image() -> Any:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise ContractError("SelfCap import requires Pillow") from exc


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _regular_member(root: Path, relative: str, *, label: str) -> Path:
    member = PurePosixPath(relative)
    if member.is_absolute() or not member.parts or ".." in member.parts or "." in member.parts:
        raise ContractError(f"{label} must be a contained POSIX relative path")
    unresolved = root.joinpath(*member.parts)
    if unresolved.is_symlink():
        raise ContractError(f"{label} must not be a symlink")
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.is_symlink():
        raise ContractError(f"{label} is not a regular file inside the SelfCap root")
    return resolved


def _finite_number(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ContractError(f"{label} must be a finite number")
    return float(value)


def _matrix_tuple(
    value: Any, *, rows: int, columns: int, label: str
) -> tuple[tuple[float, ...], ...]:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be a finite {rows}x{columns} matrix") from exc
    if matrix.shape != (rows, columns) or not np.isfinite(matrix).all():
        raise ContractError(f"{label} must be a finite {rows}x{columns} matrix")
    return tuple(tuple(float(item) for item in row) for row in matrix)


def _source_position(spec: CameraSpec, target_frame: int) -> tuple[int, int, float]:
    position = spec.source_start_frame + target_frame + spec.source_fps * spec.sync_seconds
    rounded = round(position)
    if math.isclose(position, rounded, rel_tol=0.0, abs_tol=1e-12):
        position = float(rounded)
    floor = math.floor(position)
    fraction = position - floor
    ceiling = floor if fraction == 0.0 else floor + 1
    return floor, ceiling, fraction


def _stable_file_record(path: Path, *, relative: str) -> dict[str, Any]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ContractError(f"SelfCap source changed while hashing: {relative}")
    return {"path": relative, "bytes": after.st_size, "sha256": digest}


def _runtime_identity() -> dict[str, str]:
    cv2 = _opencv()
    build_information = str(cv2.getBuildInformation())
    try:
        pillow_version = importlib.metadata.version("Pillow")
    except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover - dependency
        raise ContractError("SelfCap import requires Pillow") from exc
    return {
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "numpy": np.__version__,
        "opencv": str(cv2.__version__),
        "opencv_build_sha256": sha256_bytes(build_information.encode("utf-8")),
        "pillow": pillow_version,
    }


def _read_source_inventory(root: Path) -> tuple[dict[str, Any], dict[str, float]]:
    sync_path = _regular_member(root, "optimized/sync.json", label="sync file")
    intri_path = _regular_member(root, "optimized/intri.yml", label="intrinsic calibration")
    extri_path = _regular_member(root, "optimized/extri.yml", label="extrinsic calibration")
    raw_sync = _json_object(sync_path, label="SelfCap sync")
    sync: dict[str, float] = {}
    for raw_camera_id, raw_offset in raw_sync.items():
        if not raw_camera_id.isdigit():
            raise ContractError("SelfCap sync contains an invalid camera identifier")
        sync[raw_camera_id] = _finite_number(
            raw_offset, label=f"sync offset for camera {raw_camera_id}"
        )
    if len(sync) < 3:
        raise ContractError("SelfCap import requires at least three synchronized cameras")

    files: list[dict[str, Any]] = []
    for relative, path in (
        ("optimized/extri.yml", extri_path),
        ("optimized/intri.yml", intri_path),
        ("optimized/sync.json", sync_path),
    ):
        files.append(_stable_file_record(path, relative=relative))
    for camera_id in sorted(sync):
        relative = f"videos/{camera_id}.mp4"
        path = _regular_member(root, relative, label=f"source video {camera_id}")
        files.append(_stable_file_record(path, relative=relative))
    inventory: dict[str, Any] = {
        "schema_version": "p2g.selfcap_source_selection.v1",
        "source_repository": SELFCAP_REPOSITORY,
        "source_license": SELFCAP_LICENSE,
        "files": files,
    }
    inventory["logical_sha256"] = sha256_json(inventory)
    return inventory, sync


def _inventory_files(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_files_object: object = inventory.get("files")
    if not isinstance(raw_files_object, list):
        raise ContractError("SelfCap source inventory has no file list")
    raw_files = cast(list[object], raw_files_object)
    result: dict[str, dict[str, Any]] = {}
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ContractError("SelfCap source inventory contains an invalid file record")
        record = cast(dict[str, Any], raw)
        path_value: object = record.get("path")
        if not isinstance(path_value, str):
            raise ContractError("SelfCap source inventory contains an invalid file record")
        result[path_value] = record
    return result


def _read_calibration(
    root: Path,
    sync: Mapping[str, float],
    inventory: Mapping[str, Any],
    *,
    source_start_frame: int,
    frame_count: int,
    fps: float,
    scale: float,
    runtime_sha256: str,
) -> tuple[tuple[CameraSpec, ...], dict[str, Any]]:
    cv2 = _opencv()
    intri_path = root / "optimized/intri.yml"
    extri_path = root / "optimized/extri.yml"
    intri = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)
    if not intri.isOpened() or not extri.isOpened():
        intri.release()
        extri.release()
        raise ContractError("cannot open SelfCap EasyMocap calibration")
    try:
        intri_names_node = intri.getNode("names")
        extri_names_node = extri.getNode("names")
        intri_names = tuple(
            intri_names_node.at(index).string() for index in range(intri_names_node.size())
        )
        extri_names = tuple(
            extri_names_node.at(index).string() for index in range(extri_names_node.size())
        )
        if (
            not intri_names
            or len(set(intri_names)) != len(intri_names)
            or any(not item.isdigit() for item in intri_names)
            or len(set(extri_names)) != len(extri_names)
            or any(not item.isdigit() for item in extri_names)
        ):
            raise ContractError("SelfCap calibration contains invalid camera identifiers")
        camera_ids = tuple(sorted(intri_names, key=lambda item: (int(item), item)))
        if camera_ids != tuple(sorted(extri_names, key=lambda item: (int(item), item))):
            raise ContractError("SelfCap intrinsic and extrinsic camera inventories differ")
        if set(camera_ids) != set(sync):
            raise ContractError("SelfCap sync and calibration camera inventories differ")

        raw: dict[str, dict[str, Any]] = {}
        rois: list[tuple[int, int, int, int]] = []
        for camera_id in camera_ids:
            intrinsic = np.asarray(intri.getNode(f"K_{camera_id}").mat(), dtype=np.float64)
            distortion = np.asarray(
                intri.getNode(f"D_{camera_id}").mat(), dtype=np.float64
            ).reshape(-1)
            rotation = np.asarray(extri.getNode(f"Rot_{camera_id}").mat(), dtype=np.float64)
            translation = np.asarray(
                extri.getNode(f"T_{camera_id}").mat(), dtype=np.float64
            ).reshape(-1)
            height = round(float(intri.getNode(f"H_{camera_id}").real()))
            width = round(float(intri.getNode(f"W_{camera_id}").real()))
            if (
                intrinsic.shape != (3, 3)
                or rotation.shape != (3, 3)
                or translation.shape != (3,)
                or distortion.size not in {4, 5, 8, 12, 14}
                or not all(
                    np.isfinite(value).all()
                    for value in (intrinsic, distortion, rotation, translation)
                )
                or width <= 0
                or height <= 0
            ):
                raise ContractError(f"invalid SelfCap calibration for camera {camera_id}")
            if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=1e-6, atol=1e-6):
                raise ContractError(f"SelfCap camera {camera_id} rotation is not orthonormal")
            if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
                raise ContractError(f"SelfCap camera {camera_id} rotation determinant is not +1")
            new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
                intrinsic, distortion.reshape(-1, 1), (width, height), 0, (width, height)
            )
            roi_tuple = tuple(int(item) for item in roi)
            if (
                len(roi_tuple) != 4
                or roi_tuple[0] < 0
                or roi_tuple[1] < 0
                or roi_tuple[2] <= 0
                or roi_tuple[3] <= 0
                or roi_tuple[0] + roi_tuple[2] > width
                or roi_tuple[1] + roi_tuple[3] > height
            ):
                raise ContractError(f"SelfCap camera {camera_id} produced an invalid ROI")
            rois.append(roi_tuple)
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :3] = rotation
            world_to_camera[:3, 3] = translation
            raw[camera_id] = {
                "intrinsic": intrinsic,
                "distortion": distortion,
                "new_intrinsic": np.asarray(new_intrinsic, dtype=np.float64),
                "world_to_camera": world_to_camera,
                "width": width,
                "height": height,
            }

        source_shapes = {(item["width"], item["height"]) for item in raw.values()}
        if len(source_shapes) != 1:
            raise ContractError("SelfCap calibrated source image shapes differ")
        source_width, source_height = next(iter(source_shapes))
        x0 = max(roi[0] for roi in rois)
        y0 = max(roi[1] for roi in rois)
        x1 = min(roi[0] + roi[2] for roi in rois)
        y1 = min(roi[1] + roi[3] for roi in rois)
        if x1 <= x0 or y1 <= y0:
            raise ContractError("SelfCap cameras have no common valid undistortion ROI")
        common_roi = (x0, y0, x1 - x0, y1 - y0)
        output_width = int(scale * common_roi[2])
        output_height = int(scale * common_roi[3])
        if output_width <= 0 or output_height <= 0:
            raise ContractError("SelfCap scale produces an empty image")

        files = _inventory_files(inventory)
        specs: list[CameraSpec] = []
        camera_metadata: list[dict[str, Any]] = []
        for camera_id in camera_ids:
            video_path = root / f"videos/{camera_id}.mp4"
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                capture.release()
                raise ContractError(f"cannot open SelfCap source video {camera_id}")
            try:
                actual_width = round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
                actual_height = round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                actual_frames = round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
            finally:
                capture.release()
            if (actual_width, actual_height) != (source_width, source_height):
                raise ContractError(f"SelfCap video/calibration shape mismatch for {camera_id}")
            if actual_frames <= 0 or not math.isclose(
                actual_fps, fps, rel_tol=0.0, abs_tol=max(1e-6, fps * 1e-6)
            ):
                raise ContractError(f"SelfCap video metadata mismatch for {camera_id}")
            first_position = source_start_frame + fps * sync[camera_id]
            last_position = first_position + frame_count - 1
            if math.floor(first_position) < 0 or math.ceil(last_position) >= actual_frames:
                raise ContractError(f"SelfCap synchronized interval leaves video {camera_id}")

            item = raw[camera_id]
            output_intrinsic = item["new_intrinsic"].copy()
            output_intrinsic[0, 2] -= common_roi[0]
            output_intrinsic[1, 2] -= common_roi[1]
            output_intrinsic[:2] *= scale
            output_intrinsic[2, 2] = 1.0
            video = files[f"videos/{camera_id}.mp4"]
            spec = CameraSpec(
                camera_id=camera_id,
                video_path=str(video_path),
                video_bytes=cast(int, video["bytes"]),
                video_sha256=cast(str, video["sha256"]),
                sync_seconds=sync[camera_id],
                intrinsic=_matrix_tuple(item["intrinsic"], rows=3, columns=3, label="intrinsic"),
                distortion=tuple(float(value) for value in item["distortion"]),
                output_intrinsic=_matrix_tuple(
                    output_intrinsic, rows=3, columns=3, label="output intrinsic"
                ),
                world_to_camera=_matrix_tuple(
                    item["world_to_camera"], rows=4, columns=4, label="world_to_camera"
                ),
                source_width=source_width,
                source_height=source_height,
                source_fps=fps,
                source_frame_count=actual_frames,
                source_start_frame=source_start_frame,
                output_frame_count=frame_count,
                scale=scale,
                roi=common_roi,
                output_width=output_width,
                output_height=output_height,
                runtime_sha256=runtime_sha256,
            )
            specs.append(spec)
            camera_metadata.append(
                {
                    "camera_id": camera_id,
                    "source_frame_count": actual_frames,
                    "first_source_position": first_position,
                    "last_source_position": last_position,
                }
            )
    finally:
        intri.release()
        extri.release()
    return tuple(specs), {
        "camera_ids": list(camera_ids),
        "source_shape": [source_height, source_width, 3],
        "common_valid_roi_xywh": list(common_roi),
        "output_shape": [output_height, output_width, 3],
        "cameras": camera_metadata,
    }


def _png_bytes(rgb: np.ndarray[Any, np.dtype[np.uint8]]) -> bytes:
    stream = io.BytesIO()
    image_module = _pillow_image()
    image_module.fromarray(rgb, mode="RGB").save(
        stream, format="PNG", compress_level=6, optimize=False
    )
    return stream.getvalue()


def _camera_spec_sha256(spec: CameraSpec) -> str:
    portable = asdict(spec)
    portable["video_path"] = f"videos/{spec.camera_id}.mp4"
    return sha256_json(portable)


def _verify_camera_directory(root: Path, spec: CameraSpec) -> dict[str, Any]:
    receipt_path = root / "camera.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ContractError(f"camera {spec.camera_id} receipt is not a regular file")
    receipt = _json_object(receipt_path, label=f"camera {spec.camera_id} receipt")
    if (
        receipt.get("schema_version") != _CAMERA_RECEIPT
        or receipt.get("status") != "PASS"
        or receipt.get("camera_id") != spec.camera_id
        or receipt.get("spec_sha256") != _camera_spec_sha256(spec)
    ):
        raise ContractError(f"camera {spec.camera_id} receipt differs from the import request")
    receipt_id = receipt.get("receipt_id")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if receipt_id != sha256_json(unsigned):
        raise ContractError(f"camera {spec.camera_id} receipt identity changed")
    raw_images_object: object = receipt.get("images")
    if not isinstance(raw_images_object, list):
        raise ContractError(f"camera {spec.camera_id} receipt has an invalid image inventory")
    raw_images = cast(list[object], raw_images_object)
    if len(raw_images) != spec.output_frame_count:
        raise ContractError(f"camera {spec.camera_id} receipt has an invalid image inventory")
    expected_names = {f"{frame_id:06d}.png" for frame_id in range(spec.output_frame_count)}
    observed_names: list[str] = []
    for raw in raw_images:
        if not isinstance(raw, dict):
            raise ContractError(f"camera {spec.camera_id} image receipt is invalid")
        image = cast(dict[str, Any], raw)
        name = image.get("path")
        if not isinstance(name, str) or name not in expected_names or name in observed_names:
            raise ContractError(f"camera {spec.camera_id} image inventory is invalid")
        observed_names.append(name)
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"camera {spec.camera_id} output image is missing")
        if path.stat().st_size != image.get("bytes") or sha256_file(path) != image.get("sha256"):
            raise ContractError(f"camera {spec.camera_id} output image changed")
        probe = probe_image(path)
        if (
            probe.container != "png"
            or probe.channel_order != "RGB"
            or probe.bit_depth != 8
            or (probe.width, probe.height) != (spec.output_width, spec.output_height)
        ):
            raise ContractError(f"camera {spec.camera_id} output image contract changed")
    if observed_names != sorted(expected_names):
        raise ContractError(f"camera {spec.camera_id} image inventory is incomplete")
    return receipt


def _materialize_camera(spec: CameraSpec, output_root: str) -> dict[str, Any]:
    cv2 = _opencv()
    cv2.setNumThreads(1)
    output = Path(output_root)
    final = output / "rgb" / spec.camera_id
    if final.is_dir() and not final.is_symlink():
        return _verify_camera_directory(final, spec)
    if os.path.lexists(final):
        raise ContractError(f"camera output exists but is not a resumable directory: {final}")
    staging_parent = output / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = staging_parent / f"{spec.camera_id}.{os.getpid()}"
    if os.path.lexists(staging):
        raise ContractError(f"camera staging path already exists: {staging}")
    staging.mkdir()

    capture = cv2.VideoCapture(spec.video_path)
    if not capture.isOpened():
        capture.release()
        raise ContractError(f"camera {spec.camera_id}: cannot open source video")
    intrinsic = np.asarray(spec.intrinsic, dtype=np.float64)
    distortion = np.asarray(spec.distortion, dtype=np.float64).reshape(-1, 1)
    output_intrinsic = np.asarray(spec.output_intrinsic, dtype=np.float64)
    full_new_intrinsic = output_intrinsic.copy()
    full_new_intrinsic[:2] /= spec.scale
    full_new_intrinsic[0, 2] += spec.roi[0]
    full_new_intrinsic[1, 2] += spec.roi[1]
    map_x, map_y = cv2.initUndistortRectifyMap(
        intrinsic,
        distortion,
        np.eye(3, dtype=np.float64),
        full_new_intrinsic,
        (spec.source_width, spec.source_height),
        cv2.CV_32FC1,
    )
    positions = [_source_position(spec, frame) for frame in range(spec.output_frame_count)]
    first_decode = min(item[0] for item in positions)
    last_decode = max(item[1] for item in positions)
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(first_decode)):
        capture.release()
        raise ContractError(f"camera {spec.camera_id}: video backend rejected seek")
    observed_position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if not math.isfinite(observed_position) or abs(observed_position - first_decode) > 0.25:
        capture.release()
        raise ContractError(f"camera {spec.camera_id}: source seek is unverifiable")
    decoded: dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
    next_index = first_decode
    images: list[dict[str, Any]] = []

    def decode_through(index: int) -> None:
        nonlocal next_index
        while next_index <= index:
            ok, frame = capture.read()
            after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if (
                not ok
                or frame is None
                or frame.dtype != np.uint8
                or frame.shape != (spec.source_height, spec.source_width, 3)
                or not math.isfinite(after)
                or abs(after - (next_index + 1)) > 0.25
            ):
                raise ContractError(
                    f"camera {spec.camera_id}: failed verified decode at frame {next_index}"
                )
            decoded[next_index] = frame
            next_index += 1

    try:
        x, y, roi_width, roi_height = spec.roi
        for target_frame, (floor, ceiling, fraction) in enumerate(positions):
            decode_through(ceiling)
            work = decoded[floor].astype(np.float32)
            if ceiling != floor:
                work += np.float32(fraction) * (decoded[ceiling].astype(np.float32) - work)
            undistorted = cv2.remap(
                work,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            cropped = undistorted[y : y + roi_height, x : x + roi_width]
            resized = cv2.resize(
                cropped,
                (spec.output_width, spec.output_height),
                interpolation=cv2.INTER_AREA,
            )
            quantized_bgr = np.floor(np.clip(resized, 0.0, 255.0) + np.float32(0.5)).astype(
                np.uint8
            )
            encoded = _png_bytes(quantized_bgr[..., ::-1])
            name = f"{target_frame:06d}.png"
            path = staging / name
            write_new_bytes(path, encoded)
            images.append(
                {
                    "path": name,
                    "bytes": len(encoded),
                    "sha256": sha256_bytes(encoded),
                }
            )
            for stale in [index for index in decoded if index < floor]:
                del decoded[stale]
        if next_index - 1 != last_decode:
            raise ContractError(f"camera {spec.camera_id}: decoded range ended unexpectedly")
        source_video = Path(spec.video_path)
        if (
            source_video.stat().st_size != spec.video_bytes
            or sha256_file(source_video) != spec.video_sha256
        ):
            raise ContractError(f"camera {spec.camera_id}: source video changed during import")
        receipt_unsigned: dict[str, Any] = {
            "schema_version": _CAMERA_RECEIPT,
            "status": "PASS",
            "camera_id": spec.camera_id,
            "spec_sha256": _camera_spec_sha256(spec),
            "source_video": {
                "path": f"videos/{spec.camera_id}.mp4",
                "bytes": spec.video_bytes,
                "sha256": spec.video_sha256,
            },
            "source_decode_range_inclusive": [first_decode, last_decode],
            "output_shape": [spec.output_height, spec.output_width, 3],
            "images": images,
        }
        receipt = {**receipt_unsigned, "receipt_id": sha256_json(receipt_unsigned)}
        write_new_json(staging / "camera.json", receipt)
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(staging, final)
        except FileExistsError as exc:
            raise OutputExistsError(f"camera output appeared concurrently: {final}") from exc
        return _verify_camera_directory(final, spec)
    finally:
        capture.release()


def _camera_payload(spec: CameraSpec) -> dict[str, Any]:
    intrinsic = np.asarray(spec.output_intrinsic, dtype=np.float32).tolist()
    world_to_camera = np.asarray(spec.world_to_camera, dtype=np.float32).tolist()
    return {
        "model": "pinhole",
        "pixel_domain": "undistorted",
        "intrinsic": intrinsic,
        "world_to_camera": world_to_camera,
        "distortion": [],
    }


def _build_manifest(
    *,
    output: Path,
    dataset_id: str,
    specs: Sequence[CameraSpec],
    camera_receipts: Mapping[str, Mapping[str, Any]],
    source_inventory: Mapping[str, Any],
    diagnostic_camera: str,
    sealed_camera: str,
) -> dict[str, Any]:
    roles = {
        spec.camera_id: (
            "diagnostic"
            if spec.camera_id == diagnostic_camera
            else "sealed"
            if spec.camera_id == sealed_camera
            else "train"
        )
        for spec in specs
    }
    by_camera = {spec.camera_id: spec for spec in specs}
    image_records: dict[tuple[str, int], Mapping[str, Any]] = {}
    for camera_id, receipt in camera_receipts.items():
        images = cast(list[dict[str, Any]], receipt["images"])
        for frame_id, image in enumerate(images):
            image_records[(camera_id, frame_id)] = image

    observations: list[dict[str, Any]] = []
    frame_count = specs[0].output_frame_count
    for frame_id in range(frame_count):
        for camera_id in sorted(by_camera):
            spec = by_camera[camera_id]
            image = image_records[(camera_id, frame_id)]
            relative = f"rgb/{camera_id}/{image['path']}"
            observations.append(
                {
                    "observation_id": f"{dataset_id}_c{camera_id}_f{frame_id:06d}",
                    "camera_id": camera_id,
                    "frame_id": frame_id,
                    "timestamp_seconds": frame_id / spec.source_fps,
                    "role": roles[camera_id],
                    "image": {
                        "path": relative,
                        "sha256": image["sha256"],
                        "width": spec.output_width,
                        "height": spec.output_height,
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
                    "camera": _camera_payload(spec),
                }
            )
    implementation_path = Path(__file__).resolve()
    conversion_config = {
        **SELFCAP_CONVERSION,
        "source_start_frame": specs[0].source_start_frame,
        "frame_count": frame_count,
        "source_fps": specs[0].source_fps,
        "scale": specs[0].scale,
        "camera_ids": sorted(by_camera),
        "diagnostic_camera": diagnostic_camera,
        "sealed_camera": sealed_camera,
        "common_valid_roi_xywh": list(specs[0].roi),
        "output_shape": [specs[0].output_height, specs[0].output_width, 3],
    }
    manifest: dict[str, Any] = {
        "schema_version": "p2g.observation_manifest.v2",
        "dataset_id": dataset_id,
        "source": {
            "description": (
                f"SelfCap synchronized RGB observations from {SELFCAP_REPOSITORY}; "
                f"source frames {specs[0].source_start_frame}.."
                f"{specs[0].source_start_frame + frame_count - 1}"
            ),
            "license": SELFCAP_LICENSE,
            "license_status": "restricted",
            "root_sha256": source_inventory["logical_sha256"],
        },
        "coordinate_conventions": {
            "handedness": "right",
            "extrinsic": "world_to_camera",
            "pixel_center": "half_pixel",
            "time_unit": "seconds",
            "photometric_space": "srgb_reference_profile",
        },
        "sync": {
            "variant": "selfcap_fractional_common_time_v1",
            "tolerance_seconds": 0.0,
            "per_camera_offset_seconds": {spec.camera_id: spec.sync_seconds for spec in specs},
        },
        "transforms": [
            {
                "name": SELFCAP_CONVERSION["algorithm"],
                "config_sha256": sha256_json(conversion_config),
                "implementation_sha256": sha256_file(implementation_path),
            }
        ],
        "observations": observations,
    }
    validate_payload("observation", manifest)
    audit = audit_observation_manifest(manifest, base_dir=output, verify_files=True)
    if audit.status != "PASS":
        failures = [
            check.name for check in audit.checks if check.required and check.status == "FAIL"
        ]
        raise ContractError(f"generated SelfCap manifest failed public audit: {failures}")
    return manifest


def _validate_arguments(
    *,
    dataset_id: str,
    source_start_frame: int,
    frame_count: int,
    fps: float,
    scale: float,
    workers: int,
    diagnostic_camera: str,
    sealed_camera: str,
) -> None:
    if (
        not dataset_id
        or len(dataset_id) > 96
        or not dataset_id[0].isalnum()
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
            for character in dataset_id
        )
    ):
        raise ContractError("dataset_id is outside the public identifier contract")
    if isinstance(source_start_frame, bool) or source_start_frame < 0:
        raise ContractError("source_start_frame must be a non-negative integer")
    if isinstance(frame_count, bool) or frame_count < 2:
        raise ContractError("frame_count must be an integer of at least two")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ContractError("fps must be a positive finite number")
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ContractError("scale must be finite and in (0, 1]")
    if isinstance(workers, bool) or workers < 1:
        raise ContractError("workers must be a positive integer")
    if (
        not diagnostic_camera
        or not sealed_camera
        or diagnostic_camera == sealed_camera
        or not diagnostic_camera.isdigit()
        or not sealed_camera.isdigit()
    ):
        raise ContractError("diagnostic and sealed cameras must be distinct numeric IDs")


def import_selfcap(
    dataset_root: Path,
    *,
    output: Path,
    dataset_id: str,
    source_start_frame: int,
    frame_count: int,
    fps: float,
    scale: float,
    diagnostic_camera: str,
    sealed_camera: str,
    workers: int = 1,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Convert one local SelfCap capture into ordinary PNG observations."""

    _validate_arguments(
        dataset_id=dataset_id,
        source_start_frame=source_start_frame,
        frame_count=frame_count,
        fps=fps,
        scale=scale,
        workers=workers,
        diagnostic_camera=diagnostic_camera,
        sealed_camera=sealed_camera,
    )
    root = dataset_root.expanduser().resolve()
    destination = output.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ContractError("SelfCap dataset root must be a regular directory")
    if destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ContractError("SelfCap output and source roots must be disjoint")

    runtime = _runtime_identity()
    runtime_sha256 = sha256_json(runtime)
    source_inventory, sync = _read_source_inventory(root)
    specs, calibration = _read_calibration(
        root,
        sync,
        source_inventory,
        source_start_frame=source_start_frame,
        frame_count=frame_count,
        fps=fps,
        scale=scale,
        runtime_sha256=runtime_sha256,
    )
    camera_ids = {spec.camera_id for spec in specs}
    if diagnostic_camera not in camera_ids or sealed_camera not in camera_ids:
        raise ContractError("diagnostic or sealed camera is absent from SelfCap source")

    request: dict[str, Any] = {
        "schema_version": _REQUEST,
        "adapter": SELFCAP_ADAPTER,
        "dataset_id": dataset_id,
        "source_root_sha256": source_inventory["logical_sha256"],
        "source_start_frame": source_start_frame,
        "frame_count": frame_count,
        "fps": fps,
        "scale": scale,
        "diagnostic_camera": diagnostic_camera,
        "sealed_camera": sealed_camera,
        "camera_ids": [spec.camera_id for spec in specs],
        "calibration": calibration,
        "conversion_sha256": sha256_json(SELFCAP_CONVERSION),
        "runtime": runtime,
        "runtime_sha256": runtime_sha256,
    }
    if os.path.lexists(destination) and not destination.is_dir():
        raise OutputExistsError(f"SelfCap output exists and is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    request_path = destination / "request.json"
    if request_path.is_file() and not request_path.is_symlink():
        if _json_object(request_path, label="existing SelfCap request") != request:
            raise ContractError("existing SelfCap request differs from current arguments")
    elif os.path.lexists(request_path):
        raise ContractError("SelfCap request path is not a regular file")
    else:
        write_new_json(request_path, request)
    inventory_path = destination / "source_inventory.json"
    if inventory_path.is_file() and not inventory_path.is_symlink():
        if _json_object(inventory_path, label="existing source inventory") != source_inventory:
            raise ContractError("existing SelfCap source inventory changed")
    elif os.path.lexists(inventory_path):
        raise ContractError("SelfCap source inventory path is not a regular file")
    else:
        write_new_json(inventory_path, source_inventory)

    receipt_path = destination / "import.json"
    manifest_path = destination / "observation_manifest.json"
    if receipt_path.is_file() and not receipt_path.is_symlink():
        receipt = _json_object(receipt_path, label="completed SelfCap import")
        receipt_unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
        if (
            receipt.get("schema_version") != _IMPORT_RECEIPT
            or receipt.get("status") != "PASS"
            or receipt.get("receipt_id") != sha256_json(receipt_unsigned)
            or receipt.get("request_sha256") != sha256_file(request_path)
            or receipt.get("source_inventory_sha256") != sha256_file(inventory_path)
            or not manifest_path.is_file()
            or receipt.get("observation_manifest_sha256") != sha256_file(manifest_path)
        ):
            raise ContractError("completed SelfCap import receipt no longer verifies")
        manifest = _json_object(manifest_path, label="completed observation manifest")
        validate_payload("observation", manifest)
        audit = audit_observation_manifest(manifest, base_dir=destination, verify_files=True)
        if audit.status != "PASS":
            raise ContractError("completed SelfCap observation manifest no longer verifies")
        return receipt
    if os.path.lexists(receipt_path) or os.path.lexists(manifest_path):
        raise ContractError("partial terminal SelfCap artifacts exist without a valid receipt")

    camera_receipts: dict[str, dict[str, Any]] = {}
    if workers == 1:
        for index, spec in enumerate(specs, start=1):
            camera_receipts[spec.camera_id] = _materialize_camera(spec, str(destination))
            if progress is not None:
                progress(f"materialized {index}/{len(specs)} SelfCap cameras")
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as executor:
            futures = {
                executor.submit(_materialize_camera, spec, str(destination)): spec.camera_id
                for spec in specs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                camera_id = futures[future]
                camera_receipts[camera_id] = future.result()
                if progress is not None:
                    progress(f"materialized {completed}/{len(specs)} SelfCap cameras")

    manifest = _build_manifest(
        output=destination,
        dataset_id=dataset_id,
        specs=specs,
        camera_receipts=camera_receipts,
        source_inventory=source_inventory,
        diagnostic_camera=diagnostic_camera,
        sealed_camera=sealed_camera,
    )
    write_new_json(manifest_path, manifest)
    role_counts = Counter(cast(str, item["role"]) for item in manifest["observations"])
    unsigned: dict[str, Any] = {
        "schema_version": _IMPORT_RECEIPT,
        "status": "PASS",
        "adapter": SELFCAP_ADAPTER,
        "dataset_id": dataset_id,
        "request_sha256": sha256_file(request_path),
        "source_inventory_sha256": sha256_file(inventory_path),
        "source_root_sha256": source_inventory["logical_sha256"],
        "observation_manifest_sha256": sha256_file(manifest_path),
        "camera_count": len(specs),
        "frame_count": frame_count,
        "observation_count": len(manifest["observations"]),
        "role_counts": dict(sorted(role_counts.items())),
        "source_frame_range_inclusive": [source_start_frame, source_start_frame + frame_count - 1],
        "output_shape": calibration["output_shape"],
        "camera_receipt_ids": {
            camera_id: camera_receipts[camera_id]["receipt_id"]
            for camera_id in sorted(camera_receipts)
        },
        "claim_boundary": (
            "PASS proves deterministic source binding, synchronized RGB materialization, "
            "public-manifest audit, and role partitioning; it does not prove training quality "
            "or grant permission to redistribute source or derived media."
        ),
    }
    receipt = {**unsigned, "receipt_id": sha256_json(unsigned)}
    write_new_json(receipt_path, receipt)
    return receipt


__all__ = [
    "SELFCAP_ADAPTER",
    "SELFCAP_CONVERSION",
    "SELFCAP_LICENSE",
    "SELFCAP_REPOSITORY",
    "CameraSpec",
    "import_selfcap",
]
