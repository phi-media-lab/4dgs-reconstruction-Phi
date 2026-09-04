"""Render a portable AssetBundle from an explicit, hash-bound camera path."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import torch

from p2g import __version__
from p2g.canonical import sha256_bytes, sha256_file, write_new_json
from p2g.errors import ContractError, OutputExistsError
from p2g.schema import validate_payload
from p2g.training.asset import VerifiedAssetBundle, load_asset_bundle
from p2g.training.config import RendererConfig
from p2g.training.dataset import TrainingBatch
from p2g.training.evaluate import write_render
from p2g.training.renderer import GsplatRenderer

CAMERA_PATH_SCHEMA = "p2g.camera_path.v1"
ASSET_VIDEO_RECEIPT_SCHEMA = "p2g.asset_video_render.v1"
ASSET_VIDEO_RENDERER_ABI = "p2g.asset_video_renderer.v1"


@dataclass(frozen=True)
class CameraPathFrame:
    timestamp_seconds: float
    intrinsic: tuple[tuple[float, float, float], ...]
    world_to_camera: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class CameraPath:
    asset_bundle_id: str
    width: int
    height: int
    fps: int
    frames: tuple[CameraPathFrame, ...]
    source_sha256: str


@dataclass(frozen=True)
class CameraTrajectory:
    width: int
    height: int
    fps: int
    frames: tuple[CameraPathFrame, ...]
    source_sha256: str


def _matrix(
    value: Any,
    *,
    rows: int,
    columns: int,
    label: str,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be an array")
    raw_rows = cast(list[Any], value)
    if len(raw_rows) != rows:
        raise ContractError(f"{label} must have {rows} rows")
    result: list[tuple[float, ...]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, list):
            raise ContractError(f"{label} rows must be arrays")
        values = cast(list[Any], raw_row)
        if len(values) != columns:
            raise ContractError(f"{label} must have shape [{rows},{columns}]")
        try:
            row = tuple(float(item) for item in values)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{label} must contain numbers") from exc
        if not all(math.isfinite(item) for item in row):
            raise ContractError(f"{label} must be finite")
        result.append(row)
    return tuple(result)


def _validate_camera_frame(frame: CameraPathFrame, *, index: int) -> None:
    intrinsic = torch.tensor(frame.intrinsic, dtype=torch.float64)
    world_to_camera = torch.tensor(frame.world_to_camera, dtype=torch.float64)
    if float(intrinsic[0, 0]) <= 0.0 or float(intrinsic[1, 1]) <= 0.0:
        raise ContractError(f"camera-path frame {index} has non-positive focal length")
    expected_intrinsic_row = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    if not torch.allclose(intrinsic[2], expected_intrinsic_row, atol=1.0e-8, rtol=0.0):
        raise ContractError(f"camera-path frame {index} has an invalid intrinsic last row")
    if not math.isclose(float(intrinsic[0, 1]), 0.0, abs_tol=1.0e-8) or not math.isclose(
        float(intrinsic[1, 0]), 0.0, abs_tol=1.0e-8
    ):
        raise ContractError(
            f"camera-path frame {index} uses skew unsupported by the renderer ABI"
        )
    expected_extrinsic_row = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    if not torch.allclose(
        world_to_camera[3], expected_extrinsic_row, atol=1.0e-8, rtol=0.0
    ):
        raise ContractError(f"camera-path frame {index} has an invalid extrinsic last row")
    rotation = world_to_camera[:3, :3]
    identity = torch.eye(3, dtype=torch.float64)
    if not torch.allclose(rotation @ rotation.T, identity, atol=1.0e-4, rtol=1.0e-4):
        raise ContractError(f"camera-path frame {index} rotation is not orthonormal")
    raw = frame.world_to_camera
    determinant = (
        raw[0][0] * (raw[1][1] * raw[2][2] - raw[1][2] * raw[2][1])
        - raw[0][1] * (raw[1][0] * raw[2][2] - raw[1][2] * raw[2][0])
        + raw[0][2] * (raw[1][0] * raw[2][1] - raw[1][1] * raw[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1.0e-4):
        raise ContractError(f"camera-path frame {index} rotation is not right-handed")


def _load_camera_payload(path: Path, *, schema: str, label: str) -> tuple[dict[str, Any], bytes]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ContractError(f"{label} must be a regular non-symlink file")
    path = unresolved.resolve()
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} must be a regular non-symlink file")
    try:
        source = path.read_bytes()
        payload: Any = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc
    validate_payload(schema, payload)
    return cast(dict[str, Any], payload), source


def _camera_frames(value: dict[str, Any], *, label: str) -> tuple[CameraPathFrame, ...]:
    raw_frames = cast(list[dict[str, Any]], value["frames"])
    frames: list[CameraPathFrame] = []
    for index, raw in enumerate(raw_frames):
        timestamp = float(raw["timestamp_seconds"])
        if not math.isfinite(timestamp):
            raise ContractError(f"{label} frame {index} has a non-finite time")
        frame = CameraPathFrame(
            timestamp_seconds=timestamp,
            intrinsic=cast(
                tuple[tuple[float, float, float], ...],
                _matrix(raw["intrinsic"], rows=3, columns=3, label="intrinsic"),
            ),
            world_to_camera=cast(
                tuple[tuple[float, float, float, float], ...],
                _matrix(
                    raw["world_to_camera"],
                    rows=4,
                    columns=4,
                    label="world_to_camera",
                ),
            ),
        )
        _validate_camera_frame(frame, index=index)
        frames.append(frame)
    if any(
        second.timestamp_seconds < first.timestamp_seconds
        for first, second in pairwise(frames)
    ):
        raise ContractError(f"{label} timestamps must be nondecreasing")
    return tuple(frames)


def load_camera_path(path: Path) -> CameraPath:
    """Load and semantically validate exactly the bytes later named by the receipt."""

    value, source = _load_camera_payload(
        path,
        schema="camera_path",
        label="camera path",
    )
    return CameraPath(
        asset_bundle_id=cast(str, value["asset_bundle_id"]),
        width=int(value["width"]),
        height=int(value["height"]),
        fps=int(value["fps"]),
        frames=_camera_frames(value, label="camera-path"),
        source_sha256=sha256_bytes(source),
    )


def load_camera_trajectory(path: Path) -> CameraTrajectory:
    """Load camera geometry and time samples that are not yet bound to an asset."""

    value, source = _load_camera_payload(
        path,
        schema="camera_trajectory",
        label="camera trajectory",
    )
    return CameraTrajectory(
        width=int(value["width"]),
        height=int(value["height"]),
        fps=int(value["fps"]),
        frames=_camera_frames(value, label="camera-trajectory"),
        source_sha256=sha256_bytes(source),
    )


def renderer_config_from_asset(bundle: VerifiedAssetBundle) -> RendererConfig:
    raw = cast(dict[str, Any], bundle.metadata["renderer"])
    background = cast(list[float], raw["background_linear_rgb"])
    config = RendererConfig(
        backend="gsplat_rocm",
        near_plane=float(raw["near_plane"]),
        far_plane=float(raw["far_plane"]),
        eps2d=float(raw["eps2d"]),
        radius_clip=float(raw["radius_clip"]),
        tile_size=int(raw["tile_size"]),
        packed=bool(raw["packed"]),
        background=(float(background[0]), float(background[1]), float(background[2])),
        clamp_rgb=bool(raw["clamp_rgb"]),
        require_gfx942=raw["required_architecture"] == "gfx942",
    )
    if config.near_plane <= 0.0 or config.far_plane <= config.near_plane:
        raise ContractError("asset renderer near/far interval is invalid")
    if config.eps2d < 0.0 or config.radius_clip < 0.0:
        raise ContractError("asset renderer filter parameters are invalid")
    if config.tile_size != 8:
        raise ContractError("asset renderer tile size is unsupported")
    return config


def _validate_path_against_asset(path: CameraPath, bundle: VerifiedAssetBundle) -> None:
    bundle_id = cast(str, bundle.manifest["bundle_id"])
    if path.asset_bundle_id != bundle_id:
        raise ContractError(
            "camera path is bound to a different AssetBundle: "
            f"expected {bundle_id}, found {path.asset_bundle_id}"
        )
    time_metadata = cast(dict[str, Any], bundle.metadata["time"])
    interval = cast(list[float], time_metadata["valid_interval"])
    start, stop = float(interval[0]), float(interval[1])
    if not math.isfinite(start) or not math.isfinite(stop) or start >= stop:
        raise ContractError("asset valid-time interval is invalid")
    for index, frame in enumerate(path.frames):
        if not start <= frame.timestamp_seconds <= stop:
            raise ContractError(
                f"camera-path frame {index} time is outside asset interval [{start}, {stop}]"
            )


def bind_camera_trajectory(
    asset: Path,
    *,
    trajectory_file: Path,
    output: Path,
) -> dict[str, Any]:
    """Bind asset-independent camera geometry to one already-published AssetBundle."""

    bundle = load_asset_bundle(asset)
    unresolved_destination = output.expanduser()
    if unresolved_destination.is_symlink():
        raise ContractError("camera-path output must not be a symlink")
    destination = unresolved_destination.resolve()
    if destination.suffix.casefold() != ".json":
        raise ContractError("camera-path output must use a .json filename")
    if destination == bundle.root or bundle.root in destination.parents:
        raise ContractError("camera path must be published outside the AssetBundle")
    trajectory = load_camera_trajectory(trajectory_file)
    bundle_id = cast(str, bundle.manifest["bundle_id"])
    candidate = CameraPath(
        asset_bundle_id=bundle_id,
        width=trajectory.width,
        height=trajectory.height,
        fps=trajectory.fps,
        frames=trajectory.frames,
        source_sha256="",
    )
    _validate_path_against_asset(candidate, bundle)
    payload = {
        "schema_version": CAMERA_PATH_SCHEMA,
        "asset_bundle_id": bundle_id,
        "time_unit": "seconds",
        "camera_model": "pinhole",
        "pixel_domain": "pre-undistorted",
        "intrinsic_matrix": "3x3_pixel_center",
        "extrinsic_matrix": "4x4_world_to_camera",
        "camera_axes": "opencv_x_right_y_down_z_forward",
        "width": trajectory.width,
        "height": trajectory.height,
        "fps": trajectory.fps,
        "frames": [
            {
                "timestamp_seconds": frame.timestamp_seconds,
                "intrinsic": [list(row) for row in frame.intrinsic],
                "world_to_camera": [list(row) for row in frame.world_to_camera],
            }
            for frame in trajectory.frames
        ],
    }
    validate_payload("camera_path", payload)
    write_new_json(destination, payload)
    bound = load_camera_path(destination)
    _validate_path_against_asset(bound, bundle)
    time_values = [frame.timestamp_seconds for frame in bound.frames]
    return {
        "schema_version": "p2g.camera_path_binding.v1",
        "status": "PASS",
        "asset_bundle_id": bundle_id,
        "asset_model_sha256": bundle.metadata["model"]["sha256"],
        "trajectory_sha256": trajectory.source_sha256,
        "camera_path_sha256": bound.source_sha256,
        "frame_count": len(bound.frames),
        "fps": bound.fps,
        "resolution": [bound.width, bound.height],
        "time_interval_seconds": [min(time_values), max(time_values)],
        "claim_boundary": (
            "The new camera path is bound to this exact AssetBundle and admitted time "
            "interval; no render, visual-quality, or redistribution claim was made."
        ),
    }


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError(f"asset video runtime distribution is missing: {name}") from exc


def _validate_runtime_against_asset(
    bundle: VerifiedAssetBundle,
    runtime: dict[str, Any],
) -> None:
    producer = cast(dict[str, Any], bundle.metadata["producer"])
    dependencies = cast(dict[str, str], producer["dependencies"])
    gsplat_revision = runtime.get("gsplat_source_revision")
    if not isinstance(gsplat_revision, str) or not gsplat_revision:
        raise ContractError("asset video runtime is missing the amd-gsplat source revision")
    observed = {
        "amd-gsplat": (
            f"{cast(str, runtime['gsplat_distribution'])}@{gsplat_revision}"
        ),
        "torch": cast(str, runtime["torch"]),
        "torch-hip": cast(str, runtime["hip"]),
        "safetensors": _distribution_version("safetensors"),
    }
    mismatches = {
        name: {"asset": expected, "runtime": observed[name]}
        for name, expected in dependencies.items()
        if name in observed and observed[name] != expected
    }
    if mismatches:
        raise ContractError(f"asset dependency identity mismatch: {mismatches}")


def _implementation_hashes() -> dict[str, str]:
    source_root = Path(__file__).resolve().parent
    names = ("asset", "asset_render", "config", "dataset", "evaluate", "model", "renderer")
    paths = {name: source_root / f"{name}.py" for name in names}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ContractError(f"asset video renderer source identity is unavailable: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _flush_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _flush_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _tool_version(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ContractError(f"cannot execute {Path(executable).name}: {exc}") from exc
    lines = completed.stdout.splitlines()
    if completed.returncode != 0 or not lines:
        raise ContractError(f"cannot identify {Path(executable).name}")
    return lines[0]


def _encode_video(
    *,
    ffmpeg: str,
    frames_root: Path,
    destination: Path,
    fps: int,
    frame_count: int,
    crf: int,
) -> None:
    completed = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            str(frames_root / "frame_%06d.png"),
            "-frames:v",
            str(frame_count),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1000:]
        raise ContractError(f"ffmpeg video encoding failed: {detail or 'unknown error'}")


def _probe_video(
    ffprobe: str,
    video: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_fps: int,
    expected_frames: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(video),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1000:]
        raise ContractError(f"ffprobe validation failed: {detail or 'unknown error'}")
    try:
        payload: Any = json.loads(completed.stdout)
        streams = cast(list[dict[str, Any]], cast(dict[str, Any], payload)["streams"])
        stream = streams[0]
        width = int(stream["width"])
        height = int(stream["height"])
        frame_count = int(stream["nb_read_frames"])
        frame_rate = Fraction(cast(str, stream["avg_frame_rate"]))
        codec = cast(str, stream["codec_name"])
        pixel_format = cast(str, stream["pix_fmt"])
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ContractError("ffprobe returned an invalid video description") from exc
    if (width, height) != (expected_width, expected_height):
        raise ContractError(
            f"encoded video resolution is {(width, height)}, expected "
            f"{(expected_width, expected_height)}"
        )
    if frame_count != expected_frames:
        raise ContractError(f"encoded video has {frame_count} frames, expected {expected_frames}")
    if frame_rate != expected_fps:
        raise ContractError(f"encoded video rate is {frame_rate}, expected {expected_fps}")
    if codec != "h264" or pixel_format != "yuv420p":
        raise ContractError(
            f"encoded video format is {codec}/{pixel_format}, expected h264/yuv420p"
        )
    return {
        "codec": codec,
        "pixel_format": pixel_format,
        "verified_frame_count": frame_count,
    }


@torch.inference_mode()
def render_asset_video(
    asset: Path,
    *,
    camera_path_file: Path,
    output: Path,
    receipt: Path | None = None,
    device: str = "cuda",
    crf: int = 18,
) -> dict[str, Any]:
    """Render a video without consulting a training run or source SceneBundle."""

    output = output.expanduser().resolve()
    receipt = (
        output.with_suffix(".render.json")
        if receipt is None
        else receipt.expanduser().resolve()
    )
    if output.suffix.lower() != ".mp4":
        raise ContractError("asset video output must use an .mp4 filename")
    if output == receipt:
        raise ContractError("asset video and receipt paths must be distinct")
    if _path_exists(output) or _path_exists(receipt):
        raise OutputExistsError("refusing to overwrite asset video or receipt")
    if not 0 <= crf <= 51:
        raise ContractError("H.264 CRF must be in [0, 51]")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise ContractError("ffmpeg and ffprobe are required to encode and verify an asset video")
    ffmpeg_version = _tool_version(ffmpeg)

    asset_load_started = time.perf_counter()
    bundle = load_asset_bundle(asset)
    asset_load_ms = (time.perf_counter() - asset_load_started) * 1_000.0
    camera_path = load_camera_path(camera_path_file)
    _validate_path_against_asset(camera_path, bundle)
    renderer = GsplatRenderer(renderer_config_from_asset(bundle))
    target = torch.device(device)
    runtime = renderer.validate_environment(target)
    runtime["safetensors"] = _distribution_version("safetensors")
    runtime["imageio"] = _distribution_version("imageio")
    _validate_runtime_against_asset(bundle, runtime)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    transfer_started = time.perf_counter()
    model = bundle.model.to(target).eval()
    _sync(target)
    model_to_device_ms = (time.perf_counter() - transfer_started) * 1_000.0
    appearance = cast(dict[str, Any], bundle.metadata["appearance"])
    sh_degree = int(appearance["default_sh_degree"])
    photometric_space = cast(str, appearance["output_photometric_space"])

    output.parent.mkdir(parents=True, exist_ok=True)
    render_times: list[float] = []
    pipeline_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}.", dir=output.parent) as raw_temp:
        temporary = Path(raw_temp)
        frames_root = temporary / "frames"
        frames_root.mkdir()
        dummy_rgb = torch.empty(
            (camera_path.height, camera_path.width, 3),
            dtype=torch.float32,
            device=target,
        )
        for index, frame in enumerate(camera_path.frames):
            batch = TrainingBatch(
                observation_id=f"asset_camera_path_{index:06d}",
                camera_id="asset_camera_path",
                frame_id=index,
                role="free_view",
                timestamp=torch.tensor(
                    frame.timestamp_seconds,
                    dtype=torch.float32,
                    device=target,
                ),
                rgb=dummy_rgb,
                intrinsic=torch.tensor(
                    frame.intrinsic,
                    dtype=torch.float32,
                    device=target,
                )[None].contiguous(),
                world_to_camera=torch.tensor(
                    frame.world_to_camera,
                    dtype=torch.float32,
                    device=target,
                )[None].contiguous(),
                radial_coeffs=None,
                tangential_coeffs=None,
            )
            _sync(target)
            started = time.perf_counter()
            rendered = renderer.render(model, batch, sh_degree=sh_degree)
            _sync(target)
            render_times.append((time.perf_counter() - started) * 1_000.0)
            if not bool(torch.isfinite(rendered.image).all()):
                raise ContractError(f"asset video frame {index} contains non-finite pixels")
            write_render(
                frames_root / f"frame_{index:06d}.png",
                rendered.image,
                photometric_space=photometric_space,
            )
        frame_pipeline_ms = (time.perf_counter() - pipeline_started) * 1_000.0

        temporary_video = temporary / output.name
        encode_started = time.perf_counter()
        _encode_video(
            ffmpeg=ffmpeg,
            frames_root=frames_root,
            destination=temporary_video,
            fps=camera_path.fps,
            frame_count=len(camera_path.frames),
            crf=crf,
        )
        encode_ms = (time.perf_counter() - encode_started) * 1_000.0
        _flush_file(temporary_video)
        encoded_width = camera_path.width + camera_path.width % 2
        encoded_height = camera_path.height + camera_path.height % 2
        video_probe = _probe_video(
            ffprobe,
            temporary_video,
            expected_width=encoded_width,
            expected_height=encoded_height,
            expected_fps=camera_path.fps,
            expected_frames=len(camera_path.frames),
        )
        video_bytes = temporary_video.stat().st_size
        video_sha256 = sha256_file(temporary_video)

        sorted_times = sorted(render_times)
        p95_index = min(len(sorted_times) - 1, math.ceil(0.95 * len(sorted_times)) - 1)
        rights = cast(dict[str, Any], bundle.metadata["rights"])
        result: dict[str, Any] = {
            "schema_version": ASSET_VIDEO_RECEIPT_SCHEMA,
            "status": "PASS",
            "asset_bundle_id": bundle.manifest["bundle_id"],
            "model_sha256": bundle.metadata["model"]["sha256"],
            "camera_path_sha256": camera_path.source_sha256,
            "frame_count": len(camera_path.frames),
            "fps": camera_path.fps,
            "duration_seconds": len(camera_path.frames) / camera_path.fps,
            "render_resolution": [camera_path.width, camera_path.height],
            "encoded_resolution": [encoded_width, encoded_height],
            "sh_degree": sh_degree,
            "phase_ms": {
                "asset_load": asset_load_ms,
                "model_to_device": model_to_device_ms,
                "frame_pipeline": frame_pipeline_ms,
                "encode": encode_ms,
            },
            "render_ms": {
                "mean": statistics.fmean(render_times),
                "median": statistics.median(render_times),
                "p95": sorted_times[p95_index],
            },
            "runtime": runtime,
            "consumer": {
                "name": "pixel4dgs",
                "version": __version__,
                "renderer_abi": ASSET_VIDEO_RENDERER_ABI,
                "implementation_sha256": _implementation_hashes(),
            },
            "encoder": {"ffmpeg_version": ffmpeg_version, **video_probe},
            "rights": {"redistribution": rights["redistribution"]},
            "video": {
                "file": output.name,
                "bytes": video_bytes,
                "sha256": video_sha256,
            },
        }
        if target.type == "cuda":
            result["peak_memory_bytes"] = {
                "allocated": torch.cuda.max_memory_allocated(target),
                "reserved": torch.cuda.max_memory_reserved(target),
            }
        validate_payload("asset_video_render", result)

        try:
            os.link(temporary_video, output)
        except FileExistsError as exc:
            raise OutputExistsError(f"refusing to overwrite asset video: {output}") from exc
        _flush_directory(output.parent)
        try:
            write_new_json(receipt, result)
        except Exception:
            if output.is_file() and os.path.samefile(output, temporary_video):
                output.unlink()
                _flush_directory(output.parent)
            raise
    return result
