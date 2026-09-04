# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false

"""Audited observation loading for the public Pixel4DGS training path.

The loader has two interchangeable pixel sources: ordinary RGB image files and
a project-owned NumPy tensor cache.  Both are bound to the same v2 observation
manifest.  The cache changes I/O only; camera geometry, timestamps, roles, and
photometric decoding continue to come from the public manifest contract.

Routine callers can load only ``train`` and ``diagnostic`` observations.
Accessing ``sealed`` or ``free_view`` records requires an explicit access mode,
which prevents the ordinary training/evaluation loop from consuming a sealed
view by accidentally passing the wrong index.
"""

from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, cast

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image
from torch import Tensor

from p2g.audit import AuditReport, audit_observation_manifest
from p2g.canonical import sha256_file
from p2g.errors import ContractError
from p2g.schema import validate_payload
from p2g.training.config import DataConfig, TensorMemmapConfig

CACHE_MANIFEST_NAME = "tensor_cache.json"
CACHE_SCHEMA = "p2g.tensor_cache.v1"

BatchAccess = Literal["routine", "sealed", "free_view"]
SamplingPolicy = Literal["shuffled_epoch", "frame_camera_with_replacement"]


@dataclass(frozen=True, slots=True)
class SceneObservation:
    """One immutable, audited camera/image observation."""

    observation_id: str
    camera_id: str
    frame_id: int
    timestamp_seconds: float
    role: str
    image_path: Path
    image_sha256: str
    image_width: int
    image_height: int
    image_color_space: str
    decode_profile: str
    intrinsic: tuple[tuple[float, float, float], ...]
    world_to_camera: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """One observation in the exact tensor layout consumed by the renderer."""

    observation_id: str
    camera_id: str
    frame_id: int
    role: str
    timestamp: Tensor
    rgb: Tensor
    intrinsic: Tensor
    world_to_camera: Tensor
    radial_coeffs: Tensor | None = None
    tangential_coeffs: Tensor | None = None

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def distorted(self) -> bool:
        return self.radial_coeffs is not None or self.tangential_coeffs is not None

    def to(self, device: str | torch.device, *, non_blocking: bool = False) -> Self:
        """Return a batch whose tensors live on ``device``."""

        def move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(device, non_blocking=non_blocking)

        return replace(
            self,
            timestamp=cast(Tensor, move(self.timestamp)),
            rgb=cast(Tensor, move(self.rgb)),
            intrinsic=cast(Tensor, move(self.intrinsic)),
            world_to_camera=cast(Tensor, move(self.world_to_camera)),
            radial_coeffs=move(self.radial_coeffs),
            tangential_coeffs=move(self.tangential_coeffs),
        )


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object: {path}")
    return cast(dict[str, Any], value)


def _finite_matrix(
    value: Any,
    *,
    rows: int,
    columns: int,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise ContractError(f"{name} must have shape [{rows},{columns}]")
    output: list[tuple[float, ...]] = []
    for raw_row in cast(list[Any], value):
        if not isinstance(raw_row, list) or len(raw_row) != columns:
            raise ContractError(f"{name} must have shape [{rows},{columns}]")
        row = cast(list[Any], raw_row)
        if any(
            not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item)
            for item in row
        ):
            raise ContractError(f"{name} must contain only finite numbers")
        output.append(tuple(float(item) for item in row))
    return tuple(output)


def _safe_member(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ContractError(f"{label} must be a non-empty POSIX relative path")
    member = Path(relative)
    if member.is_absolute() or ".." in member.parts:
        raise ContractError(f"{label} escapes its declared root: {relative!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / member).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ContractError(f"{label} escapes its declared root: {relative!r}")
    return resolved


def _srgb_to_linear(rgb: Tensor) -> Tensor:
    return torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
    )


def _linear_to_srgb(rgb: Tensor) -> Tensor:
    return torch.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * torch.pow(rgb.clamp_min(0.0), 1.0 / 2.4) - 0.055,
    )


def _decode_rgb8(path: Path, observation: SceneObservation) -> np.ndarray[Any, np.dtype[np.uint8]]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != observation.image_sha256:
        raise ContractError(f"observation image changed after audit: {observation.observation_id}")
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                raise ContractError(
                    f"observation {observation.observation_id} must decode as RGB8, "
                    f"got Pillow mode {image.mode!r}"
                )
            array = np.array(image, dtype=np.uint8, copy=True)
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"cannot decode observation image {path}: {exc}") from exc
    if array.shape != (observation.image_height, observation.image_width, 3):
        raise ContractError(
            f"decoded image dimensions changed after audit: {observation.observation_id}"
        )
    return array


class _TensorCacheStore:
    """Read a hash-bound C-contiguous NumPy cache without dataset-specific adapters."""

    _DTYPES: Mapping[str, np.dtype[Any]] = {
        "rgb": np.dtype("uint8"),
        "intrinsic": np.dtype("float32"),
        "world_to_camera": np.dtype("float32"),
        "timestamp_seconds": np.dtype("float64"),
    }

    def __init__(
        self,
        config: TensorMemmapConfig,
        observations: Sequence[SceneObservation],
        *,
        observation_manifest: Path,
    ) -> None:
        config.validate()
        root = config.root.resolve()
        if not root.is_dir():
            raise ContractError(f"tensor cache root is not a directory: {root}")
        cache_manifest_path = root / CACHE_MANIFEST_NAME
        metadata = _read_json_object(cache_manifest_path, label="tensor cache manifest")
        validate_payload("tensor_cache", metadata)
        if metadata["schema_version"] != CACHE_SCHEMA:  # schema documents intent here
            raise ContractError(f"unsupported tensor cache schema: {metadata['schema_version']}")

        camera_ids = tuple(cast(list[str], metadata["camera_ids"]))
        frame_ids = tuple(cast(list[int], metadata["frame_ids"]))
        if camera_ids != config.camera_ids or frame_ids != config.frame_ids:
            raise ContractError("tensor cache axes differ from the resolved run configuration")
        if metadata["observation_manifest_sha256"] != sha256_file(observation_manifest):
            raise ContractError("tensor cache is bound to a different observation manifest")

        dimensions = {(item.image_height, item.image_width) for item in observations}
        if len(dimensions) != 1:
            raise ContractError("tensor cache requires one RGB resolution across the scene")
        height, width = next(iter(dimensions))
        axes = (len(frame_ids), len(camera_ids))
        expected_shapes = {
            "rgb": (*axes, height, width, 3),
            "intrinsic": (*axes, 3, 3),
            "world_to_camera": (*axes, 4, 4),
            "timestamp_seconds": axes,
        }
        arrays_metadata = cast(dict[str, Any], metadata["arrays"])
        self._arrays = {
            name: self._open_array(
                root,
                name=name,
                record=cast(dict[str, Any], arrays_metadata[name]),
                expected_shape=shape,
                expected_dtype=self._DTYPES[name],
                verify_sha256=config.verify_transport_sha256,
            )
            for name, shape in expected_shapes.items()
        }
        self._camera_to_index = {value: index for index, value in enumerate(camera_ids)}
        self._frame_to_index = {value: index for index, value in enumerate(frame_ids)}
        self._validate_manifest_coordinates(observations)

    @staticmethod
    def _open_array(
        root: Path,
        *,
        name: str,
        record: dict[str, Any],
        expected_shape: tuple[int, ...],
        expected_dtype: np.dtype[Any],
        verify_sha256: bool,
    ) -> np.ndarray[Any, Any]:
        if tuple(cast(list[int], record["shape"])) != expected_shape:
            raise ContractError(
                f"tensor cache {name} shape {record['shape']} != {list(expected_shape)}"
            )
        if record["dtype"] != expected_dtype.name:
            raise ContractError(
                f"tensor cache {name} dtype {record['dtype']!r} != {expected_dtype.name!r}"
            )
        path = _safe_member(root, record["path"], label=f"tensor cache {name} path")
        if path.suffix != ".npy" or not path.is_file() or path.is_symlink():
            raise ContractError(f"tensor cache {name} must be a regular .npy file")
        if verify_sha256 and sha256_file(path) != record["sha256"]:
            raise ContractError(f"tensor cache {name} SHA-256 mismatch")
        try:
            untyped: Any = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ContractError(f"cannot memory-map tensor cache {name}: {exc}") from exc
        if not isinstance(untyped, np.ndarray):
            raise ContractError(f"tensor cache {name} is not an ndarray")
        array = cast(np.ndarray[Any, Any], untyped)
        if array.shape != expected_shape or array.dtype != expected_dtype:
            raise ContractError(f"tensor cache {name} header differs from its manifest")
        if not array.flags.c_contiguous:
            raise ContractError(f"tensor cache {name} must use C order")
        return array

    def _coordinates(self, observation: SceneObservation) -> tuple[int, int]:
        try:
            return (
                self._frame_to_index[observation.frame_id],
                self._camera_to_index[observation.camera_id],
            )
        except KeyError as exc:
            raise ContractError(
                "observation lies outside the tensor cache axes: "
                f"{observation.camera_id}:{observation.frame_id}"
            ) from exc

    def _validate_manifest_coordinates(self, observations: Sequence[SceneObservation]) -> None:
        occupied: set[tuple[int, int]] = set()
        for observation in observations:
            coordinates = self._coordinates(observation)
            if coordinates in occupied:
                raise ContractError("tensor cache received a duplicate camera/frame observation")
            occupied.add(coordinates)
            expected_intrinsic = np.asarray(observation.intrinsic, dtype=np.float32)
            expected_world_to_camera = np.asarray(observation.world_to_camera, dtype=np.float32)
            if not np.array_equal(self._arrays["intrinsic"][coordinates], expected_intrinsic):
                raise ContractError(
                    f"tensor cache intrinsic differs for {observation.observation_id}"
                )
            if not np.array_equal(
                self._arrays["world_to_camera"][coordinates], expected_world_to_camera
            ):
                raise ContractError(
                    f"tensor cache world_to_camera differs for {observation.observation_id}"
                )
            if float(self._arrays["timestamp_seconds"][coordinates]) != float(
                observation.timestamp_seconds
            ):
                raise ContractError(
                    f"tensor cache timestamp differs for {observation.observation_id}"
                )
        expected = len(self._frame_to_index) * len(self._camera_to_index)
        if len(occupied) != expected:
            raise ContractError("tensor cache axes are not fully covered by scene observations")

    def load(self, observation: SceneObservation) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        coordinates = self._coordinates(observation)
        rgb = torch.from_numpy(np.array(self._arrays["rgb"][coordinates], copy=True))
        intrinsic = torch.from_numpy(np.array(self._arrays["intrinsic"][coordinates], copy=True))
        world_to_camera = torch.from_numpy(
            np.array(self._arrays["world_to_camera"][coordinates], copy=True)
        )
        timestamp = torch.tensor(
            float(self._arrays["timestamp_seconds"][coordinates]), dtype=torch.float32
        )
        return rgb, intrinsic, world_to_camera, timestamp


class PreparedScene:
    """An audited, role-partitioned scene ready for single-view training steps."""

    def __init__(
        self,
        *,
        dataset_id: str,
        observations: Sequence[SceneObservation],
        train_indices: Sequence[int],
        diagnostic_indices: Sequence[int],
        sealed_indices: Sequence[int],
        free_view_indices: Sequence[int],
        excluded_indices: Sequence[int],
        downscale: int,
        photometric_space: str,
        image_cache_size: int,
        audit_report: AuditReport,
        tensor_cache: _TensorCacheStore | None,
    ) -> None:
        self.dataset_id = dataset_id
        self.observations = tuple(observations)
        self.train_indices = tuple(train_indices)
        # Compatibility name intentionally means diagnostic-only evaluation.
        self.eval_indices = tuple(diagnostic_indices)
        self.diagnostic_indices = tuple(diagnostic_indices)
        self.sealed_indices = tuple(sealed_indices)
        self.free_view_indices = tuple(free_view_indices)
        self.excluded_indices = tuple(excluded_indices)
        self.downscale = downscale
        self.photometric_space = photometric_space
        self.image_cache_size = image_cache_size
        self.audit_report = audit_report
        self._tensor_cache = tensor_cache
        self._batch_cache: OrderedDict[int, TrainingBatch] = OrderedDict()
        self._validate_partitions()

    def _validate_partitions(self) -> None:
        partitions = {
            "train": self.train_indices,
            "diagnostic": self.diagnostic_indices,
            "sealed": self.sealed_indices,
            "free_view": self.free_view_indices,
        }
        if not self.observations or not self.train_indices or not self.diagnostic_indices:
            raise ContractError("PreparedScene requires non-empty train and diagnostic partitions")
        flattened = [
            index for indices in (*partitions.values(), self.excluded_indices) for index in indices
        ]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(
            range(len(self.observations))
        ):
            raise ContractError("PreparedScene role partitions must cover each observation once")
        for role, indices in partitions.items():
            if any(self.observations[index].role != role for index in indices):
                raise ContractError(f"PreparedScene {role} partition contains another role")

    @classmethod
    def load(cls, config: DataConfig) -> Self:
        """Audit and open a resolved data configuration."""

        config.validate()
        manifest_path = config.manifest
        manifest = _read_json_object(manifest_path, label="observation manifest")
        image_root = config.image_root or manifest_path.parent
        report = audit_observation_manifest(
            manifest,
            base_dir=image_root,
            verify_files=True,
        )
        if report.status != "PASS":
            failures = sorted(
                check.name for check in report.checks if check.required and check.status == "FAIL"
            )
            raise ContractError(
                "observation manifest semantic audit failed: " + ", ".join(failures)
            )

        conventions = cast(dict[str, Any], manifest["coordinate_conventions"])
        observations = tuple(
            cls._observation_from_payload(raw, image_root=image_root)
            for raw in cast(list[dict[str, Any]], manifest["observations"])
        )
        by_role = {
            role: [index for index, item in enumerate(observations) if item.role == role]
            for role in ("train", "diagnostic", "sealed", "free_view")
        }
        train_indices = by_role["train"]
        diagnostic_indices = by_role["diagnostic"]
        if config.max_train_observations is not None:
            train_indices = train_indices[: config.max_train_observations]
        if config.max_eval_observations is not None:
            diagnostic_indices = diagnostic_indices[: config.max_eval_observations]
        # Truncation is selection, so unselected observations remain explicit
        # rather than disappearing from the scene inventory.
        selected_train = set(train_indices)
        selected_diagnostic = set(diagnostic_indices)
        omitted = [
            index
            for index, item in enumerate(observations)
            if (item.role == "train" and index not in selected_train)
            or (item.role == "diagnostic" and index not in selected_diagnostic)
        ]
        tensor_cache = (
            None
            if config.tensor_memmap is None
            else _TensorCacheStore(
                config.tensor_memmap,
                observations,
                observation_manifest=manifest_path,
            )
        )
        return cls(
            dataset_id=cast(str, manifest["dataset_id"]),
            observations=observations,
            train_indices=train_indices,
            diagnostic_indices=diagnostic_indices,
            sealed_indices=by_role["sealed"],
            free_view_indices=by_role["free_view"],
            excluded_indices=omitted,
            downscale=config.downscale,
            photometric_space=cast(str, conventions["photometric_space"]),
            image_cache_size=config.image_cache_size,
            audit_report=report,
            tensor_cache=tensor_cache,
        )

    @staticmethod
    def _observation_from_payload(raw: dict[str, Any], *, image_root: Path) -> SceneObservation:
        image = cast(dict[str, Any], raw["image"])
        encoding = cast(dict[str, Any], image["encoding"])
        camera = cast(dict[str, Any], raw["camera"])
        observation_id = cast(str, raw["observation_id"])
        if (
            encoding["container"] not in {"png", "jpeg"}
            or encoding["channel_order"] != "RGB"
            or encoding["bit_depth"] != 8
            or encoding["stored_range"] != "full"
        ):
            raise ContractError(
                f"observation {observation_id} is outside the public RGB8 decode contract"
            )
        color_space = cast(str, image["color_space"])
        decode_profile = cast(str, encoding["canonical_decode_profile"])
        expected_profiles = {
            "srgb_encoded": {"srgb_eotf_v1", "srgb_reference_assumption_v1"},
            "linear_rgb": {"linear_passthrough_v1"},
        }
        if decode_profile not in expected_profiles.get(color_space, set()):
            raise ContractError(f"observation {observation_id} has no trainable photometric decode")
        if (
            camera["model"] != "pinhole"
            or camera["pixel_domain"] != "undistorted"
            or camera["distortion"] != []
        ):
            raise ContractError(
                f"observation {observation_id} must be offline-undistorted pinhole RGB"
            )
        return SceneObservation(
            observation_id=observation_id,
            camera_id=cast(str, raw["camera_id"]),
            frame_id=cast(int, raw["frame_id"]),
            timestamp_seconds=float(raw["timestamp_seconds"]),
            role=cast(str, raw["role"]),
            image_path=_safe_member(
                image_root,
                image["path"],
                label=f"observation {observation_id} image path",
            ),
            image_sha256=cast(str, image["sha256"]),
            image_width=cast(int, image["width"]),
            image_height=cast(int, image["height"]),
            image_color_space=color_space,
            decode_profile=decode_profile,
            intrinsic=cast(
                tuple[tuple[float, float, float], ...],
                _finite_matrix(camera["intrinsic"], rows=3, columns=3, name="camera.intrinsic"),
            ),
            world_to_camera=cast(
                tuple[tuple[float, float, float, float], ...],
                _finite_matrix(
                    camera["world_to_camera"],
                    rows=4,
                    columns=4,
                    name="camera.world_to_camera",
                ),
            ),
        )

    @staticmethod
    def _check_access(observation: SceneObservation, access: BatchAccess) -> None:
        if access not in {"routine", "sealed", "free_view"}:
            raise ContractError(f"unknown observation access mode: {access!r}")
        expected_roles = {
            "routine": {"train", "diagnostic"},
            "sealed": {"sealed"},
            "free_view": {"free_view"},
        }
        if observation.role not in expected_roles[access]:
            raise ContractError(f"{access} access cannot load a {observation.role} observation")

    def load_batch(self, index: int, *, access: BatchAccess = "routine") -> TrainingBatch:
        """Load one observation while enforcing its role capability."""

        if type(index) is not int or index < 0 or index >= len(self.observations):
            raise ContractError(f"observation index is outside PreparedScene: {index!r}")
        observation = self.observations[index]
        self._check_access(observation, access)
        cached = self._batch_cache.get(index)
        if cached is not None:
            self._batch_cache.move_to_end(index)
            return cached

        if self._tensor_cache is None:
            rgb_array = _decode_rgb8(observation.image_path, observation)
            rgb = torch.from_numpy(rgb_array)
            intrinsic = torch.tensor(observation.intrinsic, dtype=torch.float32)
            world_to_camera = torch.tensor(observation.world_to_camera, dtype=torch.float32)
            timestamp = torch.tensor(observation.timestamp_seconds, dtype=torch.float32)
        else:
            rgb, intrinsic, world_to_camera, timestamp = self._tensor_cache.load(observation)
        if rgb.dtype != torch.uint8 or tuple(rgb.shape) != (
            observation.image_height,
            observation.image_width,
            3,
        ):
            raise ContractError(
                f"pixel tensor differs from the RGB8 manifest: {observation.observation_id}"
            )

        pixels = rgb.to(torch.float32).div_(255.0)
        if self.photometric_space == "linear_rgb":
            if observation.image_color_space == "srgb_encoded":
                pixels = _srgb_to_linear(pixels)
        elif self.photometric_space == "srgb_reference_profile":
            if observation.image_color_space == "linear_rgb":
                pixels = _linear_to_srgb(pixels)
        else:  # protected by the observation schema, retained as a runtime invariant
            raise ContractError(f"unsupported scene photometric space: {self.photometric_space}")

        height = max(1, observation.image_height // self.downscale)
        width = max(1, observation.image_width // self.downscale)
        if (height, width) != (observation.image_height, observation.image_width):
            pixels = (
                torch_functional.interpolate(
                    pixels.permute(2, 0, 1).unsqueeze(0),
                    size=(height, width),
                    mode="area",
                )
                .squeeze(0)
                .permute(1, 2, 0)
            )
            intrinsic = intrinsic.clone()
            intrinsic[0, :] *= width / observation.image_width
            intrinsic[1, :] *= height / observation.image_height

        batch = TrainingBatch(
            observation_id=observation.observation_id,
            camera_id=observation.camera_id,
            frame_id=observation.frame_id,
            role=observation.role,
            timestamp=timestamp,
            rgb=pixels.contiguous(),
            intrinsic=intrinsic.unsqueeze(0).contiguous(),
            world_to_camera=world_to_camera.unsqueeze(0).contiguous(),
        )
        if not all(
            torch.isfinite(value).all()
            for value in (
                batch.timestamp,
                batch.rgb,
                batch.intrinsic,
                batch.world_to_camera,
            )
        ):
            raise ContractError(f"non-finite training batch: {observation.observation_id}")
        if self.image_cache_size:
            self._batch_cache[index] = batch
            self._batch_cache.move_to_end(index)
            while len(self._batch_cache) > self.image_cache_size:
                self._batch_cache.popitem(last=False)
        return batch

    def camera_extent(self) -> float:
        """Return the maximum training-camera radius around their centroid."""

        centers: list[Tensor] = []
        seen: set[str] = set()
        for index in self.train_indices:
            observation = self.observations[index]
            if observation.camera_id in seen:
                continue
            seen.add(observation.camera_id)
            transform = torch.tensor(observation.world_to_camera, dtype=torch.float64)
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            centers.append(-(rotation.transpose(0, 1) @ translation))
        if len(centers) < 2:
            return 1.0
        stacked = torch.stack(centers)
        centroid = stacked.mean(dim=0)
        radius = cast(Tensor, torch.linalg.vector_norm(stacked - centroid, dim=1).max())
        return max(float(radius), 1.0e-6)

    def train_frame_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return deterministic frame-major groups for frame/camera sampling."""

        frames = sorted({self.observations[index].frame_id for index in self.train_indices})
        groups = tuple(
            tuple(
                sorted(
                    (
                        index
                        for index in self.train_indices
                        if self.observations[index].frame_id == frame_id
                    ),
                    key=lambda index: self.observations[index].camera_id,
                )
            )
            for frame_id in frames
        )
        if not groups or any(not group for group in groups):
            raise ContractError("PreparedScene has no training frame groups")
        return groups


def _encode_random_state(state: object) -> dict[str, Any]:
    if not isinstance(state, tuple) or len(state) != 3:
        raise ContractError("Python random state has an unsupported layout")
    version, words, gauss_next = cast(tuple[Any, Any, Any], state)
    if type(version) is not int or not isinstance(words, tuple):
        raise ContractError("Python random state has an unsupported layout")
    word_values = cast(tuple[Any, ...], words)
    if any(type(word) is not int for word in word_values):
        raise ContractError("Python random state words must be integers")
    if gauss_next is not None and (
        not isinstance(gauss_next, (int, float))
        or isinstance(gauss_next, bool)
        or not math.isfinite(gauss_next)
    ):
        raise ContractError("Python random Gaussian cache must be finite")
    return {
        "version": version,
        "words": list(cast(tuple[int, ...], word_values)),
        "gauss_next": gauss_next,
    }


def _decode_random_state(raw: Any) -> tuple[int, tuple[int, ...], float | None]:
    if not isinstance(raw, dict) or set(raw) != {"version", "words", "gauss_next"}:
        raise ContractError("sampler random state has invalid fields")
    value = cast(dict[str, Any], raw)
    version = value["version"]
    words = value["words"]
    gauss_next = value["gauss_next"]
    if type(version) is not int or not isinstance(words, list) or not words:
        raise ContractError("sampler random state has an invalid layout")
    if any(type(word) is not int for word in cast(list[Any], words)):
        raise ContractError("sampler random state words must be integers")
    if gauss_next is not None and (
        not isinstance(gauss_next, (int, float))
        or isinstance(gauss_next, bool)
        or not math.isfinite(gauss_next)
    ):
        raise ContractError("sampler random Gaussian cache must be finite")
    return (
        version,
        tuple(cast(list[int], words)),
        None if gauss_next is None else float(gauss_next),
    )


class SceneSampler:
    """Deterministic sampler with JSON-safe, validated resume state."""

    _STATE_FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "policy",
        "seed",
        "indices",
        "frame_groups",
        "order",
        "cursor",
        "epoch",
        "random_state",
    }

    def __init__(
        self,
        indices: Iterable[int],
        *,
        seed: int,
        policy: SamplingPolicy = "shuffled_epoch",
        frame_groups: Iterable[Iterable[int]] | None = None,
    ) -> None:
        values = tuple(indices)
        if (
            not values
            or any(type(index) is not int or index < 0 for index in values)
            or len(values) != len(set(values))
        ):
            raise ContractError("SceneSampler requires unique, non-negative indices")
        if type(seed) is not int or seed < 0:
            raise ContractError("SceneSampler seed must be a non-negative integer")
        if policy not in {"shuffled_epoch", "frame_camera_with_replacement"}:
            raise ContractError(f"unsupported public sampling policy: {policy!r}")
        groups = None if frame_groups is None else tuple(tuple(group) for group in frame_groups)
        if policy == "shuffled_epoch" and groups is not None:
            raise ContractError("shuffled_epoch does not accept frame groups")
        if policy == "frame_camera_with_replacement":
            flattened = [] if groups is None else [index for group in groups for index in group]
            if (
                not groups
                or any(not group for group in groups)
                or any(type(index) is not int for index in flattened)
                or len(flattened) != len(set(flattened))
                or set(flattened) != set(values)
            ):
                raise ContractError("frame groups must partition every sampler index exactly")

        self.indices = values
        self.seed = seed
        self.policy = policy
        self.frame_groups = groups
        self._random = random.Random(seed)
        self.order = list(values)
        if policy == "shuffled_epoch":
            self._random.shuffle(self.order)
        self.cursor = 0
        self.epoch = 0

    def next_index(self) -> int:
        if self.policy == "frame_camera_with_replacement":
            if self.frame_groups is None:  # pragma: no cover - constructor invariant
                raise AssertionError("replacement sampler has no frame groups")
            group = self.frame_groups[self._random.randrange(len(self.frame_groups))]
            result = group[self._random.randrange(len(group))]
            self.cursor += 1
            if self.cursor == len(self.indices):
                self.cursor = 0
                self.epoch += 1
            return result

        if self.cursor == len(self.order):
            self.epoch += 1
            self.order = list(self.indices)
            self._random.shuffle(self.order)
            self.cursor = 0
        result = self.order[self.cursor]
        self.cursor += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, implementation-identified state."""

        return {
            "schema_version": "p2g.scene_sampler_state.v1",
            "policy": self.policy,
            "seed": self.seed,
            "indices": list(self.indices),
            "frame_groups": (
                None if self.frame_groups is None else [list(group) for group in self.frame_groups]
            ),
            "order": list(self.order),
            "cursor": self.cursor,
            "epoch": self.epoch,
            "random_state": _encode_random_state(self._random.getstate()),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate a complete state before atomically replacing sampler state."""

        if set(state) != self._STATE_FIELDS:
            raise ContractError("checkpoint sampler state has invalid fields")
        if state["schema_version"] != "p2g.scene_sampler_state.v1":
            raise ContractError("checkpoint sampler state has an unsupported schema")
        if (
            state["policy"] != self.policy
            or state["seed"] != self.seed
            or state["indices"] != list(self.indices)
        ):
            raise ContractError("checkpoint sampler does not match this scene and policy")
        raw_groups = state["frame_groups"]
        if raw_groups is None:
            groups = None
        elif isinstance(raw_groups, list):
            raw_group_list = cast(list[Any], raw_groups)
            if not all(isinstance(group, list) for group in raw_group_list):
                raise ContractError("checkpoint sampler frame groups are malformed")
            groups = tuple(tuple(cast(list[int], group)) for group in raw_group_list)
        else:
            raise ContractError("checkpoint sampler frame groups are malformed")
        if groups != self.frame_groups:
            raise ContractError("checkpoint sampler frame groups differ from this scene")

        order = state["order"]
        cursor = state["cursor"]
        epoch = state["epoch"]
        if not isinstance(order, list):
            raise ContractError("checkpoint sampler contains an invalid cursor or order")
        order_values = cast(list[Any], order)
        if (
            any(type(index) is not int for index in order_values)
            or len(order_values) != len(set(cast(list[int], order_values)))
            or set(cast(list[int], order_values)) != set(self.indices)
            or type(cursor) is not int
            or cursor < 0
            or cursor > len(self.indices)
            or type(epoch) is not int
            or epoch < 0
            or (self.policy == "frame_camera_with_replacement" and cursor == len(self.indices))
        ):
            raise ContractError("checkpoint sampler contains an invalid cursor or order")
        random_state = _decode_random_state(state["random_state"])
        candidate_random = random.Random()
        try:
            candidate_random.setstate(random_state)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"checkpoint sampler random state is invalid: {exc}") from exc

        self.order = list(cast(list[int], order_values))
        self.cursor = cursor
        self.epoch = epoch
        self._random = candidate_random


__all__ = [
    "CACHE_MANIFEST_NAME",
    "PreparedScene",
    "SceneObservation",
    "SceneSampler",
    "TrainingBatch",
]
