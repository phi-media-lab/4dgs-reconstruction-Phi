# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

"""Auditable dense two-view point proposals from the public tensor cache.

The geometry and artifact format in this module are owned by Pixel4DGS. RoMa
is an optional, separately licensed correspondence provider loaded only after
its installed source identity, local weights, environment lock, ROCm runtime,
and MI300X device have passed explicit checks. This module never downloads or
redistributes model weights.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.resources
import inspect
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from p2g.audit import audit_observation_manifest
from p2g.canonical import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
    write_new_json,
)
from p2g.errors import ContractError, OutputExistsError
from p2g.evidence.vector_geometry import evaluate_two_view_diagnostics
from p2g.schema import validate_payload

ROMA_POINT_PROVIDER_SCHEMA = "p2g.roma_point_proposals.v1"
PROVENANCE_SCHEMA = "p2g.roma_point_provenance.v1"
PROVENANCE_DIGEST_SCHEMA = "p2g.roma_point_provenance_canonical_digest.v1"
ROMA_PROVIDER_REGISTRY_SCHEMA = "p2g.roma_provider_registry.v1"
ROMA_PROVIDER_REGISTRY_RESOURCE = "registries/roma_provider_v1.json"
TENSOR_CACHE_SCHEMA = "p2g.tensor_cache.v1"
TENSOR_CACHE_MANIFEST = "tensor_cache.json"
OBSERVATION_MANIFEST_SCHEMA = "p2g.observation_manifest.v2"
ROLE_ADMISSION_SCHEMA = "p2g.observation_role_admission.v1"
FRAME_TIMESTAMP_OPERATOR = "arithmetic_mean_of_train_observation_timestamps_v1"

Array = np.ndarray[Any, np.dtype[Any]]


@dataclass(frozen=True, slots=True)
class TensorCacheFrame:
    """One complete same-time, multi-camera frame from tensor-cache v1."""

    frame_id: int
    rgb: Array
    world_to_camera: Array
    intrinsic: Array
    camera_timestamp_seconds: Array
    timestamp_seconds: float
    camera_ids: tuple[str, ...]
    source_receipt: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ObservationAuthority:
    """Audited role authority bound byte-for-byte to a tensor cache."""

    path: Path
    sha256: str
    camera_ids: tuple[str, ...]
    frame_ids: tuple[int, ...]
    observations: dict[tuple[int, str], dict[str, Any]]


class PairSampler(Protocol):
    """Minimal correspondence-provider surface consumed by the geometry path."""

    @property
    def identity(self) -> dict[str, Any]: ...

    def sample_pair(
        self,
        source_rgb: Array,
        target_rgb: Array,
        *,
        count: int,
        seed: int,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class _RomatchBinding:
    factory: Callable[..., object]
    identity: dict[str, Any]


def _json_object_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _sha256_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _https_url(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("https://"):
        raise ContractError(f"{label} must be an HTTPS URL")
    return value


def _validate_registry(registry: dict[str, Any]) -> None:
    _exact_keys(
        registry,
        {"schema_version", "provider", "runtime", "weights", "policy"},
        label="RoMa provider registry",
    )
    if registry["schema_version"] != ROMA_PROVIDER_REGISTRY_SCHEMA:
        raise ContractError("unsupported RoMa provider registry schema")

    provider = registry.get("provider")
    if not isinstance(provider, dict):
        raise ContractError("RoMa registry provider must be an object")
    _exact_keys(
        provider,
        {
            "distribution",
            "distribution_version",
            "factory",
            "license",
            "repository",
            "revision",
        },
        label="RoMa registry provider",
    )
    if provider["distribution"] != "romatch" or provider["distribution_version"] != "0.1.2":
        raise ContractError("RoMa registry has an unsupported distribution identity")
    if provider["factory"] != "romatch.roma_indoor":
        raise ContractError("RoMa registry has an unsupported factory")
    _https_url(provider["repository"], label="RoMa repository")
    revision = provider["revision"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ContractError("RoMa revision must be a full Git commit")
    code_license = provider["license"]
    if not isinstance(code_license, dict):
        raise ContractError("RoMa code license must be an object")
    _exact_keys(
        code_license,
        {"spdx", "copyright", "url", "sha256"},
        label="RoMa code license",
    )
    if code_license["spdx"] != "MIT" or not isinstance(code_license["copyright"], str):
        raise ContractError("RoMa registry must retain the upstream MIT identity")
    _https_url(code_license["url"], label="RoMa license URL")
    _sha256_text(code_license["sha256"], label="RoMa license SHA-256")

    runtime = registry.get("runtime")
    if not isinstance(runtime, dict):
        raise ContractError("RoMa registry runtime must be an object")
    _exact_keys(
        runtime,
        {
            "architecture",
            "hip",
            "operating_system",
            "python",
            "torch",
            "torch_index",
            "torchvision",
            "visible_device_count",
        },
        label="RoMa registry runtime",
    )
    if runtime != {
        "architecture": "gfx942",
        "hip": "7.0.51831",
        "operating_system": "Linux x86_64",
        "python": "3.12",
        "torch": "2.10.0+rocm7.0",
        "torch_index": "https://download.pytorch.org/whl/rocm7.0",
        "torchvision": "0.25.0+rocm7.0",
        "visible_device_count": 1,
    }:
        raise ContractError("RoMa registry runtime differs from the admitted MI300X profile")

    weights = registry.get("weights")
    if not isinstance(weights, dict) or set(weights) != {"roma_indoor", "dinov2_vitl14"}:
        raise ContractError("RoMa registry must declare exactly the two external weights")
    for weight_id, record_value in weights.items():
        if not isinstance(record_value, dict):
            raise ContractError(f"weight record {weight_id} must be an object")
        record = cast(dict[str, Any], record_value)
        _exact_keys(
            record,
            {"filename", "url", "bytes", "sha256", "license", "redistribution"},
            label=f"weight record {weight_id}",
        )
        filename = record["filename"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ContractError(f"weight record {weight_id} has an unsafe filename")
        _https_url(record["url"], label=f"weight URL {weight_id}")
        _positive_integer(record["bytes"], label=f"weight size {weight_id}")
        _sha256_text(record["sha256"], label=f"weight SHA-256 {weight_id}")
        if record["redistribution"] != "external_only_not_bundled":
            raise ContractError(f"weight record {weight_id} permits unsupported bundling")
        license_record = record["license"]
        if not isinstance(license_record, dict):
            raise ContractError(f"weight license {weight_id} must be an object")
        _exact_keys(
            license_record,
            {"spdx", "status", "url"},
            label=f"weight license {weight_id}",
        )
        if license_record["status"] not in {"declared", "review_required"}:
            raise ContractError(f"weight license {weight_id} has an invalid status")
        if not isinstance(license_record["spdx"], str):
            raise ContractError(f"weight license {weight_id} lacks an explicit SPDX value")
        _https_url(license_record["url"], label=f"weight license URL {weight_id}")

    policy = registry.get("policy")
    if policy != {
        "automatic_download": False,
        "bundle_weights": False,
        "require_local_hash_match": True,
        "record_environment_lock": True,
    }:
        raise ContractError("RoMa registry policy must remain fail-closed")


def load_roma_provider_registry() -> tuple[dict[str, Any], str]:
    """Load and validate the package-owned provider/weight registry."""

    try:
        resource = importlib.resources.files("p2g").joinpath(ROMA_PROVIDER_REGISTRY_RESOURCE)
        data = resource.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise ContractError(f"RoMa provider registry is unavailable: {exc}") from exc
    registry = _json_object_bytes(data, label="RoMa provider registry")
    _validate_registry(registry)
    return registry, sha256_bytes(data)


def _read_json_file(path: Path, *, label: str) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"{label} is unavailable: {path}: {exc}") from exc
    return _json_object_bytes(data, label=label)


def _safe_member(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ContractError(f"{label} must be a non-empty POSIX relative path")
    member = Path(relative)
    if member.is_absolute() or ".." in member.parts:
        raise ContractError(f"{label} escapes the tensor-cache root")
    resolved = (root / member).resolve()
    if not resolved.is_relative_to(root):
        raise ContractError(f"{label} escapes the tensor-cache root")
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError(f"{label} must resolve to a regular non-symlink file")
    return resolved


def _open_cache_array(
    root: Path,
    metadata: dict[str, Any],
    *,
    name: str,
    expected_dtype: np.dtype[Any],
) -> Array:
    record_value = metadata["arrays"].get(name)
    if not isinstance(record_value, dict):
        raise ContractError(f"tensor cache lacks the {name} array record")
    record = cast(dict[str, Any], record_value)
    path = _safe_member(root, record["path"], label=f"tensor cache {name}")
    if path.suffix != ".npy":
        raise ContractError(f"tensor cache {name} must use a .npy container")
    if sha256_file(path) != record["sha256"]:
        raise ContractError(f"tensor cache {name} SHA-256 mismatch")
    try:
        value: Any = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot memory-map tensor cache {name}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise ContractError(f"tensor cache {name} is not an ndarray")
    array = cast(Array, value)
    if (
        list(array.shape) != record["shape"]
        or array.dtype != expected_dtype
        or record["dtype"] != expected_dtype.name
        or record["order"] != "C"
        or not array.flags.c_contiguous
    ):
        raise ContractError(f"tensor cache {name} header differs from its manifest")
    return array


def _sha256_array(value: Array) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def load_observation_authority(path: Path) -> ObservationAuthority:
    """Validate the source manifest that remains authoritative for roles."""

    supplied = path.expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise ContractError("observation manifest must be a regular non-symlink file")
    resolved = supplied.resolve()
    manifest = _read_json_file(resolved, label="observation manifest")
    validate_payload("observation", manifest)
    if manifest.get("schema_version") != OBSERVATION_MANIFEST_SCHEMA:
        raise ContractError("unsupported observation manifest schema")
    report = audit_observation_manifest(
        manifest,
        base_dir=resolved.parent,
        verify_files=False,
    )
    if report.status != "PASS":
        failures = sorted(
            check.name for check in report.checks if check.required and check.status == "FAIL"
        )
        raise ContractError(
            "observation manifest semantic audit failed: " + ", ".join(failures)
        )

    observations: dict[tuple[int, str], dict[str, Any]] = {}
    for item in cast(list[dict[str, Any]], manifest["observations"]):
        coordinate = (cast(int, item["frame_id"]), cast(str, item["camera_id"]))
        if coordinate in observations:
            raise ContractError("observation manifest contains a duplicate camera/frame")
        observations[coordinate] = item
    camera_ids = tuple(sorted({camera_id for _, camera_id in observations}))
    frame_ids = tuple(sorted({frame_id for frame_id, _ in observations}))
    expected = {(frame_id, camera_id) for frame_id in frame_ids for camera_id in camera_ids}
    if set(observations) != expected:
        raise ContractError("observation manifest does not cover the complete cache grid")
    return ObservationAuthority(
        path=resolved,
        sha256=sha256_file(resolved),
        camera_ids=camera_ids,
        frame_ids=frame_ids,
        observations=observations,
    )


def build_train_role_admission(
    authority: ObservationAuthority,
    *,
    frame_id: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    if frame_id not in authority.frame_ids:
        raise ContractError(f"frame {frame_id} is outside the observation manifest")
    admitted_indices: list[int] = []
    admitted_camera_ids: list[str] = []
    admitted_observation_ids: list[str] = []
    excluded: dict[str, list[str]] = {
        "diagnostic": [],
        "sealed": [],
        "free_view": [],
    }
    for camera_index, camera_id in enumerate(authority.camera_ids):
        observation = authority.observations[(frame_id, camera_id)]
        role = cast(str, observation["role"])
        if role == "train":
            admitted_indices.append(camera_index)
            admitted_camera_ids.append(camera_id)
            admitted_observation_ids.append(cast(str, observation["observation_id"]))
        else:
            excluded[role].append(camera_id)
    if not admitted_indices:
        raise ContractError(f"frame {frame_id} has no train-role observations")
    unsigned = {
        "schema": ROLE_ADMISSION_SCHEMA,
        "role": "train",
        "observation_manifest_sha256": authority.sha256,
        "frame_id": frame_id,
        "frame_timestamp_operator": FRAME_TIMESTAMP_OPERATOR,
        "cache_camera_count": len(authority.camera_ids),
        "admitted_camera_ids": admitted_camera_ids,
        "admitted_observation_ids": admitted_observation_ids,
        "excluded_camera_ids_by_role": excluded,
    }
    return tuple(admitted_indices), {**unsigned, "logical_sha256": sha256_json(unsigned)}


def load_tensor_cache_frame(cache_root: Path, *, frame_id: int) -> TensorCacheFrame:
    """Load and validate one frame from the public hash-bound tensor cache."""

    root = cache_root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ContractError(f"tensor cache root must be a regular directory: {root}")
    manifest_path = root / TENSOR_CACHE_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ContractError(f"tensor cache manifest is unavailable: {manifest_path}")
    metadata = _read_json_file(manifest_path, label="tensor cache manifest")
    validate_payload("tensor_cache", metadata)
    if metadata["schema_version"] != TENSOR_CACHE_SCHEMA:
        raise ContractError("unsupported tensor cache schema")

    camera_ids = tuple(cast(list[str], metadata["camera_ids"]))
    frame_ids = tuple(cast(list[int], metadata["frame_ids"]))
    if tuple(sorted(frame_ids)) != frame_ids:
        raise ContractError("tensor cache frame IDs must be ascending")
    try:
        frame_index = frame_ids.index(frame_id)
    except ValueError as exc:
        raise ContractError(f"frame {frame_id} is outside the tensor cache") from exc

    arrays = {
        "rgb": _open_cache_array(root, metadata, name="rgb", expected_dtype=np.dtype("uint8")),
        "intrinsic": _open_cache_array(
            root, metadata, name="intrinsic", expected_dtype=np.dtype("float32")
        ),
        "world_to_camera": _open_cache_array(
            root, metadata, name="world_to_camera", expected_dtype=np.dtype("float32")
        ),
        "timestamp_seconds": _open_cache_array(
            root, metadata, name="timestamp_seconds", expected_dtype=np.dtype("float64")
        ),
    }
    frame_count = len(frame_ids)
    camera_count = len(camera_ids)
    rgb_all = arrays["rgb"]
    if (
        rgb_all.ndim != 5
        or rgb_all.shape[:2] != (frame_count, camera_count)
        or rgb_all.shape[-1] != 3
        or min(rgb_all.shape[2:4]) <= 0
        or arrays["intrinsic"].shape != (frame_count, camera_count, 3, 3)
        or arrays["world_to_camera"].shape != (frame_count, camera_count, 4, 4)
        or arrays["timestamp_seconds"].shape != (frame_count, camera_count)
    ):
        raise ContractError("tensor cache arrays do not share the declared frame/camera axes")

    rgb = np.ascontiguousarray(rgb_all[frame_index])
    intrinsic = np.ascontiguousarray(arrays["intrinsic"][frame_index], dtype=np.float64)
    world_to_camera = np.ascontiguousarray(arrays["world_to_camera"][frame_index], dtype=np.float64)
    timestamps = np.ascontiguousarray(arrays["timestamp_seconds"][frame_index], dtype=np.float64)
    if (
        not np.isfinite(intrinsic).all()
        or not np.isfinite(world_to_camera).all()
        or not np.isfinite(timestamps).all()
        or not np.all(intrinsic[:, 0, 0] > 0.0)
        or not np.all(intrinsic[:, 1, 1] > 0.0)
    ):
        raise ContractError("tensor-cache frame contains invalid calibration or timestamps")
    if not np.allclose(
        intrinsic[:, 2, :],
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ContractError("tensor-cache intrinsics have invalid homogeneous rows")
    if not np.allclose(
        world_to_camera[:, 3, :],
        np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ContractError("tensor-cache world-to-camera matrices have invalid rows")
    rotations = world_to_camera[:, :3, :3]
    gram = rotations @ np.transpose(rotations, (0, 2, 1))
    determinants = np.linalg.det(rotations)
    if not np.allclose(gram, np.eye(3), rtol=1.0e-5, atol=1.0e-5) or not np.allclose(
        determinants, 1.0, rtol=1.0e-5, atol=1.0e-5
    ):
        raise ContractError("tensor-cache world-to-camera rotations are not proper rotations")

    array_receipts = {
        name: {
            "path": cast(dict[str, Any], metadata["arrays"][name])["path"],
            "sha256": cast(dict[str, Any], metadata["arrays"][name])["sha256"],
        }
        for name in sorted(arrays)
    }
    return TensorCacheFrame(
        frame_id=frame_id,
        rgb=rgb,
        world_to_camera=world_to_camera,
        intrinsic=intrinsic,
        camera_timestamp_seconds=timestamps,
        timestamp_seconds=float(np.mean(timestamps)),
        camera_ids=camera_ids,
        source_receipt={
            "schema": TENSOR_CACHE_SCHEMA,
            "manifest": {
                "path": TENSOR_CACHE_MANIFEST,
                "sha256": sha256_file(manifest_path),
                "observation_manifest_sha256": metadata["observation_manifest_sha256"],
            },
            "arrays": array_receipts,
            "frame_count": frame_count,
            "frame_ids": list(frame_ids),
            "camera_count": camera_count,
            "image_shape": list(rgb.shape[1:]),
            "frame_payload_sha256": {
                "rgb": _sha256_array(rgb),
                "intrinsic_float64": _sha256_array(intrinsic),
                "world_to_camera_float64": _sha256_array(world_to_camera),
                "timestamp_float64": _sha256_array(timestamps),
            },
        },
    )


def load_train_tensor_cache_frame(
    cache_root: Path,
    *,
    observation_manifest: Path,
    frame_id: int,
) -> TensorCacheFrame:
    """Load one frame after admitting only manifest-authorized train cameras."""

    authority = load_observation_authority(observation_manifest)
    frame = load_tensor_cache_frame(cache_root, frame_id=frame_id)
    source_manifest = cast(dict[str, Any], frame.source_receipt["manifest"])
    source_frame_ids = tuple(cast(list[int], frame.source_receipt["frame_ids"]))
    if source_manifest.get("observation_manifest_sha256") != authority.sha256:
        raise ContractError("tensor cache is bound to a different observation manifest")
    if frame.camera_ids != authority.camera_ids or source_frame_ids != authority.frame_ids:
        raise ContractError("tensor cache axes differ from the observation manifest")

    height, width = cast(tuple[int, int], frame.rgb.shape[1:3])
    for camera_index, camera_id in enumerate(authority.camera_ids):
        observation = authority.observations[(frame_id, camera_id)]
        image = cast(dict[str, Any], observation["image"])
        camera = cast(dict[str, Any], observation["camera"])
        expected_intrinsic = np.asarray(camera["intrinsic"], dtype=np.float32).astype(
            np.float64
        )
        expected_world_to_camera = np.asarray(
            camera["world_to_camera"], dtype=np.float32
        ).astype(np.float64)
        if image.get("height") != height or image.get("width") != width:
            raise ContractError(
                f"tensor cache image dimensions differ for {observation['observation_id']}"
            )
        if not np.array_equal(frame.intrinsic[camera_index], expected_intrinsic):
            raise ContractError(
                f"tensor cache intrinsic differs for {observation['observation_id']}"
            )
        if not np.array_equal(
            frame.world_to_camera[camera_index], expected_world_to_camera
        ):
            raise ContractError(
                "tensor cache world_to_camera differs for "
                f"{observation['observation_id']}"
            )
        if frame.camera_timestamp_seconds[camera_index] != float(
            observation["timestamp_seconds"]
        ):
            raise ContractError(
                f"tensor cache timestamp differs for {observation['observation_id']}"
            )

    admitted_indices, role_admission = build_train_role_admission(
        authority,
        frame_id=frame_id,
    )
    index = np.asarray(admitted_indices, dtype=np.intp)
    rgb = np.ascontiguousarray(frame.rgb[index])
    intrinsic = np.ascontiguousarray(frame.intrinsic[index])
    world_to_camera = np.ascontiguousarray(frame.world_to_camera[index])
    timestamps = np.ascontiguousarray(frame.camera_timestamp_seconds[index])
    timestamp_seconds = float(np.mean(timestamps))
    source_receipt = dict(frame.source_receipt)
    source_receipt["cache_camera_count"] = source_receipt["camera_count"]
    source_receipt["cache_frame_payload_sha256"] = source_receipt[
        "frame_payload_sha256"
    ]
    source_receipt["camera_count"] = len(admitted_indices)
    source_receipt["frame_payload_sha256"] = {
        "rgb": _sha256_array(rgb),
        "intrinsic_float64": _sha256_array(intrinsic),
        "world_to_camera_float64": _sha256_array(world_to_camera),
        "timestamp_float64": _sha256_array(timestamps),
    }
    source_receipt["role_admission"] = role_admission
    return TensorCacheFrame(
        frame_id=frame.frame_id,
        rgb=rgb,
        world_to_camera=world_to_camera,
        intrinsic=intrinsic,
        camera_timestamp_seconds=timestamps,
        timestamp_seconds=timestamp_seconds,
        camera_ids=tuple(
            cast(list[str], role_admission["admitted_camera_ids"])
        ),
        source_receipt=source_receipt,
    )


def canonical_provenance_sha256(
    planes: Mapping[str, Any],
    *,
    frame_id: int,
) -> str:
    """Hash provenance semantics independently of Safetensors map ordering."""

    normalized: dict[str, Array] = {}
    inventory: list[dict[str, Any]] = []
    for name in sorted(planes):
        raw = planes[name]
        if isinstance(raw, np.ndarray):
            array = raw
        else:
            device = getattr(raw, "device", None)
            if getattr(device, "type", None) != "cpu":
                raise ContractError("canonical provenance digest requires CPU arrays")
            try:
                array = np.asarray(raw.detach().numpy())
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                raise ContractError(f"provenance plane is not array-like: {name}") from exc
        if array.dtype.hasobject:
            raise ContractError(f"provenance plane has an object dtype: {name}")
        dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
        normalized[name] = canonical
        inventory.append(
            {
                "name": name,
                "dtype": canonical.dtype.str,
                "shape": list(canonical.shape),
                "nbytes": canonical.nbytes,
            }
        )
    header = {
        "schema": PROVENANCE_DIGEST_SCHEMA,
        "provenance_schema": PROVENANCE_SCHEMA,
        "frame_id": frame_id,
        "row_semantics": "directed_pair_then_dense_source_linear_order",
        "planes": inventory,
    }
    digest = hashlib.sha256(canonical_json_bytes(header))
    for name in sorted(normalized):
        digest.update(memoryview(normalized[name]).cast("B"))
    return digest.hexdigest()


def camera_centers(world_to_camera: Array) -> Array:
    """Return world-space camera centers from rigid world-to-camera matrices."""

    matrices = np.asarray(world_to_camera, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ContractError("camera centers require world-to-camera matrices with shape (C,4,4)")
    if not np.isfinite(matrices).all():
        raise ContractError("camera matrices contain non-finite values")
    rotations = matrices[:, :3, :3]
    translations = matrices[:, :3, 3]
    centers = -np.einsum("cij,cj->ci", np.transpose(rotations, (0, 2, 1)), translations)
    if not np.isfinite(centers).all():
        raise ContractError("computed camera centers are non-finite")
    return np.ascontiguousarray(centers)


def nearest_camera_graph(world_to_camera: Array, *, neighbors: int = 2) -> Array:
    """Build a deterministic directed graph by Euclidean camera-center distance."""

    centers = camera_centers(world_to_camera)
    camera_count = len(centers)
    if neighbors <= 0 or neighbors >= camera_count:
        raise ContractError("nearest-camera degree must be in [1, camera_count)")
    squared_distance = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    result = np.empty((camera_count, neighbors), dtype=np.int32)
    camera_ids = np.arange(camera_count)
    for source_camera in range(camera_count):
        ordered = np.lexsort((camera_ids, squared_distance[source_camera]))
        result[source_camera] = ordered[ordered != source_camera][:neighbors]
    return result


def pair_seed(
    global_seed: int,
    *,
    frame_id: int,
    source_camera: int,
    target_camera: int,
    neighbor_rank: int,
) -> int:
    """Derive one stable 64-bit seed without depending on process hash state."""

    values = (global_seed, frame_id, source_camera, target_camera, neighbor_rank)
    if any(type(value) is not int for value in values):
        raise ContractError("pair-seed coordinates must be integers")
    payload = {
        "schema": "p2g.roma_point_pair_seed.v1",
        "global_seed": global_seed,
        "frame_id": frame_id,
        "source_camera": source_camera,
        "target_camera": target_camera,
        "neighbor_rank": neighbor_rank,
    }
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:8], "big")


def normalized_matches_to_pixels(
    matches: Array,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> tuple[Array, Array]:
    """Apply RoMa's public normalized-to-pixel coordinate convention."""

    values = np.asarray(matches, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 4
        or not np.isfinite(values).all()
        or min(source_height, source_width, target_height, target_width) <= 0
    ):
        raise ContractError("normalized matches must be finite with shape (N,4)")
    source = np.column_stack(
        (
            source_width * (values[:, 0] + 1.0) * 0.5,
            source_height * (values[:, 1] + 1.0) * 0.5,
        )
    )
    target = np.column_stack(
        (
            target_width * (values[:, 2] + 1.0) * 0.5,
            target_height * (values[:, 3] + 1.0) * 0.5,
        )
    )
    return source, target


def recover_dense_source_xy(
    source_normalized: Array,
    *,
    dense_height: int,
    dense_width: int,
) -> tuple[Array, Array]:
    """Map normalized source coordinates back to align-corners-false grid cells."""

    normalized = np.asarray(source_normalized, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != 2 or min(dense_height, dense_width) <= 0:
        raise ContractError("dense-source recovery requires shape (N,2) and a non-empty grid")
    finite = np.isfinite(normalized).all(axis=1)
    safe = np.where(np.isfinite(normalized), normalized, 0.0)
    x = np.rint(((safe[:, 0] + 1.0) * dense_width - 1.0) * 0.5).astype(np.int64)
    y = np.rint(((safe[:, 1] + 1.0) * dense_height - 1.0) * 0.5).astype(np.int64)
    valid = finite & (x >= 0) & (x < dense_width) & (y >= 0) & (y < dense_height)
    return np.ascontiguousarray(np.column_stack((x, y))), np.ascontiguousarray(valid)


def _camera_record(intrinsic: Array, world_to_camera: Array) -> dict[str, Any]:
    return {
        "model": "pinhole",
        "pixel_domain": "undistorted",
        "intrinsic": intrinsic.tolist(),
        "world_to_camera": world_to_camera.tolist(),
        "distortion": [],
    }


def _inside_image(points: Array, *, height: int, width: int) -> Array:
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < height)
    )


def _bilinear_rgb8(image: Array, points: Array) -> Array:
    """Sample RGB8 at explicit pixel-center coordinates with edge clamping."""

    height, width = image.shape[:2]
    x = np.clip(points[:, 0], 0.0, width - 1.0)
    y = np.clip(points[:, 1], 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fraction_x = (x - x0)[:, None]
    fraction_y = (y - y0)[:, None]
    upper = image[y0, x0].astype(np.float64) * (1.0 - fraction_x)
    upper += image[y0, x1].astype(np.float64) * fraction_x
    lower = image[y1, x0].astype(np.float64) * (1.0 - fraction_x)
    lower += image[y1, x1].astype(np.float64) * fraction_x
    sampled = upper * (1.0 - fraction_y) + lower * fraction_y
    return np.ascontiguousarray(np.rint(np.clip(sampled, 0.0, 255.0)).astype(np.uint8))


def assemble_pair_provenance(
    frame: TensorCacheFrame,
    *,
    source_camera: int,
    target_camera: int,
    neighbor_rank: int,
    pair_ordinal: int,
    sampled: dict[str, Any],
    world_bound: float = 1_000.0,
) -> dict[str, Array]:
    """Triangulate one directed pair while preserving every sampled hypothesis."""

    try:
        matches = np.asarray(sampled["matches_normalized"], dtype=np.float64)
        raw_certainty = np.asarray(sampled["raw_certainty"], dtype=np.float64)
        selection_score = np.asarray(sampled["selection_score"], dtype=np.float64)
        dense_value = np.asarray(sampled["dense_source_xy"])
        dense_valid_value = np.asarray(sampled["dense_source_valid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"pair sampler payload is incomplete: {exc}") from exc
    if not np.issubdtype(dense_value.dtype, np.integer) or dense_valid_value.dtype != np.bool_:
        raise ContractError("pair sampler dense-source arrays have invalid dtypes")
    dense_xy = np.asarray(dense_value, dtype=np.int64)
    dense_valid = np.asarray(dense_valid_value, dtype=np.bool_)
    count = len(matches)
    camera_count = len(frame.camera_ids)
    if (
        count <= 0
        or matches.shape != (count, 4)
        or not np.isfinite(matches).all()
        or raw_certainty.shape != (count,)
        or selection_score.shape != (count,)
        or dense_xy.shape != (count, 2)
        or dense_valid.shape != (count,)
        or source_camera == target_camera
        or not 0 <= source_camera < camera_count
        or not 0 <= target_camera < camera_count
        or neighbor_rank < 0
        or pair_ordinal < 0
        or not math.isfinite(world_bound)
        or world_bound <= 0.0
    ):
        raise ContractError("pair sampler payload or pair coordinates are invalid")

    height, width = frame.rgb.shape[1:3]
    source_xy, target_xy = normalized_matches_to_pixels(
        matches,
        source_height=height,
        source_width=width,
        target_height=height,
        target_width=width,
    )
    diagnostics = evaluate_two_view_diagnostics(
        _camera_record(frame.intrinsic[source_camera], frame.world_to_camera[source_camera]),
        _camera_record(frame.intrinsic[target_camera], frame.world_to_camera[target_camera]),
        source_xy,
        target_xy,
    )
    xyz = np.asarray(diagnostics["position"], dtype=np.float64)
    finite_geometry = np.asarray(diagnostics["valid"], dtype=np.bool_).copy()
    finite_geometry &= np.isfinite(raw_certainty) & np.isfinite(selection_score)
    source_in_bounds = _inside_image(source_xy, height=height, width=width)
    target_in_bounds = _inside_image(target_xy, height=height, width=width)
    source_positive_depth = np.asarray(diagnostics["source_camera_z"]) > 0.0
    target_positive_depth = np.asarray(diagnostics["target_camera_z"]) > 0.0
    inside_world_bound = np.isfinite(xyz).all(axis=1)
    inside_world_bound &= np.max(np.abs(xyz), axis=1) < world_bound
    admitted = (
        finite_geometry
        & dense_valid
        & source_in_bounds
        & target_in_bounds
        & source_positive_depth
        & target_positive_depth
        & inside_world_bound
    )

    color_camera = min(source_camera, target_camera)
    color_xy = source_xy if color_camera == source_camera else target_xy
    rgb = _bilinear_rgb8(frame.rgb[color_camera], color_xy)
    ply_row = np.full(count, -1, dtype=np.int64)
    ply_row[admitted] = np.arange(np.count_nonzero(admitted), dtype=np.int64)

    def finite_float32(value: Any) -> Array:
        converted = np.asarray(value, dtype=np.float32)
        return np.ascontiguousarray(
            np.nan_to_num(converted, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        )

    return {
        "frame_id": np.full(count, frame.frame_id, dtype=np.int32),
        "xyz": finite_float32(xyz),
        "rgb": rgb,
        "source_xy": finite_float32(source_xy),
        "target_xy": finite_float32(target_xy),
        "matches_normalized": finite_float32(matches),
        "raw_certainty": finite_float32(raw_certainty),
        "selection_score": finite_float32(selection_score),
        "source_camera": np.full(count, source_camera, dtype=np.int32),
        "target_camera": np.full(count, target_camera, dtype=np.int32),
        "color_camera": np.full(count, color_camera, dtype=np.int32),
        "neighbor_rank": np.full(count, neighbor_rank, dtype=np.int16),
        "pair_ordinal": np.full(count, pair_ordinal, dtype=np.int32),
        "dense_source_xy": np.ascontiguousarray(dense_xy.astype(np.int32)),
        "ray_source_parameter": finite_float32(diagnostics["source_depth"]),
        "ray_target_parameter": finite_float32(diagnostics["target_depth"]),
        "source_camera_z": finite_float32(diagnostics["source_camera_z"]),
        "target_camera_z": finite_float32(diagnostics["target_camera_z"]),
        "source_reprojection_error_pixels": finite_float32(
            diagnostics["source_reprojection_error_pixels"]
        ),
        "target_reprojection_error_pixels": finite_float32(
            diagnostics["target_reprojection_error_pixels"]
        ),
        "ray_gap_world": finite_float32(diagnostics["ray_gap_world"]),
        "triangulation_angle_degrees": finite_float32(diagnostics["triangulation_angle_degrees"]),
        "epipolar_sampson_normalized": finite_float32(diagnostics["epipolar_sampson_normalized"]),
        "dense_source_valid": np.ascontiguousarray(dense_valid),
        "finite_geometry": np.ascontiguousarray(finite_geometry),
        "source_pixel_in_bounds": np.ascontiguousarray(source_in_bounds),
        "target_pixel_in_bounds": np.ascontiguousarray(target_in_bounds),
        "source_positive_depth": np.ascontiguousarray(source_positive_depth),
        "target_positive_depth": np.ascontiguousarray(target_positive_depth),
        "inside_world_bound": np.ascontiguousarray(inside_world_bound),
        "admitted": np.ascontiguousarray(admitted),
        "ply_row": ply_row,
    }


def _finite_quantiles(values: Any) -> dict[str, Any]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flattened[np.isfinite(flattened)]
    if finite.size == 0:
        return {"finite_count": 0, "total_count": int(flattened.size), "values": None}
    names = ("minimum", "p10", "median", "p90", "p99", "maximum")
    quantiles = np.quantile(finite, (0.0, 0.1, 0.5, 0.9, 0.99, 1.0))
    return {
        "finite_count": int(finite.size),
        "total_count": int(flattened.size),
        "values": {name: float(value) for name, value in zip(names, quantiles, strict=True)},
    }


def _weight_record(registry: dict[str, Any], name: str) -> dict[str, Any]:
    weights = cast(dict[str, Any], registry["weights"])
    return cast(dict[str, Any], weights[name])


def _verify_registered_file(path: Path, record: dict[str, Any], *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise ContractError(f"{label} is unavailable: {resolved}: {exc}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise ContractError(f"{label} must be a regular non-symlink file")
    if size != record["bytes"]:
        raise ContractError(f"{label} size differs from the public registry")
    digest = sha256_file(resolved)
    if digest != record["sha256"]:
        raise ContractError(f"{label} SHA-256 differs from the public registry")
    return resolved


def _one_lock_package(lock: dict[str, Any], name: str) -> dict[str, Any]:
    raw_packages = lock.get("package")
    if not isinstance(raw_packages, list):
        raise ContractError("RoMa environment lock has no package inventory")
    matches = [
        value
        for value in cast(list[Any], raw_packages)
        if isinstance(value, dict) and value.get("name") == name
    ]
    if len(matches) != 1:
        raise ContractError(f"RoMa environment lock must contain exactly one {name} package")
    return cast(dict[str, Any], matches[0])


def _validate_environment_lock(path: Path, registry: dict[str, Any]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError(f"RoMa environment lock must be a regular file: {resolved}")
    try:
        payload = resolved.read_bytes()
        lock = tomllib.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ContractError(f"RoMa environment lock is invalid: {exc}") from exc

    if lock.get("requires-python") != "==3.12.*":
        raise ContractError("RoMa environment lock must require Python 3.12 exactly")
    provider = cast(dict[str, Any], registry["provider"])
    runtime = cast(dict[str, Any], registry["runtime"])
    romatch = _one_lock_package(lock, "romatch")
    expected_git = f"{provider['repository']}?rev={provider['revision']}#{provider['revision']}"
    if romatch.get("version") != provider["distribution_version"] or romatch.get("source") != {
        "git": expected_git
    }:
        raise ContractError("RoMa environment lock has a different romatch source identity")
    for name, version in (
        ("torch", runtime["torch"]),
        ("torchvision", runtime["torchvision"]),
    ):
        package = _one_lock_package(lock, name)
        if package.get("version") != version or package.get("source") != {
            "registry": runtime["torch_index"]
        }:
            raise ContractError(f"RoMa environment lock has a different {name} runtime")
    return {
        "filename": resolved.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "requires_python": lock["requires-python"],
    }


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _distribution_member(
    distribution: importlib.metadata.Distribution,
    relative: str,
) -> Path:
    files = distribution.files
    if files is None:
        raise ContractError("romatch distribution has no installed-file catalog")
    matches = [item for item in files if str(item).replace("\\", "/") == relative]
    if len(matches) != 1:
        raise ContractError(f"romatch distribution does not own exactly one {relative}")
    path = Path(str(distribution.locate_file(matches[0])))
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"romatch distribution member is not a regular file: {relative}")
    return path.resolve()


def _module_file(module: object, *, label: str) -> Path:
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str):
        raise ContractError(f"{label} has no module file")
    path = Path(raw)
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} module file is not regular")
    return path.resolve()


def _validate_factory_signature(factory: Callable[..., object]) -> None:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as exc:
        raise ContractError("romatch indoor factory signature is not inspectable") from exc
    required = {
        "device",
        "weights",
        "dinov2_weights",
        "coarse_res",
        "upsample_res",
        "amp_dtype",
        "symmetric",
        "use_custom_corr",
        "upsample_preds",
        "with_padding",
        "do_compile",
    }
    missing = sorted(required - set(signature.parameters))
    if missing:
        raise ContractError(f"romatch indoor factory is missing parameters: {missing}")


def _load_romatch_binding(registry: dict[str, Any]) -> _RomatchBinding:
    provider = cast(dict[str, Any], registry["provider"])
    distribution_name = cast(str, provider["distribution"])
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError("the registered romatch distribution is not installed") from exc
    if (
        _normalized_distribution_name(distribution.metadata.get("Name", "")) != distribution_name
        or distribution.version != provider["distribution_version"]
    ):
        raise ContractError("installed romatch distribution identity differs from the registry")
    expected_module = _distribution_member(distribution, "romatch/__init__.py")
    direct_text = distribution.read_text("direct_url.json")
    if direct_text is None:
        raise ContractError("romatch installation lacks direct_url source provenance")
    direct = _json_object_bytes(direct_text.encode("utf-8"), label="romatch direct_url")
    vcs = direct.get("vcs_info")
    if (
        direct.get("url") != provider["repository"]
        or not isinstance(vcs, dict)
        or vcs.get("vcs") != "git"
        or vcs.get("commit_id") != provider["revision"]
    ):
        raise ContractError("romatch direct_url does not identify the registered source commit")
    try:
        module: Any = importlib.import_module("romatch")
    except (ImportError, OSError, RuntimeError) as exc:
        raise ContractError(f"cannot import the registered romatch module: {exc}") from exc
    if _module_file(module, label="romatch") != expected_module:
        raise ContractError("imported romatch module is not owned by the registered distribution")
    factory: Any = getattr(module, "roma_indoor", None)
    if not callable(factory):
        raise ContractError("romatch does not expose the registered roma_indoor factory")
    typed_factory = factory
    _validate_factory_signature(typed_factory)
    return _RomatchBinding(
        factory=typed_factory,
        identity={
            "distribution": distribution_name,
            "distribution_version": distribution.version,
            "repository": provider["repository"],
            "revision": provider["revision"],
            "factory": provider["factory"],
            "direct_url": direct,
            "code_license": provider["license"],
        },
    )


def _validate_torch_runtime(torch_runtime: Any, runtime: dict[str, Any]) -> dict[str, Any]:
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or sys.version_info[:2] != (3, 12)
    ):
        raise ContractError("RoMa provider requires Linux x86_64 with CPython 3.12")
    hip_version = getattr(getattr(torch_runtime, "version", None), "hip", None)
    if torch_runtime.__version__ != runtime["torch"] or hip_version != runtime["hip"]:
        raise ContractError("Torch/HIP runtime differs from the registered RoMa profile")
    try:
        torchvision_version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError("registered torchvision distribution is not installed") from exc
    if torchvision_version != runtime["torchvision"]:
        raise ContractError("torchvision runtime differs from the registered RoMa profile")
    if (
        not torch_runtime.cuda.is_available()
        or torch_runtime.cuda.device_count() != runtime["visible_device_count"]
    ):
        raise ContractError("RoMa provider requires exactly one visible accelerator")
    properties = torch_runtime.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", "")).split(":", maxsplit=1)[0]
    if architecture != runtime["architecture"]:
        raise ContractError(
            f"RoMa provider requires {runtime['architecture']}, got {architecture or 'unknown'}"
        )
    torch_runtime.set_float32_matmul_precision("highest")
    torch_runtime.use_deterministic_algorithms(True, warn_only=False)
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch_runtime.__version__,
        "hip": hip_version,
        "accelerator_name": torch_runtime.cuda.get_device_name(0),
        "accelerator_architecture": str(getattr(properties, "gcnArchName", "")),
        "visible_device_count": torch_runtime.cuda.device_count(),
        "deterministic_algorithms": True,
        "float32_matmul_precision": "highest",
    }


def _canonicalize_sampled_pair(
    matches: Any,
    selection_score: Any,
    certainty: Any,
    *,
    count: int,
) -> dict[str, Any]:
    match_array = np.asarray(matches, dtype=np.float64)
    selection_array = np.asarray(selection_score, dtype=np.float64).reshape(-1)
    certainty_array = np.asarray(certainty, dtype=np.float64)
    # RoMa keeps the inference batch dimension even for the single image pair
    # accepted by this adapter.  Canonicalize that exact public API shape while
    # continuing to reject multi-pair or otherwise ambiguous tensors.
    if certainty_array.ndim == 3 and certainty_array.shape[0] == 1:
        certainty_array = certainty_array[0]
    if (
        count <= 0
        or match_array.shape != (count, 4)
        or selection_array.shape != (count,)
        or certainty_array.ndim != 2
        or min(certainty_array.shape) <= 0
        or not np.isfinite(match_array).all()
    ):
        raise ContractError("romatch sample output has an invalid shape or coordinate")
    dense_height, dense_width = certainty_array.shape
    dense_xy, dense_valid = recover_dense_source_xy(
        match_array[:, :2],
        dense_height=dense_height,
        dense_width=dense_width,
    )
    safe_x = np.clip(dense_xy[:, 0], 0, dense_width - 1)
    safe_y = np.clip(dense_xy[:, 1], 0, dense_height - 1)
    raw_certainty = certainty_array[safe_y, safe_x]
    dense_linear = dense_xy[:, 1] * dense_width + dense_xy[:, 0]
    order = np.argsort(dense_linear, kind="stable")
    return {
        "matches_normalized": np.ascontiguousarray(match_array[order]),
        "selection_score": np.ascontiguousarray(selection_array[order]),
        "raw_certainty": np.ascontiguousarray(raw_certainty[order]),
        "dense_source_xy": np.ascontiguousarray(dense_xy[order]),
        "dense_source_valid": np.ascontiguousarray(dense_valid[order]),
        "certainty_summary": _finite_quantiles(certainty_array),
        "selected_certainty_summary": _finite_quantiles(raw_certainty),
        "dense_shape": [dense_height, dense_width],
    }


class RomaIndoorPairSampler:
    """Pinned RoMa-v1 indoor inference on the admitted MI300X runtime."""

    def __init__(self, *, roma_weight: Path, dino_weight: Path, environment_lock: Path) -> None:
        registry, registry_sha256 = load_roma_provider_registry()
        self.roma_weight = _verify_registered_file(
            roma_weight,
            _weight_record(registry, "roma_indoor"),
            label="RoMa indoor weight",
        )
        self.dino_weight = _verify_registered_file(
            dino_weight,
            _weight_record(registry, "dinov2_vitl14"),
            label="DINOv2 ViT-L/14 weight",
        )
        lock_identity = _validate_environment_lock(environment_lock, registry)
        binding = _load_romatch_binding(registry)
        try:
            torch_runtime: Any = importlib.import_module("torch")
        except (ImportError, OSError, RuntimeError) as exc:
            raise ContractError(f"cannot import the registered Torch runtime: {exc}") from exc
        runtime_identity = _validate_torch_runtime(
            torch_runtime, cast(dict[str, Any], registry["runtime"])
        )
        self._torch = torch_runtime
        self._factory = binding.factory
        self._model: Any | None = None
        self._identity = {
            "name": "romatch_v1_indoor_directed_proposals",
            "adapter_schema": ROMA_POINT_PROVIDER_SCHEMA,
            "adapter_source_sha256": sha256_file(Path(__file__).resolve()),
            "registry": {
                "schema": ROMA_PROVIDER_REGISTRY_SCHEMA,
                "resource": ROMA_PROVIDER_REGISTRY_RESOURCE,
                "sha256": registry_sha256,
            },
            "upstream": binding.identity,
            "model": {
                "coarse_resolution": 560,
                "upsample_resolution": 864,
                "amp_dtype": "float16",
                "symmetric": False,
                "upsample_predictions": True,
                "sample_mode": "threshold",
                "sample_threshold": 0.05,
                "use_custom_corr": False,
                "with_padding": False,
                "compile": False,
            },
            "weights": {
                "roma_indoor": {
                    **_weight_record(registry, "roma_indoor"),
                },
                "dinov2_vitl14": {
                    **_weight_record(registry, "dinov2_vitl14"),
                },
            },
            "runtime": {**runtime_identity, "environment_lock": lock_identity},
        }

    @property
    def identity(self) -> dict[str, Any]:
        return self._identity

    def _load_state_dict(self, path: Path, *, label: str) -> dict[str, Any]:
        try:
            value: Any = self._torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ContractError(f"cannot load registered {label} state dict: {exc}") from exc
        if (
            not isinstance(value, dict)
            or not value
            or not all(isinstance(key, str) for key in value)
        ):
            raise ContractError(f"registered {label} checkpoint is not a string-keyed state dict")
        return cast(dict[str, Any], value)

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            model: Any = self._factory(
                device="cuda:0",
                weights=self._load_state_dict(self.roma_weight, label="RoMa indoor"),
                dinov2_weights=self._load_state_dict(self.dino_weight, label="DINOv2"),
                coarse_res=560,
                upsample_res=864,
                amp_dtype=self._torch.float16,
                symmetric=False,
                use_custom_corr=False,
                upsample_preds=True,
                with_padding=False,
                do_compile=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ContractError(f"cannot construct registered RoMa indoor model: {exc}") from exc
        if not all(callable(getattr(model, name, None)) for name in ("match", "sample", "eval")):
            raise ContractError("constructed RoMa model lacks the required inference API")
        model.sample_mode = "threshold"
        model.sample_thresh = 0.05
        model.eval()
        self._model = model
        return model

    def sample_pair(
        self,
        source_rgb: Array,
        target_rgb: Array,
        *,
        count: int,
        seed: int,
    ) -> dict[str, Any]:
        if (
            source_rgb.dtype != np.uint8
            or target_rgb.dtype != np.uint8
            or source_rgb.shape != target_rgb.shape
            or source_rgb.ndim != 3
            or source_rgb.shape[2] != 3
            or count <= 0
            or type(seed) is not int
        ):
            raise ContractError("RoMa pair sampler received invalid images, count, or seed")
        try:
            pil_image: Any = importlib.import_module("PIL.Image")
            source_image = pil_image.fromarray(source_rgb, mode="RGB")
            target_image = pil_image.fromarray(target_rgb, mode="RGB")
        except (ImportError, OSError, TypeError, ValueError) as exc:
            raise ContractError(f"cannot construct RoMa RGB inputs: {exc}") from exc
        model = self._get_model()
        self._torch.manual_seed(seed)
        self._torch.cuda.manual_seed_all(seed)
        self._torch.cuda.reset_peak_memory_stats(0)
        started = time.monotonic()
        try:
            with self._torch.inference_mode():
                warp, certainty = model.match(source_image, target_image, device="cuda:0")
                sampled, selection = model.sample(warp, certainty, num=count)
            self._torch.cuda.synchronize()
            sampled_array = sampled.detach().cpu().numpy()
            selection_array = selection.detach().cpu().numpy()
            certainty_array = certainty.detach().float().cpu().numpy()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ContractError(f"RoMa pair inference failed: {exc}") from exc
        result = _canonicalize_sampled_pair(
            sampled_array,
            selection_array,
            certainty_array,
            count=count,
        )
        result["runtime"] = {
            "duration_seconds": time.monotonic() - started,
            "dense_shape": result.pop("dense_shape"),
            "peak_allocated_bytes": int(self._torch.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(self._torch.cuda.max_memory_reserved(0)),
            "full_certainty": result.pop("certainty_summary"),
            "full_certainty_above_0_05": int(np.count_nonzero(np.asarray(certainty_array) > 0.05)),
            "selected_raw_certainty": result.pop("selected_certainty_summary"),
        }
        return result


def _pair_summary(
    planes: dict[str, Array],
    *,
    source_camera: int,
    target_camera: int,
    neighbor_rank: int,
    pair_ordinal: int,
    seed: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    admitted = planes["admitted"]
    return {
        "pair_ordinal": pair_ordinal,
        "source_camera": source_camera,
        "target_camera": target_camera,
        "neighbor_rank": neighbor_rank,
        "seed": seed,
        "sampled_count": len(admitted),
        "admitted_count": int(np.count_nonzero(admitted)),
        "rejection_counts_nonexclusive": {
            "dense_source_invalid": int(np.count_nonzero(~planes["dense_source_valid"])),
            "nonfinite_geometry_or_score": int(np.count_nonzero(~planes["finite_geometry"])),
            "source_out_of_bounds": int(np.count_nonzero(~planes["source_pixel_in_bounds"])),
            "target_out_of_bounds": int(np.count_nonzero(~planes["target_pixel_in_bounds"])),
            "source_nonpositive_depth": int(np.count_nonzero(~planes["source_positive_depth"])),
            "target_nonpositive_depth": int(np.count_nonzero(~planes["target_positive_depth"])),
            "outside_world_bound": int(np.count_nonzero(~planes["inside_world_bound"])),
        },
        "diagnostics_all_samples": {
            "raw_certainty": _finite_quantiles(planes["raw_certainty"]),
            "source_reprojection_error_pixels": _finite_quantiles(
                planes["source_reprojection_error_pixels"]
            ),
            "target_reprojection_error_pixels": _finite_quantiles(
                planes["target_reprojection_error_pixels"]
            ),
            "ray_gap_world": _finite_quantiles(planes["ray_gap_world"]),
            "triangulation_angle_degrees": _finite_quantiles(planes["triangulation_angle_degrees"]),
            "epipolar_sampson_normalized": _finite_quantiles(planes["epipolar_sampson_normalized"]),
        },
        "runtime": runtime,
    }


def _write_point_ply(path: Path, xyz: Array, rgb: Array) -> None:
    if (
        xyz.dtype != np.float32
        or rgb.dtype != np.uint8
        or xyz.ndim != 2
        or xyz.shape != rgb.shape
        or xyz.shape[1:] != (3,)
    ):
        raise ContractError("point PLY requires aligned (N,3) float32 XYZ and uint8 RGB")
    row_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    rows = np.empty(len(xyz), dtype=row_dtype)
    rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rows["red"], rows["green"], rows["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(rows)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("xb") as stream:
        stream.write(header)
        rows.tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_roma_point_proposals(
    output: Path,
    *,
    memmap_root: Path,
    observation_manifest: Path,
    frame_id: int,
    roma_weight: Path,
    dino_weight: Path,
    environment_lock: Path,
    num_points_per_frame: int = 700_000,
    nearest_cameras: int = 2,
    seed: int = 0,
    world_bound: float = 1_000.0,
    sampler: PairSampler | None = None,
) -> dict[str, Any]:
    """Build one append-only frame proposal artifact from public scene pixels."""

    destination = output.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise OutputExistsError(f"refusing to overwrite RoMa point proposals: {destination}")
    if (
        type(frame_id) is not int
        or frame_id < 0
        or num_points_per_frame <= 0
        or nearest_cameras <= 0
        or type(seed) is not int
        or not math.isfinite(world_bound)
        or world_bound <= 0.0
    ):
        raise ContractError("RoMa point-proposal configuration is invalid")
    frame = load_train_tensor_cache_frame(
        memmap_root,
        observation_manifest=observation_manifest,
        frame_id=frame_id,
    )
    camera_count = len(frame.camera_ids)
    if nearest_cameras >= camera_count:
        raise ContractError("nearest-camera degree must be smaller than the camera count")
    samples_per_pair = num_points_per_frame // (camera_count * nearest_cameras)
    if samples_per_pair <= 0:
        raise ContractError("point budget is too small for the directed camera graph")
    expected_samples = samples_per_pair * camera_count * nearest_cameras
    graph = nearest_camera_graph(frame.world_to_camera, neighbors=nearest_cameras)
    active_sampler: PairSampler = sampler or RomaIndoorPairSampler(
        roma_weight=roma_weight,
        dino_weight=dino_weight,
        environment_lock=environment_lock,
    )

    accumulated: dict[str, list[Array]] = {}
    pair_summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    pair_ordinal = 0
    for source_camera in range(camera_count):
        for neighbor_rank, target_value in enumerate(graph[source_camera]):
            target_camera = int(target_value)
            current_seed = pair_seed(
                seed,
                frame_id=frame_id,
                source_camera=source_camera,
                target_camera=target_camera,
                neighbor_rank=neighbor_rank,
            )
            sampled = active_sampler.sample_pair(
                frame.rgb[source_camera],
                frame.rgb[target_camera],
                count=samples_per_pair,
                seed=current_seed,
            )
            planes = assemble_pair_provenance(
                frame,
                source_camera=source_camera,
                target_camera=target_camera,
                neighbor_rank=neighbor_rank,
                pair_ordinal=pair_ordinal,
                sampled=sampled,
                world_bound=world_bound,
            )
            for name, values in planes.items():
                accumulated.setdefault(name, []).append(values)
            pair_summaries.append(
                _pair_summary(
                    planes,
                    source_camera=source_camera,
                    target_camera=target_camera,
                    neighbor_rank=neighbor_rank,
                    pair_ordinal=pair_ordinal,
                    seed=current_seed,
                    runtime=cast(dict[str, Any], sampled.get("runtime", {})),
                )
            )
            pair_ordinal += 1

    combined = {
        name: np.ascontiguousarray(np.concatenate(parts, axis=0))
        for name, parts in accumulated.items()
    }
    if not combined or len(combined["admitted"]) != expected_samples:
        raise ContractError("directed camera graph did not produce the complete sample inventory")
    admitted = combined["admitted"]
    combined["ply_row"] = np.full(expected_samples, -1, dtype=np.int64)
    combined["ply_row"][admitted] = np.arange(np.count_nonzero(admitted), dtype=np.int64)
    ply_xyz = np.ascontiguousarray(combined["xyz"][admitted], dtype=np.float32)
    ply_rgb = np.ascontiguousarray(combined["rgb"][admitted], dtype=np.uint8)
    if len(ply_xyz) == 0:
        raise ContractError("RoMa provider admitted no finite two-view points")

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        ply_name = f"f{frame_id:06d}.ply"
        provenance_name = "provenance.safetensors"
        _write_point_ply(stage / ply_name, ply_xyz, ply_rgb)
        provenance_sha256 = canonical_provenance_sha256(combined, frame_id=frame_id)
        try:
            safetensors_numpy: Any = importlib.import_module("safetensors.numpy")
            safetensors_numpy.save_file(
                combined,
                stage / provenance_name,
                metadata={
                    "schema": PROVENANCE_SCHEMA,
                    "frame_id": str(frame_id),
                    "row_semantics": "directed_pair_then_dense_source_linear_order",
                },
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ContractError(f"cannot encode point-proposal provenance: {exc}") from exc
        with (stage / provenance_name).open("rb") as stream:
            os.fsync(stream.fileno())

        aggregate = {
            "sampled_count": expected_samples,
            "admitted_count": len(ply_xyz),
            "admitted_fraction": float(len(ply_xyz) / expected_samples),
            "diagnostics_all_samples": {
                "raw_certainty": _finite_quantiles(combined["raw_certainty"]),
                "source_reprojection_error_pixels": _finite_quantiles(
                    combined["source_reprojection_error_pixels"]
                ),
                "target_reprojection_error_pixels": _finite_quantiles(
                    combined["target_reprojection_error_pixels"]
                ),
                "ray_gap_world": _finite_quantiles(combined["ray_gap_world"]),
                "triangulation_angle_degrees": _finite_quantiles(
                    combined["triangulation_angle_degrees"]
                ),
                "epipolar_sampson_normalized": _finite_quantiles(
                    combined["epipolar_sampson_normalized"]
                ),
            },
            "source_camera_sample_counts": [
                int(np.count_nonzero(combined["source_camera"] == camera))
                for camera in range(camera_count)
            ],
            "source_camera_admitted_counts": [
                int(np.count_nonzero((combined["source_camera"] == camera) & combined["admitted"]))
                for camera in range(camera_count)
            ],
        }
        receipt: dict[str, Any] = {
            "schema": ROMA_POINT_PROVIDER_SCHEMA,
            "status": "COMPLETE",
            "frame": {
                "frame_id": frame_id,
                "timestamp_seconds": frame.timestamp_seconds,
                "role": "train",
                "camera_ids": list(frame.camera_ids),
            },
            "source": frame.source_receipt,
            "provider": active_sampler.identity,
            "policy": {
                "num_points_per_frame_requested": num_points_per_frame,
                "nearest_cameras": nearest_cameras,
                "matches_per_directed_pair": samples_per_pair,
                "sampled_count_after_equal_pair_partition": expected_samples,
                "global_seed": seed,
                "pair_seed_schema": "p2g.roma_point_pair_seed.v1",
                "pair_order": "source_camera_index_then_neighbor_rank",
                "within_pair_order": "dense_source_linear_index_stable_ascending",
                "color": "minimum_camera_index_bilinear_rgb8_pixel_centers_v1",
                "world_coordinate_absolute_bound": world_bound,
                "admission": {
                    "hard_requirements": [
                        "finite_two_view_geometry_and_scores",
                        "recoverable_dense_source_cell",
                        "both_pixels_inside_their_images",
                        "positive_depth_in_both_cameras",
                        "position_inside_declared_world_bound",
                    ],
                    "diagnostic_only": [
                        "raw_certainty",
                        "reprojection_error_pixels",
                        "ray_gap_world",
                        "triangulation_angle_degrees",
                        "epipolar_sampson_normalized",
                    ],
                },
            },
            "nearest_camera_graph": graph.astype(int).tolist(),
            "aggregate": aggregate,
            "pairs": pair_summaries,
            "artifacts": {
                "point_ply": {
                    "path": ply_name,
                    "vertex_count": len(ply_xyz),
                    "size_bytes": (stage / ply_name).stat().st_size,
                    "sha256": sha256_file(stage / ply_name),
                },
                "provenance": {
                    "path": provenance_name,
                    "row_count": expected_samples,
                    "plane_names": sorted(combined),
                    "size_bytes": (stage / provenance_name).stat().st_size,
                    "sha256": sha256_file(stage / provenance_name),
                    "canonical_tensor_sha256": provenance_sha256,
                    "canonical_digest_schema": PROVENANCE_DIGEST_SCHEMA,
                },
            },
            "elapsed_seconds": time.monotonic() - started,
            "limitations": [
                "Each row is a two-view hypothesis, not a persistent temporal identity.",
                "RoMa certainty is retained as an upstream score, not a calibrated probability.",
                "Confidence and residuals are diagnostics rather than hidden capacity filters.",
                "The registered pure-PyTorch correlation path is used on MI300X.",
            ],
        }
        write_new_json(stage / "receipt.json", receipt)
        _fsync_directory(stage)
        if destination.exists() or destination.is_symlink():
            raise OutputExistsError(
                f"RoMa point-proposal destination appeared during publication: {destination}"
            )
        os.rename(stage, destination)
        _fsync_directory(destination.parent)
        return receipt
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
