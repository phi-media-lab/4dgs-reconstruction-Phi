# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

"""Build a fixed-capacity Gaussian initialization from first-party proposals.

The public entry point accepts one complete ``p2g.roma_point_proposal_sequence``
and the exact tensor cache from which it was produced.  A loose PLY directory is
not an input contract: PLY is only the sequence's hash-bound XYZ/RGB payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json, write_new_json
from p2g.errors import ContractError, OutputExistsError
from p2g.schema import validate_payload
from p2g.training.initialization import INITIALIZATION_SCHEMA
from p2g.training.roma_point_provider import (
    FRAME_TIMESTAMP_OPERATOR,
    PROVENANCE_SCHEMA,
    ROLE_ADMISSION_SCHEMA,
    ROMA_POINT_PROVIDER_SCHEMA,
)
from p2g.training.roma_point_sequence import ROMA_POINT_SEQUENCE_SCHEMA

RECEIPT_SCHEMA = "p2g.gaussian_initialization_receipt.v1"
CANONICAL_TENSOR_SCHEMA = "p2g.gaussian_initialization_canonical_tensor.v1"
TENSOR_FILENAME = "initialization.safetensors"
RECEIPT_FILENAME = "initialization.json"
TENSOR_CACHE_SCHEMA = "p2g.tensor_cache.v1"
SH_C0 = 0.28209479177387814

SamplingMode = Literal[
    "raw_candidate_uniform",
    "occupied_voxel_uniform",
    "triangulation_information_mixture",
    "paired_matcher_support_rank_mixture",
    "paired_multiview_consensus_rank_mixture",
]

_SAMPLING_MODES = frozenset(
    {
        "raw_candidate_uniform",
        "occupied_voxel_uniform",
        "triangulation_information_mixture",
        "paired_matcher_support_rank_mixture",
        "paired_multiview_consensus_rank_mixture",
    }
)
_EVIDENCE_MODES = frozenset(
    {
        "triangulation_information_mixture",
        "paired_matcher_support_rank_mixture",
        "paired_multiview_consensus_rank_mixture",
    }
)
_OUTPUT_PLANES = frozenset(
    {
        "means",
        "log_scales",
        "quaternions",
        "opacity_logits",
        "sh0",
        "center_times",
        "duration_logits",
        "velocities",
        "runtime_ids",
    }
)
_PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


@dataclass(frozen=True, slots=True)
class PointFrame:
    frame_id: int
    xyz: np.ndarray[Any, np.dtype[np.float32]]
    rgb: np.ndarray[Any, np.dtype[np.uint8]]
    ply_sha256: str
    provenance_sha256: str


@dataclass(frozen=True, slots=True)
class PointEvidence:
    angle_degrees: np.ndarray[Any, np.dtype[np.float32]]
    certainty: np.ndarray[Any, np.dtype[np.float32]]
    pair_ordinal: np.ndarray[Any, np.dtype[np.int32]]
    source_camera: np.ndarray[Any, np.dtype[np.int32]] | None = None
    target_camera: np.ndarray[Any, np.dtype[np.int32]] | None = None
    ray_gap_world: np.ndarray[Any, np.dtype[np.float32]] | None = None
    source_reprojection_pixels: np.ndarray[Any, np.dtype[np.float32]] | None = None
    target_reprojection_pixels: np.ndarray[Any, np.dtype[np.float32]] | None = None


@dataclass(frozen=True, slots=True)
class SampledFrame:
    tensors: dict[str, Tensor]
    sampling: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Collection:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    frame_ids: tuple[int, ...]
    rows: dict[int, dict[str, Any]]
    timestamps: dict[int, float]
    tensor_cache_manifest_sha256: str
    observation_manifest_sha256: str
    provider_identity_sha256: str


@dataclass(frozen=True, slots=True)
class _Consensus:
    score: np.ndarray[Any, np.dtype[np.float64]]
    matcher_rank: np.ndarray[Any, np.dtype[np.float64]]
    triangulation_information: np.ndarray[Any, np.dtype[np.float64]]
    inverse_gap_rank: np.ndarray[Any, np.dtype[np.float64]]
    inverse_reprojection_rank: np.ndarray[Any, np.dtype[np.float64]]
    pair_support: np.ndarray[Any, np.dtype[np.int64]]
    camera_support: np.ndarray[Any, np.dtype[np.int64]]
    occupied_voxels: int


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} must be a regular non-symlink file")
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain one JSON object")
    return cast(dict[str, Any], value)


def _sha256_text(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _safe_relative(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} must be a non-empty POSIX relative path")
    relative = Path(value)
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        raise ContractError(f"{label} escapes its artifact root")
    return relative


def _bound_file(root: Path, relative: object, *, label: str) -> Path:
    member = _safe_relative(relative, label=label)
    path = (root / member).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} is missing, unsafe, or outside its artifact root")
    return path


def _bound_directory(root: Path, relative: object, *, label: str) -> Path:
    member = _safe_relative(relative, label=label)
    path = (root / member).resolve()
    if not path.is_relative_to(root) or not path.is_dir() or path.is_symlink():
        raise ContractError(f"{label} is missing, unsafe, or outside its artifact root")
    return path


def _quantiles(values: np.ndarray[Any, np.dtype[Any]]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not bool(np.isfinite(array).all()):
        raise ContractError("initialization diagnostics require a finite non-empty population")
    names = ("minimum", "p10", "median", "p90", "p99", "maximum")
    result = np.quantile(array, (0.0, 0.1, 0.5, 0.9, 0.99, 1.0))
    return {name: float(value) for name, value in zip(names, result, strict=True)}


def _read_point_ply(
    path: Path,
    *,
    expected_sha256: str,
    expected_count: int,
    frame_id: int,
    provenance_sha256: str,
) -> PointFrame:
    """Read only the exact XYZ/uint8-RGB payload emitted by the public provider."""

    digest_before = sha256_file(path)
    if digest_before != expected_sha256:
        raise ContractError(f"proposal point payload SHA-256 mismatch for frame {frame_id}")
    expected_header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {expected_count}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    try:
        with path.open("rb") as stream:
            header: list[str] = []
            while len(header) <= len(expected_header):
                raw = stream.readline(4097)
                if not raw or len(raw) > 4096:
                    raise ContractError(f"proposal PLY header is incomplete for frame {frame_id}")
                header.append(raw.decode("ascii").rstrip("\r\n"))
                if header[-1] == "end_header":
                    break
            offset = stream.tell()
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractError(f"cannot read proposal PLY for frame {frame_id}: {exc}") from exc
    expected_size = offset + expected_count * _PLY_DTYPE.itemsize
    if header != expected_header or path.stat().st_size != expected_size:
        raise ContractError(
            f"proposal PLY layout differs from the public provider for frame {frame_id}"
        )
    rows = np.memmap(path, mode="r", dtype=_PLY_DTYPE, offset=offset, shape=(expected_count,))
    xyz = np.ascontiguousarray(np.stack((rows["x"], rows["y"], rows["z"]), axis=1))
    rgb = np.ascontiguousarray(np.stack((rows["red"], rows["green"], rows["blue"]), axis=1))
    del rows
    if xyz.dtype != np.float32 or rgb.dtype != np.uint8 or not bool(np.isfinite(xyz).all()):
        raise ContractError(f"proposal PLY values are invalid for frame {frame_id}")
    if sha256_file(path) != digest_before:
        raise ContractError(f"proposal point payload changed while reading frame {frame_id}")
    return PointFrame(
        frame_id=frame_id,
        xyz=xyz,
        rgb=rgb,
        ply_sha256=digest_before,
        provenance_sha256=provenance_sha256,
    )


def _load_collection(root: Path) -> _Collection:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ContractError("proposal sequence must be a regular non-symlink directory")
    manifest_path = resolved / "collection.json"
    manifest = _json_object(manifest_path, label="proposal collection")
    if manifest.get("schema") != ROMA_POINT_SEQUENCE_SCHEMA or manifest.get("status") != "COMPLETE":
        raise ContractError("proposal collection is not a complete public RoMa sequence")
    raw_frame_ids = manifest.get("frame_ids")
    raw_rows = manifest.get("frames")
    if not isinstance(raw_frame_ids, list) or not isinstance(raw_rows, list):
        raise ContractError("proposal collection lacks its frame inventory")
    frame_ids = tuple(raw_frame_ids)
    if (
        len(frame_ids) < 2
        or any(type(frame_id) is not int or frame_id < 0 for frame_id in frame_ids)
        or tuple(sorted(frame_ids)) != frame_ids
        or len(set(frame_ids)) != len(frame_ids)
        or manifest.get("frame_count") != len(frame_ids)
        or len(raw_rows) != len(frame_ids)
    ):
        raise ContractError("proposal collection frame inventory is invalid")
    declared_observation_sha256 = _sha256_text(
        manifest.get("observation_manifest_sha256"),
        label="proposal collection observation-manifest SHA-256",
    )
    if manifest.get("admitted_observation_role") != "train":
        raise ContractError("proposal collection is not restricted to train observations")
    points_root = _bound_directory(resolved, manifest.get("points_root"), label="points root")
    rows: dict[int, dict[str, Any]] = {}
    timestamps: dict[int, float] = {}
    cache_hashes: set[str] = set()
    observation_hashes: set[str] = set()
    sampled_total = 0
    admitted_total = 0
    for expected_frame_id, raw_row in zip(frame_ids, raw_rows, strict=True):
        if not isinstance(raw_row, dict):
            raise ContractError("proposal collection frame row must be an object")
        row = cast(dict[str, Any], raw_row)
        frame_id_value = row.get("frame_id")
        admitted_count = _positive_integer(
            row.get("admitted_count"), label=f"frame {expected_frame_id} admitted count"
        )
        sampled_count = _positive_integer(
            row.get("sampled_count"), label=f"frame {expected_frame_id} sampled count"
        )
        if (
            frame_id_value != expected_frame_id
            or admitted_count > sampled_count
            or expected_frame_id in rows
        ):
            raise ContractError("proposal collection rows do not match the declared frame order")
        frame_id = expected_frame_id
        expected_ply_name = f"f{frame_id:06d}.ply"
        if row.get("point_ply") != expected_ply_name:
            raise ContractError(f"frame {frame_id} has a non-canonical point payload name")
        ply = _bound_file(points_root, expected_ply_name, label=f"frame {frame_id} point payload")
        ply_sha256 = _sha256_text(row.get("point_ply_sha256"), label="point payload SHA-256")
        if sha256_file(ply) != ply_sha256:
            raise ContractError(f"frame {frame_id} point payload is not collection-bound")
        frame_root = _bound_directory(
            resolved, row.get("frame_root"), label=f"frame {frame_id} artifact root"
        )
        frame_receipt = _json_object(frame_root / "receipt.json", label=f"frame {frame_id} receipt")
        point_record = cast(dict[str, Any], frame_receipt.get("artifacts", {}).get("point_ply", {}))
        provenance_record = cast(
            dict[str, Any], frame_receipt.get("artifacts", {}).get("provenance", {})
        )
        frame_record = cast(dict[str, Any], frame_receipt.get("frame", {}))
        source_record = cast(dict[str, Any], frame_receipt.get("source", {}))
        source_manifest = cast(dict[str, Any], source_record.get("manifest", {}))
        source_payload = cast(dict[str, Any], source_record.get("frame_payload_sha256", {}))
        role_admission = cast(dict[str, Any], source_record.get("role_admission", {}))
        aggregate_record = cast(dict[str, Any], frame_receipt.get("aggregate", {}))
        provenance_sha256 = _sha256_text(
            row.get("provenance_sha256"), label="provenance SHA-256"
        )
        canonical_sha256 = _sha256_text(
            row.get("provenance_canonical_tensor_sha256"),
            label="canonical provenance SHA-256",
        )
        cache_sha256 = _sha256_text(
            source_manifest.get("sha256"), label="source tensor-cache manifest SHA-256"
        )
        observation_sha256 = _sha256_text(
            source_manifest.get("observation_manifest_sha256"),
            label="source observation manifest SHA-256",
        )
        source_rgb_sha256 = _sha256_text(
            row.get("source_rgb_sha256"), label="source RGB frame SHA-256"
        )
        role_admission_sha256 = _sha256_text(
            row.get("role_admission_sha256"), label="role-admission SHA-256"
        )
        unsigned_role_admission = dict(role_admission)
        embedded_role_admission_sha256 = unsigned_role_admission.pop(
            "logical_sha256", None
        )
        admitted_camera_ids = role_admission.get("admitted_camera_ids")
        admitted_observation_ids = role_admission.get("admitted_observation_ids")
        excluded_camera_ids = role_admission.get("excluded_camera_ids_by_role")
        valid_excluded_roles = (
            isinstance(excluded_camera_ids, dict)
            and set(excluded_camera_ids) == {"diagnostic", "sealed", "free_view"}
            and all(isinstance(value, list) for value in excluded_camera_ids.values())
        )
        excluded_flat = (
            [
                camera_id
                for role in ("diagnostic", "sealed", "free_view")
                for camera_id in cast(dict[str, list[Any]], excluded_camera_ids)[role]
            ]
            if valid_excluded_roles
            else []
        )
        timestamp = frame_record.get("timestamp_seconds")
        if (
            frame_receipt.get("schema") != ROMA_POINT_PROVIDER_SCHEMA
            or frame_receipt.get("status") != "COMPLETE"
            or frame_record.get("frame_id") != frame_id
            or frame_record.get("role") != "train"
            or not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(timestamp)
            or point_record.get("path") != expected_ply_name
            or point_record.get("sha256") != ply_sha256
            or point_record.get("vertex_count") != admitted_count
            or provenance_record.get("path") != "provenance.safetensors"
            or provenance_record.get("sha256") != provenance_sha256
            or provenance_record.get("canonical_tensor_sha256") != canonical_sha256
            or aggregate_record.get("sampled_count") != sampled_count
            or aggregate_record.get("admitted_count") != admitted_count
            or source_payload.get("rgb") != source_rgb_sha256
            or role_admission.get("schema") != ROLE_ADMISSION_SCHEMA
            or role_admission.get("role") != "train"
            or role_admission.get("frame_id") != frame_id
            or role_admission.get("frame_timestamp_operator")
            != FRAME_TIMESTAMP_OPERATOR
            or role_admission.get("observation_manifest_sha256") != observation_sha256
            or admitted_camera_ids != frame_record.get("camera_ids")
            or not isinstance(admitted_camera_ids, list)
            or not admitted_camera_ids
            or any(not isinstance(value, str) for value in admitted_camera_ids)
            or admitted_camera_ids != sorted(admitted_camera_ids)
            or len(set(admitted_camera_ids)) != len(admitted_camera_ids)
            or not isinstance(admitted_observation_ids, list)
            or len(admitted_observation_ids) != len(admitted_camera_ids)
            or any(not isinstance(value, str) for value in admitted_observation_ids)
            or len(set(admitted_observation_ids)) != len(admitted_observation_ids)
            or type(role_admission.get("cache_camera_count")) is not int
            or role_admission["cache_camera_count"] < len(admitted_camera_ids)
            or not valid_excluded_roles
            or any(not isinstance(value, str) for value in excluded_flat)
            or len(set(excluded_flat)) != len(excluded_flat)
            or bool(set(admitted_camera_ids) & set(excluded_flat))
            or len(admitted_camera_ids) + len(excluded_flat)
            != role_admission["cache_camera_count"]
            or embedded_role_admission_sha256 != role_admission_sha256
            or sha256_json(unsigned_role_admission) != role_admission_sha256
        ):
            raise ContractError(f"frame {frame_id} receipt disagrees with the proposal collection")
        provenance = _bound_file(
            frame_root, "provenance.safetensors", label=f"frame {frame_id} provenance"
        )
        if sha256_file(provenance) != provenance_sha256:
            raise ContractError(f"frame {frame_id} provenance is not collection-bound")
        row["_points_path"] = ply
        row["_provenance_path"] = provenance
        rows[frame_id] = row
        timestamps[frame_id] = float(timestamp)
        cache_hashes.add(cache_sha256)
        observation_hashes.add(observation_sha256)
        sampled_total += sampled_count
        admitted_total += admitted_count
    if len(cache_hashes) != 1 or len(observation_hashes) != 1:
        raise ContractError("proposal frames do not share one tensor-cache source")
    if next(iter(observation_hashes)) != declared_observation_sha256:
        raise ContractError("proposal collection observation-manifest binding is inconsistent")
    aggregate = manifest.get("aggregate")
    provider = manifest.get("provider")
    if not isinstance(aggregate, dict) or not isinstance(provider, dict) or not provider:
        raise ContractError("proposal collection lacks aggregate or provider identity")
    expected_fraction = admitted_total / sampled_total
    if (
        aggregate.get("sampled_count") != sampled_total
        or aggregate.get("admitted_count") != admitted_total
        or not isinstance(aggregate.get("admitted_fraction"), (int, float))
        or not math.isclose(
            float(aggregate["admitted_fraction"]), expected_fraction, rel_tol=0.0, abs_tol=1.0e-15
        )
    ):
        raise ContractError("proposal collection aggregate is invalid")
    ordered_times = np.asarray([timestamps[frame_id] for frame_id in frame_ids], dtype=np.float64)
    if not bool(np.isfinite(ordered_times).all()) or bool(
        np.any(ordered_times[1:] <= ordered_times[:-1])
    ):
        raise ContractError("proposal frame timestamps are not strictly increasing")
    return _Collection(
        root=resolved,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        frame_ids=cast(tuple[int, ...], frame_ids),
        rows=rows,
        timestamps=timestamps,
        tensor_cache_manifest_sha256=next(iter(cache_hashes)),
        observation_manifest_sha256=next(iter(observation_hashes)),
        provider_identity_sha256=sha256_json(provider),
    )


def _load_cache_binding(
    root: Path,
    *,
    collection: _Collection,
) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ContractError("tensor cache must be a regular non-symlink directory")
    manifest_path = resolved / "tensor_cache.json"
    manifest = _json_object(manifest_path, label="tensor-cache manifest")
    validate_payload("tensor_cache", manifest)
    manifest_sha256 = sha256_file(manifest_path)
    if (
        manifest.get("schema_version") != TENSOR_CACHE_SCHEMA
        or manifest_sha256 != collection.tensor_cache_manifest_sha256
        or manifest.get("observation_manifest_sha256") != collection.observation_manifest_sha256
        or tuple(manifest.get("frame_ids", ())) != collection.frame_ids
    ):
        raise ContractError("tensor cache does not match the proposal sequence source")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or not isinstance(arrays.get("timestamp_seconds"), dict):
        raise ContractError("tensor cache lacks its timestamp array record")
    timestamp_record = cast(dict[str, Any], arrays["timestamp_seconds"])
    timestamp_path = _bound_file(
        resolved, timestamp_record.get("path"), label="tensor-cache timestamp array"
    )
    timestamp_sha256 = _sha256_text(
        timestamp_record.get("sha256"), label="tensor-cache timestamp SHA-256"
    )
    if sha256_file(timestamp_path) != timestamp_sha256:
        raise ContractError("tensor-cache timestamp array SHA-256 mismatch")
    try:
        raw: Any = np.load(timestamp_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot memory-map tensor-cache timestamps: {exc}") from exc
    if not isinstance(raw, np.ndarray):
        raise ContractError("tensor-cache timestamp payload is not an ndarray")
    timestamps = cast(np.ndarray[Any, np.dtype[Any]], raw)
    camera_ids = manifest.get("camera_ids")
    expected_shape = (
        len(collection.frame_ids),
        len(camera_ids) if isinstance(camera_ids, list) else 0,
    )
    if (
        timestamps.dtype != np.dtype("float64")
        or tuple(timestamps.shape) != expected_shape
        or timestamp_record.get("dtype") != "float64"
        or timestamp_record.get("shape") != list(expected_shape)
        or timestamp_record.get("order") != "C"
        or not timestamps.flags.c_contiguous
        or not bool(np.isfinite(timestamps).all())
        or not bool(np.all(timestamps == timestamps[:, :1]))
    ):
        raise ContractError("tensor-cache timestamp array differs from its public contract")
    axis = np.asarray(timestamps[:, 0], dtype=np.float64)
    expected_axis = np.asarray(
        [collection.timestamps[frame_id] for frame_id in collection.frame_ids], dtype=np.float64
    )
    # The receipt records the arithmetic mean of the admitted camera timestamps.
    # Even when every source timestamp is bit-identical, a floating-point reduction
    # may differ from the unreduced cache value by a few ULPs.  Both sides are
    # already bound to the same cache SHA-256 above, so admit only the reduction
    # error bound implied by the number of camera operands rather than a loose
    # application-level epsilon.
    timestamp_scale = np.maximum(np.abs(axis), np.abs(expected_axis))
    timestamp_tolerance = expected_shape[1] * np.spacing(timestamp_scale)
    if not bool(np.all(np.abs(axis - expected_axis) <= timestamp_tolerance)):
        raise ContractError("tensor-cache timestamps differ from the proposal receipts")
    return {
        "schema_version": TENSOR_CACHE_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "observation_manifest_sha256": collection.observation_manifest_sha256,
        "timestamp_sha256": timestamp_sha256,
        "frame_count": len(collection.frame_ids),
        "camera_count": expected_shape[1],
    }


def _load_evidence(
    path: Path,
    *,
    frame_id: int,
    expected_sha256: str,
    expected_points: int,
    require_multiview: bool,
) -> PointEvidence:
    if sha256_file(path) != expected_sha256:
        raise ContractError(f"proposal provenance SHA-256 mismatch for frame {frame_id}")
    required = {
        "admitted": np.dtype("bool"),
        "ply_row": np.dtype("int64"),
        "triangulation_angle_degrees": np.dtype("float32"),
        "raw_certainty": np.dtype("float32"),
        "pair_ordinal": np.dtype("int32"),
    }
    if require_multiview:
        required.update(
            {
                "source_camera": np.dtype("int32"),
                "target_camera": np.dtype("int32"),
                "ray_gap_world": np.dtype("float32"),
                "source_reprojection_error_pixels": np.dtype("float32"),
                "target_reprojection_error_pixels": np.dtype("float32"),
            }
        )
    try:
        with safe_open(str(path), framework="np") as stream:
            metadata = stream.metadata() or {}
            if (
                metadata.get("schema") != PROVENANCE_SCHEMA
                or metadata.get("frame_id") != str(frame_id)
                or metadata.get("row_semantics")
                != "directed_pair_then_dense_source_linear_order"
            ):
                raise ContractError(f"proposal provenance metadata is invalid for frame {frame_id}")
            if not set(required).issubset(stream.keys()):
                raise ContractError(
                    f"proposal provenance planes are incomplete for frame {frame_id}"
                )
            values = {name: np.asarray(stream.get_tensor(name)) for name in required}
    except ContractError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ContractError(f"cannot read proposal provenance for frame {frame_id}: {exc}") from exc
    if sha256_file(path) != expected_sha256:
        raise ContractError(f"proposal provenance changed while reading frame {frame_id}")
    row_count = len(values["admitted"])
    for name, dtype in required.items():
        value = values[name]
        if value.dtype != dtype or value.shape != (row_count,):
            raise ContractError(f"proposal provenance plane {name} is invalid for frame {frame_id}")
    admitted = values["admitted"]
    ply_row = values["ply_row"]
    if (
        int(np.count_nonzero(admitted)) != expected_points
        or not np.array_equal(ply_row[admitted], np.arange(expected_points, dtype=np.int64))
    ):
        raise ContractError(f"proposal provenance does not map to PLY rows for frame {frame_id}")

    def admitted_plane(name: str) -> np.ndarray[Any, np.dtype[Any]]:
        return np.ascontiguousarray(values[name][admitted])

    angle = admitted_plane("triangulation_angle_degrees")
    certainty = admitted_plane("raw_certainty")
    pairs = admitted_plane("pair_ordinal")
    if (
        not bool(np.isfinite(angle).all())
        or not bool(np.isfinite(certainty).all())
        or bool(np.any(angle < 0.0))
        or bool(np.any(pairs < 0))
    ):
        raise ContractError(f"proposal evidence is non-finite or out of range for frame {frame_id}")
    return PointEvidence(
        angle_degrees=cast(np.ndarray[Any, np.dtype[np.float32]], angle),
        certainty=cast(np.ndarray[Any, np.dtype[np.float32]], certainty),
        pair_ordinal=cast(np.ndarray[Any, np.dtype[np.int32]], pairs),
        source_camera=(
            cast(np.ndarray[Any, np.dtype[np.int32]], admitted_plane("source_camera"))
            if require_multiview
            else None
        ),
        target_camera=(
            cast(np.ndarray[Any, np.dtype[np.int32]], admitted_plane("target_camera"))
            if require_multiview
            else None
        ),
        ray_gap_world=(
            cast(np.ndarray[Any, np.dtype[np.float32]], admitted_plane("ray_gap_world"))
            if require_multiview
            else None
        ),
        source_reprojection_pixels=(
            cast(
                np.ndarray[Any, np.dtype[np.float32]],
                admitted_plane("source_reprojection_error_pixels"),
            )
            if require_multiview
            else None
        ),
        target_reprojection_pixels=(
            cast(
                np.ndarray[Any, np.dtype[np.float32]],
                admitted_plane("target_reprojection_error_pixels"),
            )
            if require_multiview
            else None
        ),
    )


def _midrank(values: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    data = np.asarray(values)
    if data.ndim != 1 or data.size == 0:
        raise ContractError("rank evidence must be a non-empty vector")
    order = np.argsort(data, kind="stable")
    ordered = data[order]
    starts = np.flatnonzero(
        np.concatenate((np.asarray((True,)), ordered[1:] != ordered[:-1]))
    )
    stops = np.concatenate((starts[1:], np.asarray((len(data),))))
    ranked = np.repeat((starts + stops + 1.0) / (2.0 * len(data)), stops - starts)
    result = np.empty(len(data), dtype=np.float64)
    result[order] = ranked
    return result


def _group_midrank(
    values: np.ndarray[Any, np.dtype[Any]],
    groups: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if values.shape != groups.shape or values.ndim != 1 or values.size == 0:
        raise ContractError("group-ranked evidence must contain aligned non-empty vectors")
    result = np.empty(len(values), dtype=np.float64)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        result[indices] = _midrank(values[indices])
    return result


def _geometric_mean(
    *components: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if not components:
        raise ContractError("evidence fusion requires at least one component")
    shape = components[0].shape
    logarithm = np.zeros(shape, dtype=np.float64)
    floor = np.finfo(np.float64).tiny
    for raw in components:
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != shape or not bool(np.isfinite(values).all()) or bool(np.any(values < 0)):
            raise ContractError("evidence fusion received an invalid component")
        logarithm += np.log(np.maximum(values, floor))
    result = np.exp(logarithm / len(components))
    if not bool(np.isfinite(result).all()):
        raise ContractError("evidence fusion produced non-finite scores")
    return result


def _voxel_inverse(
    xyz: np.ndarray[Any, np.dtype[np.float32]], voxel_size: float
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], int]:
    coordinates = np.floor(xyz.astype(np.float64) / voxel_size)
    limit = np.iinfo(np.int64)
    if (
        not bool(np.isfinite(coordinates).all())
        or float(coordinates.min()) < limit.min
        or float(coordinates.max()) > limit.max
    ):
        raise ContractError("proposal voxel coordinates exceed the int64 domain")
    _, inverse = np.unique(coordinates.astype(np.int64), axis=0, return_inverse=True)
    typed = np.ascontiguousarray(inverse, dtype=np.int64)
    return typed, int(typed.max()) + 1


def _consensus(
    xyz: np.ndarray[Any, np.dtype[np.float32]],
    evidence: PointEvidence,
    *,
    voxel_size: float,
) -> _Consensus:
    optional = (
        evidence.source_camera,
        evidence.target_camera,
        evidence.ray_gap_world,
        evidence.source_reprojection_pixels,
        evidence.target_reprojection_pixels,
    )
    if any(value is None for value in optional):
        raise ContractError("multi-view consensus requires complete geometric evidence")
    source_camera = np.asarray(evidence.source_camera, dtype=np.int64)
    target_camera = np.asarray(evidence.target_camera, dtype=np.int64)
    gap = np.asarray(evidence.ray_gap_world, dtype=np.float64)
    source_error = np.asarray(evidence.source_reprojection_pixels, dtype=np.float64)
    target_error = np.asarray(evidence.target_reprojection_pixels, dtype=np.float64)
    angle = np.asarray(evidence.angle_degrees, dtype=np.float64)
    certainty = np.asarray(evidence.certainty, dtype=np.float64)
    pair = np.asarray(evidence.pair_ordinal, dtype=np.int64)
    expected = (len(xyz),)
    if (
        any(
            value.shape != expected
            for value in (
                source_camera,
                target_camera,
                gap,
                source_error,
                target_error,
                angle,
                certainty,
                pair,
            )
        )
        or any(
            not bool(np.isfinite(value).all())
            for value in (gap, source_error, target_error, angle, certainty)
        )
        or any(
            bool(np.any(value < 0))
            for value in (
                source_camera,
                target_camera,
                pair,
                gap,
                source_error,
                target_error,
                angle,
            )
        )
        or bool(np.any(source_camera == target_camera))
    ):
        raise ContractError("multi-view consensus evidence is invalid")
    camera_domain = int(max(source_camera.max(), target_camera.max())) + 1
    pair_domain = int(pair.max()) + 1
    directed_pair = source_camera * camera_domain + target_camera
    for ordinal in np.unique(pair):
        if len(np.unique(directed_pair[pair == ordinal])) != 1:
            raise ContractError("one pair ordinal maps to multiple directed camera pairs")
    inverse, occupied = _voxel_inverse(xyz, voxel_size)
    unique_voxel_pair = np.unique(inverse * pair_domain + pair)
    pairs_per_voxel = np.bincount(
        unique_voxel_pair // pair_domain, minlength=occupied
    ).astype(np.int64)
    unique_voxel_camera = np.unique(
        np.concatenate(
            (
                inverse * camera_domain + source_camera,
                inverse * camera_domain + target_camera,
            )
        )
    )
    cameras_per_voxel = np.bincount(
        unique_voxel_camera // camera_domain, minlength=occupied
    ).astype(np.int64)
    pair_support = pairs_per_voxel[inverse]
    camera_support = cameras_per_voxel[inverse]
    matcher_rank = _group_midrank(certainty, pair)
    information = np.square(np.sin(np.deg2rad(np.clip(angle, 0.0, 90.0))))
    if not bool(np.any(information > 0.0)):
        raise ContractError("multi-view consensus has zero triangulation information")
    inverse_gap_rank = _group_midrank(-gap, pair)
    inverse_reprojection_rank = _group_midrank(-np.maximum(source_error, target_error), pair)
    candidate_quality = _geometric_mean(
        matcher_rank, information, inverse_gap_rank, inverse_reprojection_rank
    )
    support = _geometric_mean(_midrank(pair_support), _midrank(camera_support))
    score = _geometric_mean(candidate_quality, support)
    if float(score.sum(dtype=np.float64)) <= 0.0:
        raise ContractError("multi-view consensus has zero probability mass")
    return _Consensus(
        score=score,
        matcher_rank=matcher_rank,
        triangulation_information=information,
        inverse_gap_rank=inverse_gap_rank,
        inverse_reprojection_rank=inverse_reprojection_rank,
        pair_support=pair_support,
        camera_support=camera_support,
        occupied_voxels=occupied,
    )


def _evidence_seed(
    global_seed: int,
    *,
    frame_id: int,
    sampling_mode: SamplingMode,
) -> int:
    """Derive an evidence-only stream without perturbing the raw baseline stream."""

    payload = json.dumps(
        {
            "schema": "p2g.point_sampling_evidence_seed.v1",
            "global_seed": global_seed,
            "frame_id": frame_id,
            "sampling_mode": sampling_mode,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _weighted_draw(
    weights: np.ndarray[Any, np.dtype[Any]],
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], float]:
    probabilities = np.asarray(weights, dtype=np.float64)
    total = float(probabilities.sum(dtype=np.float64))
    if (
        probabilities.ndim != 1
        or probabilities.size == 0
        or not bool(np.isfinite(probabilities).all())
        or bool(np.any(probabilities < 0.0))
        or not math.isfinite(total)
        or total <= 0.0
    ):
        raise ContractError("weighted proposal population has invalid probability mass")
    probabilities = probabilities / total
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative[-1] = 1.0
    indices = np.searchsorted(cumulative, rng.random(count), side="right")
    effective_population = 1.0 / float(np.square(probabilities).sum(dtype=np.float64))
    return np.ascontiguousarray(indices, dtype=np.int64), effective_population


def _selection_quantiles(
    values: np.ndarray[Any, np.dtype[Any]],
    indices: np.ndarray[Any, np.dtype[np.int64]],
) -> dict[str, dict[str, float]]:
    return {"source": _quantiles(values), "selected": _quantiles(values[indices])}


def _draw_indices(
    xyz: np.ndarray[Any, np.dtype[np.float32]],
    *,
    count: int,
    rng: np.random.Generator,
    sampling_mode: SamplingMode,
    voxel_size: float,
    evidence: PointEvidence | None,
    evidence_fraction: float,
    evidence_seed: int,
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], dict[str, Any]]:
    candidate_count = len(xyz)
    if candidate_count <= 0 or count <= 0:
        raise ContractError("proposal sampling requires non-empty source and target populations")

    if sampling_mode == "raw_candidate_uniform":
        indices = rng.integers(0, candidate_count, size=count, endpoint=False, dtype=np.int64)
        return indices, {
            "mode": sampling_mode,
            "replacement": "with_replacement",
            "source_candidate_count": candidate_count,
            "selected_count": count,
            "selected_unique_source_count": int(np.unique(indices).size),
        }

    if sampling_mode == "occupied_voxel_uniform":
        inverse, occupied = _voxel_inverse(xyz, voxel_size)
        voxel_counts = np.bincount(inverse, minlength=occupied).astype(np.int64)
        grouped = np.argsort(inverse, kind="stable")
        starts = np.zeros(occupied, dtype=np.int64)
        if occupied > 1:
            starts[1:] = np.cumsum(voxel_counts[:-1], dtype=np.int64)
        selected_voxels = rng.integers(
            0, occupied, size=count, endpoint=False, dtype=np.int64
        )
        offsets = (rng.random(count) * voxel_counts[selected_voxels]).astype(np.int64)
        indices = np.ascontiguousarray(
            grouped[starts[selected_voxels] + offsets], dtype=np.int64
        )
        return indices, {
            "mode": sampling_mode,
            "replacement": "with_replacement_at_voxel_and_member_levels",
            "source_candidate_count": candidate_count,
            "source_occupied_voxel_count": occupied,
            "source_candidates_per_occupied_voxel": _quantiles(voxel_counts),
            "selected_count": count,
            "selected_unique_source_count": int(np.unique(indices).size),
            "selected_occupied_voxel_count": int(np.unique(selected_voxels).size),
            "voxel_size_world": voxel_size,
        }

    if evidence is None:
        raise ContractError(f"{sampling_mode} requires hash-bound proposal evidence")
    angle = np.asarray(evidence.angle_degrees, dtype=np.float64)
    certainty = np.asarray(evidence.certainty, dtype=np.float64)
    pairs = np.asarray(evidence.pair_ordinal, dtype=np.int64)
    expected = (candidate_count,)
    if (
        angle.shape != expected
        or certainty.shape != expected
        or pairs.shape != expected
        or not bool(np.isfinite(angle).all())
        or not bool(np.isfinite(certainty).all())
        or bool(np.any(angle < 0.0))
        or bool(np.any(pairs < 0))
    ):
        raise ContractError("proposal sampling evidence is invalid")

    if sampling_mode == "triangulation_information_mixture":
        information = np.square(np.sin(np.deg2rad(np.clip(angle, 0.0, 90.0))))
        information_total = float(information.sum(dtype=np.float64))
        if not math.isfinite(information_total) or information_total <= 0.0:
            raise ContractError("triangulation-information population has zero information")
        weights = np.full(candidate_count, 1.0 - evidence_fraction, dtype=np.float64)
        weights += evidence_fraction * candidate_count * information / information_total
        indices, effective_population = _weighted_draw(weights, count=count, rng=rng)
        return indices, {
            "mode": sampling_mode,
            "replacement": "with_replacement",
            "source_candidate_count": candidate_count,
            "selected_count": count,
            "selected_unique_source_count": int(np.unique(indices).size),
            "evidence": {
                "quantity": "sin_squared_triangulation_angle_clipped_0_90_degrees",
                "mixture_fraction": evidence_fraction,
                "effective_population": effective_population,
                "triangulation_angle_degrees": _selection_quantiles(angle, indices),
                "triangulation_information": _selection_quantiles(information, indices),
            },
        }

    if sampling_mode == "paired_matcher_support_rank_mixture":
        scores = _group_midrank(certainty, pairs)
        score_name = "within_directed_pair_matcher_midrank"
        consensus: _Consensus | None = None
    elif sampling_mode == "paired_multiview_consensus_rank_mixture":
        consensus = _consensus(xyz, evidence, voxel_size=voxel_size)
        scores = consensus.score
        score_name = "multiview_geometric_consensus_rank"
    else:  # pragma: no cover - guarded by the public builder and type surface
        raise ContractError(f"unsupported proposal sampling mode: {sampling_mode}")

    if evidence_seed < 0:
        raise ContractError("proposal evidence seed must be non-negative")
    raw_indices = rng.integers(
        0, candidate_count, size=count, endpoint=False, dtype=np.int64
    )
    evidence_rng = np.random.default_rng(evidence_seed)
    replacement_count = math.floor(count * evidence_fraction + 0.5)
    replacement_slots = np.sort(
        evidence_rng.choice(count, size=replacement_count, replace=False).astype(np.int64)
    )
    replacement_indices, effective_population = _weighted_draw(
        scores,
        count=replacement_count,
        rng=evidence_rng,
    )
    indices = raw_indices.copy()
    indices[replacement_slots] = replacement_indices
    sampling: dict[str, Any] = {
        "mode": sampling_mode,
        "replacement": "paired_slot_replacement_from_evidence_population",
        "source_candidate_count": candidate_count,
        "selected_count": count,
        "selected_unique_source_count": int(np.unique(indices).size),
        "pairing": {
            "main_rng_contract": "raw_candidate_uniform_indices_then_time_jitter",
            "evidence_seed_schema": "p2g.point_sampling_evidence_seed.v1",
            "evidence_seed_namespace": "paired_matcher_support_rank_mixture",
            "evidence_seed": evidence_seed,
            "preserved_raw_slot_count": count - replacement_count,
            "replaced_slot_count": replacement_count,
            "exact_raw_index_matches_after_replacement": int(
                np.count_nonzero(indices == raw_indices)
            ),
        },
        "evidence": {
            "quantity": score_name,
            "rank_weighted_replacement_fraction": evidence_fraction,
            "effective_population": effective_population,
            "raw_certainty": _selection_quantiles(certainty, indices),
            "score": _selection_quantiles(scores, indices),
        },
    }
    if consensus is not None:
        maximum_reprojection = np.maximum(
            np.asarray(evidence.source_reprojection_pixels, dtype=np.float64),
            np.asarray(evidence.target_reprojection_pixels, dtype=np.float64),
        )
        sampling.update(
            {
                "voxel_size_world": voxel_size,
                "source_occupied_voxel_count": consensus.occupied_voxels,
                "density_contract": (
                    "proposal_rows_retain_provider_density; voxels_measure_only_local_"
                    "multiview_corroboration"
                ),
            }
        )
        sampling["evidence"].update(
            {
                "score_model": (
                    "geomean(geomean(pair_matcher_rank,sin2_angle,pair_inverse_gap_rank,"
                    "pair_inverse_max_reprojection_rank),"
                    "geomean(voxel_pair_support_rank,voxel_camera_support_rank))"
                ),
                "triangulation_angle_degrees": _selection_quantiles(angle, indices),
                "ray_gap_world": _selection_quantiles(
                    np.asarray(evidence.ray_gap_world, dtype=np.float64), indices
                ),
                "maximum_bidirectional_reprojection_pixels": _selection_quantiles(
                    maximum_reprojection, indices
                ),
                "distinct_pair_support": _selection_quantiles(
                    consensus.pair_support, indices
                ),
                "distinct_camera_support": _selection_quantiles(
                    consensus.camera_support, indices
                ),
            }
        )
    return np.ascontiguousarray(indices, dtype=np.int64), sampling


def _sample_point_frame(
    *,
    xyz: np.ndarray[Any, np.dtype[np.float32]],
    rgb: np.ndarray[Any, np.dtype[np.uint8]],
    reference_xyz: np.ndarray[Any, np.dtype[np.float32]],
    center_time: float,
    delta_time: float,
    count: int,
    rng: np.random.Generator,
    velocity_neighbors: int,
    scale_multiplier: float,
    sampling_mode: SamplingMode,
    sampling_voxel_size: float,
    sampling_evidence: PointEvidence | None,
    sampling_evidence_fraction: float,
    sampling_evidence_seed: int,
) -> SampledFrame:
    if (
        xyz.dtype != np.float32
        or reference_xyz.dtype != np.float32
        or rgb.dtype != np.uint8
        or xyz.ndim != 2
        or xyz.shape[1:] != (3,)
        or reference_xyz.ndim != 2
        or reference_xyz.shape[1:] != (3,)
        or rgb.shape != xyz.shape
        or not bool(np.isfinite(xyz).all())
        or not bool(np.isfinite(reference_xyz).all())
        or not math.isfinite(center_time)
        or not math.isfinite(delta_time)
        or delta_time == 0.0
        or count <= 0
        or velocity_neighbors <= 0
        or velocity_neighbors > len(reference_xyz)
        or not math.isfinite(scale_multiplier)
        or scale_multiplier <= 0.0
        or sampling_mode not in _SAMPLING_MODES
        or not math.isfinite(sampling_voxel_size)
        or sampling_voxel_size <= 0.0
        or not math.isfinite(sampling_evidence_fraction)
        or not 0.0 < sampling_evidence_fraction <= 1.0
    ):
        raise ContractError("point-bank frame inputs are invalid")
    try:
        spatial: Any = __import__("scipy.spatial", fromlist=["KDTree"])
        tree_type: Any = spatial.KDTree
    except (ImportError, AttributeError) as exc:
        raise ContractError("point-bank assembly requires scipy.spatial.KDTree") from exc

    indices, sampling = _draw_indices(
        xyz,
        count=count,
        rng=rng,
        sampling_mode=sampling_mode,
        voxel_size=sampling_voxel_size,
        evidence=sampling_evidence,
        evidence_fraction=sampling_evidence_fraction,
        evidence_seed=sampling_evidence_seed,
    )
    means = np.ascontiguousarray(xyz[indices], dtype=np.float32)
    colors = np.ascontiguousarray(rgb[indices].astype(np.float32) / 255.0)
    reference_tree = tree_type(reference_xyz)
    _, neighbor_indices = reference_tree.query(means, k=velocity_neighbors)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]
    velocities = (
        np.mean(reference_xyz[neighbor_indices], axis=1, dtype=np.float64)
        - means.astype(np.float64)
    ) / delta_time

    local_tree = tree_type(means)
    local_distances, _ = local_tree.query(means, k=min(3, len(means)))
    local_distances = np.asarray(local_distances, dtype=np.float64)
    if local_distances.ndim == 1:
        local_distances = local_distances[:, None]
    spacing = np.sqrt(
        np.maximum(np.mean(np.square(local_distances), axis=1), 1.0e-7)
    )
    scales = np.asarray(scale_multiplier * spacing, dtype=np.float32)
    center_times = center_time + (
        rng.random(count, dtype=np.float32) - np.float32(0.5)
    ) * np.float32(delta_time)
    tensors = {
        "means": torch.from_numpy(means),
        "log_scales": torch.from_numpy(
            np.ascontiguousarray(np.repeat(np.log(scales)[:, None], 3, axis=1))
        ),
        "sh0": torch.from_numpy(
            np.ascontiguousarray(((colors - 0.5) / SH_C0)[:, None, :])
        ),
        "center_times": torch.from_numpy(
            np.ascontiguousarray(center_times[:, None], dtype=np.float32)
        ),
        "velocities": torch.from_numpy(
            np.ascontiguousarray(velocities, dtype=np.float32)
        ),
    }
    if any(not bool(value.isfinite().all()) for value in tensors.values()):
        raise ContractError("point-bank frame produced non-finite parameters")
    return SampledFrame(tensors=tensors, sampling=sampling)


def sample_point_frame(
    *,
    xyz: np.ndarray[Any, np.dtype[np.float32]],
    rgb: np.ndarray[Any, np.dtype[np.uint8]],
    reference_xyz: np.ndarray[Any, np.dtype[np.float32]],
    center_time: float,
    delta_time: float,
    count: int,
    rng: np.random.Generator,
    velocity_neighbors: int = 3,
    scale_multiplier: float = 0.1,
    sampling_mode: SamplingMode = "raw_candidate_uniform",
    sampling_voxel_size: float = 0.02,
    sampling_evidence: PointEvidence | None = None,
    sampling_evidence_fraction: float = 0.5,
    sampling_evidence_seed: int = 0,
) -> dict[str, Tensor]:
    """Sample one temporal bank using explicit KNN motion and local scale rules."""

    return _sample_point_frame(
        xyz=xyz,
        rgb=rgb,
        reference_xyz=reference_xyz,
        center_time=center_time,
        delta_time=delta_time,
        count=count,
        rng=rng,
        velocity_neighbors=velocity_neighbors,
        scale_multiplier=scale_multiplier,
        sampling_mode=sampling_mode,
        sampling_voxel_size=sampling_voxel_size,
        sampling_evidence=sampling_evidence,
        sampling_evidence_fraction=sampling_evidence_fraction,
        sampling_evidence_seed=sampling_evidence_seed,
    ).tensors


def _tensor_dtype(value: Tensor) -> str:
    if value.dtype == torch.float32:
        return "float32"
    if value.dtype == torch.int64:
        return "int64"
    raise ContractError(f"initialization has unsupported tensor dtype: {value.dtype}")


def _validate_output_tensors(tensors: dict[str, Tensor]) -> int:
    catalog = frozenset(tensors)
    if catalog != _OUTPUT_PLANES:
        raise ContractError(
            "initialization tensor catalog mismatch: "
            f"missing={sorted(_OUTPUT_PLANES - catalog)}, "
            f"unknown={sorted(catalog - _OUTPUT_PLANES)}"
        )
    means = tensors["means"]
    count = int(means.shape[0]) if means.ndim == 2 else 0
    shapes = {
        "means": (count, 3),
        "log_scales": (count, 3),
        "quaternions": (count, 4),
        "opacity_logits": (count, 1),
        "sh0": (count, 1, 3),
        "center_times": (count, 1),
        "duration_logits": (count, 1),
        "velocities": (count, 3),
        "runtime_ids": (count,),
    }
    if count <= 0:
        raise ContractError("initialization output population must be non-empty")
    for name in sorted(tensors):
        value = tensors[name]
        expected_dtype = torch.int64 if name == "runtime_ids" else torch.float32
        if (
            tuple(value.shape) != shapes[name]
            or value.dtype != expected_dtype
            or value.device.type != "cpu"
            or not value.is_contiguous()
            or (value.is_floating_point() and not bool(value.isfinite().all()))
        ):
            raise ContractError(f"initialization output plane is invalid: {name}")
    if int(torch.unique(tensors["runtime_ids"]).numel()) != count:
        raise ContractError("initialization runtime IDs must be unique")
    if not bool((torch.linalg.vector_norm(tensors["quaternions"], dim=1) > 1.0e-12).all()):
        raise ContractError("initialization contains a zero quaternion")
    return count


def canonical_initialization_tensor_sha256(tensors: dict[str, Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and little-endian row-major values."""

    _validate_output_tensors(tensors)
    digest = hashlib.sha256()
    digest.update(CANONICAL_TENSOR_SCHEMA.encode("ascii") + b"\0")
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        array = value.numpy()
        descriptor = {
            "name": name,
            "dtype": _tensor_dtype(value),
            "shape": list(value.shape),
        }
        digest.update(canonical_json_bytes(descriptor))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _canonicalize_safetensors_header(path: Path) -> None:
    """Sort the JSON header so equal tensor assets have equal container bytes."""

    try:
        with path.open("r+b") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ContractError("initialization Safetensors header is truncated")
            header_length = int.from_bytes(prefix, byteorder="little", signed=False)
            if not 2 <= header_length <= 100_000_000:
                raise ContractError("initialization Safetensors header length is invalid")
            encoded = stream.read(header_length)
            if len(encoded) != header_length:
                raise ContractError("initialization Safetensors header is truncated")
            decoded: Any = json.loads(encoded.rstrip(b" "))
            if not isinstance(decoded, dict):
                raise ContractError("initialization Safetensors header is not an object")
            canonical = json.dumps(
                decoded,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            if len(canonical) > header_length:
                raise ContractError("canonical Safetensors header exceeds its container extent")
            stream.seek(8)
            stream.write(canonical)
            stream.write(b" " * (header_length - len(canonical)))
            stream.flush()
            os.fsync(stream.fileno())
    except ContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(
            f"cannot canonicalize initialization Safetensors header: {exc}"
        ) from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_point_bank_initialization(
    output: Path,
    *,
    proposal_sequence: Path,
    tensor_cache: Path,
    num_gaussians: int = 500_000,
    seed: int = 0,
    velocity_neighbors: int = 3,
    scale_multiplier: float = 0.1,
    sampling_mode: SamplingMode = "paired_multiview_consensus_rank_mixture",
    sampling_voxel_size: float = 0.02,
    sampling_evidence_fraction: float = 0.5,
    opacity: float = 0.5,
    duration_seconds: float = 0.1,
    duration_min_seconds: float = 1.0 / 600.0,
    duration_max_seconds: float = 1.0,
    time_offset_seconds: float = 0.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build an append-only trainer input from one verified proposal sequence."""

    destination = output.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise OutputExistsError(f"refusing to overwrite Gaussian initialization: {destination}")
    if (
        type(num_gaussians) is not int
        or num_gaussians <= 0
        or type(seed) is not int
        or not 0 <= seed < 2**64
        or type(velocity_neighbors) is not int
        or velocity_neighbors <= 0
        or not math.isfinite(scale_multiplier)
        or scale_multiplier <= 0.0
        or sampling_mode not in _SAMPLING_MODES
        or not math.isfinite(sampling_voxel_size)
        or sampling_voxel_size <= 0.0
        or not math.isfinite(sampling_evidence_fraction)
        or not 0.0 < sampling_evidence_fraction <= 1.0
        or not math.isfinite(opacity)
        or not 0.0 < opacity < 1.0
        or not all(
            math.isfinite(value)
            for value in (
                duration_seconds,
                duration_min_seconds,
                duration_max_seconds,
                time_offset_seconds,
            )
        )
        or not 0.0 < duration_min_seconds < duration_seconds < duration_max_seconds
    ):
        raise ContractError("point-bank initialization configuration is invalid")

    collection = _load_collection(proposal_sequence)
    cache_binding = _load_cache_binding(tensor_cache, collection=collection)
    frame_count = len(collection.frame_ids)
    if num_gaussians < frame_count:
        raise ContractError("Gaussian budget must allocate at least one slot per frame")
    per_frame = num_gaussians // frame_count
    assembled_count = per_frame * frame_count
    rng = np.random.default_rng(seed)
    accumulated: dict[str, list[Tensor]] = {
        "means": [],
        "log_scales": [],
        "sh0": [],
        "center_times": [],
        "velocities": [],
    }
    frame_receipts: list[dict[str, Any]] = []
    seed_mode: SamplingMode = (
        "paired_matcher_support_rank_mixture"
        if sampling_mode == "paired_multiview_consensus_rank_mixture"
        else sampling_mode
    )
    for position, frame_id in enumerate(collection.frame_ids):
        reference_position = position + 1 if position + 1 < frame_count else position - 1
        reference_id = collection.frame_ids[reference_position]
        row = collection.rows[frame_id]
        reference_row = collection.rows[reference_id]
        source = _read_point_ply(
            cast(Path, row["_points_path"]),
            expected_sha256=cast(str, row["point_ply_sha256"]),
            expected_count=cast(int, row["admitted_count"]),
            frame_id=frame_id,
            provenance_sha256=cast(str, row["provenance_sha256"]),
        )
        reference = _read_point_ply(
            cast(Path, reference_row["_points_path"]),
            expected_sha256=cast(str, reference_row["point_ply_sha256"]),
            expected_count=cast(int, reference_row["admitted_count"]),
            frame_id=reference_id,
            provenance_sha256=cast(str, reference_row["provenance_sha256"]),
        )
        evidence = (
            _load_evidence(
                cast(Path, row["_provenance_path"]),
                frame_id=frame_id,
                expected_sha256=source.provenance_sha256,
                expected_points=len(source.xyz),
                require_multiview=(
                    sampling_mode == "paired_multiview_consensus_rank_mixture"
                ),
            )
            if sampling_mode in _EVIDENCE_MODES
            else None
        )
        delta_time = collection.timestamps[reference_id] - collection.timestamps[frame_id]
        sampled = _sample_point_frame(
            xyz=source.xyz,
            rgb=source.rgb,
            reference_xyz=reference.xyz,
            center_time=collection.timestamps[frame_id],
            delta_time=delta_time,
            count=per_frame,
            rng=rng,
            velocity_neighbors=velocity_neighbors,
            scale_multiplier=scale_multiplier,
            sampling_mode=sampling_mode,
            sampling_voxel_size=sampling_voxel_size,
            sampling_evidence=evidence,
            sampling_evidence_fraction=sampling_evidence_fraction,
            sampling_evidence_seed=_evidence_seed(
                seed, frame_id=frame_id, sampling_mode=seed_mode
            ),
        )
        for name, value in sampled.tensors.items():
            accumulated[name].append(value)
        frame_receipts.append(
            {
                "frame_id": frame_id,
                "timestamp_seconds": collection.timestamps[frame_id],
                "point_ply_sha256": source.ply_sha256,
                "provenance_sha256": source.provenance_sha256,
                "source_candidate_count": len(source.xyz),
                "selected_count": per_frame,
                "reference_frame_id": reference_id,
                "delta_time_seconds": delta_time,
                "sampling": sampled.sampling,
            }
        )
        if progress is not None:
            progress(
                f"FRAME [{position + 1}/{frame_count}] {frame_id:06d}: "
                f"{len(source.xyz):,} proposals -> {per_frame:,} Gaussian slots"
            )

    tensors = {
        name: torch.cat(parts, dim=0).contiguous()
        for name, parts in accumulated.items()
    }
    quaternions = torch.zeros((assembled_count, 4), dtype=torch.float32)
    quaternions[:, 0] = 1.0
    duration_fraction = (duration_seconds - duration_min_seconds) / (
        duration_max_seconds - duration_min_seconds
    )
    duration_logit = math.log(duration_fraction / (1.0 - duration_fraction))
    tensors.update(
        {
            "quaternions": quaternions,
            "opacity_logits": torch.full(
                (assembled_count, 1),
                math.log(opacity / (1.0 - opacity)),
                dtype=torch.float32,
            ),
            "duration_logits": torch.full(
                (assembled_count, 1), duration_logit, dtype=torch.float32
            ),
            "runtime_ids": torch.arange(assembled_count, dtype=torch.int64),
        }
    )
    _validate_output_tensors(tensors)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        tensor_path = stage / TENSOR_FILENAME
        save_file(
            tensors,
            str(tensor_path),
            metadata={
                "schema_version": INITIALIZATION_SCHEMA,
                "builder_receipt_schema": RECEIPT_SCHEMA,
                "proposal_sequence_sha256": collection.manifest_sha256,
                "tensor_cache_manifest_sha256": collection.tensor_cache_manifest_sha256,
                "sampling_mode": sampling_mode,
                "duration_min_seconds": repr(duration_min_seconds),
                "duration_max_seconds": repr(duration_max_seconds),
                "duration_seconds": repr(duration_seconds),
                "time_offset_seconds": repr(time_offset_seconds),
                "higher_order_sh": "loader_zero_fill_at_configured_degree",
            },
        )
        _canonicalize_safetensors_header(tensor_path)
        scale_values = torch.exp(tensors["log_scales"][:, 0]).numpy()
        velocity_norms = torch.linalg.vector_norm(tensors["velocities"], dim=1).numpy()
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "status": "COMPLETE",
            "trainer_eligible": True,
            "source": {
                "proposal_sequence": {
                    "schema": ROMA_POINT_SEQUENCE_SCHEMA,
                    "manifest_sha256": collection.manifest_sha256,
                    "provider_identity_sha256": collection.provider_identity_sha256,
                    "frame_ids": list(collection.frame_ids),
                },
                "tensor_cache": cache_binding,
            },
            "policy": {
                "sampling_mode": sampling_mode,
                "sampling_voxel_size_world": (
                    sampling_voxel_size
                    if sampling_mode
                    in {
                        "occupied_voxel_uniform",
                        "paired_multiview_consensus_rank_mixture",
                    }
                    else None
                ),
                "sampling_evidence_fraction": (
                    sampling_evidence_fraction
                    if sampling_mode in _EVIDENCE_MODES
                    else None
                ),
                "seed": seed,
                "requested_gaussians": num_gaussians,
                "gaussians_per_frame": per_frame,
                "assembled_gaussians": assembled_count,
                "velocity": f"adjacent_frame_scipy_kdtree_mean_k{velocity_neighbors}",
                "scale": "sampled_frame_knn3_rms_times_multiplier",
                "scale_multiplier": scale_multiplier,
                "opacity": opacity,
                "duration_seconds": duration_seconds,
                "duration_min_seconds": duration_min_seconds,
                "duration_max_seconds": duration_max_seconds,
                "time_offset_seconds": time_offset_seconds,
                "higher_order_sh": "loader_zero_fill_at_configured_degree",
                "runtime_ids": "contiguous_zero_based_v1",
            },
            "population": {
                "frame_count": frame_count,
                "source_candidate_count": sum(
                    int(row["source_candidate_count"]) for row in frame_receipts
                ),
                "requested_gaussians": num_gaussians,
                "assembled_gaussians": assembled_count,
                "discarded_budget_remainder": num_gaussians - assembled_count,
            },
            "statistics": {
                "center_time_seconds": _quantiles(tensors["center_times"].numpy()),
                "scale_world": _quantiles(scale_values),
                "velocity_norm_world_per_second": _quantiles(velocity_norms),
            },
            "frames": frame_receipts,
            "tensor": {
                "path": TENSOR_FILENAME,
                "container_sha256": sha256_file(tensor_path),
                "canonical_digest_schema": CANONICAL_TENSOR_SCHEMA,
                "canonical_tensor_sha256": canonical_initialization_tensor_sha256(tensors),
                "planes": {
                    name: {
                        "dtype": _tensor_dtype(value),
                        "shape": list(value.shape),
                    }
                    for name, value in sorted(tensors.items())
                },
            },
            "limitations": [
                (
                    "Adjacent-frame KNN velocity is an explicit initialization prior, not "
                    "a persistent cross-frame identity or learned scene flow."
                ),
                (
                    "Multi-view evidence changes sampling probability only; it does not "
                    "certify ground-truth geometry."
                ),
                (
                    "The output contains SH0 appearance; configured higher-order SH starts "
                    "at zero and is learned from pixels."
                ),
            ],
            "claim_boundary": (
                "This receipt proves deterministic, hash-bound initialization assembly from "
                "the declared first-party proposal sequence. It does not prove downstream "
                "training quality, convergence, or dataset redistribution rights."
            ),
        }
        receipt["logical_sha256"] = sha256_json(receipt)
        validate_payload("gaussian_initialization_receipt", receipt)
        write_new_json(stage / RECEIPT_FILENAME, receipt)
        _fsync_directory(stage)
        if destination.exists() or destination.is_symlink():
            raise OutputExistsError(
                f"Gaussian initialization destination appeared during publication: {destination}"
            )
        os.rename(stage, destination)
        _fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return receipt
