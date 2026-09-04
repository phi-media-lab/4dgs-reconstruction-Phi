"""Portable, hash-closed 4D Gaussian asset bundles.

An asset is an inference artifact, not a training checkpoint. It contains no
pickle, optimizer state, RNG state, dataset path, or machine path. The model is
stored in Safetensors and every published byte is covered by a manifest.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json
from p2g.errors import ContractError, OutputExistsError
from p2g.schema import validate_payload
from p2g.training.config import RendererConfig
from p2g.training.model import DynamicGaussianModel

ASSET_BUNDLE_SCHEMA = "p2g.asset_bundle.v1"
ASSET_MANIFEST_SCHEMA = "p2g.asset_bundle_manifest.v1"
ASSET_MODEL_SCHEMA = "p2g.asset_model.v1"
MODEL_EQUATION_VERSION = "p2g.linear_motion_gaussian_gate.v1"
RENDERER_ABI = "p2g.gsplat_rocm.v1"
MODEL_FILENAME = "model.safetensors"
METADATA_FILENAME = "asset.json"
MANIFEST_FILENAME = "manifest.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")
Redistribution = Literal["allowed", "restricted", "review_required"]


@dataclass(frozen=True)
class AssetBundleSpec:
    """Scene and rights metadata that cannot be inferred from model tensors."""

    valid_time_start_seconds: float
    valid_time_stop_seconds: float
    reference_time_seconds: float
    world_coordinate_convention: str
    world_unit: str
    calibration_scale: float
    photometric_space: Literal["linear_rgb", "srgb_reference_profile"]
    default_sh_degree: int
    final_step: int
    source_bundle_digests: dict[str, str]
    producer_version: str
    producer_git_revision: str
    dependency_identities: dict[str, str]
    asset_license: str
    source_data_license: str
    redistribution: Redistribution
    provenance_summary: str

    def validate(self, *, model: DynamicGaussianModel) -> None:
        times = (
            self.valid_time_start_seconds,
            self.valid_time_stop_seconds,
            self.reference_time_seconds,
        )
        if not all(math.isfinite(value) for value in times):
            raise ContractError("asset time values must be finite")
        if (
            not (
                self.valid_time_start_seconds
                <= self.reference_time_seconds
                <= self.valid_time_stop_seconds
            )
            or self.valid_time_start_seconds == self.valid_time_stop_seconds
        ):
            raise ContractError("asset reference time must lie inside a non-empty interval")
        if not math.isfinite(self.calibration_scale) or self.calibration_scale <= 0.0:
            raise ContractError("asset calibration scale must be positive and finite")
        if not 0 <= self.default_sh_degree <= model.max_sh_degree:
            raise ContractError("asset default SH degree is outside the model catalog")
        if self.final_step < 0:
            raise ContractError("asset final training step cannot be negative")
        if not _GIT_REVISION.fullmatch(self.producer_git_revision):
            raise ContractError("asset producer Git revision must be a full lowercase hash")
        if not self.source_bundle_digests:
            raise ContractError("asset must bind at least one source bundle digest")
        for name, digest in self.source_bundle_digests.items():
            if not name or not _SHA256.fullmatch(digest):
                raise ContractError("asset source bundle identities must be named SHA-256 values")
        if not self.dependency_identities or any(
            not name or not value for name, value in self.dependency_identities.items()
        ):
            raise ContractError("asset dependency identities must be non-empty")
        required_text = (
            self.world_coordinate_convention,
            self.world_unit,
            self.producer_version,
            self.asset_license,
            self.source_data_license,
            self.provenance_summary,
        )
        if any(not value.strip() for value in required_text):
            raise ContractError(
                "asset convention, producer, rights, and provenance text is required"
            )
        if self.redistribution not in {"allowed", "restricted", "review_required"}:
            raise ContractError("asset redistribution state is invalid")


@dataclass(frozen=True)
class VerifiedAssetBundle:
    root: Path
    model: DynamicGaussianModel
    metadata: dict[str, Any]
    manifest: dict[str, Any]


def _flush_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _flush_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tensor_catalog(tensors: dict[str, Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "dtype": str(tensors[name].dtype),
            "shape": list(tensors[name].shape),
            "bytes": tensors[name].numel() * tensors[name].element_size(),
        }
        for name in sorted(tensors)
    ]


def _model_tensors(model: DynamicGaussianModel) -> dict[str, Tensor]:
    tensors = {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in model.state_dict().items()
    }
    for name, value in tensors.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ContractError(f"asset model contains a non-finite tensor: {name}")
    return tensors


def _write_canonical_safetensors(
    path: Path,
    tensors: dict[str, Tensor],
    *,
    metadata: dict[str, str],
) -> None:
    """Write the small Safetensors subset used by AssetBundle v1 deterministically.

    The upstream serializer intentionally does not promise metadata-map/header
    ordering. AssetBundle needs byte-stable archives, so this writer fixes key
    order, compact JSON, eight-byte header padding, and tensor payload order.
    """

    dtype_names = {
        torch.float32: "F32",
        torch.int64: "I64",
    }
    header: dict[str, Any] = {"__metadata__": dict(sorted(metadata.items()))}
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        try:
            dtype = dtype_names[tensor.dtype]
        except KeyError as exc:
            raise ContractError(
                f"asset tensor dtype is unsupported: {name}={tensor.dtype}"
            ) from exc
        byte_count = tensor.numel() * tensor.element_size()
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + byte_count],
        }
        offset += byte_count
    encoded_header = json.dumps(
        header,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded_header += b" " * (-len(encoded_header) % 8)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded_header)))
        stream.write(encoded_header)
        for name in sorted(tensors):
            stream.write(tensors[name].numpy().tobytes(order="C"))
        stream.flush()
        os.fsync(stream.fileno())


def _reject_machine_paths(value: Any, *, location: str = "asset") -> None:
    if isinstance(value, str):
        if value.startswith(("/", "~/", "file://")) or "/home/" in value or "/mnt/" in value:
            raise ContractError(f"portable {location} contains a machine path")
        return
    if isinstance(value, dict):
        for key, item in cast(dict[str, Any], value).items():
            _reject_machine_paths(item, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(cast(list[Any], value)):
            _reject_machine_paths(item, location=f"{location}[{index}]")


def _asset_metadata(
    *,
    model: DynamicGaussianModel,
    tensors: dict[str, Tensor],
    model_bytes: int,
    model_sha256: str,
    spec: AssetBundleSpec,
    renderer: RendererConfig,
) -> dict[str, Any]:
    persistence = "learned" if model.persistence_enabled else "off"
    gate_scale = float(model.gate_logit_scale.detach().cpu())
    return {
        "schema_version": ASSET_BUNDLE_SCHEMA,
        "format_version": {"major": 1, "minor": 0},
        "model": {
            "file": MODEL_FILENAME,
            "schema_version": ASSET_MODEL_SCHEMA,
            "bytes": model_bytes,
            "sha256": model_sha256,
            "gaussian_count": model.count,
            "tensor_count": len(tensors),
            "tensors": _tensor_catalog(tensors),
        },
        "equations": {
            "version": MODEL_EQUATION_VERSION,
            "motion": "mean(t) = mean + velocity * (t - center_time)",
            "duration": ("sigma = sigma_min + (sigma_max - sigma_min) * sigmoid(duration_logit)"),
            "transient": "exp(-0.5 * ((t - center_time) / sigma)^2)",
            "persistence": persistence,
            "gate_logit_scale": gate_scale,
            "activation": (
                "persistent + (1 - persistent) * transient"
                if model.persistence_enabled
                else "transient"
            ),
            "persistent": (
                "sigmoid(gate_logit_scale * persistence_logit)"
                if model.persistence_enabled
                else "disabled"
            ),
        },
        "appearance": {
            "representation": "real_spherical_harmonics",
            "convention": "gsplat_real_sh_v1",
            "max_sh_degree": model.max_sh_degree,
            "default_sh_degree": spec.default_sh_degree,
            "coefficient_color_space": "linear_rgb",
            "output_photometric_space": spec.photometric_space,
        },
        "time": {
            "unit": "seconds",
            "valid_interval": [
                spec.valid_time_start_seconds,
                spec.valid_time_stop_seconds,
            ],
            "reference_time": spec.reference_time_seconds,
        },
        "coordinates": {
            "world_convention": spec.world_coordinate_convention,
            "world_unit": spec.world_unit,
            "calibration_scale": spec.calibration_scale,
            "extrinsic": "world_to_camera",
            "camera_axes": "opencv_x_right_y_down_z_forward",
        },
        "camera": {
            "model": "pinhole",
            "distortion": "pre-undistorted",
            "intrinsic_matrix": "3x3_pixel_center",
            "extrinsic_matrix": "4x4_world_to_camera",
        },
        "renderer": {
            "abi": RENDERER_ABI,
            "backend": renderer.backend,
            "required_architecture": "gfx942" if renderer.require_gfx942 else "unspecified",
            "near_plane": renderer.near_plane,
            "far_plane": renderer.far_plane,
            "eps2d": renderer.eps2d,
            "radius_clip": renderer.radius_clip,
            "tile_size": renderer.tile_size,
            "packed": renderer.packed,
            "background_linear_rgb": list(renderer.background),
            "clamp_rgb": renderer.clamp_rgb,
        },
        "training": {
            "final_step": spec.final_step,
            "source_bundle_digests": dict(sorted(spec.source_bundle_digests.items())),
        },
        "producer": {
            "name": "pixel4dgs",
            "version": spec.producer_version,
            "git_revision": spec.producer_git_revision,
            "dependencies": dict(sorted(spec.dependency_identities.items())),
        },
        "rights": {
            "asset_license": spec.asset_license,
            "source_data_license": spec.source_data_license,
            "redistribution": spec.redistribution,
            "provenance_summary": spec.provenance_summary,
        },
    }


def write_asset_bundle(
    destination: Path,
    *,
    model: DynamicGaussianModel,
    spec: AssetBundleSpec,
    renderer: RendererConfig,
) -> Path:
    """Atomically publish a portable AssetBundle v1 without overwriting."""

    destination = destination.expanduser().resolve()
    if destination.exists():
        raise OutputExistsError(f"refusing to overwrite asset bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    spec.validate(model=model)
    tensors = _model_tensors(model)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        model_path = temporary / MODEL_FILENAME
        _write_canonical_safetensors(
            model_path,
            tensors,
            metadata={
                "schema_version": ASSET_MODEL_SCHEMA,
                "equation_version": MODEL_EQUATION_VERSION,
                "persistence": "learned" if model.persistence_enabled else "off",
                "gate_logit_scale": repr(float(model.gate_logit_scale.detach().cpu())),
            },
        )
        _flush_file(model_path)
        metadata = _asset_metadata(
            model=model,
            tensors=tensors,
            model_bytes=model_path.stat().st_size,
            model_sha256=sha256_file(model_path),
            spec=spec,
            renderer=renderer,
        )
        _reject_machine_paths(metadata)
        validate_payload("asset_bundle", metadata)
        metadata_path = temporary / METADATA_FILENAME
        metadata_path.write_bytes(canonical_json_bytes(metadata))
        _flush_file(metadata_path)
        files = [
            {
                "path": METADATA_FILENAME,
                "bytes": metadata_path.stat().st_size,
                "sha256": sha256_file(metadata_path),
            },
            {
                "path": MODEL_FILENAME,
                "bytes": model_path.stat().st_size,
                "sha256": sha256_file(model_path),
            },
        ]
        manifest = {
            "schema_version": ASSET_MANIFEST_SCHEMA,
            "bundle_id": sha256_json(files),
            "files": files,
        }
        manifest_path = temporary / MANIFEST_FILENAME
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        _flush_file(manifest_path)
        _flush_directory(temporary)
        if destination.exists():
            raise OutputExistsError(f"asset destination appeared during publication: {destination}")
        os.rename(temporary, destination)
        _flush_directory(destination.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _verify_manifest(root: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != ASSET_MANIFEST_SCHEMA:
        raise ContractError("asset manifest schema is unsupported")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ContractError("asset manifest files must be an array")
    raw_file_list = cast(list[Any], raw_files)
    if len(raw_file_list) != 2 or not all(isinstance(item, dict) for item in raw_file_list):
        raise ContractError("asset manifest must contain exactly model and metadata")
    files = cast(list[dict[str, Any]], raw_file_list)
    expected_names = {METADATA_FILENAME, MODEL_FILENAME}
    names = {item.get("path") for item in files}
    if names != expected_names:
        raise ContractError("asset manifest file catalog is invalid")
    if manifest.get("bundle_id") != sha256_json(files):
        raise ContractError("asset manifest bundle ID is invalid")
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names | {MANIFEST_FILENAME}:
        raise ContractError("asset bundle contains an undeclared or missing path")
    for item in files:
        name = cast(str, item["path"])
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"asset member is missing or unsafe: {name}")
        if item.get("bytes") != path.stat().st_size or item.get("sha256") != sha256_file(path):
            raise ContractError(f"asset member digest mismatch: {name}")


def load_asset_bundle(root: Path) -> VerifiedAssetBundle:
    """Verify every asset byte and load the safe inference model on CPU."""

    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ContractError(f"asset bundle is not a regular directory: {root}")
    manifest = _read_json_object(root / MANIFEST_FILENAME, label="asset manifest")
    _verify_manifest(root, manifest)
    metadata = _read_json_object(root / METADATA_FILENAME, label="asset metadata")
    validate_payload("asset_bundle", metadata)
    _reject_machine_paths(metadata)
    model_path = root / MODEL_FILENAME
    model_record = cast(dict[str, Any], metadata["model"])
    if (
        model_record.get("file") != MODEL_FILENAME
        or model_record.get("bytes") != model_path.stat().st_size
        or model_record.get("sha256") != sha256_file(model_path)
    ):
        raise ContractError("asset metadata does not bind the model file")
    try:
        safetensors_module: Any = importlib.import_module("safetensors")
        safetensors_torch: Any = importlib.import_module("safetensors.torch")
        load_file_api: Any = safetensors_torch.load_file
        tensors = cast(dict[str, Tensor], load_file_api(str(model_path), device="cpu"))
        safe_open_api: Any = safetensors_module.safe_open
        with safe_open_api(model_path, framework="pt", device="cpu") as stream:
            stream_api: Any = stream
            tensor_metadata = cast(dict[str, str], stream_api.metadata() or {})
    except Exception as exc:
        raise ContractError(f"cannot load asset Safetensors model: {exc}") from exc
    if tensor_metadata.get("schema_version") != ASSET_MODEL_SCHEMA:
        raise ContractError("asset tensor schema is unsupported")
    if tensor_metadata.get("equation_version") != MODEL_EQUATION_VERSION:
        raise ContractError("asset tensor equation version is unsupported")
    equations = cast(dict[str, Any], metadata["equations"])
    appearance = cast(dict[str, Any], metadata["appearance"])
    persistence = cast(str, equations["persistence"])
    if persistence not in {"off", "learned"} or tensor_metadata.get("persistence") != persistence:
        raise ContractError("asset persistence semantics disagree between files")
    gate_scale = float(cast(float, equations["gate_logit_scale"]))
    tensor_gate_scale = tensor_metadata.get("gate_logit_scale")
    if tensor_gate_scale is None or float(tensor_gate_scale) != gate_scale:
        raise ContractError("asset gate scale disagrees between files")
    catalog = _tensor_catalog(tensors)
    if model_record.get("tensors") != catalog or model_record.get("tensor_count") != len(catalog):
        raise ContractError("asset tensor catalog disagrees with Safetensors")
    model = DynamicGaussianModel.from_checkpoint_state(
        tensors,
        persistence=persistence == "learned",
        gate_logit_scale=gate_scale,
    )
    if (
        model.count != model_record.get("gaussian_count")
        or model.max_sh_degree != appearance.get("max_sh_degree")
        or not 0 <= cast(int, appearance["default_sh_degree"]) <= model.max_sh_degree
    ):
        raise ContractError("asset model shape disagrees with metadata")
    return VerifiedAssetBundle(root=root, model=model, metadata=metadata, manifest=manifest)


def asset_summary(bundle: VerifiedAssetBundle) -> dict[str, Any]:
    metadata = bundle.metadata
    return {
        "schema_version": metadata["schema_version"],
        "bundle_id": bundle.manifest["bundle_id"],
        "gaussian_count": metadata["model"]["gaussian_count"],
        "tensor_count": metadata["model"]["tensor_count"],
        "model_sha256": metadata["model"]["sha256"],
        "equation_version": metadata["equations"]["version"],
        "time": metadata["time"],
        "camera": metadata["camera"],
        "renderer": metadata["renderer"],
        "rights": metadata["rights"],
        "status": "PASS",
    }
