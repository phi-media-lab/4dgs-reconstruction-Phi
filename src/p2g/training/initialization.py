"""Safe, explicit construction of the public Gaussian initialization state."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, cast

import torch
from torch import Tensor

from p2g.errors import ContractError
from p2g.training.config import InitializationConfig

INITIALIZATION_SCHEMA = "p2g.gaussian_initialization.v1"

_REQUIRED_FILE_TENSORS = frozenset(
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
_OPTIONAL_FILE_TENSORS = frozenset({"sh_rest"})
_BUILDER_RECEIPT_SCHEMA = "p2g.gaussian_initialization_receipt.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _SafeTensorReader(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def metadata(self) -> dict[str, str] | None: ...

    def keys(self) -> list[str]: ...


def _tensor(value: object, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ContractError(f"GaussianInit.{name} must be a tensor")
    return value


def _portable_source(source: object) -> None:
    if not isinstance(source, dict) or not source:
        raise ContractError("GaussianInit.source must be a non-empty string map")
    for key, value in cast(dict[Any, Any], source).items():
        if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
            raise ContractError("GaussianInit.source must be a non-empty string map")
        if (
            value.startswith(("/", "~/", "file://"))
            or "/home/" in value
            or "/mnt/" in value
        ):
            raise ContractError("GaussianInit.source must not contain a machine path")


def _float_plane(value: object, *, name: str, shape: tuple[int, ...]) -> Tensor:
    value = _tensor(value, name=name)
    if tuple(value.shape) != shape:
        raise ContractError(f"GaussianInit.{name} must have shape {shape}")
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ContractError(f"GaussianInit.{name} must be CPU float32")
    if not value.is_contiguous():
        raise ContractError(f"GaussianInit.{name} must be contiguous")
    if not bool(torch.isfinite(value).all()):
        raise ContractError(f"GaussianInit.{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class GaussianInit:
    """Fully resolved struct-of-arrays state used to construct the trainable model."""

    means: Tensor
    log_scales: Tensor
    quaternions: Tensor
    opacity_logits: Tensor
    sh0: Tensor
    sh_rest: Tensor
    center_times: Tensor
    duration_logits: Tensor
    velocities: Tensor
    persistence_logits: Tensor
    duration_min_seconds: Tensor
    duration_max_seconds: Tensor
    runtime_ids: Tensor
    source: dict[str, str]

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @property
    def max_sh_degree(self) -> int:
        coefficient_count = int(self.sh_rest.shape[1]) + 1
        root = math.isqrt(coefficient_count)
        if root * root != coefficient_count:
            raise ContractError("GaussianInit SH coefficient count must be a perfect square")
        return root - 1

    def validate(self) -> None:
        means = _tensor(self.means, name="means")
        if means.ndim != 2:
            raise ContractError("GaussianInit.means must be a rank-two tensor")
        count = int(means.shape[0])
        if count <= 0:
            raise ContractError("GaussianInit must contain at least one Gaussian")
        expected = {
            "means": (count, 3),
            "log_scales": (count, 3),
            "quaternions": (count, 4),
            "opacity_logits": (count, 1),
            "sh0": (count, 1, 3),
            "center_times": (count, 1),
            "duration_logits": (count, 1),
            "velocities": (count, 3),
            "persistence_logits": (count, 1),
            "duration_min_seconds": (count, 1),
            "duration_max_seconds": (count, 1),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            _float_plane(value, name=name, shape=shape)
        sh_rest = _tensor(self.sh_rest, name="sh_rest")
        if sh_rest.ndim != 3:
            raise ContractError("GaussianInit.sh_rest must be a rank-three tensor")
        _float_plane(
            sh_rest,
            name="sh_rest",
            shape=(count, int(sh_rest.shape[1]), 3),
        )
        if not 0 <= self.max_sh_degree <= 3:
            raise ContractError("GaussianInit SH degree must be in [0, 3]")
        runtime_ids = _tensor(self.runtime_ids, name="runtime_ids")
        if tuple(runtime_ids.shape) != (count,):
            raise ContractError(f"GaussianInit.runtime_ids must have shape ({count},)")
        if (
            runtime_ids.device.type != "cpu"
            or runtime_ids.dtype != torch.int64
            or not runtime_ids.is_contiguous()
        ):
            raise ContractError("GaussianInit.runtime_ids must be contiguous CPU int64")
        unique_ids = cast(Tensor, torch.unique(runtime_ids))  # pyright: ignore[reportUnknownMemberType]
        if int(unique_ids.numel()) != count:
            raise ContractError("GaussianInit.runtime_ids must be unique")
        quaternion_norm = self.quaternions.square().sum(dim=1).sqrt()
        if not bool((quaternion_norm > 1.0e-12).all()):
            raise ContractError("GaussianInit contains a zero quaternion")
        if not bool(
            (
                (self.duration_min_seconds > 0.0)
                & (self.duration_min_seconds < self.duration_max_seconds)
            ).all()
        ):
            raise ContractError("GaussianInit duration bounds must satisfy 0 < min < max")
        _portable_source(self.source)


def _loaded_plane(
    tensors: dict[str, Tensor],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    value = tensors[name]
    if tuple(value.shape) != shape or value.dtype != dtype or value.device.type != "cpu":
        raise ContractError(
            f"initialization tensor {name} must be CPU {dtype} with shape {shape}"
        )
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ContractError(f"initialization tensor {name} must be finite")
    return value.detach().clone().contiguous()


def _regular_safetensors_path(path: Path) -> Path:
    if not path.is_absolute():
        raise ContractError("initialization path must be absolute")
    if path.suffix != ".safetensors":
        raise ContractError("initialization must use the .safetensors suffix")
    if not path.is_file() or path.is_symlink():
        raise ContractError("initialization must be a regular non-symlink file")
    return path.resolve(strict=True)


def _metadata_float(metadata: dict[str, str], name: str) -> float:
    try:
        value = float(metadata[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"initialization builder metadata {name} is invalid") from exc
    if not math.isfinite(value):
        raise ContractError(f"initialization builder metadata {name} must be finite")
    return value


def _validate_builder_metadata(
    metadata: dict[str, str],
    tensors: dict[str, Tensor],
    config: InitializationConfig,
) -> None:
    if "builder_receipt_schema" not in metadata:
        return
    if metadata["builder_receipt_schema"] != _BUILDER_RECEIPT_SCHEMA:
        raise ContractError("initialization declares an unsupported builder receipt schema")
    required = {
        "proposal_sequence_sha256",
        "tensor_cache_manifest_sha256",
        "sampling_mode",
        "duration_min_seconds",
        "duration_max_seconds",
        "duration_seconds",
        "time_offset_seconds",
        "higher_order_sh",
    }
    missing = required - set(metadata)
    if missing:
        raise ContractError(f"initialization builder metadata is incomplete: {sorted(missing)}")
    for name in ("proposal_sequence_sha256", "tensor_cache_manifest_sha256"):
        value = metadata[name]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ContractError(f"initialization builder metadata {name} is not a SHA-256")
    if metadata["higher_order_sh"] != "loader_zero_fill_at_configured_degree":
        raise ContractError("initialization has an unsupported higher-order SH policy")
    minimum = _metadata_float(metadata, "duration_min_seconds")
    maximum = _metadata_float(metadata, "duration_max_seconds")
    target = _metadata_float(metadata, "duration_seconds")
    offset = _metadata_float(metadata, "time_offset_seconds")
    if (
        minimum != config.duration_min_seconds
        or maximum != config.duration_max_seconds
        or offset != config.time_offset_seconds
    ):
        raise ContractError("initialization builder policy differs from the run configuration")
    if not minimum < target < maximum:
        raise ContractError("initialization builder duration target lies outside its bounds")
    fraction = (target - minimum) / (maximum - minimum)
    expected_logit = math.log(fraction / (1.0 - fraction))
    duration_logits = tensors.get("duration_logits")
    if duration_logits is None or not bool(
        torch.allclose(
            duration_logits,
            torch.full_like(duration_logits, expected_logit),
            rtol=0.0,
            atol=1.0e-6,
        )
    ):
        raise ContractError("initialization duration logits disagree with builder metadata")


def load_p2g_safetensors(path: Path, config: InitializationConfig) -> GaussianInit:
    """Load the exact public initialization tensor contract without pickle fallback."""

    config.validate()
    resolved = _regular_safetensors_path(path)
    digest_before = _sha256(resolved)
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file  # pyright: ignore[reportUnknownVariableType]

        reader = cast(Callable[..., _SafeTensorReader], safe_open)
        loader = cast(Callable[..., dict[str, Tensor]], load_file)
        with reader(str(resolved), framework="pt", device="cpu") as stream:
            metadata = stream.metadata() or {}
            catalog = frozenset(stream.keys())
        tensors = loader(str(resolved), device="cpu")
    except Exception as exc:
        raise ContractError(f"cannot load public Safetensors initialization: {exc}") from exc
    digest_after = _sha256(resolved)
    if digest_before != digest_after:
        raise ContractError("initialization changed while it was being loaded")
    if metadata.get("schema_version") != INITIALIZATION_SCHEMA:
        raise ContractError(f"initialization must declare {INITIALIZATION_SCHEMA}")
    if catalog != frozenset(tensors):
        raise ContractError("initialization tensor catalog changed while loading")
    missing = _REQUIRED_FILE_TENSORS - catalog
    unknown = catalog - _REQUIRED_FILE_TENSORS - _OPTIONAL_FILE_TENSORS
    if missing or unknown:
        raise ContractError(
            f"initialization tensor catalog mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    _validate_builder_metadata(metadata, tensors, config)
    means = tensors["means"]
    if means.ndim != 2 or means.shape[1] != 3 or means.shape[0] <= 0:
        raise ContractError("initialization tensor means must have shape [N,3] with N > 0")
    count = int(means.shape[0])
    coefficient_count = (config.sh_degree + 1) ** 2 - 1
    if "sh_rest" in tensors:
        sh_rest = _loaded_plane(
            tensors,
            "sh_rest",
            shape=(count, coefficient_count, 3),
        )
    else:
        sh_rest = torch.zeros((count, coefficient_count, 3), dtype=torch.float32)
    result = GaussianInit(
        means=_loaded_plane(tensors, "means", shape=(count, 3)),
        log_scales=_loaded_plane(tensors, "log_scales", shape=(count, 3)),
        quaternions=_loaded_plane(tensors, "quaternions", shape=(count, 4)),
        opacity_logits=_loaded_plane(tensors, "opacity_logits", shape=(count, 1)),
        sh0=_loaded_plane(tensors, "sh0", shape=(count, 1, 3)),
        sh_rest=sh_rest,
        center_times=(
            _loaded_plane(tensors, "center_times", shape=(count, 1))
            - config.time_offset_seconds
        ).contiguous(),
        duration_logits=_loaded_plane(tensors, "duration_logits", shape=(count, 1)),
        velocities=_loaded_plane(tensors, "velocities", shape=(count, 3)),
        persistence_logits=torch.full(
            (count, 1), config.persistence_initial_logit, dtype=torch.float32
        ),
        duration_min_seconds=torch.full(
            (count, 1), config.duration_min_seconds, dtype=torch.float32
        ),
        duration_max_seconds=torch.full(
            (count, 1), config.duration_max_seconds, dtype=torch.float32
        ),
        runtime_ids=_loaded_plane(
            tensors,
            "runtime_ids",
            shape=(count,),
            dtype=torch.int64,
        ),
        source={
            "format": "p2g_safetensors",
            "schema_version": INITIALIZATION_SCHEMA,
            "sha256": digest_after,
        },
    )
    result.validate()
    return result


def load_gaussian_init(config: InitializationConfig) -> GaussianInit:
    config.validate()
    if config.format != "p2g_safetensors":
        raise ContractError("only the public p2g_safetensors initialization format is supported")
    return load_p2g_safetensors(config.path, config)
