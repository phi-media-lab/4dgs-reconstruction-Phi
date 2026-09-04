# pyright: reportUnnecessaryIsInstance=false

"""Fail-closed adapter for the pinned AMD gsplat MI300X runtime.

The project owns this adapter and its validation rules.  Projection, spherical
harmonics, sorting, and alpha compositing are supplied by the separately
licensed AMD gsplat distribution identified below.  No compatibility import or
reference-project renderer is accepted at this boundary.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import math
import platform
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import Tensor

from p2g.errors import ContractError
from p2g.training.config import RendererConfig
from p2g.training.dataset import TrainingBatch
from p2g.training.model import DynamicGaussianModel, MaterializedGaussians

RENDERER_ABI = "p2g.gsplat_rocm.v1"
AMD_GSPLAT_DISTRIBUTION = "amd-gsplat"
AMD_GSPLAT_MODULE_VERSION = "1.5.3"
AMD_GSPLAT_REVISION = "b01acd43e3c7fa942f95fda0974e9125e4de7395"
AMD_GSPLAT_VERSION = f"{AMD_GSPLAT_MODULE_VERSION}+{AMD_GSPLAT_REVISION}"
EXPECTED_TORCH_VERSION = "2.10.0+rocm7.0"
EXPECTED_HIP_VERSION = "7.0.51831"
EXPECTED_ARCHITECTURE = "gfx942"

_RASTERIZATION_KEYWORDS = (
    "means",
    "quats",
    "scales",
    "opacities",
    "colors",
    "viewmats",
    "Ks",
    "width",
    "height",
    "near_plane",
    "far_plane",
    "radius_clip",
    "eps2d",
    "sh_degree",
    "packed",
    "tile_size",
    "backgrounds",
    "render_mode",
    "sparse_grad",
    "absgrad",
    "rasterize_mode",
    "channel_chunk",
    "distributed",
    "camera_model",
    "segmented",
    "covars",
    "with_ut",
    "with_eval3d",
    "radial_coeffs",
    "tangential_coeffs",
    "thin_prism_coeffs",
    "ftheta_coeffs",
    "rolling_shutter",
    "viewmats_rs",
)


@dataclass(frozen=True, slots=True)
class RenderResult:
    """One rendered RGB/alpha image plus differentiable projection metadata."""

    image: Tensor
    alpha: Tensor
    aux: dict[str, Any]
    materialized: MaterializedGaussians


@dataclass(frozen=True, slots=True)
class _Provider:
    rasterization: Callable[..., object]
    global_shutter: object
    identity: dict[str, str]


@dataclass(frozen=True, slots=True)
class _RuntimeBinding:
    provider: _Provider
    device: torch.device
    identity: dict[str, Any]


class _DeviceProperties(Protocol):
    name: str
    gcnArchName: str


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _distribution_member(
    distribution: importlib.metadata.Distribution,
    relative: str,
) -> Path:
    files = distribution.files
    if files is None:
        raise ContractError("AMD gsplat distribution has no installed-file catalog")
    matches = [item for item in files if str(item).replace("\\", "/") == relative]
    if len(matches) != 1:
        raise ContractError(f"AMD gsplat distribution is missing the registered {relative}")
    candidate = Path(str(distribution.locate_file(matches[0])))
    if not candidate.is_file() or candidate.is_symlink():
        raise ContractError(f"AMD gsplat distribution member is not a regular file: {relative}")
    return candidate.resolve()


def _module_file(module: object, *, label: str) -> Path:
    value: object = getattr(module, "__file__", None)
    if not isinstance(value, str):
        raise ContractError(f"{label} has no regular module file")
    candidate = Path(value)
    if not candidate.is_file() or candidate.is_symlink():
        raise ContractError(f"{label} has no regular module file")
    return candidate.resolve()


def _validate_rasterization_signature(function: Callable[..., object]) -> object:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise ContractError("AMD gsplat rasterization signature is not inspectable") from exc
    missing: list[str] = []
    for name in _RASTERIZATION_KEYWORDS:
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            missing.append(name)
    if missing:
        raise ContractError(f"AMD gsplat rasterization ABI is missing keywords: {missing}")
    shutter = signature.parameters["rolling_shutter"].default
    if shutter is inspect.Parameter.empty or getattr(shutter, "name", None) != "GLOBAL":
        raise ContractError("AMD gsplat rasterization ABI has no global-shutter default")
    return shutter


def _load_provider() -> _Provider:
    """Load only the exact wheel-owned Python module and native provider."""

    try:
        distribution = importlib.metadata.distribution(AMD_GSPLAT_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError("the registered amd-gsplat distribution is not installed") from exc
    project_name = distribution.metadata.get("Name", "")
    if _normalized_distribution_name(project_name) != AMD_GSPLAT_DISTRIBUTION:
        raise ContractError("AMD gsplat distribution identity is invalid")
    if distribution.version != AMD_GSPLAT_VERSION:
        raise ContractError(f"renderer requires {AMD_GSPLAT_DISTRIBUTION}=={AMD_GSPLAT_VERSION}")

    expected_module = _distribution_member(distribution, "gsplat/__init__.py")
    # Requiring the prebuilt provider before importing gsplat.cuda._backend
    # prevents that upstream module from falling back to an unregistered JIT build.
    expected_native = _distribution_member(distribution, "gsplat/csrc.so")
    try:
        module: Any = importlib.import_module("gsplat")
    except (ImportError, OSError, RuntimeError) as exc:
        raise ContractError(f"cannot import the registered AMD gsplat module: {exc}") from exc
    if _module_file(module, label="AMD gsplat") != expected_module:
        raise ContractError("the imported gsplat module is not owned by amd-gsplat")
    if getattr(module, "__version__", None) != AMD_GSPLAT_MODULE_VERSION:
        raise ContractError("AMD gsplat module version differs from its registered API")
    rasterization: object = getattr(module, "rasterization", None)
    if not callable(rasterization):
        raise ContractError("AMD gsplat does not expose a callable rasterization API")

    try:
        backend: Any = importlib.import_module("gsplat.cuda._backend")
    except (ImportError, OSError, RuntimeError) as exc:
        raise ContractError(f"cannot import the registered AMD gsplat provider: {exc}") from exc
    native: object = getattr(backend, "_C", None)
    if (
        native is None
        or _module_file(native, label="AMD gsplat native provider") != expected_native
    ):
        raise ContractError("AMD gsplat loaded an unregistered native provider")

    typed_rasterization = rasterization
    shutter = _validate_rasterization_signature(typed_rasterization)
    return _Provider(
        rasterization=typed_rasterization,
        global_shutter=shutter,
        identity={
            "distribution": AMD_GSPLAT_DISTRIBUTION,
            "version": AMD_GSPLAT_VERSION,
            "module_version": AMD_GSPLAT_MODULE_VERSION,
            "source_revision": AMD_GSPLAT_REVISION,
        },
    )


def _require_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    contiguous: bool = True,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ContractError(f"{name} must be a tensor")
    if tuple(value.shape) != shape:
        raise ContractError(f"{name} must have shape {shape}, found {tuple(value.shape)}")
    if value.dtype != dtype:
        raise ContractError(f"{name} must use {dtype}, found {value.dtype}")
    if value.device != device:
        raise ContractError(f"{name} must be on {device}, found {value.device}")
    if contiguous and not value.is_contiguous():
        raise ContractError(f"{name} must be contiguous")
    return value


def _validate_inputs(
    materialized: MaterializedGaussians,
    batch: TrainingBatch,
    *,
    sh_degree: int,
) -> tuple[torch.device, int, int]:
    if batch.distorted:
        raise ContractError("renderer accepts only offline-undistorted pinhole observations")
    if type(sh_degree) is not int or not 0 <= sh_degree <= 3:
        raise ContractError("renderer SH degree must be an integer in [0, 3]")
    if not isinstance(materialized.means, Tensor) or materialized.means.ndim != 2:
        raise ContractError("materialized means must have shape [N,3]")
    count = int(materialized.means.shape[0])
    if count <= 0 or tuple(materialized.means.shape) != (count, 3):
        raise ContractError("materialized means must have non-empty shape [N,3]")
    device = materialized.means.device
    coefficient_count = (sh_degree + 1) ** 2
    planes = {
        "materialized.means": (materialized.means, (count, 3)),
        "materialized.quaternions": (materialized.quaternions, (count, 4)),
        "materialized.scales": (materialized.scales, (count, 3)),
        "materialized.opacities": (materialized.opacities, (count,)),
        "materialized.colors": (materialized.colors, (count, coefficient_count, 3)),
        "materialized.temporal_activation": (materialized.temporal_activation, (count, 1)),
        "materialized.temporal_sigma": (materialized.temporal_sigma, (count, 1)),
        "materialized.time_delta": (materialized.time_delta, (count, 1)),
    }
    for name, (value, shape) in planes.items():
        _require_tensor(value, name=name, shape=shape, device=device)

    rgb = batch.rgb
    if not isinstance(rgb, Tensor) or rgb.ndim != 3 or int(rgb.shape[-1]) != 3:
        raise ContractError("batch RGB must have shape [H,W,3]")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if height <= 0 or width <= 0:
        raise ContractError("batch RGB dimensions must be positive")
    _require_tensor(rgb, name="batch.rgb", shape=(height, width, 3), device=device)
    _require_tensor(
        batch.intrinsic,
        name="batch.intrinsic",
        shape=(1, 3, 3),
        device=device,
    )
    _require_tensor(
        batch.world_to_camera,
        name="batch.world_to_camera",
        shape=(1, 4, 4),
        device=device,
    )
    timestamp = batch.timestamp
    if not isinstance(timestamp, Tensor) or tuple(timestamp.shape) not in {(), (1,)}:
        raise ContractError("batch timestamp must be one scalar tensor")
    if timestamp.dtype != torch.float32 or timestamp.device != device:
        raise ContractError("batch timestamp must match the float32 render device")
    if batch.intrinsic.requires_grad or batch.world_to_camera.requires_grad:
        raise ContractError("the public renderer does not expose camera gradients")
    return device, height, width


def _metadata_tensor(
    metadata: Mapping[str, Any],
    name: str,
    *,
    device: torch.device,
) -> Tensor:
    value: object = metadata.get(name)
    if not isinstance(value, Tensor) or value.device != device:
        raise ContractError(f"AMD gsplat metadata {name} is missing or on the wrong device")
    return value


def _validated_metadata(
    metadata: Mapping[str, Any],
    *,
    device: torch.device,
    width: int,
    height: int,
    tile_size: int,
) -> dict[str, Any]:
    camera_ids = _metadata_tensor(metadata, "camera_ids", device=device)
    gaussian_ids = _metadata_tensor(metadata, "gaussian_ids", device=device)
    means2d = _metadata_tensor(metadata, "means2d", device=device)
    radii = _metadata_tensor(metadata, "radii", device=device)
    depths = _metadata_tensor(metadata, "depths", device=device)
    projected_opacities = _metadata_tensor(metadata, "opacities", device=device)
    tiles = _metadata_tensor(metadata, "tiles_per_gauss", device=device)
    flatten_ids = _metadata_tensor(metadata, "flatten_ids", device=device)
    packed_count = int(gaussian_ids.numel())
    if gaussian_ids.dtype != torch.int64 or tuple(gaussian_ids.shape) != (packed_count,):
        raise ContractError("AMD gsplat packed gaussian_ids metadata is invalid")
    if camera_ids.dtype != torch.int64 or tuple(camera_ids.shape) != (packed_count,):
        raise ContractError("AMD gsplat packed camera_ids metadata is invalid")
    if means2d.dtype != torch.float32 or tuple(means2d.shape) != (packed_count, 2):
        raise ContractError("AMD gsplat packed means2d metadata is invalid")
    if radii.is_floating_point() or tuple(radii.shape) != (packed_count, 2):
        raise ContractError("AMD gsplat packed radii metadata is invalid")
    for name, value in (
        ("depths", depths),
        ("opacities", projected_opacities),
    ):
        if value.dtype != torch.float32 or tuple(value.shape) != (packed_count,):
            raise ContractError(f"AMD gsplat packed {name} metadata is invalid")
    if tiles.is_floating_point() or tuple(tiles.shape) != (packed_count,):
        raise ContractError("AMD gsplat packed tiles_per_gauss metadata is invalid")
    if flatten_ids.is_floating_point() or flatten_ids.ndim != 1:
        raise ContractError("AMD gsplat flatten_ids metadata is invalid")

    expected_tile_width = math.ceil(width / tile_size)
    expected_tile_height = math.ceil(height / tile_size)
    if type(metadata.get("tile_width")) is not int or metadata["tile_width"] != expected_tile_width:
        raise ContractError("AMD gsplat tile_width metadata differs from the render contract")
    if (
        type(metadata.get("tile_height")) is not int
        or metadata["tile_height"] != expected_tile_height
    ):
        raise ContractError("AMD gsplat tile_height metadata differs from the render contract")
    for name, expected in (("width", width), ("height", height)):
        value = metadata.get(name)
        if value is not None and (type(value) is not int or value != expected):
            raise ContractError(f"AMD gsplat {name} metadata differs from the render contract")
    if means2d.requires_grad:
        means2d.retain_grad()
    result = dict(metadata)
    result.update(
        {
            "width": width,
            "height": height,
            "n_cameras": 1,
            "renderer_abi": RENDERER_ABI,
        }
    )
    return result


class GsplatRenderer:
    """Rasterize the project's materialized state through one pinned provider."""

    def __init__(self, config: RendererConfig) -> None:
        config.validate()
        if not config.require_gfx942:
            raise ContractError("the public renderer requires require_gfx942=true")
        self.config = config
        self._binding: _RuntimeBinding | None = None

    def validate_environment(self, device: str | torch.device) -> dict[str, Any]:
        """Bind the exact single-MI300X ABI and return a path-free identity."""

        try:
            target = torch.device(device)
        except (RuntimeError, TypeError) as exc:
            raise ContractError(f"invalid renderer device: {device!r}") from exc
        if target.type != "cuda" or target.index not in {None, 0}:
            raise ContractError("gsplat_rocm requires the sole visible device cuda:0")
        resolved = torch.device("cuda", 0)
        if self._binding is not None:
            if self._binding.device != resolved:
                raise ContractError("renderer was validated for a different device")
            return dict(self._binding.identity)
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ContractError("gsplat_rocm requires Linux x86_64")
        if sys.version_info[:2] != (3, 12):
            raise ContractError("gsplat_rocm requires CPython 3.12")
        if str(torch.__version__) != EXPECTED_TORCH_VERSION:
            raise ContractError(
                f"gsplat_rocm requires torch {EXPECTED_TORCH_VERSION}, found {torch.__version__}"
            )
        if str(torch.version.hip) != EXPECTED_HIP_VERSION:
            raise ContractError(
                f"gsplat_rocm requires HIP {EXPECTED_HIP_VERSION}, found {torch.version.hip}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ContractError("gsplat_rocm requires exactly one visible ROCm accelerator")
        try:
            get_device_properties = cast(
                Callable[[int], _DeviceProperties],
                vars(torch.cuda)["get_device_properties"],
            )
            properties = get_device_properties(0)
        except RuntimeError as exc:
            raise ContractError(f"cannot query the visible ROCm accelerator: {exc}") from exc
        architecture = str(properties.gcnArchName).split(":", maxsplit=1)[0]
        if architecture != EXPECTED_ARCHITECTURE:
            raise ContractError(
                f"gsplat_rocm requires {EXPECTED_ARCHITECTURE}, found {architecture or 'unknown'}"
            )
        provider = _load_provider()
        identity: dict[str, Any] = {
            "renderer_abi": RENDERER_ABI,
            "backend": "gsplat_rocm",
            "python_abi": "cp312",
            "torch": EXPECTED_TORCH_VERSION,
            "hip": EXPECTED_HIP_VERSION,
            "device": str(properties.name),
            "architecture": architecture,
            "visible_device_count": 1,
            "gsplat_distribution_name": provider.identity["distribution"],
            "gsplat_distribution": provider.identity["version"],
            "gsplat_module": provider.identity["module_version"],
            "gsplat_source_revision": provider.identity["source_revision"],
        }
        self._binding = _RuntimeBinding(provider=provider, device=resolved, identity=identity)
        return dict(identity)

    def _binding_for(self, device: torch.device) -> _RuntimeBinding:
        if self._binding is None:
            self.validate_environment(device)
        binding = self._binding
        if binding is None:  # pragma: no cover - validate_environment postcondition
            raise AssertionError("renderer binding was not installed")
        if binding.device != device:
            raise ContractError(
                f"renderer tensors must use validated device {binding.device}, found {device}"
            )
        return binding

    def render(
        self,
        model: DynamicGaussianModel,
        batch: TrainingBatch,
        *,
        sh_degree: int,
    ) -> RenderResult:
        if batch.distorted:
            raise ContractError("renderer accepts only offline-undistorted pinhole observations")
        materialized = model.materialize(batch.timestamp, sh_degree=sh_degree)
        return self.render_materialized(materialized, batch, sh_degree=sh_degree)

    def render_materialized(
        self,
        materialized: MaterializedGaussians,
        batch: TrainingBatch,
        *,
        sh_degree: int,
    ) -> RenderResult:
        """Rasterize a physical state without changing any Gaussian parameter."""

        device, height, width = _validate_inputs(materialized, batch, sh_degree=sh_degree)
        binding = self._binding_for(device)
        background = materialized.means.new_tensor(self.config.background)
        try:
            output = binding.provider.rasterization(
                means=materialized.means,
                quats=materialized.quaternions,
                scales=materialized.scales,
                opacities=materialized.opacities,
                colors=materialized.colors,
                viewmats=batch.world_to_camera,
                Ks=batch.intrinsic,
                width=width,
                height=height,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                radius_clip=self.config.radius_clip,
                eps2d=self.config.eps2d,
                sh_degree=sh_degree,
                packed=True,
                tile_size=8,
                backgrounds=background,
                render_mode="RGB",
                sparse_grad=False,
                absgrad=False,
                rasterize_mode="classic",
                channel_chunk=32,
                distributed=False,
                camera_model="pinhole",
                segmented=False,
                covars=None,
                with_ut=False,
                with_eval3d=False,
                radial_coeffs=None,
                tangential_coeffs=None,
                thin_prism_coeffs=None,
                ftheta_coeffs=None,
                rolling_shutter=binding.provider.global_shutter,
                viewmats_rs=None,
            )
        except (AssertionError, TypeError, ValueError) as exc:
            raise ContractError(f"AMD gsplat rejected the admitted renderer ABI: {exc}") from exc
        if not isinstance(output, tuple):
            raise ContractError("AMD gsplat must return RGB, alpha, and metadata")
        output_items = cast(tuple[object, ...], output)
        if len(output_items) != 3:
            raise ContractError("AMD gsplat must return RGB, alpha, and metadata")
        rendered, alpha, metadata = output_items
        if not isinstance(rendered, Tensor) or not isinstance(alpha, Tensor):
            raise ContractError("AMD gsplat RGB and alpha outputs must be tensors")
        if tuple(rendered.shape) != (1, height, width, 3):
            raise ContractError(f"AMD gsplat returned unexpected RGB shape {tuple(rendered.shape)}")
        if tuple(alpha.shape) != (1, height, width, 1):
            raise ContractError(f"AMD gsplat returned unexpected alpha shape {tuple(alpha.shape)}")
        for name, value in (("RGB", rendered), ("alpha", alpha)):
            if value.dtype != torch.float32 or value.device != device:
                raise ContractError(f"AMD gsplat {name} output has the wrong execution layout")
        if not isinstance(metadata, Mapping):
            raise ContractError("AMD gsplat metadata must be a mapping")
        image = rendered[0]
        if self.config.clamp_rgb:
            image = image.clamp(0.0, 1.0)
        aux = _validated_metadata(
            cast(Mapping[str, Any], metadata),
            device=device,
            width=width,
            height=height,
            tile_size=8,
        )
        return RenderResult(
            image=image,
            alpha=alpha[0],
            aux=aux,
            materialized=materialized,
        )
