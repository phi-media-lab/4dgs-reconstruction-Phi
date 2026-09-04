"""Explicit reconstruction metrics and independently owned training losses."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.resources
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import Tensor
from torch.nn import functional as F

from p2g.canonical import sha256_bytes, sha256_file
from p2g.errors import ContractError
from p2g.training.config import LossConfig
from p2g.training.model import DynamicGaussianModel, MaterializedGaussians

FUSED_SSIM_DISTRIBUTION = "fused-ssim"
FUSED_SSIM_VERSION = "1.0.0"
FUSED_SSIM_REVISION = "a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"
SSIM_EQUATION_VERSION = "p2g.gaussian_ssim_rgb_unit_range.v1"
PSNR_EQUATION_VERSION = "p2g.psnr_rgb_unit_range.v1"
LOSS_EQUATION_VERSION = "p2g.pixel4dgs_objective.v1"
LPIPS_REGISTRY_SCHEMA = "p2g.lpips_alex_provider_registry.v1"
LPIPS_REGISTRY_RESOURCE = "registries/lpips_alex_v1.json"
LPIPS_INPUT_CONTRACT = "clamp_rgb_unit_range_then_nchw_view_with_normalize_false_v1"

LOSS_TERM_NAMES = (
    "l1",
    "ssim",
    "lpips",
    "opacity",
    "scale",
    "persistence",
    "gate",
    "color_correction",
)


class _FusedSsim(Protocol):
    def __call__(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        padding: str,
    ) -> object: ...


class _Lpips(Protocol):
    def __call__(self, prediction: Tensor, target: Tensor) -> object: ...

    def reset(self) -> None: ...


def _validate_image_pair(prediction: Tensor, target: Tensor) -> None:
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ContractError("prediction must be one HWC RGB image")
    if target.shape != prediction.shape:
        raise ContractError("prediction and target image shapes must match exactly")
    if prediction.dtype != torch.float32 or target.dtype != torch.float32:
        raise ContractError("loss images must use float32")
    if prediction.device != target.device:
        raise ContractError("prediction and target must be on the same device")
    if prediction.shape[0] <= 0 or prediction.shape[1] <= 0:
        raise ContractError("loss images must have non-empty spatial dimensions")


def _to_nchw(image: Tensor) -> Tensor:
    return image.permute(2, 0, 1).unsqueeze(0).contiguous()


def _to_lpips_nchw(image: Tensor) -> Tensor:
    # Preserve the versioned strided HWC->NCHW view exactly. Fused SSIM has a
    # different ABI and deliberately uses the contiguous helper above.
    return image.permute(2, 0, 1).unsqueeze(0)


def _window_size(image: Tensor) -> int:
    spatial_minimum = min(int(image.shape[0]), int(image.shape[1]))
    candidate = min(11, spatial_minimum)
    return candidate if candidate % 2 == 1 else candidate - 1


def _gaussian_kernel(reference: Tensor, window_size: int) -> Tensor:
    sigma = 1.5 * window_size / 11.0
    coordinates = (
        torch.arange(
            window_size,
            dtype=reference.dtype,
            device=reference.device,
        )
        - (window_size - 1) / 2.0
    )
    kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    return (
        torch.outer(kernel_1d, kernel_1d)
        .reshape(1, 1, window_size, window_size)
        .expand(3, 1, -1, -1)
    )


def _structural_similarity_with_kernel(
    prediction: Tensor,
    target: Tensor,
    *,
    padding: str,
    kernel: Tensor,
) -> Tensor:
    window_size = int(kernel.shape[-1])
    convolution_padding = window_size // 2 if padding == "same" else 0
    prediction_nchw = _to_nchw(prediction)
    target_nchw = _to_nchw(target)

    def local_mean(value: Tensor) -> Tensor:
        return F.conv2d(value, kernel, padding=convolution_padding, groups=3)

    mean_prediction = local_mean(prediction_nchw)
    mean_target = local_mean(target_nchw)
    mean_prediction_squared = mean_prediction.square()
    mean_target_squared = mean_target.square()
    mean_product = mean_prediction * mean_target
    variance_prediction = local_mean(prediction_nchw.square()) - mean_prediction_squared
    variance_target = local_mean(target_nchw.square()) - mean_target_squared
    covariance = local_mean(prediction_nchw * target_nchw) - mean_product
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mean_product + c1) * (2.0 * covariance + c2)
    denominator = (mean_prediction_squared + mean_target_squared + c1) * (
        variance_prediction + variance_target + c2
    )
    return (numerator / denominator).mean()


def structural_similarity(
    prediction: Tensor,
    target: Tensor,
    *,
    padding: str = "same",
) -> Tensor:
    """Return RGB SSIM for unit-range HWC float32 images.

    The equation uses a Gaussian window of at most 11 pixels, sigma 1.5 at
    width 11, population moments, and constants for a unit photometric range.
    Values are not clamped, preserving the exact diagnostic and gradient.
    """

    _validate_image_pair(prediction, target)
    if padding not in {"same", "valid"}:
        raise ContractError("SSIM padding must be 'same' or 'valid'")
    window_size = _window_size(prediction)
    kernel = _gaussian_kernel(prediction, window_size)
    return _structural_similarity_with_kernel(
        prediction,
        target,
        padding=padding,
        kernel=kernel,
    )


def psnr(prediction: Tensor, target: Tensor) -> Tensor:
    """Return unit-range PSNR, with exact matches capped at 120 dB."""

    _validate_image_pair(prediction, target)
    mean_squared_error = (prediction - target).square().mean().clamp_min(1.0e-12)
    return -10.0 * torch.log10(mean_squared_error)


def _distribution_file_is_owned(
    distribution: importlib.metadata.Distribution,
    module_file: Path,
) -> bool:
    files = distribution.files
    if files is None:
        return False
    if not module_file.is_file() or module_file.is_symlink():
        return False
    module_file = module_file.resolve()
    for item in files:
        if str(item).replace("\\", "/") != "fused_ssim/__init__.py":
            continue
        if Path(str(distribution.locate_file(item))).resolve() == module_file:
            return True
    return False


def _load_fused_ssim() -> tuple[_FusedSsim, dict[str, str]]:
    try:
        distribution = importlib.metadata.distribution(FUSED_SSIM_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError("fused SSIM requires the registered fused-ssim distribution") from exc
    project_name = distribution.metadata.get("Name", "")
    if project_name.casefold().replace("_", "-") != FUSED_SSIM_DISTRIBUTION:
        raise ContractError("fused SSIM distribution identity is invalid")
    if distribution.version != FUSED_SSIM_VERSION:
        raise ContractError(f"fused SSIM requires {FUSED_SSIM_DISTRIBUTION}=={FUSED_SSIM_VERSION}")
    try:
        module = importlib.import_module("fused_ssim")
    except (ImportError, RuntimeError) as exc:
        raise ContractError(f"cannot import the registered fused SSIM provider: {exc}") from exc
    module_name: Any = getattr(module, "__file__", None)
    function: Any = getattr(module, "fused_ssim", None)
    if not isinstance(module_name, str) or not callable(function):
        raise ContractError("registered fused SSIM provider has an invalid Python API")
    if not _distribution_file_is_owned(distribution, Path(module_name)):
        raise ContractError("fused SSIM module is not owned by its registered distribution")
    return cast(_FusedSsim, function), {
        "backend": "fused",
        "distribution": FUSED_SSIM_DISTRIBUTION,
        "version": FUSED_SSIM_VERSION,
        "required_source_revision": FUSED_SSIM_REVISION,
    }


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_record(value: object, *, label: str) -> dict[str, Any]:
    record = _object(value, label=label)
    _exact_keys(record, {"path", "bytes", "sha256"}, label=label)
    relative = record["path"]
    byte_count = record["bytes"]
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith(("/", "~"))
        or "\\" in relative
        or ".." in Path(relative).parts
    ):
        raise ContractError(f"{label}.path must be a safe distribution-relative path")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
        raise ContractError(f"{label}.bytes must be a positive integer")
    _sha256(record["sha256"], label=f"{label}.sha256")
    return record


def _validate_lpips_registry(registry: dict[str, Any]) -> None:
    _exact_keys(
        registry,
        {"schema_version", "provider", "runtime", "external_weights", "metric", "policy"},
        label="LPIPS registry",
    )
    if registry["schema_version"] != LPIPS_REGISTRY_SCHEMA:
        raise ContractError("unsupported LPIPS provider registry schema")

    provider = _object(registry["provider"], label="LPIPS provider")
    _exact_keys(
        provider,
        {
            "distribution",
            "distribution_version",
            "factory",
            "repository",
            "release",
            "license",
            "distribution_artifacts",
            "files",
        },
        label="LPIPS provider",
    )
    if (
        provider["distribution"] != "torchmetrics"
        or provider["distribution_version"] != "1.9.0"
        or provider["factory"]
        != "torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity"
        or provider["release"] != "v1.9.0"
    ):
        raise ContractError("LPIPS provider identity differs from the admitted implementation")
    if not isinstance(provider["repository"], str) or not provider["repository"].startswith(
        "https://"
    ):
        raise ContractError("LPIPS provider repository must be an HTTPS URL")
    license_record = _object(provider["license"], label="LPIPS provider license")
    _exact_keys(
        license_record,
        {"spdx", "url", "path", "bytes", "sha256"},
        label="LPIPS provider license",
    )
    if license_record["spdx"] != "Apache-2.0" or not isinstance(
        license_record["url"], str
    ) or not license_record["url"].startswith("https://"):
        raise ContractError("LPIPS provider license identity is invalid")
    _file_record(
        {key: license_record[key] for key in ("path", "bytes", "sha256")},
        label="LPIPS provider license file",
    )
    artifacts = _object(
        provider["distribution_artifacts"], label="LPIPS distribution artifacts"
    )
    _exact_keys(
        artifacts,
        {"wheel_sha256", "sdist_sha256"},
        label="LPIPS distribution artifacts",
    )
    for name, digest in artifacts.items():
        _sha256(digest, label=f"LPIPS distribution artifact {name}")
    files = _object(provider["files"], label="LPIPS provider files")
    _exact_keys(files, {"metric", "functional", "linear_weights"}, label="LPIPS files")
    for name, record in files.items():
        _file_record(record, label=f"LPIPS provider file {name}")

    runtime = _object(registry["runtime"], label="LPIPS runtime")
    _exact_keys(
        runtime,
        {"python", "torch", "torchvision", "torchvision_files"},
        label="LPIPS runtime",
    )
    if runtime["python"] != "3.12" or runtime["torch"] != "2.10.0+rocm7.0" or runtime[
        "torchvision"
    ] != "0.25.0+rocm7.0":
        raise ContractError("LPIPS runtime differs from the admitted MI300X profile")
    torchvision_files = _object(
        runtime["torchvision_files"], label="LPIPS torchvision files"
    )
    _exact_keys(
        torchvision_files,
        {"alexnet", "license"},
        label="LPIPS torchvision files",
    )
    for name, record in torchvision_files.items():
        _file_record(record, label=f"LPIPS torchvision file {name}")

    weights = _object(registry["external_weights"], label="LPIPS external weights")
    _exact_keys(weights, {"alexnet_features"}, label="LPIPS external weights")
    alexnet = _object(weights["alexnet_features"], label="LPIPS AlexNet weight")
    _exact_keys(
        alexnet,
        {"filename", "url", "bytes", "sha256", "license", "redistribution"},
        label="LPIPS AlexNet weight",
    )
    filename = alexnet["filename"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ContractError("LPIPS AlexNet weight filename is unsafe")
    if not isinstance(alexnet["url"], str) or not alexnet["url"].startswith("https://"):
        raise ContractError("LPIPS AlexNet weight URL must use HTTPS")
    if (
        not isinstance(alexnet["bytes"], int)
        or isinstance(alexnet["bytes"], bool)
        or alexnet["bytes"] <= 0
    ):
        raise ContractError("LPIPS AlexNet weight size must be positive")
    _sha256(alexnet["sha256"], label="LPIPS AlexNet weight SHA-256")
    weight_license = _object(alexnet["license"], label="LPIPS AlexNet weight license")
    _exact_keys(weight_license, {"spdx", "status", "url"}, label="LPIPS weight license")
    if weight_license != {
        "spdx": "NOASSERTION",
        "status": "review_required",
        "url": "https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.alexnet.html",
    } or alexnet["redistribution"] != "external_only_not_bundled":
        raise ContractError("LPIPS AlexNet weight rights policy is invalid")

    metric = _object(registry["metric"], label="LPIPS metric")
    if metric != {
        "net_type": "alex",
        "reduction": "mean",
        "normalize": False,
        "input_contract": LPIPS_INPUT_CONTRACT,
    }:
        raise ContractError("LPIPS metric configuration differs from the frozen recipe")
    policy = _object(registry["policy"], label="LPIPS policy")
    if policy != {
        "automatic_download": False,
        "bundle_weights": False,
        "require_local_hash_match": True,
        "record_provider_identity": True,
    }:
        raise ContractError("LPIPS provider policy is not fail-closed")


def _load_lpips_registry() -> tuple[dict[str, Any], str]:
    try:
        resource = importlib.resources.files("p2g").joinpath(
            *LPIPS_REGISTRY_RESOURCE.split("/")
        )
        payload = resource.read_bytes()
        decoded: Any = json.loads(payload)
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load the packaged LPIPS registry: {exc}") from exc
    registry = _object(decoded, label="LPIPS registry")
    _validate_lpips_registry(registry)
    return registry, sha256_bytes(payload)


def _distribution(
    name: str,
    version: str,
    *,
    label: str,
) -> importlib.metadata.Distribution:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError(f"{label} requires {name}=={version}") from exc
    project_name = distribution.metadata.get("Name", "")
    if project_name.casefold().replace("_", "-") != name or distribution.version != version:
        raise ContractError(f"{label} requires {name}=={version}")
    return distribution


def _owned_distribution_file(
    distribution: importlib.metadata.Distribution,
    record_value: object,
    *,
    label: str,
) -> Path:
    record = _file_record(record_value, label=label)
    files = distribution.files
    if files is None:
        raise ContractError(f"{label} cannot be bound to its installed distribution")
    relative = cast(str, record["path"])
    matches = [item for item in files if str(item).replace("\\", "/") == relative]
    if len(matches) != 1:
        raise ContractError(f"{label} is not uniquely owned by its installed distribution")
    path = Path(str(distribution.locate_file(matches[0])))
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} is not a regular installed file")
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ContractError(f"{label} bytes differ from the provider registry")
    return path.resolve()


def _verify_external_weight(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} is absent; automatic download is disabled")
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ContractError(f"{label} differs from the provider registry")


def _load_lpips(device: str | torch.device) -> tuple[_Lpips, dict[str, str]]:
    registry, registry_sha256 = _load_lpips_registry()
    provider = cast(dict[str, Any], registry["provider"])
    runtime = cast(dict[str, Any], registry["runtime"])
    if f"{sys.version_info.major}.{sys.version_info.minor}" != runtime["python"]:
        raise ContractError(f"LPIPS requires CPython {runtime['python']}")
    if str(torch.__version__) != runtime["torch"]:
        raise ContractError(f"LPIPS requires torch=={runtime['torch']}")
    try:
        if importlib.metadata.version("torch") != runtime["torch"]:
            raise ContractError(f"LPIPS requires torch=={runtime['torch']}")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError(f"LPIPS requires torch=={runtime['torch']}") from exc

    torchmetrics_distribution = _distribution(
        cast(str, provider["distribution"]),
        cast(str, provider["distribution_version"]),
        label="LPIPS",
    )
    torchvision_distribution = _distribution(
        "torchvision",
        cast(str, runtime["torchvision"]),
        label="LPIPS",
    )
    provider_files = cast(dict[str, Any], provider["files"])
    bound_provider_files = {
        name: _owned_distribution_file(
            torchmetrics_distribution,
            record,
            label=f"LPIPS provider file {name}",
        )
        for name, record in provider_files.items()
    }
    license_record = cast(dict[str, Any], provider["license"])
    _owned_distribution_file(
        torchmetrics_distribution,
        {key: license_record[key] for key in ("path", "bytes", "sha256")},
        label="LPIPS provider license file",
    )
    torchvision_files = cast(dict[str, Any], runtime["torchvision_files"])
    for name, record in torchvision_files.items():
        _owned_distribution_file(
            torchvision_distribution,
            record,
            label=f"LPIPS torchvision file {name}",
        )

    alexnet_record = cast(
        dict[str, Any],
        cast(dict[str, Any], registry["external_weights"])["alexnet_features"],
    )
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / cast(str, alexnet_record["filename"])
    _verify_external_weight(checkpoint, alexnet_record, label="LPIPS AlexNet checkpoint")

    try:
        module = importlib.import_module("torchmetrics.image.lpip")
    except (ImportError, OSError, RuntimeError) as exc:
        raise ContractError(f"cannot import the registered LPIPS provider: {exc}") from exc
    module_path = getattr(module, "__file__", None)
    factory = getattr(module, "LearnedPerceptualImagePatchSimilarity", None)
    if (
        not isinstance(module_path, str)
        or Path(module_path).resolve() != bound_provider_files["metric"]
        or not callable(factory)
    ):
        raise ContractError("registered LPIPS provider has an invalid Python API")

    metric_config = cast(dict[str, Any], registry["metric"])
    download_attribute = "download_url_to_file"
    original_download = getattr(torch.hub, download_attribute, None)
    if not callable(original_download):
        raise ContractError("Torch Hub download guard is unavailable")

    def reject_automatic_download(*_args: Any, **_kwargs: Any) -> None:
        raise ContractError("LPIPS automatic download is disabled")

    setattr(torch.hub, download_attribute, reject_automatic_download)
    try:
        try:
            metric_object: Any = factory(
                net_type=metric_config["net_type"],
                reduction=metric_config["reduction"],
                normalize=metric_config["normalize"],
            )
        except ContractError:
            raise
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ContractError(f"cannot construct the registered LPIPS provider: {exc}") from exc
    finally:
        setattr(torch.hub, download_attribute, original_download)
    if not isinstance(metric_object, torch.nn.Module) or not callable(
        getattr(metric_object, "reset", None)
    ):
        raise ContractError("registered LPIPS provider did not construct a resettable module")
    try:
        metric_object = metric_object.to(torch.device(device))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ContractError(
            f"cannot place the LPIPS provider on its execution device: {exc}"
        ) from exc
    metric_object.eval()
    for parameter in metric_object.parameters():
        parameter.requires_grad_(False)

    for name, path in bound_provider_files.items():
        record = cast(dict[str, Any], provider_files[name])
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ContractError(f"LPIPS provider file {name} changed during construction")
    _verify_external_weight(checkpoint, alexnet_record, label="LPIPS AlexNet checkpoint")
    return cast(_Lpips, metric_object), {
        "backend": "torchmetrics_alex",
        "distribution": cast(str, provider["distribution"]),
        "version": cast(str, provider["distribution_version"]),
        "torch": cast(str, runtime["torch"]),
        "torchvision": cast(str, runtime["torchvision"]),
        "registry_sha256": registry_sha256,
        "linear_weights_sha256": cast(str, provider_files["linear_weights"]["sha256"]),
        "alexnet_weights_sha256": cast(str, alexnet_record["sha256"]),
        "input_contract": LPIPS_INPUT_CONTRACT,
    }


class LossFunction:
    """Compute an exact, stable catalog of weighted scalar loss terms."""

    def __init__(self, config: LossConfig, *, device: str | torch.device | None = None) -> None:
        config.validate()
        self.config = config
        self._fused_ssim: _FusedSsim | None = None
        self._lpips: _Lpips | None = None
        self._kernel_cache: dict[tuple[int, str, int | None, torch.dtype], Tensor] = {}
        self.provider_identity: dict[str, str] = {
            "equation": LOSS_EQUATION_VERSION,
            "ssim_equation": SSIM_EQUATION_VERSION,
            "ssim_backend": "disabled" if config.ssim == 0.0 else config.ssim_backend,
            "lpips_backend": "disabled" if config.lpips == 0.0 else "torchmetrics_alex",
        }
        if config.ssim > 0.0 and config.ssim_backend == "fused":
            if device is None or torch.device(device).type != "cuda":
                raise ContractError("fused SSIM requires an explicit CUDA/HIP device")
            self._fused_ssim, identity = _load_fused_ssim()
            self.provider_identity.update({f"ssim_{key}": value for key, value in identity.items()})
        if config.lpips > 0.0:
            if device is None:
                raise ContractError("LPIPS requires an explicit execution device")
            self._lpips, identity = _load_lpips(device)
            self.provider_identity.update(
                {f"lpips_{key}": value for key, value in identity.items()}
            )

    def _reference_ssim(self, prediction: Tensor, target: Tensor) -> Tensor:
        window_size = _window_size(prediction)
        key = (
            window_size,
            prediction.device.type,
            prediction.device.index,
            prediction.dtype,
        )
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = _gaussian_kernel(prediction, window_size)
            self._kernel_cache[key] = kernel
        return _structural_similarity_with_kernel(
            prediction,
            target,
            padding=self.config.ssim_padding,
            kernel=kernel,
        )

    def _ssim_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self._fused_ssim is None:
            score = self._reference_ssim(prediction, target)
        else:
            score_value: object = self._fused_ssim(
                _to_nchw(prediction),
                _to_nchw(target),
                padding=self.config.ssim_padding,
            )
            if not isinstance(score_value, Tensor) or score_value.numel() != 1:
                raise ContractError("fused SSIM provider must return one scalar tensor")
            score = score_value
            if score.device != prediction.device or score.dtype != prediction.dtype:
                raise ContractError("fused SSIM result device or dtype is invalid")
        return 1.0 - score.reshape(())

    def _lpips_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self._lpips is None:
            return prediction.new_zeros(())
        prediction_nchw = _to_lpips_nchw(prediction.clamp(0.0, 1.0))
        target_nchw = _to_lpips_nchw(target.clamp(0.0, 1.0))
        score_value: object = self._lpips(prediction_nchw, target_nchw)
        self._lpips.reset()
        if not isinstance(score_value, Tensor) or score_value.numel() != 1:
            raise ContractError("LPIPS provider must return one scalar tensor")
        if score_value.device != prediction.device or score_value.dtype != prediction.dtype:
            raise ContractError("LPIPS result device or dtype is invalid")
        return score_value.reshape(())

    def __call__(
        self,
        *,
        model: DynamicGaussianModel,
        materialized: MaterializedGaussians,
        prediction: Tensor,
        target: Tensor,
        color_correction_regularization: Tensor | None = None,
    ) -> dict[str, Tensor]:
        _validate_image_pair(prediction, target)
        zero = prediction.new_zeros(())
        terms = {
            "l1": self.config.l1 * F.l1_loss(prediction, target) if self.config.l1 > 0.0 else zero,
            "ssim": self.config.ssim * self._ssim_loss(prediction, target)
            if self.config.ssim > 0.0
            else zero,
            "lpips": self.config.lpips * self._lpips_loss(prediction, target)
            if self.config.lpips > 0.0
            else zero,
            "opacity": zero,
            "scale": zero,
            "persistence": zero,
            "gate": zero,
            "color_correction": zero,
        }
        if self.config.opacity > 0.0:
            if materialized.temporal_activation.shape != model.opacity_logits.shape:
                raise ContractError("temporal activation shape differs from opacity logits")
            if (
                model.opacity_logits.device != prediction.device
                or materialized.temporal_activation.device != prediction.device
                or model.opacity_logits.dtype != prediction.dtype
                or materialized.temporal_activation.dtype != prediction.dtype
            ):
                raise ContractError("opacity regularization tensors differ from the image layout")
            base_opacity = torch.sigmoid(model.opacity_logits)
            active_base_opacity = base_opacity * materialized.temporal_activation.detach()
            terms["opacity"] = self.config.opacity * active_base_opacity.mean()
        if self.config.scale > 0.0:
            if materialized.scales.ndim != 2 or materialized.scales.shape != model.log_scales.shape:
                raise ContractError("materialized scale shape differs from model log-scales")
            if (
                materialized.scales.device != prediction.device
                or materialized.scales.dtype != prediction.dtype
            ):
                raise ContractError("scale regularization tensors differ from the image layout")
            terms["scale"] = self.config.scale * materialized.scales.mean()
        persistence_weighted = self.config.persistence > 0.0 or self.config.gate > 0.0
        if persistence_weighted and not model.persistence_enabled:
            raise ContractError("persistence regularization requires learned persistence")
        if model.persistence_enabled:
            persistent_fraction = model.gate()
            if self.config.persistence > 0.0:
                terms["persistence"] = self.config.persistence * persistent_fraction.mean()
            if self.config.gate > 0.0:
                terms["gate"] = self.config.gate * (1.0 - persistent_fraction).mean()
        if color_correction_regularization is not None:
            if (
                color_correction_regularization.numel() != 1
                or color_correction_regularization.device != prediction.device
                or color_correction_regularization.dtype != prediction.dtype
            ):
                raise ContractError("color-correction regularization must be one compatible scalar")
            terms["color_correction"] = color_correction_regularization.reshape(())
        if tuple(terms) != LOSS_TERM_NAMES:
            raise AssertionError("loss term catalog changed internally")
        return terms
