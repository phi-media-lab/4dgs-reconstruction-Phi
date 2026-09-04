#!/usr/bin/env python3
"""Capture and compare the pinned MI300X renderer, loss, and training runtime.

The command is intentionally independent of an experiment directory.  It uses
small deterministic tensors, records only content identities (never machine
paths), and writes NumPy archives with ``allow_pickle=False`` compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import torch

GSPLAT_DISTRIBUTION = "amd-gsplat"
GSPLAT_VERSION = "1.5.3+b01acd43e3c7fa942f95fda0974e9125e4de7395"
GSPLAT_REVISION = "b01acd43e3c7fa942f95fda0974e9125e4de7395"
FUSED_SSIM_DISTRIBUTION = "fused-ssim"
FUSED_SSIM_VERSION = "1.0.0"
FUSED_SSIM_REVISION = "a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"
EXPECTED_TORCH = "2.10.0+rocm7.0"
EXPECTED_HIP = "7.0.51831"
EXPECTED_ARCH = "gfx942"

FORWARD_MAX_ABS = 1.0e-6
GRADIENT_MAX_ABS = 1.0e-5
GRADIENT_L2_REL = 1.0e-5
SSIM_VALUE_MAX_ABS = 5.0e-6
SSIM_GRADIENT_MAX_ABS = 2.0e-6
SSIM_GRADIENT_L2_REL = 1.0e-4
SSIM_GRADIENT_COSINE_MIN = 0.99999
TRAINING_STEP_SCALAR_MAX_ABS = 2.0e-6
TRAINING_STEP_GRADIENT_MAX_ABS = 1.0e-6
TRAINING_STEP_GRADIENT_RELATIVE_ERROR = 1.0e-4
TRAINING_STEP_EXACT_FIELDS = (
    "step",
    "next_step",
    "observation_id",
    "camera_id",
    "frame_id",
    "timestamp_seconds",
    "sh_degree",
    "active_gaussians",
    "intersection_count",
)
TRAINING_STEP_SCALAR_FIELDS = ("loss", "psnr", "raw_psnr")
TRAINING_STEP_OPERATIONAL_FIELDS = (
    "iteration_ms",
    "relocation_ms",
    "screen_guard_ms",
    "allocated_bytes",
    "reserved_bytes",
)
TRAINING_STEP_CONFIG_DELTA_CLASSES = {
    "operational_only",
    "inactive_at_compared_step",
    "takes_effect_after_compared_step",
}


class ValidationError(RuntimeError):
    """The runtime does not satisfy the preregistered public-build gate."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValidationError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _direct_wheel(distribution: importlib.metadata.Distribution) -> Path:
    text = distribution.read_text("direct_url.json")
    if text is None:
        raise ValidationError(f"{distribution.metadata['Name']} has no direct_url.json")
    try:
        payload = json.loads(text)
        parsed = urlparse(payload["url"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValidationError("invalid wheel direct_url metadata") from exc
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValidationError("runtime must be installed from a local, hashable wheel")
    path = Path(unquote(parsed.path)).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValidationError("runtime wheel is absent or is not a regular file")
    return path


def _code_architectures(provider: Path) -> list[str]:
    tool = Path("/opt/rocm/bin/roc-obj-ls")
    if not tool.is_file():
        candidate = Path("/usr/bin/roc-obj-ls")
        tool = candidate if candidate.is_file() else tool
    if not tool.is_file():
        raise ValidationError("roc-obj-ls is required to inspect GPU code objects")
    completed = subprocess.run(
        [str(tool), str(provider)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise ValidationError("roc-obj-ls failed for the runtime provider")
    architectures = sorted(set(re.findall(r"gfx[0-9a-z]+", completed.stdout)))
    if architectures != [EXPECTED_ARCH]:
        raise ValidationError(
            f"runtime provider must contain only {EXPECTED_ARCH}, found {architectures}"
        )
    return architectures


def _runtime_base() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise ValidationError("a visible ROCm accelerator is required")
    if str(torch.__version__) != EXPECTED_TORCH or str(torch.version.hip) != EXPECTED_HIP:
        raise ValidationError(
            f"expected Torch/HIP {EXPECTED_TORCH}/{EXPECTED_HIP}, found "
            f"{torch.__version__}/{torch.version.hip}"
        )
    properties = torch.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", "")).split(":", maxsplit=1)[0]
    if architecture != EXPECTED_ARCH:
        raise ValidationError(f"expected {EXPECTED_ARCH}, found {architecture or 'unknown'}")
    return {
        "python_abi": f"cp{os.sys.version_info.major}{os.sys.version_info.minor}",
        "torch": str(torch.__version__),
        "hip": str(torch.version.hip),
        "device": str(properties.name),
        "architecture": architecture,
        "visible_device_count": int(torch.cuda.device_count()),
    }


def _distribution_identity(
    distribution_name: str,
    expected_version: str,
    provider: Path,
) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(distribution_name)
    if distribution.version != expected_version:
        raise ValidationError(
            f"expected {distribution_name} {expected_version}, found {distribution.version}"
        )
    wheel = _direct_wheel(distribution)
    return {
        "distribution": distribution_name,
        "version": distribution.version,
        "wheel": {
            "filename": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": _sha256(wheel),
        },
        "provider": {
            "filename": provider.name,
            "bytes": provider.stat().st_size,
            "sha256": _sha256(provider),
            "code_object_architectures": _code_architectures(provider),
        },
    }


def _assert_module_root(module_file: Path, required_root: Path) -> None:
    try:
        module_file.resolve().relative_to(required_root.resolve())
    except ValueError as exc:
        raise ValidationError("loaded module is outside the required isolated root") from exc


def _gsplat_inputs(*, spherical_harmonics: bool) -> dict[str, torch.Tensor]:
    count = 12
    index = torch.arange(count, dtype=torch.float32, device="cuda")
    means = torch.stack(
        (
            ((index.remainder(4)) - 1.5) * 0.14,
            ((torch.floor(index / 4)) - 1.0) * 0.12,
            2.2 + index * 0.045,
        ),
        dim=1,
    ).requires_grad_(True)
    quaternions = torch.stack(
        (
            torch.ones_like(index),
            (index - 5.5) * 0.003,
            (index.remainder(3) - 1.0) * 0.008,
            (index.remainder(5) - 2.0) * 0.005,
        ),
        dim=1,
    )
    quaternions = torch.nn.functional.normalize(quaternions, dim=1).detach().requires_grad_(True)
    scales = torch.stack(
        (
            0.045 + index.remainder(3) * 0.004,
            0.052 + index.remainder(4) * 0.003,
            0.040 + index.remainder(2) * 0.005,
        ),
        dim=1,
    ).requires_grad_(True)
    opacities = (0.25 + index * 0.035).requires_grad_(True)
    base_color = torch.stack(
        (
            0.15 + index * 0.045,
            0.72 - index * 0.035,
            0.25 + index.remainder(4) * 0.12,
        ),
        dim=1,
    )
    if spherical_harmonics:
        colors = torch.zeros((count, 16, 3), dtype=torch.float32, device="cuda")
        colors[:, 0, :] = base_color
        coefficient = torch.arange(1, 16, dtype=torch.float32, device="cuda")[None, :, None]
        channel = torch.tensor([1.0, -0.75, 0.5], device="cuda")[None, None, :]
        row = (index[:, None, None].remainder(3) - 1.0) * 0.001
        colors[:, 1:, :] = coefficient * channel * 0.0007 + row
    else:
        colors = base_color
    return {
        "means": means,
        "quats": quaternions,
        "scales": scales,
        "opacities": opacities,
        "colors": colors.detach().requires_grad_(True),
    }


def _capture_gsplat_case(
    gsplat: Any,
    *,
    spherical_harmonics: bool,
) -> dict[str, np.ndarray]:
    tensors = _gsplat_inputs(spherical_harmonics=spherical_harmonics)
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")[None]
    intrinsic = torch.tensor(
        [[[58.0, 0.0, 32.0], [0.0, 58.0, 32.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device="cuda",
    )
    rendered, alpha, _ = gsplat.rasterization(
        means=tensors["means"],
        quats=tensors["quats"],
        scales=tensors["scales"],
        opacities=tensors["opacities"],
        colors=tensors["colors"],
        viewmats=viewmat,
        Ks=intrinsic,
        width=64,
        height=64,
        near_plane=0.01,
        far_plane=1.0e10,
        radius_clip=0.0,
        eps2d=0.3,
        sh_degree=3 if spherical_harmonics else None,
        packed=True,
        tile_size=8,
        backgrounds=torch.zeros(3, dtype=torch.float32, device="cuda"),
        render_mode="RGB",
        sparse_grad=False,
        absgrad=False,
        rasterize_mode="classic",
        camera_model="pinhole",
        with_ut=False,
        radial_coeffs=None,
        tangential_coeffs=None,
    )
    weights = torch.linspace(
        0.25,
        1.25,
        rendered.numel(),
        dtype=rendered.dtype,
        device=rendered.device,
    ).reshape_as(rendered)
    loss = (rendered * weights).mean() + 0.13 * alpha.mean()
    loss.backward()
    result = {
        "rendered": rendered.detach().cpu().numpy(),
        "alpha": alpha.detach().cpu().numpy(),
    }
    for name, tensor in tensors.items():
        if tensor.grad is None:
            raise ValidationError(f"gsplat did not produce the {name} gradient")
        result[f"gradient_{name}"] = tensor.grad.detach().cpu().numpy()
        result[f"input_{name}"] = tensor.detach().cpu().numpy()
    return result


def capture_gsplat(output: Path, module_root: Path) -> None:
    if output.exists() or output.is_symlink():
        raise ValidationError(f"refusing to overwrite {output}")
    gsplat: Any = importlib.import_module("gsplat")
    module_file = Path(gsplat.__file__).resolve()
    _assert_module_root(module_file, module_root)
    backend: Any = importlib.import_module("gsplat.cuda._backend")
    provider = Path(backend._C.__file__).resolve()
    _assert_module_root(provider, module_root)

    arrays: dict[str, np.ndarray] = {}
    for case_name, spherical_harmonics in (("rgb", False), ("sh3", True)):
        case = _capture_gsplat_case(gsplat, spherical_harmonics=spherical_harmonics)
        arrays.update({f"{case_name}__{name}": value for name, value in case.items()})
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValidationError("gsplat capture contains non-finite values")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        tensor_path = temporary / "tensors.npz"
        np.savez_compressed(tensor_path, **arrays)
        tensor_catalog = {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            }
            for name, value in sorted(arrays.items())
        }
        payload: dict[str, Any] = {
            "schema_version": "p2g.mi300x_gsplat_capture.v1",
            "source_revision": GSPLAT_REVISION,
            "runtime": _runtime_base(),
            "dependency": _distribution_identity(
                GSPLAT_DISTRIBUTION,
                GSPLAT_VERSION,
                provider,
            ),
            "profile": {
                "cases": ["rgb", "sh3"],
                "camera_count": 1,
                "gaussian_count": 12,
                "image_size": [64, 64],
                "packed": True,
                "tile_size": 8,
                "precision": "float32",
                "gradients": ["means", "quats", "scales", "opacities", "colors"],
            },
            "tensor_archive": {
                "filename": tensor_path.name,
                "bytes": tensor_path.stat().st_size,
                "sha256": _sha256(tensor_path),
            },
            "tensors": tensor_catalog,
        }
        payload["capture_id"] = _canonical_digest(payload)
        _write_json(temporary / "capture.json", payload)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()


def _load_capture(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    try:
        payload = json.loads((root / "capture.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load gsplat capture {root}") from exc
    if payload.get("schema_version") != "p2g.mi300x_gsplat_capture.v1":
        raise ValidationError("unexpected gsplat capture schema")
    claimed_id = payload.pop("capture_id", None)
    if claimed_id != _canonical_digest(payload):
        raise ValidationError("gsplat capture identity mismatch")
    payload["capture_id"] = claimed_id
    archive = root / payload["tensor_archive"]["filename"]
    if (
        archive.stat().st_size != payload["tensor_archive"]["bytes"]
        or _sha256(archive) != payload["tensor_archive"]["sha256"]
    ):
        raise ValidationError("gsplat tensor archive identity mismatch")
    with np.load(archive, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    return payload, arrays


def compare_gsplat(baseline: Path, candidate: Path, output: Path) -> None:
    baseline_payload, baseline_arrays = _load_capture(baseline)
    candidate_payload, candidate_arrays = _load_capture(candidate)
    if set(baseline_arrays) != set(candidate_arrays):
        raise ValidationError("baseline and candidate tensor catalogs differ")
    comparisons: list[dict[str, Any]] = []
    passed = True
    for name in sorted(baseline_arrays):
        reference = baseline_arrays[name]
        observed = candidate_arrays[name]
        if reference.shape != observed.shape or reference.dtype != observed.dtype:
            raise ValidationError(f"tensor type mismatch for {name}")
        difference = np.abs(reference.astype(np.float64) - observed.astype(np.float64))
        exact = bool(np.array_equal(reference, observed))
        maximum_absolute = float(difference.max(initial=0.0))
        reference_norm = float(np.linalg.norm(reference.astype(np.float64).ravel()))
        relative_l2 = float(np.linalg.norm(difference.ravel()) / max(reference_norm, 1.0e-30))
        if "__input_" in name:
            tensor_pass = exact
            kind = "input"
        elif "__gradient_" in name:
            tensor_pass = maximum_absolute <= GRADIENT_MAX_ABS and relative_l2 <= GRADIENT_L2_REL
            kind = "gradient"
        else:
            tensor_pass = maximum_absolute <= FORWARD_MAX_ABS
            kind = "forward"
        passed = passed and tensor_pass
        comparisons.append(
            {
                "tensor": name,
                "kind": kind,
                "exact": exact,
                "maximum_absolute_error": maximum_absolute,
                "relative_l2_error": relative_l2,
                "status": "PASS" if tensor_pass else "FAIL",
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "p2g.mi300x_gsplat_public_build_parity.v1",
        "status": "PASS" if passed else "FAIL",
        "source_revision": GSPLAT_REVISION,
        "baseline_capture_id": baseline_payload["capture_id"],
        "candidate_capture_id": candidate_payload["capture_id"],
        "baseline_dependency": baseline_payload["dependency"],
        "candidate_dependency": candidate_payload["dependency"],
        "thresholds": {
            "forward_maximum_absolute_error": FORWARD_MAX_ABS,
            "gradient_maximum_absolute_error": GRADIENT_MAX_ABS,
            "gradient_relative_l2_error": GRADIENT_L2_REL,
            "input_equality": "bit_exact",
        },
        "comparisons": comparisons,
        "claim": (
            "The checked source matches the baseline MI300X renderer "
            "for the registered RGB/SH3 single-camera packed forward/backward profile."
        ),
        "claim_boundary": (
            "This deterministic parity gate does not establish quality for arbitrary scenes, "
            "unsupported gsplat modes, or other GPU architectures."
        ),
    }
    payload["receipt_id"] = _canonical_digest(payload)
    _write_json(output, payload)
    if not passed:
        raise ValidationError("gsplat public-build numerical parity failed")


def _ssim_images(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(0.0, 1.0, height, device="cuda")[:, None, None]
    x = torch.linspace(0.0, 1.0, width, device="cuda")[None, :, None]
    channels = torch.tensor([0.8, 1.0, 1.2], device="cuda")[None, None, :]
    base = (0.2 + 0.5 * y + 0.2 * x) * channels
    row = torch.arange(height, dtype=torch.float32, device="cuda")[:, None, None]
    column = torch.arange(width, dtype=torch.float32, device="cuda")[None, :, None]
    channel = torch.arange(3, dtype=torch.float32, device="cuda")[None, None, :]
    noise = torch.remainder(row * 17.0 + column * 13.0 + channel * 7.0, 29.0) / 400.0
    prediction = (base + noise).clamp(0.0, 1.0)
    target = (base + torch.flip(noise, dims=[1]) * 0.05).clamp(0.0, 1.0)
    return prediction, target


def _ssim_evaluation(
    fused_ssim: Any,
    structural_similarity: Any,
    prediction: torch.Tensor,
    target: torch.Tensor,
    padding: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reference_input = prediction.clone().requires_grad_(True)
    reference = structural_similarity(reference_input, target, padding=padding)
    reference.backward()
    if reference_input.grad is None:
        raise ValidationError("reference SSIM produced no prediction gradient")
    reference_gradient = reference_input.grad.detach().clone()

    fused_input = prediction.clone().requires_grad_(True)
    fused = fused_ssim(
        fused_input.permute(2, 0, 1)[None],
        target.permute(2, 0, 1)[None],
        padding=padding,
    )
    fused.backward()
    if fused_input.grad is None:
        raise ValidationError("fused SSIM produced no prediction gradient")
    return (
        reference.detach(),
        fused.detach(),
        reference_gradient,
        fused_input.grad.detach().clone(),
    )


def _load_structural_similarity(p2g_source_root: Path) -> Any:
    package = p2g_source_root / "p2g"
    if not package.is_dir():
        raise ValidationError("--p2g-source-root must contain the p2g package")
    # The active train environment may contain an editable finder for a
    # different worktree.  A release check must import only the explicit source
    # root and wheel overlay supplied on its command line.
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if type(finder).__module__ != "_pixel4dgs_editable"
    ]
    sys.path.insert(0, str(p2g_source_root.resolve()))
    module: Any = importlib.import_module("p2g.training.losses")
    function: Any = getattr(module, "structural_similarity", None)
    if not callable(function):
        raise ValidationError("explicit p2g source does not expose structural_similarity")
    source_file_name = inspect.getsourcefile(function)
    if source_file_name is None:
        raise ValidationError("cannot identify the structural-similarity reference source")
    try:
        Path(source_file_name).resolve().relative_to(p2g_source_root.resolve())
    except ValueError as exc:
        raise ValidationError(
            "structural-similarity reference came from another source tree"
        ) from exc
    return function


def validate_fused_ssim(output: Path, module_root: Path, p2g_source_root: Path) -> None:
    fused_module: Any = importlib.import_module("fused_ssim")
    module_file = Path(fused_module.__file__).resolve()
    _assert_module_root(module_file, module_root)
    provider_module: Any = importlib.import_module("fused_ssim_cuda")
    provider = Path(provider_module.__file__).resolve()
    _assert_module_root(provider, module_root)
    structural_similarity = _load_structural_similarity(p2g_source_root)
    source_file_name = inspect.getsourcefile(structural_similarity)
    if source_file_name is None:
        raise ValidationError("cannot identify the structural-similarity reference source")
    source_file = Path(source_file_name).resolve()
    cases = (("small", 16, 17), ("medium", 32, 40), ("square", 64, 64))
    rows: list[dict[str, Any]] = []
    passed = True
    for case_name, height, width in cases:
        prediction, target = _ssim_images(height, width)
        for padding in ("valid", "same"):
            reference, fused, reference_gradient, fused_gradient = _ssim_evaluation(
                fused_module.fused_ssim,
                structural_similarity,
                prediction,
                target,
                padding,
            )
            _, repeat, _, repeat_gradient = _ssim_evaluation(
                fused_module.fused_ssim,
                structural_similarity,
                prediction,
                target,
                padding,
            )
            difference = (reference_gradient - fused_gradient).abs()
            value_error = float((reference - fused).abs().cpu())
            gradient_maximum_absolute = float(difference.max().cpu())
            reference_norm = torch.linalg.vector_norm(reference_gradient)
            gradient_relative_l2 = float(
                (
                    torch.linalg.vector_norm(reference_gradient - fused_gradient) / reference_norm
                ).cpu()
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    reference_gradient.flatten(),
                    fused_gradient.flatten(),
                    dim=0,
                ).cpu()
            )
            repeat_value_exact = bool(torch.equal(fused, repeat))
            repeat_gradient_exact = bool(torch.equal(fused_gradient, repeat_gradient))
            finite = bool(torch.isfinite(fused).all() and torch.isfinite(fused_gradient).all())
            row_pass = (
                finite
                and value_error <= SSIM_VALUE_MAX_ABS
                and gradient_maximum_absolute <= SSIM_GRADIENT_MAX_ABS
                and gradient_relative_l2 <= SSIM_GRADIENT_L2_REL
                and cosine >= SSIM_GRADIENT_COSINE_MIN
                and repeat_value_exact
                and repeat_gradient_exact
            )
            passed = passed and row_pass
            rows.append(
                {
                    "case": case_name,
                    "height": height,
                    "width": width,
                    "padding": padding,
                    "reference_value": float(reference.cpu()),
                    "fused_value": float(fused.cpu()),
                    "value_absolute_error": value_error,
                    "gradient_maximum_absolute_error": gradient_maximum_absolute,
                    "gradient_relative_l2_error": gradient_relative_l2,
                    "gradient_cosine_similarity": cosine,
                    "repeat_value_bit_exact": repeat_value_exact,
                    "repeat_gradient_bit_exact": repeat_gradient_exact,
                    "finite": finite,
                    "status": "PASS" if row_pass else "FAIL",
                }
            )
    payload: dict[str, Any] = {
        "schema_version": "p2g.mi300x_fused_ssim_public_build_parity.v1",
        "status": "PASS" if passed else "FAIL",
        "source_revision": FUSED_SSIM_REVISION,
        "runtime": _runtime_base(),
        "dependency": _distribution_identity(
            FUSED_SSIM_DISTRIBUTION,
            FUSED_SSIM_VERSION,
            provider,
        ),
        "reference": {
            "implementation": "p2g.training.losses.structural_similarity",
            "source_sha256": _sha256(source_file),
            "window": "11x11 Gaussian, sigma 1.5 for every registered case",
        },
        "thresholds": {
            "value_maximum_absolute_error": SSIM_VALUE_MAX_ABS,
            "gradient_maximum_absolute_error": SSIM_GRADIENT_MAX_ABS,
            "gradient_relative_l2_error": SSIM_GRADIENT_L2_REL,
            "gradient_cosine_similarity_minimum": SSIM_GRADIENT_COSINE_MIN,
            "repeatability": "bit_exact",
        },
        "cases": rows,
        "claim": (
            "The clean public-source gfx942 fused-SSIM build matches the explicit "
            "PyTorch SSIM equation for the registered value and prediction-gradient cases."
        ),
        "claim_boundary": (
            "This numerical gate does not by itself reproduce a full 30k training run or "
            "establish behavior for unsupported devices and tensor layouts."
        ),
    }
    payload["receipt_id"] = _canonical_digest(payload)
    _write_json(output, payload)
    if not passed:
        raise ValidationError("fused-SSIM public-build numerical parity failed")


def _regular_file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValidationError(f"input must be a regular non-symlink file: {path}")
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _first_jsonl_record(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _regular_file_identity(path)
    with path.open("rb") as stream:
        first_record = next((line for line in stream if line.strip()), None)
    if first_record is None:
        raise ValidationError(f"metrics JSONL is empty: {path}")
    try:
        decoded = json.loads(first_record)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"first metrics record is invalid JSON: {path}") from exc
    if not isinstance(decoded, dict):
        raise ValidationError("first metrics record must be a JSON object")
    identity.update(
        {
            "first_nonempty_record_index": 0,
            "first_record_bytes": len(first_record),
            "first_record_sha256": hashlib.sha256(first_record).hexdigest(),
        }
    )
    return decoded, identity


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        field = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_mapping(item, field))
        else:
            flattened[field] = item
    return flattened


def _load_training_config_delta_profile(
    path: Path,
    *,
    compared_step: int,
) -> tuple[dict[str, tuple[Any, Any]], dict[str, str], dict[str, Any]]:
    identity = _regular_file_identity(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("training config-delta profile is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValidationError("training config-delta profile must be an object")
    claimed_id = payload.pop("profile_id", None)
    if payload.get("schema_version") != "p2g.training_step_config_delta_profile.v1":
        raise ValidationError("unexpected training config-delta profile schema")
    if claimed_id != _canonical_digest(payload):
        raise ValidationError("training config-delta profile identity mismatch")
    if payload.get("comparison_step") != compared_step:
        raise ValidationError("config-delta profile targets a different training step")
    raw_differences = payload.get("differences")
    if not isinstance(raw_differences, list) or not raw_differences:
        raise ValidationError("config-delta profile must contain differences")

    expected: dict[str, tuple[Any, Any]] = {}
    classifications: dict[str, str] = {}
    for index, raw in enumerate(raw_differences):
        if not isinstance(raw, dict) or set(raw) != {
            "field",
            "baseline",
            "candidate",
            "classification",
            "rationale",
        }:
            raise ValidationError(f"invalid config-delta profile row {index}")
        field = raw["field"]
        classification = raw["classification"]
        rationale = raw["rationale"]
        if (
            not isinstance(field, str)
            or not field
            or field.startswith(".")
            or field.endswith(".")
            or ".." in field
        ):
            raise ValidationError(f"invalid config-delta field in row {index}")
        if field in expected:
            raise ValidationError(f"duplicate config-delta field: {field}")
        if classification not in TRAINING_STEP_CONFIG_DELTA_CLASSES:
            raise ValidationError(f"invalid config-delta classification for {field}")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValidationError(f"config-delta rationale is empty for {field}")
        expected[field] = (raw["baseline"], raw["candidate"])
        classifications[field] = classification
    identity["profile_id"] = claimed_id
    identity["difference_count"] = len(expected)
    return expected, classifications, identity


def _training_config_comparison(
    baseline_path: Path,
    candidate_path: Path,
    delta_profile_path: Path,
    *,
    compared_step: int,
) -> tuple[dict[str, Any], bool]:
    expected_changes, classifications, profile_identity = _load_training_config_delta_profile(
        delta_profile_path,
        compared_step=compared_step,
    )
    baseline_identity = _regular_file_identity(baseline_path)
    candidate_identity = _regular_file_identity(candidate_path)
    with baseline_path.open("rb") as stream:
        baseline = _flatten_mapping(tomllib.load(stream))
    with candidate_path.open("rb") as stream:
        candidate = _flatten_mapping(tomllib.load(stream))
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        added = sorted(set(candidate) - set(baseline))
        raise ValidationError(
            f"training configuration fields differ; missing={missing}, added={added}"
        )

    observed_changes = {
        field: (baseline[field], candidate[field])
        for field in baseline
        if baseline[field] != candidate[field]
    }
    expected_change_fields = set(expected_changes)
    if set(observed_changes) != expected_change_fields:
        missing = sorted(expected_change_fields - set(observed_changes))
        added = sorted(set(observed_changes) - expected_change_fields)
        raise ValidationError(
            f"unregistered training configuration delta; missing={missing}, added={added}"
        )

    rows: list[dict[str, Any]] = []
    passed = True
    for field, expected_values in expected_changes.items():
        observed_values = observed_changes[field]
        row_pass = observed_values == expected_values
        passed = passed and row_pass
        rows.append(
            {
                "field": field,
                "baseline": observed_values[0],
                "candidate": observed_values[1],
                "classification": classifications[field],
                "status": "PASS" if row_pass else "FAIL",
            }
        )
    return (
        {
            "baseline": baseline_identity,
            "candidate": candidate_identity,
            "delta_profile": profile_identity,
            "common_field_count": len(baseline) - len(observed_changes),
            "registered_differences": rows,
            "status": "PASS" if passed else "FAIL",
        },
        passed,
    )


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"training metric {field} must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise ValidationError(f"training metric {field} must be finite")
    return number


def compare_training_step(
    baseline_metrics: Path,
    candidate_metrics: Path,
    baseline_config: Path,
    candidate_config: Path,
    config_delta_profile: Path,
    output: Path,
) -> None:
    baseline, baseline_identity = _first_jsonl_record(baseline_metrics)
    candidate, candidate_identity = _first_jsonl_record(candidate_metrics)
    expected_fields = (
        set(TRAINING_STEP_EXACT_FIELDS)
        | set(TRAINING_STEP_SCALAR_FIELDS)
        | set(TRAINING_STEP_OPERATIONAL_FIELDS)
        | {"loss_terms", "gradient_norms"}
    )
    for label, row in (("baseline", baseline), ("candidate", candidate)):
        if set(row) != expected_fields:
            missing = sorted(expected_fields - set(row))
            added = sorted(set(row) - expected_fields)
            raise ValidationError(
                f"{label} training metric schema differs; missing={missing}, added={added}"
            )
        for field in TRAINING_STEP_OPERATIONAL_FIELDS:
            _finite_number(row[field], field=f"{label}.{field}")

    exact_rows: list[dict[str, Any]] = []
    passed = True
    for field in TRAINING_STEP_EXACT_FIELDS:
        row_pass = baseline[field] == candidate[field]
        passed = passed and row_pass
        exact_rows.append(
            {
                "field": field,
                "baseline": baseline[field],
                "candidate": candidate[field],
                "status": "PASS" if row_pass else "FAIL",
            }
        )

    scalar_rows: list[dict[str, Any]] = []
    for field in TRAINING_STEP_SCALAR_FIELDS:
        reference = _finite_number(baseline[field], field=f"baseline.{field}")
        observed = _finite_number(candidate[field], field=f"candidate.{field}")
        absolute_error = abs(reference - observed)
        row_pass = absolute_error <= TRAINING_STEP_SCALAR_MAX_ABS
        passed = passed and row_pass
        scalar_rows.append(
            {
                "field": field,
                "baseline": reference,
                "candidate": observed,
                "absolute_error": absolute_error,
                "status": "PASS" if row_pass else "FAIL",
            }
        )

    baseline_loss_terms = baseline["loss_terms"]
    candidate_loss_terms = candidate["loss_terms"]
    baseline_gradients = baseline["gradient_norms"]
    candidate_gradients = candidate["gradient_norms"]
    for field, value in (
        ("baseline.loss_terms", baseline_loss_terms),
        ("candidate.loss_terms", candidate_loss_terms),
        ("baseline.gradient_norms", baseline_gradients),
        ("candidate.gradient_norms", candidate_gradients),
    ):
        if not isinstance(value, dict):
            raise ValidationError(f"training metric {field} must be an object")
    if set(baseline_loss_terms) != set(candidate_loss_terms):
        raise ValidationError("baseline and candidate loss-term catalogs differ")
    if set(baseline_gradients) != set(candidate_gradients):
        raise ValidationError("baseline and candidate gradient-norm catalogs differ")

    loss_rows: list[dict[str, Any]] = []
    for field in sorted(baseline_loss_terms):
        reference = _finite_number(baseline_loss_terms[field], field=f"baseline.loss_terms.{field}")
        observed = _finite_number(
            candidate_loss_terms[field], field=f"candidate.loss_terms.{field}"
        )
        absolute_error = abs(reference - observed)
        row_pass = absolute_error <= TRAINING_STEP_SCALAR_MAX_ABS
        passed = passed and row_pass
        loss_rows.append(
            {
                "field": field,
                "baseline": reference,
                "candidate": observed,
                "absolute_error": absolute_error,
                "status": "PASS" if row_pass else "FAIL",
            }
        )

    gradient_rows: list[dict[str, Any]] = []
    for field in sorted(baseline_gradients):
        reference = _finite_number(
            baseline_gradients[field], field=f"baseline.gradient_norms.{field}"
        )
        observed = _finite_number(
            candidate_gradients[field], field=f"candidate.gradient_norms.{field}"
        )
        absolute_error = abs(reference - observed)
        relative_error = absolute_error / max(abs(reference), 1.0e-30)
        row_pass = (
            absolute_error <= TRAINING_STEP_GRADIENT_MAX_ABS
            and relative_error <= TRAINING_STEP_GRADIENT_RELATIVE_ERROR
        )
        passed = passed and row_pass
        gradient_rows.append(
            {
                "field": field,
                "baseline": reference,
                "candidate": observed,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "status": "PASS" if row_pass else "FAIL",
            }
        )

    configuration, configuration_pass = _training_config_comparison(
        baseline_config,
        candidate_config,
        config_delta_profile,
        compared_step=int(baseline["step"]),
    )
    passed = passed and configuration_pass
    payload: dict[str, Any] = {
        "schema_version": "p2g.mi300x_public_runtime_training_step_parity.v1",
        "status": "PASS" if passed else "FAIL",
        "records_compared": [0],
        "metrics_sources": {
            "baseline": baseline_identity,
            "candidate": candidate_identity,
        },
        "configuration": configuration,
        "thresholds": {
            "identity_fields": "exact",
            "scalar_and_loss_term_maximum_absolute_error": (TRAINING_STEP_SCALAR_MAX_ABS),
            "gradient_norm_maximum_absolute_error": (TRAINING_STEP_GRADIENT_MAX_ABS),
            "gradient_norm_relative_error": (TRAINING_STEP_GRADIENT_RELATIVE_ERROR),
        },
        "exact_field_comparisons": exact_rows,
        "scalar_comparisons": scalar_rows,
        "loss_term_comparisons": loss_rows,
        "gradient_norm_comparisons": gradient_rows,
        "operational_fields_excluded_from_parity": list(TRAINING_STEP_OPERATIONAL_FIELDS),
        "summary": {
            "maximum_scalar_absolute_error": max(row["absolute_error"] for row in scalar_rows),
            "maximum_loss_term_absolute_error": max(row["absolute_error"] for row in loss_rows),
            "maximum_gradient_norm_absolute_error": max(
                row["absolute_error"] for row in gradient_rows
            ),
            "maximum_gradient_norm_relative_error": max(
                row["relative_error"] for row in gradient_rows
            ),
        },
        "claim": (
            "For the registered observation, initialization, and config-delta profile, "
            "the candidate MI300X runtime reproduces the baseline step losses, PSNR, "
            "Gaussian counts, and per-parameter gradient norms within the registered bounds."
        ),
        "claim_boundary": (
            "Only the first optimization step and gradient norms are compared. The short "
            "candidate configuration changes cache and run length and disables relocation "
            "and screen guard, which are inactive at step 0. This receipt does not establish "
            "later optimizer state, sampling trajectory, relocation, screen guard, final "
            "30k quality, performance parity, or rights to redistribute source data."
        ),
    }
    payload["receipt_id"] = _canonical_digest(payload)
    _write_json(output, payload)
    if not passed:
        raise ValidationError("public-runtime training step parity failed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture-gsplat")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--module-root", type=Path, required=True)

    compare = subparsers.add_parser("compare-gsplat")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    fused = subparsers.add_parser("validate-fused-ssim")
    fused.add_argument("--output", type=Path, required=True)
    fused.add_argument("--module-root", type=Path, required=True)
    fused.add_argument("--p2g-source-root", type=Path, required=True)

    training_step = subparsers.add_parser("compare-training-step")
    training_step.add_argument("--baseline-metrics", type=Path, required=True)
    training_step.add_argument("--candidate-metrics", type=Path, required=True)
    training_step.add_argument("--baseline-config", type=Path, required=True)
    training_step.add_argument("--candidate-config", type=Path, required=True)
    training_step.add_argument("--config-delta-profile", type=Path, required=True)
    training_step.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        if parsed.command == "capture-gsplat":
            capture_gsplat(parsed.output, parsed.module_root)
        elif parsed.command == "compare-gsplat":
            compare_gsplat(parsed.baseline, parsed.candidate, parsed.output)
        elif parsed.command == "validate-fused-ssim":
            validate_fused_ssim(parsed.output, parsed.module_root, parsed.p2g_source_root)
        elif parsed.command == "compare-training-step":
            compare_training_step(
                parsed.baseline_metrics,
                parsed.candidate_metrics,
                parsed.baseline_config,
                parsed.candidate_config,
                parsed.config_delta_profile,
                parsed.output,
            )
        else:  # pragma: no cover - argparse enforces the choices
            raise ValidationError(f"unknown command {parsed.command}")
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
