"""Hash-closed local resume checkpoints and safe tensor-only model exports."""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, cast

import numpy as np
import torch
from torch import Tensor

from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json
from p2g.errors import ContractError, OutputExistsError
from p2g.training.config import RunConfig
from p2g.training.dataset import SceneSampler
from p2g.training.model import MODEL_EQUATION_VERSION, DynamicGaussianModel
from p2g.training.optim import OptimizerBundle
from p2g.training.photometric import CameraColorCorrectors

CHECKPOINT_SCHEMA = "p2g.train_checkpoint.v1"
CHECKPOINT_MANIFEST_SCHEMA = "p2g.train_checkpoint_manifest.v1"
MODEL_EXPORT_SCHEMA = "p2g.dynamic_gaussian_model.v1"
COLOR_EXPORT_SCHEMA = "p2g.camera_color_correctors.v1"

STATE_FILENAME = "state.pt"
CONFIG_FILENAME = "config.toml"
METADATA_FILENAME = "metadata.json"
MANIFEST_FILENAME = "manifest.json"

_CHECKPOINT_NAME = re.compile(r"step_([0-9]{8})\Z")
_STATE_FIELDS = {
    "schema_version",
    "next_step",
    "model",
    "optimizers",
    "sampler",
    "rng",
    "relocation",
    "color_correctors",
}


class _SafeTensorReader(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def metadata(self) -> dict[str, str] | None: ...


def _step(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 99_999_999:
        raise ContractError("checkpoint next_step must be an integer in [0, 99999999]")
    return value


def _flush_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _flush_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value: Any = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    result = cast(dict[str, Any], value)
    if raw != canonical_json_bytes(result):
        raise ContractError(f"{label} must use canonical JSON encoding")
    return result


def _safe_state(value: object, *, location: str = "state", depth: int = 0) -> None:
    if depth > 64:
        raise ContractError("checkpoint state nesting is too deep")
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ContractError(f"checkpoint {location} contains a non-finite float")
        return
    if isinstance(value, Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ContractError(f"checkpoint {location} contains a non-finite tensor")
        return
    if isinstance(value, Mapping):
        for key, child in cast(Mapping[object, object], value).items():
            if type(key) not in {str, int}:
                raise ContractError(f"checkpoint {location} has an unsafe mapping key")
            _safe_state(child, location=f"{location}.{key}", depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(cast(Sequence[object], value)):
            _safe_state(child, location=f"{location}[{index}]", depth=depth + 1)
        return
    raise ContractError(f"checkpoint {location} has unsupported type {type(value).__name__}")


def _encode_python_rng() -> dict[str, Any]:
    version, words, gaussian = random.getstate()
    return {
        "version": version,
        "words": list(words),
        "gauss_next": gaussian,
    }


def _decode_python_rng(value: object) -> tuple[int, tuple[int, ...], float | None]:
    if not isinstance(value, dict):
        raise ContractError("checkpoint Python RNG state has invalid fields")
    raw = cast(dict[str, Any], value)
    if set(raw) != {"version", "words", "gauss_next"}:
        raise ContractError("checkpoint Python RNG state has invalid fields")
    version = raw["version"]
    words = raw["words"]
    gaussian = raw["gauss_next"]
    if type(version) is not int or not isinstance(words, list) or not words:
        raise ContractError("checkpoint Python RNG state has invalid values")
    if any(type(word) is not int for word in cast(list[Any], words)):
        raise ContractError("checkpoint Python RNG words must be integers")
    if gaussian is not None and (
        not isinstance(gaussian, (int, float))
        or isinstance(gaussian, bool)
        or not math.isfinite(gaussian)
    ):
        raise ContractError("checkpoint Python RNG Gaussian cache must be finite")
    result = (
        version,
        tuple(cast(list[int], words)),
        None if gaussian is None else float(gaussian),
    )
    candidate = random.Random()
    try:
        candidate.setstate(result)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"checkpoint Python RNG state is invalid: {exc}") from exc
    return result


def _encode_numpy_rng() -> dict[str, Any]:
    name, keys, position, has_gaussian, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": name,
        "keys": [int(value) for value in keys],
        "position": int(position),
        "has_gaussian": int(has_gaussian),
        "cached_gaussian": float(cached_gaussian),
    }


def _decode_numpy_rng(
    value: object,
) -> tuple[str, np.ndarray[Any, np.dtype[np.uint32]], int, int, float]:
    fields = {"bit_generator", "keys", "position", "has_gaussian", "cached_gaussian"}
    if not isinstance(value, dict):
        raise ContractError("checkpoint NumPy RNG state has invalid fields")
    raw = cast(dict[str, Any], value)
    if set(raw) != fields:
        raise ContractError("checkpoint NumPy RNG state has invalid fields")
    name = raw["bit_generator"]
    keys = raw["keys"]
    position = raw["position"]
    has_gaussian = raw["has_gaussian"]
    cached_gaussian = raw["cached_gaussian"]
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(keys, list)
        or not keys
        or any(type(item) is not int or not 0 <= item < 2**32 for item in cast(list[Any], keys))
        or type(position) is not int
        or type(has_gaussian) is not int
        or has_gaussian not in {0, 1}
        or not isinstance(cached_gaussian, (int, float))
        or isinstance(cached_gaussian, bool)
        or not math.isfinite(cached_gaussian)
    ):
        raise ContractError("checkpoint NumPy RNG state has invalid values")
    result = (
        name,
        np.asarray(cast(list[int], keys), dtype=np.uint32),
        position,
        has_gaussian,
        float(cached_gaussian),
    )
    candidate = np.random.RandomState()
    try:
        candidate.set_state(result)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"checkpoint NumPy RNG state is invalid: {exc}") from exc
    return result


def capture_rng_state() -> dict[str, Any]:
    """Capture global generators using tensors and primitive containers only."""

    return {
        "schema_version": "p2g.rng_state.v1",
        "python": _encode_python_rng(),
        "numpy": _encode_numpy_rng(),
        "torch_cpu": torch.get_rng_state().cpu().contiguous(),
        "torch_cuda": [
            value.cpu().contiguous()
            for value in (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        ],
    }


def _byte_rng_tensor(value: object, *, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.uint8
        or value.ndim != 1
        or value.numel() == 0
    ):
        raise ContractError(f"checkpoint {name} must be a non-empty CPU uint8 vector")
    return value.contiguous()


def restore_rng_state(state: Mapping[str, Any]) -> None:
    fields = {"schema_version", "python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != fields or state.get("schema_version") != "p2g.rng_state.v1":
        raise ContractError("checkpoint RNG state has invalid fields or schema")
    python_state = _decode_python_rng(state["python"])
    numpy_state = _decode_numpy_rng(state["numpy"])
    torch_cpu = _byte_rng_tensor(state["torch_cpu"], name="torch CPU RNG")
    raw_cuda = state["torch_cuda"]
    if not isinstance(raw_cuda, list):
        raise ContractError("checkpoint torch CUDA RNG state must be a list")
    torch_cuda = [
        _byte_rng_tensor(item, name=f"torch CUDA RNG {index}")
        for index, item in enumerate(cast(list[Any], raw_cuda))
    ]
    if torch.cuda.is_available() and len(torch_cuda) != torch.cuda.device_count():
        raise ContractError("checkpoint CUDA RNG count differs from the visible device count")
    try:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_cpu)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(torch_cuda)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ContractError(f"checkpoint RNG state is incompatible: {exc}") from exc


def checkpoint_path(run_dir: Path, next_step: int) -> Path:
    return run_dir.resolve() / "checkpoints" / f"step_{_step(next_step):08d}"


def latest_checkpoint(run_dir: Path) -> Path:
    root = run_dir.resolve() / "checkpoints"
    if not root.is_dir() or root.is_symlink():
        raise ContractError("run has no regular checkpoint directory")
    candidates: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match is not None and path.is_dir() and not path.is_symlink():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ContractError("run has no completed checkpoint")
    selected = max(candidates, key=lambda item: item[0])[1]
    _verify_checkpoint_manifest(selected)
    return selected


def _model_matches_config(model: DynamicGaussianModel, config: RunConfig) -> None:
    expected_persistence = config.model.persistence == "learned"
    if model.persistence_enabled is not expected_persistence:
        raise ContractError("model persistence mode differs from the run config")
    scale = float(model.gate_logit_scale.detach().cpu().item())
    if not math.isclose(scale, config.model.gate_logit_scale, rel_tol=1.0e-6, abs_tol=1.0e-8):
        raise ContractError("model gate scale differs from the run config")
    if model.max_sh_degree != config.initialization.sh_degree:
        raise ContractError("model SH degree differs from the run config")


def _checkpoint_files(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / STATE_FILENAME,
        root / CONFIG_FILENAME,
        root / METADATA_FILENAME,
        root / MANIFEST_FILENAME,
    )


def save_checkpoint(
    run_dir: Path,
    *,
    next_step: int,
    config: RunConfig,
    model: DynamicGaussianModel,
    optimizers: OptimizerBundle,
    sampler: SceneSampler,
    relocation_state: dict[str, Any] | None = None,
    color_correctors: CameraColorCorrectors | None = None,
) -> Path:
    """Atomically write a hash-closed local-only resume checkpoint."""

    step = _step(next_step)
    config.validate()
    if step > config.training.iterations:
        raise ContractError("checkpoint step is outside the configured training interval")
    _model_matches_config(model, config)
    run_path = run_dir.expanduser()
    if run_path.exists() and (run_path.is_symlink() or not run_path.is_dir()):
        raise ContractError("checkpoint run path must be a regular directory")
    run_path = run_path.resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoint_root = run_path / "checkpoints"
    if checkpoint_root.exists() and (checkpoint_root.is_symlink() or not checkpoint_root.is_dir()):
        raise ContractError("checkpoint root must be a regular directory")
    checkpoint_root.mkdir(exist_ok=True)
    target = checkpoint_root / f"step_{step:08d}"
    if target.exists():
        raise OutputExistsError(f"refusing to overwrite training checkpoint: {target}")
    color_state = (
        None
        if color_correctors is None
        else {
            "camera_ids": list(color_correctors.camera_ids),
            "state": color_correctors.state_dict(),
        }
    )
    if (color_correctors is None) != (config.color_correction.mode == "off"):
        raise ContractError("color-correction module differs from the run config")
    state: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "next_step": step,
        "model": model.state_dict(),
        "optimizers": optimizers.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": capture_rng_state(),
        "relocation": {} if relocation_state is None else relocation_state,
        "color_correctors": color_state,
    }
    _safe_state(state)
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "trust_scope": "local_resume_only",
        "redistributable": False,
        "next_step": step,
        "gaussian_count": model.count,
        "max_sh_degree": model.max_sh_degree,
        "persistence": config.model.persistence,
        "gate_logit_scale": config.model.gate_logit_scale,
        "color_correction": config.color_correction.mode,
        "torch": str(torch.__version__),
        "hip": None if torch.version.hip is None else str(torch.version.hip),
    }
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        state_path, config_path, metadata_path, manifest_path = _checkpoint_files(temporary)
        torch.save(state, state_path)
        config_path.write_bytes(config.to_toml_bytes())
        metadata_path.write_bytes(canonical_json_bytes(metadata))
        for path in (state_path, config_path, metadata_path):
            _flush_file(path)
        files = [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (config_path, metadata_path, state_path)
        ]
        manifest = {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
            "checkpoint_id": sha256_json(files),
            "files": files,
        }
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        _flush_file(manifest_path)
        _flush_directory(temporary)
        if target.exists():
            raise OutputExistsError(f"checkpoint destination appeared during write: {target}")
        os.rename(temporary, target)
        _flush_directory(target.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _verify_checkpoint_manifest(root: Path) -> None:
    state_path, config_path, metadata_path, manifest_path = _checkpoint_files(root)
    expected_names = {path.name for path in (state_path, config_path, metadata_path, manifest_path)}
    if {path.name for path in root.iterdir()} != expected_names:
        raise ContractError("checkpoint contains an undeclared or missing path")
    for path in (state_path, config_path, metadata_path, manifest_path):
        if not path.is_file() or path.is_symlink():
            raise ContractError("checkpoint contains an unsafe file")
    manifest = _read_json_object(manifest_path, label="checkpoint manifest")
    if manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA:
        raise ContractError("checkpoint manifest schema is unsupported")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ContractError("checkpoint manifest must bind exactly three payload files")
    files = cast(list[Any], raw_files)
    if len(files) != 3:
        raise ContractError("checkpoint manifest must bind exactly three payload files")
    if not all(isinstance(item, dict) for item in files):
        raise ContractError("checkpoint manifest file records must be objects")
    records = cast(list[dict[str, Any]], files)
    if manifest.get("checkpoint_id") != sha256_json(records):
        raise ContractError("checkpoint manifest identity is invalid")
    expected_payloads = {CONFIG_FILENAME, METADATA_FILENAME, STATE_FILENAME}
    if {item.get("path") for item in records} != expected_payloads:
        raise ContractError("checkpoint manifest file catalog is invalid")
    for item in records:
        path = root / cast(str, item["path"])
        if item.get("bytes") != path.stat().st_size or item.get("sha256") != sha256_file(path):
            raise ContractError(f"checkpoint payload digest mismatch: {path.name}")


def read_checkpoint(checkpoint: Path) -> tuple[RunConfig, dict[str, Any], dict[str, Any]]:
    """Verify a checkpoint before using Torch's restricted weights-only loader."""

    checkpoint = checkpoint.expanduser()
    if checkpoint.is_symlink():
        raise ContractError("training checkpoint must be a regular directory")
    try:
        checkpoint = checkpoint.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"cannot resolve training checkpoint: {exc}") from exc
    if not checkpoint.is_dir():
        raise ContractError("training checkpoint must be a regular directory")
    _verify_checkpoint_manifest(checkpoint)
    config = RunConfig.load(checkpoint / CONFIG_FILENAME)
    metadata = _read_json_object(checkpoint / METADATA_FILENAME, label="checkpoint metadata")
    try:
        loaded: Any = torch.load(
            checkpoint / STATE_FILENAME,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ContractError(f"cannot load restricted checkpoint state: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ContractError("checkpoint state must be a mapping")
    state = cast(dict[str, Any], loaded)
    if set(state) != _STATE_FIELDS or state.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ContractError("checkpoint state has invalid fields or schema")
    _safe_state(state)
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ContractError("checkpoint metadata schema is unsupported")
    step = _step(state["next_step"])
    if step != metadata.get("next_step") or step > config.training.iterations:
        raise ContractError("checkpoint step disagrees with metadata or config")
    if (
        metadata.get("trust_scope") != "local_resume_only"
        or metadata.get("redistributable") is not False
    ):
        raise ContractError("checkpoint trust scope is invalid")
    raw_gate_scale = metadata.get("gate_logit_scale")
    if (
        isinstance(raw_gate_scale, bool)
        or not isinstance(raw_gate_scale, (int, float))
        or not math.isfinite(raw_gate_scale)
    ):
        raise ContractError("checkpoint gate scale is not a finite JSON number")
    if metadata.get("persistence") != config.model.persistence or not math.isclose(
        float(raw_gate_scale),
        config.model.gate_logit_scale,
        rel_tol=1.0e-6,
        abs_tol=1.0e-8,
    ):
        raise ContractError("checkpoint model policy disagrees with the config")
    model_state = state.get("model")
    if not isinstance(model_state, dict) or "means" not in model_state:
        raise ContractError("checkpoint model state is incomplete")
    typed_model_state = cast(dict[str, Any], model_state)
    means = typed_model_state["means"]
    if not isinstance(means, Tensor) or means.ndim != 2:
        raise ContractError("checkpoint model means are invalid")
    if metadata.get("gaussian_count") != int(means.shape[0]):
        raise ContractError("checkpoint Gaussian count disagrees with its model")
    _verify_checkpoint_manifest(checkpoint)
    return config, state, metadata


def restore_training_state(
    state: dict[str, Any],
    *,
    model: DynamicGaussianModel,
    optimizers: OptimizerBundle,
    sampler: SceneSampler,
    color_correctors: CameraColorCorrectors | None = None,
) -> int:
    """Restore a verified state into a freshly constructed training runtime."""

    if set(state) != _STATE_FIELDS or state.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ContractError("checkpoint state has invalid fields or schema")
    _safe_state(state)
    try:
        model_state = state["model"]
        if not isinstance(model_state, dict):
            raise ContractError("checkpoint model state must be a mapping")
        model.load_state_dict(cast(dict[str, Tensor], model_state), strict=True)
        saved_color = state["color_correctors"]
        if color_correctors is None:
            if saved_color is not None:
                raise ContractError("checkpoint contains disabled color correction state")
        else:
            if not isinstance(saved_color, dict):
                raise ContractError("checkpoint is missing enabled color correction state")
            color = cast(dict[str, Any], saved_color)
            if color.get("camera_ids") != list(color_correctors.camera_ids):
                raise ContractError("checkpoint color-correction cameras differ from the scene")
            color_state = color.get("state")
            if not isinstance(color_state, dict):
                raise ContractError("checkpoint color-correction tensors are invalid")
            color_correctors.load_state_dict(cast(dict[str, Tensor], color_state), strict=True)
        optimizer_state = state["optimizers"]
        sampler_state = state["sampler"]
        if not isinstance(optimizer_state, dict) or not isinstance(sampler_state, dict):
            raise ContractError("checkpoint optimizer or sampler state is invalid")
        optimizers.load_state_dict(cast(dict[str, Any], optimizer_state))
        sampler.load_state_dict(cast(dict[str, Any], sampler_state))
        next_step = _step(state["next_step"])
        rng = state["rng"]
        if not isinstance(rng, dict):
            raise ContractError("checkpoint RNG state must be a mapping")
        restore_rng_state(cast(dict[str, Any], rng))
    except ContractError:
        raise
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ContractError(f"checkpoint training state is incompatible: {exc}") from exc
    return next_step


def _export_tensors(model: DynamicGaussianModel) -> dict[str, Tensor]:
    tensors = {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in model.state_dict().items()
    }
    _safe_state(tensors, location="export")
    return tensors


def export_model(
    run_dir: Path,
    *,
    model: DynamicGaussianModel,
    config: RunConfig,
    final_step: int,
    color_correctors: CameraColorCorrectors | None = None,
) -> tuple[Path, Path]:
    """Write a safe local model export; AssetBundle remains the publication format."""

    from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

    step = _step(final_step)
    config.validate()
    if step > config.training.iterations:
        raise ContractError("model export step is outside the configured training interval")
    _model_matches_config(model, config)
    run_dir = run_dir.expanduser()
    if run_dir.is_symlink():
        raise ContractError("model export run directory must already exist")
    try:
        run_dir = run_dir.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"cannot resolve model export run directory: {exc}") from exc
    if not run_dir.is_dir():
        raise ContractError("model export run directory must already exist")
    tensor_target = run_dir / "model.safetensors"
    metadata_target = run_dir / "model.json"
    color_targets = (run_dir / "color_correctors.safetensors", run_dir / "color_correctors.json")
    targets = [tensor_target, metadata_target]
    if color_correctors is not None:
        targets.extend(color_targets)
    if any(path.exists() for path in targets):
        raise OutputExistsError("refusing to overwrite a model export")
    if (color_correctors is None) != (config.color_correction.mode == "off"):
        raise ContractError("export color correction differs from the run config")
    temporary = Path(tempfile.mkdtemp(prefix=".model-export.", dir=run_dir))
    published: list[Path] = []
    try:
        tensor_temp = temporary / tensor_target.name
        tensors = _export_tensors(model)
        save_file(
            tensors,
            str(tensor_temp),
            metadata={
                "schema_version": MODEL_EXPORT_SCHEMA,
                "equation_version": MODEL_EQUATION_VERSION,
                "persistence": config.model.persistence,
                "gate_logit_scale": repr(config.model.gate_logit_scale),
            },
        )
        _flush_file(tensor_temp)
        metadata = {
            "schema_version": MODEL_EXPORT_SCHEMA,
            "artifact_role": "local_training_export",
            "final_step": step,
            "gaussian_count": model.count,
            "max_sh_degree": model.max_sh_degree,
            "persistence": config.model.persistence,
            "gate_logit_scale": config.model.gate_logit_scale,
            "planes": sorted(tensors),
            "tensor": {
                "file": tensor_target.name,
                "bytes": tensor_temp.stat().st_size,
                "sha256": sha256_file(tensor_temp),
            },
        }
        metadata_temp = temporary / metadata_target.name
        metadata_temp.write_bytes(canonical_json_bytes(metadata))
        _flush_file(metadata_temp)
        staged = [(tensor_temp, tensor_target), (metadata_temp, metadata_target)]
        if color_correctors is not None:
            color_tensor_temp = temporary / color_targets[0].name
            color_tensors = {
                name: value.detach().cpu().contiguous()
                for name, value in color_correctors.state_dict().items()
                if value.is_floating_point()
            }
            _safe_state(color_tensors, location="color export")
            save_file(
                color_tensors,
                str(color_tensor_temp),
                metadata={"schema_version": COLOR_EXPORT_SCHEMA},
            )
            _flush_file(color_tensor_temp)
            color_metadata_temp = temporary / color_targets[1].name
            color_metadata_temp.write_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": COLOR_EXPORT_SCHEMA,
                        "camera_ids": list(color_correctors.camera_ids),
                        "planes": sorted(color_tensors),
                        "applied_at_inference": False,
                        "tensor": {
                            "file": color_targets[0].name,
                            "bytes": color_tensor_temp.stat().st_size,
                            "sha256": sha256_file(color_tensor_temp),
                        },
                    }
                )
            )
            _flush_file(color_metadata_temp)
            staged.extend(
                [
                    (color_tensor_temp, color_targets[0]),
                    (color_metadata_temp, color_targets[1]),
                ]
            )
        for source, target in staged:
            try:
                os.link(source, target)
            except FileExistsError as exc:
                raise OutputExistsError(f"model export destination appeared: {target}") from exc
            published.append(target)
        _flush_directory(run_dir)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return tensor_target, metadata_target


def load_exported_model(
    tensor_path: Path,
    *,
    metadata_path: Path | None = None,
) -> tuple[DynamicGaussianModel, dict[str, Any]]:
    """Hash-check and load a tensor-only local model export on CPU."""

    from safetensors import safe_open
    from safetensors.torch import load_file  # pyright: ignore[reportUnknownVariableType]

    tensor_path = tensor_path.expanduser()
    if tensor_path.is_symlink():
        raise ContractError("exported model requires regular tensor and metadata files")
    try:
        tensor_path = tensor_path.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"cannot resolve exported model tensor: {exc}") from exc
    raw_metadata_path = tensor_path.with_suffix(".json") if metadata_path is None else metadata_path
    raw_metadata_path = raw_metadata_path.expanduser()
    if raw_metadata_path.is_symlink():
        raise ContractError("exported model requires regular tensor and metadata files")
    try:
        metadata_path = raw_metadata_path.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"cannot resolve exported model metadata: {exc}") from exc
    if (
        not tensor_path.is_file()
        or tensor_path.is_symlink()
        or not metadata_path.is_file()
        or metadata_path.is_symlink()
    ):
        raise ContractError("exported model requires regular tensor and metadata files")
    metadata = _read_json_object(metadata_path, label="exported model metadata")
    tensor_record = metadata.get("tensor")
    if not isinstance(tensor_record, dict):
        raise ContractError("exported model metadata has no tensor identity")
    record = cast(dict[str, Any], tensor_record)
    digest_before = sha256_file(tensor_path)
    if (
        record.get("file") != tensor_path.name
        or record.get("bytes") != tensor_path.stat().st_size
        or record.get("sha256") != digest_before
    ):
        raise ContractError("exported model tensor identity is invalid")
    try:
        tensors = load_file(str(tensor_path), device="cpu")
        reader = cast(Callable[..., _SafeTensorReader], safe_open)
        with reader(str(tensor_path), framework="pt", device="cpu") as stream:
            tensor_metadata = stream.metadata() or {}
    except Exception as exc:
        raise ContractError(f"cannot load exported Safetensors model: {exc}") from exc
    if digest_before != sha256_file(tensor_path):
        raise ContractError("exported model changed while it was being loaded")
    if (
        metadata.get("schema_version") != MODEL_EXPORT_SCHEMA
        or tensor_metadata.get("schema_version") != MODEL_EXPORT_SCHEMA
    ):
        raise ContractError("exported model schema is unsupported")
    if tensor_metadata.get("equation_version") != MODEL_EQUATION_VERSION:
        raise ContractError("exported model equation version is unsupported")
    persistence = metadata.get("persistence")
    if persistence not in {"off", "learned"} or tensor_metadata.get("persistence") != persistence:
        raise ContractError("exported model persistence policy is invalid")
    try:
        gate_scale = float(cast(float, metadata["gate_logit_scale"]))
        tensor_scale = float(tensor_metadata["gate_logit_scale"])
        _step(metadata["final_step"])
        gaussian_count = int(metadata["gaussian_count"])
        max_sh_degree = int(metadata["max_sh_degree"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"exported model metadata is incomplete: {exc}") from exc
    if (
        metadata.get("artifact_role") != "local_training_export"
        or not math.isfinite(gate_scale)
        or gate_scale <= 0.0
        or not math.isclose(gate_scale, tensor_scale, rel_tol=1.0e-6, abs_tol=1.0e-8)
        or gaussian_count <= 0
        or not 0 <= max_sh_degree <= 3
        or metadata.get("planes") != sorted(tensors)
    ):
        raise ContractError("exported model metadata is inconsistent")
    model = DynamicGaussianModel.from_checkpoint_state(
        tensors,
        persistence=persistence == "learned",
        gate_logit_scale=gate_scale,
    )
    if model.count != gaussian_count or model.max_sh_degree != max_sh_degree:
        raise ContractError("exported model shape disagrees with metadata")
    return model, metadata
