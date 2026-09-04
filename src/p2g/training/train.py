# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnnecessaryIsInstance=false

"""Fail-closed, resumable orchestration for the public MI300X trainer.

The hot loop depends only on the first-party model, loss, optimizer, renderer,
checkpoint, and role-partitioned dataset contracts.  Population control is
loaded lazily so inspection does not silently substitute an unavailable or
unreviewed relocation algorithm.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO, cast

import numpy as np
import torch
from torch import Tensor

from p2g import __version__
from p2g.canonical import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
    write_new_json,
)
from p2g.errors import ContractError, OutputExistsError
from p2g.schema import validate_payload
from p2g.training.checkpoint import (
    checkpoint_path,
    export_model,
    latest_checkpoint,
    read_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from p2g.training.config import RunConfig
from p2g.training.dataset import CACHE_MANIFEST_NAME, PreparedScene, SceneSampler
from p2g.training.evaluate import evaluate_scene
from p2g.training.initialization import load_gaussian_init
from p2g.training.losses import LossFunction, psnr
from p2g.training.model import DynamicGaussianModel
from p2g.training.optim import OptimizerBundle, build_optimizers
from p2g.training.photometric import CameraColorCorrectors
from p2g.training.renderer import GsplatRenderer

TRAINING_INPUT_BINDING_SCHEMA = "p2g.training_input_binding.v1"
TRAINING_RUNTIME_SCHEMA = "p2g.training_runtime.v1"
TRAINING_RESULT_SCHEMA = "p2g.training_result.v1"
INITIALIZATION_RECEIPT_SCHEMA = "p2g.gaussian_initialization_receipt.v1"
INITIALIZATION_RECEIPT_FILENAME = "initialization.json"
METRICS_FILENAME = "metrics.jsonl"
RUNTIME_FILENAME = "runtime.json"
RESULT_FILENAME = "training.json"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True, slots=True)
class TrainResult:
    run_dir: Path
    completed_steps: int
    final_checkpoint: Path
    model_path: Path
    metadata_path: Path
    receipt_path: Path


@dataclass(frozen=True, slots=True)
class AssetPublication:
    """Explicit user assertions needed to turn a local run into an AssetBundle."""

    output: Path
    producer_git_revision: str
    asset_license: str
    redistribution: Literal["allowed", "restricted", "review_required"]
    provenance_summary: str
    world_unit: str = "calibration_unit"
    calibration_scale: float = 1.0
    default_sh_degree: int | None = None

    def validate(self) -> None:
        if not _GIT_REVISION.fullmatch(self.producer_git_revision):
            raise ContractError("asset producer Git revision must be a full lowercase hash")
        if self.redistribution not in {"allowed", "restricted", "review_required"}:
            raise ContractError("asset redistribution assertion is invalid")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.asset_license, self.provenance_summary, self.world_unit)
        ):
            raise ContractError("asset rights, provenance, and world-unit assertions are required")
        if not math.isfinite(self.calibration_scale) or self.calibration_scale <= 0.0:
            raise ContractError("asset calibration scale must be positive and finite")
        if self.default_sh_degree is not None and (
            type(self.default_sh_degree) is not int or not 0 <= self.default_sh_degree <= 3
        ):
            raise ContractError("asset default SH degree must be an integer in [0, 3]")


class _NoRelocation:
    """Exact no-op controller for the public ``relocation.mode=off`` policy."""

    def __init__(self, gaussian_count: int) -> None:
        self.gaussian_count = gaussian_count

    def accumulate(self, _aux: dict[str, Any]) -> None:
        return

    def maybe_apply(
        self,
        _completed_step: int,
        *,
        model: DynamicGaussianModel,
        optimizers: OptimizerBundle,
    ) -> None:
        del model, optimizers
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any], *, require_state: bool) -> None:
        if state or require_state:
            raise ContractError("disabled relocation must have empty checkpoint state")

    def lineage(self, gaussian_ids: Tensor) -> tuple[Tensor, Tensor]:
        return (
            torch.zeros_like(gaussian_ids, dtype=torch.int64),
            torch.zeros_like(gaussian_ids, dtype=torch.int8),
        )


class _NoScreenGuard:
    def runtime_contract(self, _scene: PreparedScene) -> dict[str, Any]:
        return {
            "mode": "off",
            "sentinel_observations": 0,
            "sentinel_camera_ids": [],
            "sentinel_frame_ids": [],
            "development_observations_used": 0,
        }

    def maybe_apply(self, _completed_step: int, **_kwargs: Any) -> None:
        return None


def _regular_file(path: Path, *, label: str, suffix: str | None = None) -> Path:
    if not path.is_absolute():
        raise ContractError(f"{label} path must be absolute")
    if suffix is not None and path.suffix != suffix:
        raise ContractError(f"{label} must use the {suffix} suffix")
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _read_json_object(path: Path, *, label: str, canonical: bool = False) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value: Any = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    result = cast(dict[str, Any], value)
    if canonical and raw != canonical_json_bytes(result):
        raise ContractError(f"{label} must use canonical JSON encoding")
    return result


def _verified_logical_receipt(path: Path) -> dict[str, Any]:
    receipt = _read_json_object(path, label="Gaussian initialization receipt", canonical=True)
    validate_payload("gaussian_initialization_receipt", receipt)
    if (
        receipt.get("schema") != INITIALIZATION_RECEIPT_SCHEMA
        or receipt.get("status") != "COMPLETE"
        or receipt.get("trainer_eligible") is not True
    ):
        raise ContractError("Gaussian initialization receipt is not trainer eligible")
    unsigned = dict(receipt)
    logical = unsigned.pop("logical_sha256", None)
    if not isinstance(logical, str) or sha256_json(unsigned) != logical:
        raise ContractError("Gaussian initialization receipt logical hash is invalid")
    return receipt


def _safetensors_metadata(path: Path) -> dict[str, str]:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as stream:
            return dict(stream.metadata() or {})
    except Exception as exc:
        raise ContractError(f"cannot read Gaussian initialization metadata: {exc}") from exc


def verify_training_inputs(config: RunConfig) -> dict[str, Any]:
    """Close the public manifest/cache/initialization chain before GPU setup."""

    config.validate()
    tensor_config = config.data.tensor_memmap
    if tensor_config is None:
        raise ContractError("public MI300X training requires the prepared tensor cache")
    if tensor_config.verify_transport_sha256 is not True:
        raise ContractError("public MI300X training requires tensor-cache byte verification")

    observation_path = _regular_file(
        config.data.manifest,
        label="observation manifest",
        suffix=".json",
    )
    initialization_path = _regular_file(
        config.initialization.path,
        label="Gaussian initialization",
        suffix=".safetensors",
    )
    receipt_path = _regular_file(
        initialization_path.parent / INITIALIZATION_RECEIPT_FILENAME,
        label="Gaussian initialization receipt",
        suffix=".json",
    )
    cache_root = tensor_config.root
    if not cache_root.is_dir() or cache_root.is_symlink():
        raise ContractError("tensor cache must be a regular directory")
    cache_manifest_path = _regular_file(
        cache_root / CACHE_MANIFEST_NAME,
        label="tensor-cache manifest",
        suffix=".json",
    )

    observation_sha256 = sha256_file(observation_path)
    initialization_sha256 = sha256_file(initialization_path)
    cache_sha256 = sha256_file(cache_manifest_path)
    receipt_sha256 = sha256_file(receipt_path)
    cache = _read_json_object(cache_manifest_path, label="tensor-cache manifest", canonical=True)
    validate_payload("tensor_cache", cache)
    if cache.get("observation_manifest_sha256") != observation_sha256:
        raise ContractError("tensor cache is bound to another observation manifest")
    if cache.get("camera_ids") != list(tensor_config.camera_ids) or cache.get(
        "frame_ids"
    ) != list(tensor_config.frame_ids):
        raise ContractError("tensor-cache axes differ from the resolved run")

    receipt = _verified_logical_receipt(receipt_path)
    source = cast(dict[str, Any], receipt["source"])
    source_cache = cast(dict[str, Any], source["tensor_cache"])
    proposal = cast(dict[str, Any], source["proposal_sequence"])
    tensor = cast(dict[str, Any], receipt["tensor"])
    if (
        source_cache.get("manifest_sha256") != cache_sha256
        or source_cache.get("observation_manifest_sha256") != observation_sha256
    ):
        raise ContractError("Gaussian initialization receipt is bound to another scene cache")
    if (
        tensor.get("path") != initialization_path.name
        or tensor.get("container_sha256") != initialization_sha256
    ):
        raise ContractError("Gaussian initialization receipt does not bind the configured tensor")
    proposal_sha256 = proposal.get("manifest_sha256")
    if not isinstance(proposal_sha256, str) or not _SHA256.fullmatch(proposal_sha256):
        raise ContractError("Gaussian initialization proposal identity is invalid")

    metadata = _safetensors_metadata(initialization_path)
    if (
        metadata.get("builder_receipt_schema") != INITIALIZATION_RECEIPT_SCHEMA
        or metadata.get("proposal_sequence_sha256") != proposal_sha256
        or metadata.get("tensor_cache_manifest_sha256") != cache_sha256
    ):
        raise ContractError("Gaussian initialization metadata and receipt disagree")

    return {
        "schema_version": TRAINING_INPUT_BINDING_SCHEMA,
        "observation_manifest_sha256": observation_sha256,
        "tensor_cache_manifest_sha256": cache_sha256,
        "gaussian_initialization_sha256": initialization_sha256,
        "gaussian_initialization_receipt_sha256": receipt_sha256,
        "gaussian_initialization_receipt_logical_sha256": receipt["logical_sha256"],
        "proposal_sequence_manifest_sha256": proposal_sha256,
        "optimization_roles": ["train"],
        "diagnostic_roles": ["diagnostic"],
        "sealed_roles_admitted": [],
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gradient_norm_tensors(
    model: DynamicGaussianModel,
    loss: Tensor,
    *,
    active_sh_degree: int,
    color_correctors: CameraColorCorrectors | None,
) -> dict[str, Tensor]:
    """Check every gradient with one host decision and retain device-side norms."""

    if loss.ndim != 0:
        raise FloatingPointError("training loss is not a scalar")
    named_gradients: list[tuple[str, Tensor]] = []
    color_parameters: list[tuple[str, Tensor]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            if name == "sh_rest" and active_sh_degree == 0:
                continue
            raise FloatingPointError(f"trainable parameter has no gradient: {name}")
        named_gradients.append((name, gradient))
    if color_correctors is not None:
        for name, parameter in color_correctors.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                raise FloatingPointError(f"color-correction gradient is missing: {name}")
            named_gradients.append((f"color_correctors.{name}", gradient))
            color_parameters.append((f"color_correctors.{name}", parameter))
    if not named_gradients:
        raise FloatingPointError("training produced no gradients")
    norms = [torch.linalg.vector_norm(gradient.detach()) for _, gradient in named_gradients]
    finite_flags = [torch.isfinite(loss.detach())]
    finite_flags.extend(torch.isfinite(norm) for norm in norms)
    finite_flags.extend(
        torch.isfinite(parameter.detach()).all() for _, parameter in color_parameters
    )
    if not bool(torch.stack(finite_flags).all()):
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("training loss is not finite")
        for (name, gradient), norm in zip(named_gradients, norms, strict=True):
            if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(f"trainable parameter has non-finite gradient: {name}")
        for name, parameter in color_parameters:
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f"trainable parameter is non-finite: {name}")
        raise FloatingPointError("training gradient norm is non-finite")
    return {
        name: norm for (name, _), norm in zip(named_gradients, norms, strict=True)
    }


def require_finite_loss_and_gradients(
    model: DynamicGaussianModel,
    loss: Tensor,
    *,
    active_sh_degree: int,
) -> dict[str, float]:
    """Compatibility wrapper for callers that explicitly request host-side norms."""

    norms = _gradient_norm_tensors(
        model,
        loss,
        active_sh_degree=active_sh_degree,
        color_correctors=None,
    )
    return {name: float(value) for name, value in norms.items()}


def require_finite_color_correction_gradients(
    color_correctors: CameraColorCorrectors | None,
) -> dict[str, float]:
    if color_correctors is None:
        return {}
    if not color_correctors.finite():
        raise FloatingPointError("color-correction parameters are non-finite")
    result: dict[str, float] = {}
    for name, parameter in color_correctors.named_parameters():
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"color-correction gradient is missing/non-finite: {name}")
        result[f"color_correctors.{name}"] = float(torch.linalg.vector_norm(gradient.detach()))
    return result


def _active_count(aux: dict[str, Any]) -> int | None:
    radii = aux.get("radii")
    if not isinstance(radii, Tensor):
        return None
    visible = (
        torch.all(radii > 0, dim=-1)
        if radii.ndim >= 2 and radii.shape[-1] == 2
        else radii > 0
    )
    return int(visible.sum())


def _intersection_count(aux: dict[str, Any]) -> int | None:
    flatten_ids = aux.get("flatten_ids")
    return int(flatten_ids.numel()) if isinstance(flatten_ids, Tensor) else None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_new_run(run_dir: Path, config: RunConfig) -> None:
    if run_dir.exists() or run_dir.is_symlink():
        raise OutputExistsError(f"refusing to reuse training run path: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.", dir=run_dir.parent))
    try:
        config.save(stage / "config.toml")
        if run_dir.exists() or run_dir.is_symlink():
            raise OutputExistsError(f"training run path appeared during publication: {run_dir}")
        os.rename(stage, run_dir)
        _fsync_directory(run_dir.parent)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _write_or_verify_json(path: Path, payload: dict[str, Any], *, label: str) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"existing {label} must be a regular non-symlink file")
        existing = _read_json_object(path, label=label, canonical=True)
        if existing != payload:
            raise ContractError(f"existing {label} differs from the resumed runtime")
        return
    write_new_json(path, payload)


def _metric_rows(metrics_path: Path) -> list[tuple[str, dict[str, Any]]]:
    try:
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"cannot read training metric stream: {exc}") from exc
    rows: list[tuple[str, dict[str, Any]]] = []
    previous = 0
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ContractError(f"training metric stream contains a blank line at {line_number}")
        try:
            untyped: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"training metric stream has invalid JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(untyped, dict):
            raise ContractError(f"training metric row {line_number} must be an object")
        row = cast(dict[str, Any], untyped)
        next_step = row.get("next_step")
        if (
            type(next_step) is not int
            or next_step <= previous
            or row.get("step") != next_step - 1
            or canonical_json_bytes(row).decode("utf-8").rstrip("\n") != line
        ):
            raise ContractError(
                "training metric stream must be canonical and strictly step-ordered "
                f"at line {line_number}"
            )
        previous = next_step
        rows.append((line, row))
    if rows and rows[0][1]["next_step"] != 1:
        raise ContractError("training metric stream must include the first completed step")
    return rows


def _reconcile_metrics_with_checkpoint(metrics_path: Path, *, next_step: int) -> None:
    """Trim sparse metric rows newer than the authoritative resume checkpoint."""

    if metrics_path.is_symlink():
        raise ContractError("training metric stream must be a regular non-symlink file")
    if not metrics_path.is_file():
        if next_step == 0:
            return
        raise ContractError(
            f"resume checkpoint step {next_step} has no training metric stream"
        )
    rows = _metric_rows(metrics_path)
    retained = [line for line, row in rows if cast(int, row["next_step"]) <= next_step]
    if next_step > 0 and not retained:
        raise ContractError("training metric stream has no row at or before the checkpoint")
    if len(retained) == len(rows):
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{metrics_path.name}.", suffix=".tmp", dir=metrics_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for line in retained:
                stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, metrics_path)
        _fsync_directory(metrics_path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _append_metric(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(canonical_json_bytes(payload).decode("utf-8"))
    stream.flush()


def _canonical_runtime_device(
    requested: str | torch.device,
    materialized: torch.device,
) -> torch.device:
    """Resolve an unindexed device alias to the device holding runtime tensors."""

    try:
        configured = torch.device(requested)
    except (RuntimeError, TypeError) as exc:
        raise ContractError("training device is invalid") from exc
    configured_index: int | None = getattr(configured, "index", None)
    if configured.type != materialized.type or (
        configured_index is not None and configured_index != materialized.index
    ):
        raise ContractError("configured training device differs from the materialized model")
    return materialized


def _create_population_controls(
    config: RunConfig,
    *,
    model: DynamicGaussianModel,
    scene: PreparedScene,
    renderer: GsplatRenderer,
) -> tuple[Any, Any]:
    device = _canonical_runtime_device(config.training.device, model.means.device)
    if any(parameter.device != device for parameter in model.parameters()) or any(
        buffer.device != device for buffer in model.buffers()
    ):
        raise ContractError("materialized model tensors do not share one runtime device")
    relocation_config = config.training.relocation
    if relocation_config.mode == "off":
        relocation: Any = _NoRelocation(model.count)
    else:
        try:
            relocation_module = importlib.import_module("p2g.training.relocation")
        except ModuleNotFoundError as exc:
            if exc.name != "p2g.training.relocation":
                raise
            raise ContractError(
                "relocation mode requires the p2g.training.relocation module"
            ) from exc
        relocation = cast(Any, relocation_module).RelocationController.create(
            relocation_config,
            gaussian_count=model.count,
            device=device,
        )

    if config.training.screen_guard.mode == "off":
        screen_guard: Any = _NoScreenGuard()
    else:
        try:
            screen_guard_module = importlib.import_module("p2g.training.screen_guard")
        except ModuleNotFoundError as exc:
            if exc.name != "p2g.training.screen_guard":
                raise
            raise ContractError(
                "screen guard requires the p2g.training.screen_guard module"
            ) from exc
        screen_guard = cast(Any, screen_guard_module).FormationScreenInfluenceGuard.create(
            config.training.screen_guard,
            scene=scene,
            renderer=renderer,
            device=device,
        )
    return relocation, screen_guard


def initialize_runtime(
    config: RunConfig,
    *,
    checkpoint_state: dict[str, Any] | None,
) -> tuple[
    PreparedScene,
    DynamicGaussianModel,
    GsplatRenderer,
    CameraColorCorrectors | None,
    OptimizerBundle,
    SceneSampler,
    int,
    dict[str, Any],
]:
    scene = PreparedScene.load(config.data)
    persistence = config.model.persistence == "learned"
    if checkpoint_state is None:
        initialization = load_gaussian_init(config.initialization)
        model = DynamicGaussianModel(
            initialization,
            persistence=persistence,
            gate_logit_scale=config.model.gate_logit_scale,
        )
    else:
        model_state = checkpoint_state.get("model")
        if not isinstance(model_state, dict):
            raise ContractError("checkpoint model state must be a mapping")
        model = DynamicGaussianModel.from_checkpoint_state(
            cast(dict[str, Tensor], model_state),
            persistence=persistence,
            gate_logit_scale=config.model.gate_logit_scale,
        )
    model = model.to(config.training.device)
    color_correctors: CameraColorCorrectors | None = None
    if config.color_correction.mode == "per_camera_affine":
        camera_ids = tuple(
            sorted(
                {scene.observations[index].camera_id for index in scene.train_indices},
                key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
            )
        )
        color_correctors = CameraColorCorrectors(camera_ids).to(config.training.device)
    renderer = GsplatRenderer(config.renderer)
    runtime = renderer.validate_environment(config.training.device)
    optimizers = build_optimizers(
        model,
        config.optimizer,
        iterations=config.training.iterations,
        scene_extent=scene.camera_extent(),
        color_correctors=color_correctors,
        color_correction_lr=(
            config.color_correction.learning_rate if color_correctors is not None else None
        ),
    )
    sampler = SceneSampler(
        scene.train_indices,
        seed=config.training.seed,
        policy=config.training.sampling,
        frame_groups=(
            scene.train_frame_groups()
            if config.training.sampling == "frame_camera_with_replacement"
            else None
        ),
    )
    start_step = 0
    if checkpoint_state is not None:
        start_step = restore_training_state(
            checkpoint_state,
            model=model,
            optimizers=optimizers,
            sampler=sampler,
            color_correctors=color_correctors,
        )
    return scene, model, renderer, color_correctors, optimizers, sampler, start_step, runtime


def _runtime_payload(
    *,
    run_dir: Path,
    scene: PreparedScene,
    model: DynamicGaussianModel,
    runtime: dict[str, Any],
    screen_guard: Any,
    config: RunConfig,
    input_binding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_RUNTIME_SCHEMA,
        "dataset_id": scene.dataset_id,
        "resolved_config_sha256": sha256_file(run_dir / "config.toml"),
        "input_binding": input_binding,
        "gaussian_count": model.count,
        "observation_roles": {
            "train": len(scene.train_indices),
            "diagnostic": len(scene.diagnostic_indices),
            "sealed_excluded": len(scene.sealed_indices),
            "free_view_excluded": len(scene.free_view_indices),
            "selection_excluded": len(scene.excluded_indices),
        },
        "gate_logit_scale": config.model.gate_logit_scale,
        "color_correction": config.color_correction.mode,
        "hot_loop_sync_policy": {
            "explicit_cuda_synchronize_calls_per_step": 0,
            "finite_gradient_host_decisions_per_step": 1,
            "scalar_metric_materialization": (
                "first_log_checkpoint_evaluation_control_or_final_step_v1"
            ),
        },
        "screen_guard": screen_guard.runtime_contract(scene),
        "renderer": runtime,
    }


def _evaluate_atomic(
    destination: Path,
    *,
    model: DynamicGaussianModel,
    renderer: GsplatRenderer,
    scene: PreparedScene,
    config: RunConfig,
    sh_degree: int,
) -> dict[str, Any]:
    if destination.is_dir() and not destination.is_symlink():
        evaluation_path = _regular_file(
            destination / "evaluation.json",
            label="completed evaluation",
            suffix=".json",
        )
        return _read_json_object(
            evaluation_path,
            label="completed evaluation",
            canonical=True,
        )
    if destination.exists() or destination.is_symlink():
        raise ContractError("evaluation destination is not a regular completed directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        evaluation = evaluate_scene(
            model=model,
            renderer=renderer,
            scene=scene,
            device=config.training.device,
            sh_degree=sh_degree,
            output_dir=stage / "renders",
            ssim_padding=config.loss.ssim_padding,
        )
        write_new_json(stage / "evaluation.json", evaluation)
        _fsync_directory(stage)
        if destination.exists() or destination.is_symlink():
            raise OutputExistsError(f"evaluation destination appeared: {destination}")
        os.rename(stage, destination)
        _fsync_directory(destination.parent)
        return evaluation
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _training_receipt(
    *,
    run_dir: Path,
    final_step: int,
    final_checkpoint: Path,
    model_path: Path,
    metadata_path: Path,
    final_evaluation: Path,
    input_binding: dict[str, Any],
) -> dict[str, Any]:
    metrics_path = run_dir / METRICS_FILENAME
    runtime_path = run_dir / RUNTIME_FILENAME
    artifacts = {
        "runtime_sha256": sha256_file(runtime_path),
        "metrics_sha256": sha256_file(metrics_path),
        "final_checkpoint_manifest_sha256": sha256_file(final_checkpoint / "manifest.json"),
        "model_sha256": sha256_file(model_path),
        "model_metadata_sha256": sha256_file(metadata_path),
        "final_evaluation_sha256": sha256_file(final_evaluation / "evaluation.json"),
    }
    receipt: dict[str, Any] = {
        "schema_version": TRAINING_RESULT_SCHEMA,
        "status": "COMPLETE",
        "completed_steps": final_step,
        "input_binding": input_binding,
        "artifacts": artifacts,
        "claim_boundary": (
            "This receipt proves completion and byte binding for one local training run. "
            "It does not establish sealed quality, performance, or redistribution rights."
        ),
    }
    receipt["logical_sha256"] = sha256_json(receipt)
    return receipt


def _record_failure(run_dir: Path, *, step: int, error: FloatingPointError) -> None:
    payload = {
        "schema_version": "p2g.training_failure.v1",
        "step": step,
        "error": str(error),
    }
    _write_or_verify_json(
        run_dir / f"failure_step_{step:08d}.json",
        payload,
        label="training failure receipt",
    )


def run_training(
    config: RunConfig,
    *,
    run_dir: Path,
    resume_checkpoint: Path | None = None,
) -> TrainResult:
    """Run or resume one fixed-configuration, fixed-population training job."""

    config.validate()
    input_binding = verify_training_inputs(config)
    run_dir = run_dir.expanduser().resolve()
    checkpoint_state: dict[str, Any] | None = None
    if resume_checkpoint is None:
        _prepare_new_run(run_dir, config)
        seed_everything(config.training.seed)
    else:
        checkpoint = resume_checkpoint.expanduser().resolve()
        checkpoint_config, checkpoint_state, _ = read_checkpoint(checkpoint)
        if checkpoint_config != config:
            raise ContractError("resume config differs from the checkpoint config")
        if checkpoint.parent.name != "checkpoints" or checkpoint.parent.parent != run_dir:
            raise ContractError("resume checkpoint must belong to the selected training run")
        if latest_checkpoint(run_dir).resolve() != checkpoint:
            raise ContractError("in-place resume requires the latest checkpoint in the run")
        if (run_dir / RESULT_FILENAME).is_file():
            completed_config, _ = _verify_completed_run(run_dir)
            if completed_config != config:
                raise ContractError("completed run config differs from the requested config")
            final_step = config.training.iterations
            return TrainResult(
                run_dir=run_dir,
                completed_steps=final_step,
                final_checkpoint=checkpoint_path(run_dir, final_step),
                model_path=run_dir / "model.safetensors",
                metadata_path=run_dir / "model.json",
                receipt_path=run_dir / RESULT_FILENAME,
            )

    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not config.training.deterministic
        torch.backends.cudnn.deterministic = config.training.deterministic
    (
        scene,
        model,
        renderer,
        color_correctors,
        optimizers,
        sampler,
        start_step,
        runtime,
    ) = initialize_runtime(config, checkpoint_state=checkpoint_state)
    relocation, screen_guard = _create_population_controls(
        config,
        model=model,
        scene=scene,
        renderer=renderer,
    )
    relocation.load_state_dict(
        {} if checkpoint_state is None else cast(dict[str, Any], checkpoint_state["relocation"]),
        require_state=(
            checkpoint_state is not None
            and start_step > 0
            and config.training.relocation.mode != "off"
        ),
    )
    if start_step > config.training.iterations:
        raise ContractError("checkpoint is beyond configured training iterations")
    runtime_payload = _runtime_payload(
        run_dir=run_dir,
        scene=scene,
        model=model,
        runtime=runtime,
        screen_guard=screen_guard,
        config=config,
        input_binding=input_binding,
    )
    _write_or_verify_json(
        run_dir / RUNTIME_FILENAME,
        runtime_payload,
        label="training runtime receipt",
    )
    if (
        resume_checkpoint is not None
        and start_step > 0
        and start_step % config.training.evaluate_every == 0
    ):
        _evaluate_atomic(
            run_dir / "renders" / f"step_{start_step:08d}",
            model=model,
            renderer=renderer,
            scene=scene,
            config=config,
            sh_degree=min(
                (start_step - 1) // config.training.sh_degree_interval,
                model.max_sh_degree,
            ),
        )

    loss_function = LossFunction(config.loss, device=config.training.device)
    metrics_path = run_dir / METRICS_FILENAME
    metrics_mode = "a" if resume_checkpoint is not None else "x"
    if resume_checkpoint is not None:
        _reconcile_metrics_with_checkpoint(metrics_path, next_step=start_step)
    optimizers.zero_grad(set_to_none=True)
    try:
        with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_stream:
            for step in range(start_step, config.training.iterations):
                batch = scene.load_batch(sampler.next_index()).to(config.training.device)
                if batch.role != "train":
                    raise ContractError("optimization sampler admitted a non-train observation")
                active_degree = min(step // config.training.sh_degree_interval, model.max_sh_degree)
                started = time.perf_counter()
                rendered = renderer.render(model, batch, sh_degree=active_degree)
                prediction = rendered.image
                if color_correctors is not None and step > config.color_correction.start:
                    prediction = color_correctors(batch.camera_id, prediction)
                color_regularization = (
                    None
                    if color_correctors is None
                    else config.color_correction.regularization
                    * color_correctors.regularization()
                )
                terms = loss_function(
                    model=model,
                    materialized=rendered.materialized,
                    prediction=prediction,
                    target=batch.rgb,
                    color_correction_regularization=color_regularization,
                )
                loss = torch.stack(tuple(terms.values())).sum()
                if not loss.requires_grad:
                    raise FloatingPointError("rendered loss is disconnected from model parameters")
                loss.backward()
                gradient_norms = _gradient_norm_tensors(
                    model,
                    loss,
                    active_sh_degree=active_degree,
                    color_correctors=color_correctors,
                )
                relocation.accumulate(rendered.aux)
                optimizers.step()
                optimizers.step_schedulers()
                optimizers.zero_grad(set_to_none=True)
                next_step = step + 1
                relocation_event = relocation.maybe_apply(
                    next_step,
                    model=model,
                    optimizers=optimizers,
                )
                screen_guard_event = screen_guard.maybe_apply(
                    next_step,
                    model=model,
                    optimizers=optimizers,
                    scene=scene,
                    renderer=renderer,
                    relocation=relocation,
                    sh_degree=active_degree,
                )
                checkpoint_due = next_step % config.training.checkpoint_every == 0
                evaluation_due = next_step % config.training.evaluate_every == 0
                final_step_due = next_step == config.training.iterations
                metric_due = (
                    next_step == 1
                    or next_step % config.training.log_every == 0
                    or checkpoint_due
                    or evaluation_due
                    or final_step_due
                    or relocation_event is not None
                    or screen_guard_event is not None
                )
                if metric_due:
                    metric: dict[str, Any] = {
                        "step": step,
                        "next_step": next_step,
                        "observation_id": batch.observation_id,
                        "camera_id": batch.camera_id,
                        "frame_id": batch.frame_id,
                        "observation_role": batch.role,
                        "timestamp_seconds": float(batch.timestamp),
                        "sh_degree": active_degree,
                        "loss": float(loss.detach()),
                        "psnr": float(psnr(prediction.detach(), batch.rgb)),
                        "raw_psnr": float(psnr(rendered.image.detach(), batch.rgb)),
                        "loss_terms": {
                            name: float(value.detach()) for name, value in terms.items()
                        },
                        "gradient_norms": {
                            name: float(value) for name, value in gradient_norms.items()
                        },
                        "step_wall_ms": (time.perf_counter() - started) * 1_000.0,
                        "active_gaussians": _active_count(rendered.aux),
                        "intersection_count": _intersection_count(rendered.aux),
                    }
                    if relocation_event is not None:
                        metric["relocation"] = relocation_event
                    if screen_guard_event is not None:
                        metric["screen_guard"] = screen_guard_event
                    if torch.cuda.is_available():
                        metric["allocated_bytes"] = torch.cuda.memory_allocated(
                            config.training.device
                        )
                        metric["reserved_bytes"] = torch.cuda.memory_reserved(
                            config.training.device
                        )
                    _append_metric(metrics_stream, metric)
                    print(canonical_json_bytes(metric).decode("utf-8"), end="", flush=True)
                if checkpoint_due or evaluation_due or final_step_due:
                    metrics_stream.flush()
                    os.fsync(metrics_stream.fileno())
                    target = checkpoint_path(run_dir, next_step)
                    if not target.is_dir():
                        save_checkpoint(
                            run_dir,
                            next_step=next_step,
                            config=config,
                            model=model,
                            optimizers=optimizers,
                            sampler=sampler,
                            relocation_state=relocation.state_dict(),
                            color_correctors=color_correctors,
                        )
                if evaluation_due:
                    _evaluate_atomic(
                        run_dir / "renders" / f"step_{next_step:08d}",
                        model=model,
                        renderer=renderer,
                        scene=scene,
                        config=config,
                        sh_degree=active_degree,
                    )
    except FloatingPointError as exc:
        failure_step = int(locals().get("step", start_step))
        _record_failure(run_dir, step=failure_step, error=exc)
        raise

    final_step = config.training.iterations
    final_checkpoint = checkpoint_path(run_dir, final_step)
    if not final_checkpoint.is_dir():
        final_checkpoint = save_checkpoint(
            run_dir,
            next_step=final_step,
            config=config,
            model=model,
            optimizers=optimizers,
            sampler=sampler,
            relocation_state=relocation.state_dict(),
            color_correctors=color_correctors,
        )
    final_evaluation = run_dir / "renders" / f"step_{final_step:08d}"
    _evaluate_atomic(
        final_evaluation,
        model=model,
        renderer=renderer,
        scene=scene,
        config=config,
        sh_degree=min(
            max(final_step - 1, 0) // config.training.sh_degree_interval,
            model.max_sh_degree,
        ),
    )
    model_path, metadata_path = export_model(
        run_dir,
        model=model,
        config=config,
        final_step=final_step,
        color_correctors=color_correctors,
    )
    receipt = _training_receipt(
        run_dir=run_dir,
        final_step=final_step,
        final_checkpoint=final_checkpoint,
        model_path=model_path,
        metadata_path=metadata_path,
        final_evaluation=final_evaluation,
        input_binding=input_binding,
    )
    receipt_path = run_dir / RESULT_FILENAME
    write_new_json(receipt_path, receipt)
    return TrainResult(
        run_dir=run_dir,
        completed_steps=final_step,
        final_checkpoint=final_checkpoint,
        model_path=model_path,
        metadata_path=metadata_path,
        receipt_path=receipt_path,
    )


def resume_training(checkpoint: Path) -> TrainResult:
    checkpoint = checkpoint.expanduser().resolve()
    config, _, _ = read_checkpoint(checkpoint)
    if checkpoint.parent.name != "checkpoints":
        raise ContractError("checkpoint must be inside RUN/checkpoints")
    return run_training(
        config,
        run_dir=checkpoint.parent.parent,
        resume_checkpoint=checkpoint,
    )


def _verify_completed_run(run_dir: Path) -> tuple[RunConfig, dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ContractError("completed training run must be a regular directory")
    config_path = _regular_file(run_dir / "config.toml", label="training config")
    result_path = _regular_file(
        run_dir / RESULT_FILENAME,
        label="training result receipt",
        suffix=".json",
    )
    config = RunConfig.load(config_path)
    input_binding = verify_training_inputs(config)
    receipt = _read_json_object(
        result_path,
        label="training result receipt",
        canonical=True,
    )
    if (
        receipt.get("schema_version") != TRAINING_RESULT_SCHEMA
        or receipt.get("status") != "COMPLETE"
        or receipt.get("completed_steps") != config.training.iterations
        or receipt.get("input_binding") != input_binding
    ):
        raise ContractError("training result receipt is incomplete or bound to other inputs")
    unsigned = dict(receipt)
    logical = unsigned.pop("logical_sha256", None)
    if not isinstance(logical, str) or logical != sha256_json(unsigned):
        raise ContractError("training result receipt logical hash is invalid")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContractError("training result artifact catalog is invalid")
    expected = {
        "runtime_sha256": run_dir / RUNTIME_FILENAME,
        "metrics_sha256": run_dir / METRICS_FILENAME,
        "final_checkpoint_manifest_sha256": (
            checkpoint_path(run_dir, config.training.iterations) / "manifest.json"
        ),
        "model_sha256": run_dir / "model.safetensors",
        "model_metadata_sha256": run_dir / "model.json",
        "final_evaluation_sha256": (
            run_dir
            / "renders"
            / f"step_{config.training.iterations:08d}"
            / "evaluation.json"
        ),
    }
    for name, path in expected.items():
        if not path.is_file() or path.is_symlink() or artifacts.get(name) != sha256_file(path):
            raise ContractError(f"completed training artifact is missing or changed: {name}")
    return config, receipt


def verify_completed_run(run_dir: Path) -> tuple[RunConfig, dict[str, Any]]:
    """Verify and return one complete, hash-closed local training run."""

    return _verify_completed_run(run_dir)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError(f"required asset runtime distribution is missing: {name}") from exc


def export_asset(run_dir: Path, publication: AssetPublication) -> Path:
    """Publish a portable AssetBundle from a hash-closed completed training run."""

    from p2g.training.asset import AssetBundleSpec, write_asset_bundle
    from p2g.training.checkpoint import load_exported_model

    publication.validate()
    resolved_run = run_dir.expanduser().resolve()
    resolved_output = publication.output.expanduser().resolve()
    if resolved_output == resolved_run or resolved_run in resolved_output.parents:
        raise ContractError("AssetBundle output must be outside the immutable training run")
    config, training_receipt = _verify_completed_run(resolved_run)
    model, model_metadata = load_exported_model(resolved_run / "model.safetensors")
    manifest = _read_json_object(config.data.manifest, label="observation manifest")
    validate_payload("observation", manifest)
    observations = cast(list[dict[str, Any]], manifest["observations"])
    train_timestamps = [
        float(item["timestamp_seconds"]) for item in observations if item.get("role") == "train"
    ]
    if not train_timestamps or min(train_timestamps) == max(train_timestamps):
        raise ContractError("asset export requires a non-empty train-role time interval")
    conventions = cast(dict[str, Any], manifest["coordinate_conventions"])
    source = cast(dict[str, Any], manifest["source"])
    photometric_space = conventions.get("photometric_space")
    if photometric_space not in {"linear_rgb", "srgb_reference_profile"}:
        raise ContractError("asset source photometric space is unsupported")
    binding = cast(dict[str, Any], training_receipt["input_binding"])
    artifacts = cast(dict[str, Any], training_receipt["artifacts"])
    source_digests = {
        "observation_manifest": cast(str, binding["observation_manifest_sha256"]),
        "tensor_cache_manifest": cast(str, binding["tensor_cache_manifest_sha256"]),
        "gaussian_initialization": cast(str, binding["gaussian_initialization_sha256"]),
        "gaussian_initialization_receipt": cast(
            str, binding["gaussian_initialization_receipt_sha256"]
        ),
        "training_receipt": sha256_file(resolved_run / RESULT_FILENAME),
        "final_checkpoint_manifest": cast(
            str, artifacts["final_checkpoint_manifest_sha256"]
        ),
    }
    runtime = _read_json_object(
        resolved_run / RUNTIME_FILENAME,
        label="training runtime receipt",
        canonical=True,
    )
    renderer = cast(dict[str, Any], runtime["renderer"])
    default_sh_degree = (
        model.max_sh_degree
        if publication.default_sh_degree is None
        else publication.default_sh_degree
    )
    spec = AssetBundleSpec(
        valid_time_start_seconds=min(train_timestamps),
        valid_time_stop_seconds=max(train_timestamps),
        reference_time_seconds=min(train_timestamps),
        world_coordinate_convention=f"{conventions['handedness']}_handed_calibration_world",
        world_unit=publication.world_unit,
        calibration_scale=publication.calibration_scale,
        photometric_space=photometric_space,
        default_sh_degree=default_sh_degree,
        final_step=int(model_metadata["final_step"]),
        source_bundle_digests=source_digests,
        producer_version=__version__,
        producer_git_revision=publication.producer_git_revision,
        dependency_identities={
            "amd-gsplat": (
                f"{renderer['gsplat_distribution']}@{renderer['gsplat_source_revision']}"
            ),
            "safetensors": _distribution_version("safetensors"),
            "torch": cast(str, renderer["torch"]),
            "torch-hip": cast(str, renderer["hip"]),
        },
        asset_license=publication.asset_license,
        source_data_license=cast(str, source["license"]),
        redistribution=publication.redistribution,
        provenance_summary=publication.provenance_summary,
    )
    return write_asset_bundle(
        resolved_output,
        model=model,
        spec=spec,
        renderer=config.renderer,
    )


__all__ = [
    "AssetPublication",
    "TrainResult",
    "export_asset",
    "resume_training",
    "run_training",
    "verify_completed_run",
    "verify_training_inputs",
]
