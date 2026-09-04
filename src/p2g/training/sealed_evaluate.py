# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false

"""Preregistered, write-once evaluation of the sealed observation role.

Routine training and evaluation cannot load sealed observations.  This module
is the only public path that requests the sealed dataset capability, and it
does so only after binding a complete final run to a preregistered gate.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from p2g import __version__
from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json, write_new_json
from p2g.errors import ContractError, OutputExistsError
from p2g.schema import validate_payload
from p2g.training import checkpoint as checkpoint_module
from p2g.training import config as config_module
from p2g.training import dataset as dataset_module
from p2g.training import evaluate as evaluate_module
from p2g.training import losses as losses_module
from p2g.training import model as model_module
from p2g.training import renderer as renderer_module
from p2g.training import train as train_module
from p2g.training.config import (
    DataPolicyConfig,
    InitializationPolicyConfig,
    PortableProfile,
    RunConfig,
)
from p2g.training.dataset import PreparedScene
from p2g.training.evaluate import evaluate_scene_selection, load_exported_run
from p2g.training.losses import PSNR_EQUATION_VERSION, SSIM_EQUATION_VERSION
from p2g.training.renderer import GsplatRenderer
from p2g.training.train import verify_completed_run

SEALED_GATE_SCHEMA = "p2g.sealed_quality_gate.v1"
SEALED_RECEIPT_SCHEMA = "p2g.sealed_evaluation_receipt.v1"
SEALED_VERIFY_SCHEMA = "p2g.sealed_receipt_verification.v1"
RECEIPT_FILENAME = "receipt.json"
RENDERS_DIRECTORY = "renders"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class _GateInputs:
    path: Path
    payload: dict[str, Any]
    recipe_path: Path
    recipe: dict[str, Any]
    profile_path: Path
    profile: PortableProfile


def _regular_file(path: Path, *, label: str) -> Path:
    raw = path.expanduser()
    if raw.is_symlink():
        raise ContractError(f"{label} must be a regular non-symlink file")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"cannot resolve {label}: {exc}") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError(f"{label} must be a regular non-symlink file")
    return resolved


def _read_object(path: Path, *, label: str, canonical: bool = False) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value: Any = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    payload = cast(dict[str, Any], value)
    if canonical and raw != canonical_json_bytes(payload):
        raise ContractError(f"{label} must use canonical JSON encoding")
    return payload


def _file_record(path: Path) -> dict[str, Any]:
    path = _regular_file(path, label="bound file")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _assert_file_record(path: Path, record: object, *, label: str) -> None:
    if not isinstance(record, dict):
        raise ContractError(f"{label} identity must be an object")
    expected = _file_record(path)
    if cast(dict[str, Any], record) != expected:
        raise ContractError(f"{label} bytes differ from the preregistered identity")


def _safe_gate_sibling(gate_path: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ContractError(f"{label} must be one safe filename beside the gate")
    candidate = gate_path.parent / value
    return _regular_file(candidate, label=label)


def _logical_id(payload: dict[str, Any], field: str, *, label: str) -> str:
    unsigned = dict(payload)
    claimed = unsigned.pop(field, None)
    if not isinstance(claimed, str) or claimed != sha256_json(unsigned):
        raise ContractError(f"{label} canonical identity is invalid")
    return claimed


def _metric_pair(value: object, *, label: str) -> tuple[float, float]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a metric object")
    raw = cast(dict[str, Any], value)
    if set(raw) != {"psnr", "ssim"}:
        raise ContractError(f"{label} must contain only psnr and ssim")
    psnr_value = raw["psnr"]
    ssim_value = raw["ssim"]
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(float(item))
        for item in (psnr_value, ssim_value)
    ):
        raise ContractError(f"{label} values must be finite")
    return float(psnr_value), float(ssim_value)


def _validate_gate_semantics(gate: dict[str, Any]) -> None:
    _logical_id(gate, "gate_id", label="sealed quality gate")
    protocol = cast(dict[str, Any], gate["protocol"])
    metrics = cast(dict[str, Any], protocol["metrics"])
    if metrics != {
        "psnr_equation": PSNR_EQUATION_VERSION,
        "ssim_equation": SSIM_EQUATION_VERSION,
        "ssim_padding": metrics["ssim_padding"],
        "aggregation": "arithmetic_mean_of_per_observation_scores_v1",
    }:
        raise ContractError("sealed gate metric identities are unsupported")
    quality = cast(dict[str, Any], gate["quality"])
    sealed = cast(dict[str, Any], quality["sealed"])
    anchor_psnr, anchor_ssim = _metric_pair(sealed["anchor_mean"], label="sealed anchor")
    regression_psnr, regression_ssim = _metric_pair(
        sealed["maximum_regression"], label="sealed maximum regression"
    )
    floor_psnr, floor_ssim = _metric_pair(
        sealed["minimum_mean"], label="sealed minimum mean"
    )
    if regression_psnr < 0.0 or regression_ssim < 0.0:
        raise ContractError("sealed maximum regression must be non-negative")
    if not math.isclose(
        floor_psnr,
        anchor_psnr - regression_psnr,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        floor_ssim,
        anchor_ssim - regression_ssim,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ContractError("sealed quality floors do not equal anchor minus allowed regression")


def _load_gate(path: Path) -> _GateInputs:
    gate_path = _regular_file(path, label="sealed quality gate")
    gate = _read_object(gate_path, label="sealed quality gate")
    validate_payload("sealed_quality_gate", gate)
    _validate_gate_semantics(gate)
    protocol = cast(dict[str, Any], gate["protocol"])
    recipe_record = cast(dict[str, Any], protocol["recipe"])
    profile_record = cast(dict[str, Any], protocol["profile"])
    recipe_path = _safe_gate_sibling(
        gate_path, recipe_record["file"], label="sealed recipe"
    )
    profile_path = _safe_gate_sibling(
        gate_path, profile_record["file"], label="sealed profile"
    )
    _assert_file_record(
        recipe_path,
        {name: recipe_record[name] for name in ("bytes", "sha256")},
        label="sealed recipe",
    )
    _assert_file_record(
        profile_path,
        {name: profile_record[name] for name in ("bytes", "sha256")},
        label="sealed profile",
    )
    recipe = _read_object(recipe_path, label="sealed recipe")
    recipe_id = _logical_id(recipe, "recipe_id", label="sealed recipe")
    if recipe_id != recipe_record["recipe_id"]:
        raise ContractError("sealed recipe ID differs from the gate")
    profile = PortableProfile.load(profile_path)
    return _GateInputs(
        path=gate_path,
        payload=gate,
        recipe_path=recipe_path,
        recipe=recipe,
        profile_path=profile_path,
        profile=profile,
    )


def _gate_identity(inputs: _GateInputs) -> dict[str, str]:
    return {
        "gate_id": cast(str, inputs.payload["gate_id"]),
        "gate_file_sha256": sha256_file(inputs.path),
        "recipe_id": cast(str, inputs.recipe["recipe_id"]),
        "recipe_file_sha256": sha256_file(inputs.recipe_path),
        "profile_file_sha256": sha256_file(inputs.profile_path),
    }


def _assert_profile_matches(config: RunConfig, profile: PortableProfile) -> None:
    data_policy = DataPolicyConfig(
        downscale=config.data.downscale,
        train_roles=config.data.train_roles,
        eval_roles=config.data.eval_roles,
        max_train_observations=config.data.max_train_observations,
        max_eval_observations=config.data.max_eval_observations,
        image_cache_size=config.data.image_cache_size,
    )
    initialization_policy = InitializationPolicyConfig(
        format=config.initialization.format,
        sh_degree=config.initialization.sh_degree,
        time_offset_seconds=config.initialization.time_offset_seconds,
        duration_min_seconds=config.initialization.duration_min_seconds,
        duration_max_seconds=config.initialization.duration_max_seconds,
        persistence_initial_logit=config.initialization.persistence_initial_logit,
    )
    pairs = (
        ("data", data_policy, profile.data),
        ("initialization", initialization_policy, profile.initialization),
        ("model", config.model, profile.model),
        ("renderer", config.renderer, profile.renderer),
        ("loss", config.loss, profile.loss),
        ("color correction", config.color_correction, profile.color_correction),
        ("optimizer", config.optimizer, profile.optimizer),
        ("training", config.training, profile.training),
    )
    for label, actual, expected in pairs:
        if dataclasses.asdict(actual) != dataclasses.asdict(expected):
            raise ContractError(f"candidate {label} configuration differs from the sealed profile")


def _expected_partition(
    gate: dict[str, Any], role: str
) -> tuple[tuple[str, ...], int]:
    dataset = cast(dict[str, Any], cast(dict[str, Any], gate["protocol"])["dataset"])
    partition = cast(dict[str, Any], dataset[role])
    camera_ids = tuple(cast(list[str], partition["camera_ids"]))
    if tuple(sorted(camera_ids)) != camera_ids:
        raise ContractError(f"sealed gate {role} camera IDs must be sorted")
    return camera_ids, cast(int, partition["observation_count"])


def _scene_partition(
    scene: PreparedScene, indices: tuple[int, ...], *, role: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    observations = tuple(scene.observations[index] for index in indices)
    if not observations or any(item.role != role for item in observations):
        raise ContractError(f"candidate {role} partition is empty or role-contaminated")
    camera_ids = tuple(sorted({item.camera_id for item in observations}))
    observation_ids = tuple(item.observation_id for item in observations)
    if len(observation_ids) != len(set(observation_ids)):
        raise ContractError(f"candidate {role} observation IDs are not unique")
    return camera_ids, observation_ids


def _validate_recipe_and_scene(
    inputs: _GateInputs,
    config: RunConfig,
    scene: PreparedScene,
) -> None:
    gate_protocol = cast(dict[str, Any], inputs.payload["protocol"])
    gate_dataset = cast(dict[str, Any], gate_protocol["dataset"])
    gate_candidate = cast(dict[str, Any], gate_protocol["candidate"])
    recipe_dataset = inputs.recipe.get("dataset")
    recipe_training = inputs.recipe.get("training")
    if not isinstance(recipe_dataset, dict) or not isinstance(recipe_training, dict):
        raise ContractError("sealed recipe is missing dataset or training fields")
    recipe_dataset = cast(dict[str, Any], recipe_dataset)
    recipe_training = cast(dict[str, Any], recipe_training)
    if (
        recipe_dataset.get("observation_manifest_sha256")
        != gate_dataset["observation_manifest_sha256"]
        or recipe_training.get("profile_sha256")
        != cast(dict[str, Any], gate_protocol["profile"])["sha256"]
        or recipe_training.get("iterations") != gate_candidate["final_step"]
        or recipe_training.get("gaussian_count") != gate_candidate["gaussian_count"]
    ):
        raise ContractError("sealed gate and frozen recipe describe different experiments")
    manifest_path = _regular_file(config.data.manifest, label="observation manifest")
    if sha256_file(manifest_path) != gate_dataset["observation_manifest_sha256"]:
        raise ContractError("candidate observation manifest differs from the sealed gate")
    if scene.dataset_id != gate_dataset["dataset_id"]:
        raise ContractError("candidate dataset ID differs from the sealed gate")
    for role, indices in (
        ("diagnostic", scene.diagnostic_indices),
        ("sealed", scene.sealed_indices),
    ):
        expected_cameras, expected_count = _expected_partition(inputs.payload, role)
        actual_cameras, observation_ids = _scene_partition(scene, tuple(indices), role=role)
        if actual_cameras != expected_cameras or len(observation_ids) != expected_count:
            raise ContractError(f"candidate {role} partition differs from the sealed gate")


def _finite_metric(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ContractError(f"{label} must be finite")
    return float(value)


def _mean(rows: list[dict[str, Any]], name: str) -> float:
    if not rows:
        raise ContractError("cannot aggregate an empty evaluation")
    return sum(_finite_metric(row[name], label=f"evaluation {name}") for row in rows) / len(rows)


def _object_rows(value: object, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in cast(list[object], value)
    ):
        raise ContractError(f"{label} has invalid observation rows")
    return cast(list[dict[str, Any]], value)


def _validate_diagnostic(
    evaluation: dict[str, Any],
    *,
    scene: PreparedScene,
    gate: dict[str, Any],
) -> dict[str, Any]:
    rows = _object_rows(
        evaluation.get("observations"),
        label="final diagnostic evaluation",
    )
    expected_cameras, expected_count = _expected_partition(gate, "diagnostic")
    expected_ids = tuple(
        scene.observations[index].observation_id for index in scene.diagnostic_indices
    )
    actual_ids = tuple(cast(str, row.get("observation_id")) for row in rows)
    camera_ids = tuple(sorted({cast(str, row.get("camera_id")) for row in rows}))
    if (
        evaluation.get("schema_version") != "p2g.evaluation.v1"
        or evaluation.get("dataset_id") != scene.dataset_id
        or evaluation.get("observation_count") != expected_count
        or len(rows) != expected_count
        or actual_ids != expected_ids
        or camera_ids != expected_cameras
    ):
        raise ContractError("final diagnostic evaluation differs from the sealed protocol")
    psnr_mean = _mean(rows, "psnr")
    ssim_mean = _mean(rows, "ssim")
    recorded_mean = evaluation.get("mean")
    if not isinstance(recorded_mean, dict) or not math.isclose(
        _finite_metric(
            cast(dict[str, Any], recorded_mean).get("psnr"),
            label="diagnostic mean PSNR",
        ),
        psnr_mean,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        _finite_metric(
            cast(dict[str, Any], recorded_mean).get("ssim"),
            label="diagnostic mean SSIM",
        ),
        ssim_mean,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ContractError("final diagnostic mean does not match its observation rows")
    return {
        "observation_count": expected_count,
        "camera_ids": list(camera_ids),
        "mean": {"psnr": psnr_mean, "ssim": ssim_mean},
    }


def _candidate_paths(run_dir: Path, final_step: int) -> dict[str, Path]:
    return {
        "config.toml": run_dir / "config.toml",
        "training.json": run_dir / "training.json",
        "runtime.json": run_dir / "runtime.json",
        "model.safetensors": run_dir / "model.safetensors",
        "model.json": run_dir / "model.json",
        "diagnostic_evaluation.json": (
            run_dir / "renders" / f"step_{final_step:08d}" / "evaluation.json"
        ),
    }


def _candidate_file_records(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: _file_record(path) for name, path in paths.items()}


def _implementation_files() -> dict[str, Path]:
    modules: dict[str, str | None] = {
        "sealed_evaluate": __file__,
        "evaluate": evaluate_module.__file__,
        "dataset": dataset_module.__file__,
        "losses": losses_module.__file__,
        "renderer": renderer_module.__file__,
        "model": model_module.__file__,
        "checkpoint": checkpoint_module.__file__,
        "config": config_module.__file__,
        "train": train_module.__file__,
    }
    if any(not isinstance(value, str) for value in modules.values()):
        raise ContractError("sealed evaluator implementation has an unbound module file")
    return {
        name: _regular_file(Path(value), label=f"{name} implementation")
        for name, value in modules.items()
        if value is not None
    }


def _implementation_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in _implementation_files().items()}


def _check(actual: float, threshold: float) -> dict[str, Any]:
    return {
        "actual": actual,
        "operator": ">=",
        "threshold": threshold,
        "pass": actual >= threshold,
    }


def _quality_checks(
    *, diagnostic: dict[str, Any], sealed: dict[str, Any], gate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    quality = cast(dict[str, Any], gate["quality"])
    diagnostic_floor = cast(
        dict[str, Any], cast(dict[str, Any], quality["diagnostic"])["minimum_mean"]
    )
    sealed_floor = cast(
        dict[str, Any], cast(dict[str, Any], quality["sealed"])["minimum_mean"]
    )
    diagnostic_mean = cast(dict[str, Any], diagnostic["mean"])
    sealed_mean = cast(dict[str, Any], sealed["mean"])
    return {
        "diagnostic_mean_psnr": _check(
            cast(float, diagnostic_mean["psnr"]), float(diagnostic_floor["psnr"])
        ),
        "diagnostic_mean_ssim": _check(
            cast(float, diagnostic_mean["ssim"]), float(diagnostic_floor["ssim"])
        ),
        "sealed_mean_psnr": _check(
            cast(float, sealed_mean["psnr"]), float(sealed_floor["psnr"])
        ),
        "sealed_mean_ssim": _check(
            cast(float, sealed_mean["ssim"]), float(sealed_floor["ssim"])
        ),
    }


def _sealed_result(
    evaluation: dict[str, Any], *, scene: PreparedScene, stage: Path
) -> dict[str, Any]:
    rows = _object_rows(
        evaluation.get("observations"),
        label="sealed evaluation",
    )
    expected = tuple(scene.observations[index] for index in scene.sealed_indices)
    if len(rows) != len(expected):
        raise ContractError("sealed evaluation returned the wrong observation count")
    output_rows: list[dict[str, Any]] = []
    for row, observation in zip(rows, expected, strict=True):
        if (
            row.get("observation_id") != observation.observation_id
            or row.get("camera_id") != observation.camera_id
            or row.get("frame_id") != observation.frame_id
            or not math.isclose(
                _finite_metric(row.get("timestamp_seconds"), label="sealed timestamp"),
                observation.timestamp_seconds,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
        ):
            raise ContractError("sealed evaluation row differs from its selected observation")
        render_path = stage / RENDERS_DIRECTORY / f"{observation.observation_id}.png"
        render_record = _file_record(render_path)
        if sha256_file(observation.image_path) != observation.image_sha256:
            raise ContractError("sealed target changed during evaluation")
        output_rows.append(
            {
                "observation_id": observation.observation_id,
                "camera_id": observation.camera_id,
                "frame_id": observation.frame_id,
                "timestamp_seconds": observation.timestamp_seconds,
                "target_sha256": observation.image_sha256,
                "render": {
                    "file": f"{RENDERS_DIRECTORY}/{render_path.name}",
                    **render_record,
                },
                "metrics": {
                    "psnr": _finite_metric(row.get("psnr"), label="sealed PSNR"),
                    "ssim": _finite_metric(row.get("ssim"), label="sealed SSIM"),
                },
            }
        )
    metric_rows = [cast(dict[str, Any], item["metrics"]) for item in output_rows]
    return {
        "observation_count": len(output_rows),
        "camera_ids": sorted({item["camera_id"] for item in output_rows}),
        "mean": {
            "psnr": _mean(metric_rows, "psnr"),
            "ssim": _mean(metric_rows, "ssim"),
        },
        "observations": output_rows,
    }


def _validate_receipt_scene_rows(receipt: dict[str, Any], scene: PreparedScene) -> None:
    sealed = cast(dict[str, Any], receipt["sealed"])
    rows = cast(list[dict[str, Any]], sealed["observations"])
    expected = tuple(scene.observations[index] for index in scene.sealed_indices)
    if len(rows) != len(expected):
        raise ContractError("sealed receipt rows differ from the bound scene")
    for row, observation in zip(rows, expected, strict=True):
        if (
            row["observation_id"] != observation.observation_id
            or row["camera_id"] != observation.camera_id
            or row["frame_id"] != observation.frame_id
            or row["target_sha256"] != observation.image_sha256
            or not math.isclose(
                cast(float, row["timestamp_seconds"]),
                observation.timestamp_seconds,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ContractError("sealed receipt row differs from the bound scene")


def _flush_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _flush_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def evaluate_sealed_run(
    run_dir: Path,
    *,
    gate_file: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate one complete final run and publish one PASS/FAIL receipt directory."""

    resolved_run = run_dir.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise OutputExistsError(f"refusing to overwrite sealed evaluation: {destination}")
    if destination == resolved_run or resolved_run in destination.parents:
        raise ContractError("sealed evaluation output must be outside the training run")
    gate_inputs = _load_gate(gate_file)
    gate_identity = _gate_identity(gate_inputs)
    config, training_receipt = verify_completed_run(resolved_run)
    _assert_profile_matches(config, gate_inputs.profile)
    model_config, model, model_metadata, active_degree = load_exported_run(resolved_run)
    if model_config != config:
        raise ContractError("candidate model and completed-run configurations disagree")
    scene = PreparedScene.load(config.data)
    _validate_recipe_and_scene(gate_inputs, config, scene)
    protocol = cast(dict[str, Any], gate_inputs.payload["protocol"])
    candidate_gate = cast(dict[str, Any], protocol["candidate"])
    final_step = cast(int, candidate_gate["final_step"])
    if (
        config.training.iterations != final_step
        or training_receipt.get("completed_steps") != final_step
        or model_metadata.get("final_step") != final_step
        or model.count != candidate_gate["gaussian_count"]
        or model_metadata.get("gaussian_count") != candidate_gate["gaussian_count"]
        or model.max_sh_degree != candidate_gate["max_sh_degree"]
        or model_metadata.get("max_sh_degree") != candidate_gate["max_sh_degree"]
    ):
        raise ContractError("candidate final step or Gaussian shape differs from the sealed gate")
    candidate_paths = _candidate_paths(resolved_run, final_step)
    candidate_files = _candidate_file_records(candidate_paths)
    diagnostic_evaluation = _read_object(
        candidate_paths["diagnostic_evaluation.json"],
        label="final diagnostic evaluation",
        canonical=True,
    )
    diagnostic = _validate_diagnostic(
        diagnostic_evaluation, scene=scene, gate=gate_inputs.payload
    )
    manifest_record = _file_record(config.data.manifest)
    implementation = _implementation_hashes()
    renderer = GsplatRenderer(config.renderer)
    runtime = {"pixel4dgs": __version__, **renderer.validate_environment(config.training.device)}

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        evaluation = evaluate_scene_selection(
            model=model,
            renderer=renderer,
            scene=scene,
            indices=scene.sealed_indices,
            access="sealed",
            device=config.training.device,
            sh_degree=active_degree,
            output_dir=stage / RENDERS_DIRECTORY,
            ssim_padding=cast(str, cast(dict[str, Any], protocol["metrics"])["ssim_padding"]),
        )
        sealed = _sealed_result(evaluation, scene=scene, stage=stage)
        expected_cameras, expected_count = _expected_partition(gate_inputs.payload, "sealed")
        if (
            tuple(cast(list[str], sealed["camera_ids"])) != expected_cameras
            or sealed["observation_count"] != expected_count
        ):
            raise ContractError("sealed render result differs from the preregistered partition")
        checks = _quality_checks(
            diagnostic=diagnostic,
            sealed=sealed,
            gate=gate_inputs.payload,
        )
        if _candidate_file_records(candidate_paths) != candidate_files:
            raise ContractError("candidate run changed during sealed evaluation")
        if _file_record(config.data.manifest) != manifest_record:
            raise ContractError("observation manifest changed during sealed evaluation")
        if _implementation_hashes() != implementation:
            raise ContractError("evaluator implementation changed during sealed evaluation")
        refreshed_gate = _load_gate(gate_inputs.path)
        if (
            refreshed_gate.payload != gate_inputs.payload
            or _gate_identity(refreshed_gate) != gate_identity
        ):
            raise ContractError("sealed gate inputs changed during evaluation")
        receipt: dict[str, Any] = {
            "schema_version": SEALED_RECEIPT_SCHEMA,
            "status": (
                "PASS"
                if all(cast(bool, record["pass"]) for record in checks.values())
                else "FAIL"
            ),
            "gate": gate_identity,
            "candidate": {
                "dataset_id": scene.dataset_id,
                "completed_steps": final_step,
                "gaussian_count": model.count,
                "max_sh_degree": model.max_sh_degree,
                "render_sh_degree": active_degree,
                "observation_manifest": manifest_record,
                "files": candidate_files,
            },
            "diagnostic": diagnostic,
            "sealed": sealed,
            "checks": checks,
            "runtime": runtime,
            "implementation_sha256": implementation,
            "claim_boundary": (
                "PASS means this exact final exported run met the preregistered diagnostic "
                "and sealed PSNR/SSIM floors on the bound observations. FAIL preserves the "
                "same evidence but makes no quality claim. Neither status establishes "
                "performance, generalization, source-data rights, or redistribution rights."
            ),
        }
        receipt["receipt_id"] = sha256_json(receipt)
        validate_payload("sealed_evaluation_receipt", receipt)
        write_new_json(stage / RECEIPT_FILENAME, receipt)
        for observation in cast(list[dict[str, Any]], sealed["observations"]):
            render_name = cast(dict[str, Any], observation["render"])["file"]
            _flush_file(stage / cast(str, render_name))
        _flush_file(stage / RECEIPT_FILENAME)
        _flush_directory(stage / RENDERS_DIRECTORY)
        _flush_directory(stage)
        if destination.exists() or destination.is_symlink():
            raise OutputExistsError(f"sealed evaluation destination appeared: {destination}")
        os.rename(stage, destination)
        _flush_directory(destination.parent)
        return receipt
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _validate_receipt_logic(receipt: dict[str, Any], gate: dict[str, Any]) -> None:
    _logical_id(receipt, "receipt_id", label="sealed evaluation receipt")
    diagnostic = cast(dict[str, Any], receipt["diagnostic"])
    sealed = cast(dict[str, Any], receipt["sealed"])
    observations = cast(list[dict[str, Any]], sealed["observations"])
    metric_rows = [cast(dict[str, Any], item["metrics"]) for item in observations]
    if (
        sealed["observation_count"] != len(observations)
        or sorted({cast(str, item["camera_id"]) for item in observations})
        != sealed["camera_ids"]
        or not math.isclose(
            cast(float, sealed["mean"]["psnr"]),
            _mean(metric_rows, "psnr"),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            cast(float, sealed["mean"]["ssim"]),
            _mean(metric_rows, "ssim"),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ContractError("sealed receipt observation aggregate is invalid")
    expected_checks = _quality_checks(diagnostic=diagnostic, sealed=sealed, gate=gate)
    if receipt["checks"] != expected_checks:
        raise ContractError("sealed receipt checks differ from the preregistered gate")
    expected_status = (
        "PASS"
        if all(cast(bool, record["pass"]) for record in expected_checks.values())
        else "FAIL"
    )
    if receipt["status"] != expected_status:
        raise ContractError("sealed receipt status differs from its quality checks")


def verify_sealed_receipt(
    result_dir: Path,
    *,
    run_dir: Path,
    gate_file: Path,
    expected_receipt_id: str,
) -> dict[str, Any]:
    """Verify receipt logic and every bound local byte without rerendering."""

    root = result_dir.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ContractError("sealed result must be a regular directory")
    receipt_path = _regular_file(root / RECEIPT_FILENAME, label="sealed receipt")
    receipt = _read_object(receipt_path, label="sealed receipt", canonical=True)
    validate_payload("sealed_evaluation_receipt", receipt)
    if not _SHA256.fullmatch(expected_receipt_id):
        raise ContractError("expected sealed receipt ID must be one lowercase SHA-256")
    if receipt.get("receipt_id") != expected_receipt_id:
        raise ContractError("sealed receipt differs from the externally retained receipt ID")
    gate_inputs = _load_gate(gate_file)
    _validate_receipt_logic(receipt, gate_inputs.payload)
    gate_identity = cast(dict[str, Any], receipt["gate"])
    expected_gate_identity = _gate_identity(gate_inputs)
    if gate_identity != expected_gate_identity:
        raise ContractError("sealed receipt is bound to different gate inputs")
    resolved_run = run_dir.expanduser().resolve()
    config, training_receipt = verify_completed_run(resolved_run)
    _assert_profile_matches(config, gate_inputs.profile)
    scene = PreparedScene.load(config.data)
    _validate_recipe_and_scene(gate_inputs, config, scene)
    _validate_receipt_scene_rows(receipt, scene)
    candidate = cast(dict[str, Any], receipt["candidate"])
    final_step = cast(int, candidate["completed_steps"])
    paths = _candidate_paths(resolved_run, final_step)
    if candidate["files"] != _candidate_file_records(paths):
        raise ContractError("sealed receipt candidate files have changed")
    if candidate["observation_manifest"] != _file_record(config.data.manifest):
        raise ContractError("sealed receipt observation manifest has changed")
    model_metadata = _read_object(
        paths["model.json"],
        label="candidate model metadata",
        canonical=True,
    )
    if (
        training_receipt.get("completed_steps") != candidate["completed_steps"]
        or model_metadata.get("final_step") != candidate["completed_steps"]
        or model_metadata.get("gaussian_count") != candidate["gaussian_count"]
        or model_metadata.get("max_sh_degree") != candidate["max_sh_degree"]
        or scene.dataset_id != candidate["dataset_id"]
    ):
        raise ContractError("sealed receipt candidate semantics have changed")
    diagnostic_evaluation = _read_object(
        paths["diagnostic_evaluation.json"],
        label="final diagnostic evaluation",
        canonical=True,
    )
    if receipt["diagnostic"] != _validate_diagnostic(
        diagnostic_evaluation, scene=scene, gate=gate_inputs.payload
    ):
        raise ContractError("sealed receipt diagnostic evidence has changed")
    if receipt["implementation_sha256"] != _implementation_hashes():
        raise ContractError("sealed receipt requires a different evaluator implementation")
    observations = cast(
        list[dict[str, Any]],
        cast(dict[str, Any], receipt["sealed"])["observations"],
    )
    declared_render_names: set[str] = set()
    for observation in observations:
        render = cast(dict[str, Any], observation["render"])
        relative = cast(str, render["file"])
        path = root / relative
        if relative in declared_render_names:
            raise ContractError("sealed receipt declares a render more than once")
        declared_render_names.add(relative)
        _assert_file_record(
            path,
            {name: render[name] for name in ("bytes", "sha256")},
            label="sealed render",
        )
    actual_root = {path.name for path in root.iterdir()}
    render_root = root / RENDERS_DIRECTORY
    if (
        actual_root != {RECEIPT_FILENAME, RENDERS_DIRECTORY}
        or not render_root.is_dir()
        or render_root.is_symlink()
    ):
        raise ContractError("sealed result contains undeclared root entries")
    actual_renders = {
        f"{RENDERS_DIRECTORY}/{path.name}"
        for path in render_root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_renders != declared_render_names or any(
        not path.is_file() or path.is_symlink() for path in render_root.iterdir()
    ):
        raise ContractError("sealed result contains an undeclared or missing render")
    return {
        "schema_version": SEALED_VERIFY_SCHEMA,
        "status": "PASS",
        "evaluated_status": receipt["status"],
        "gate_id": gate_inputs.payload["gate_id"],
        "receipt_id": receipt["receipt_id"],
        "render_count": len(observations),
        "claim_boundary": (
            "PASS verifies the receipt logic, externally retained receipt ID, and every "
            "currently supplied bound byte; it does not rerender the scene or broaden the "
            "original quality and rights claim."
        ),
    }


__all__ = ["evaluate_sealed_run", "verify_sealed_receipt"]
