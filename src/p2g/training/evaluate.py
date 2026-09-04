# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from p2g.errors import ContractError
from p2g.training.checkpoint import load_exported_model, read_checkpoint
from p2g.training.config import RunConfig
from p2g.training.dataset import BatchAccess, PreparedScene
from p2g.training.losses import psnr, structural_similarity
from p2g.training.model import DynamicGaussianModel
from p2g.training.renderer import GsplatRenderer


def _linear_to_srgb(rgb: Tensor) -> Tensor:
    return torch.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * torch.pow(rgb.clamp_min(0.0), 1.0 / 2.4) - 0.055,
    )


def write_render(path: Path, image: Tensor, *, photometric_space: str) -> None:
    import imageio.v3 as imageio

    output = image.detach().cpu().clamp(0.0, 1.0)
    if photometric_space == "linear_rgb":
        output = _linear_to_srgb(output)
    encoded = torch.round(output * 255.0).to(torch.uint8).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, encoded)


@torch.inference_mode()
def evaluate_scene_selection(
    *,
    model: DynamicGaussianModel,
    renderer: GsplatRenderer,
    scene: PreparedScene,
    indices: Sequence[int],
    access: BatchAccess,
    device: str | torch.device,
    sh_degree: int,
    output_dir: Path | None,
    ssim_padding: str = "same",
) -> dict[str, Any]:
    selected = tuple(indices)
    if not selected or len(selected) != len(set(selected)) or any(
        type(index) is not int or index < 0 or index >= len(scene.observations)
        for index in selected
    ):
        raise ContractError("evaluation selection must contain unique in-range indices")
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows: list[dict[str, Any]] = []
    for index in selected:
        batch = (
            scene.load_batch(index)
            if access == "routine"
            else scene.load_batch(index, access=access)
        ).to(device)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        rendered = renderer.render(model, batch, sh_degree=sh_degree)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        row = {
            "observation_id": batch.observation_id,
            "camera_id": batch.camera_id,
            "frame_id": batch.frame_id,
            "timestamp_seconds": float(batch.timestamp),
            "psnr": float(psnr(rendered.image, batch.rgb)),
            "ssim": float(structural_similarity(rendered.image, batch.rgb, padding=ssim_padding)),
            "render_ms": elapsed_ms,
        }
        rows.append(row)
        if output_dir is not None:
            write_render(
                output_dir / f"{batch.observation_id}.png",
                rendered.image,
                photometric_space=scene.photometric_space,
            )
    evaluation: dict[str, Any] = {
        "schema_version": "p2g.evaluation.v1",
        "dataset_id": scene.dataset_id,
        "observation_count": len(rows),
        "mean": {
            name: sum(float(row[name]) for row in rows) / len(rows)
            for name in ("psnr", "ssim", "render_ms")
        },
        "observations": rows,
    }
    if torch.device(device).type == "cuda":
        evaluation["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        evaluation["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return evaluation


def evaluate_scene(
    *,
    model: DynamicGaussianModel,
    renderer: GsplatRenderer,
    scene: PreparedScene,
    device: str | torch.device,
    sh_degree: int,
    output_dir: Path | None,
    ssim_padding: str = "same",
) -> dict[str, Any]:
    """Evaluate only the routine diagnostic partition.

    Sealed observations require the separate, preregistered evaluator and an
    explicit ``access="sealed"`` call to :func:`evaluate_scene_selection`.
    """

    return evaluate_scene_selection(
        model=model,
        renderer=renderer,
        scene=scene,
        indices=scene.eval_indices,
        access="routine",
        device=device,
        sh_degree=sh_degree,
        output_dir=output_dir,
        ssim_padding=ssim_padding,
    )


def evaluate_checkpoint(checkpoint: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    config, state, _ = read_checkpoint(checkpoint)
    scene = PreparedScene.load(config.data)
    persistence = config.model.persistence == "learned"
    model = DynamicGaussianModel.from_checkpoint_state(
        state["model"],
        persistence=persistence,
        gate_logit_scale=config.model.gate_logit_scale,
    ).to(config.training.device)
    model.eval()
    renderer = GsplatRenderer(config.renderer)
    renderer.validate_environment(config.training.device)
    active_degree = min(
        int(state["next_step"]) // config.training.sh_degree_interval,
        model.max_sh_degree,
    )
    destination = output_dir or checkpoint / "evaluation"
    evaluation = evaluate_scene(
        model=model,
        renderer=renderer,
        scene=scene,
        device=config.training.device,
        sh_degree=active_degree,
        output_dir=destination / "renders",
        ssim_padding=config.loss.ssim_padding,
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evaluation


def load_exported_run(
    run_dir: Path,
) -> tuple[RunConfig, DynamicGaussianModel, dict[str, Any], int]:
    run_dir = run_dir.expanduser().resolve()
    config_path = run_dir / "config.toml"
    if not config_path.is_file():
        raise ContractError(f"exported run has no training config: {run_dir}")
    config = RunConfig.load(config_path)
    model, metadata = load_exported_model(
        run_dir / "model.safetensors",
        metadata_path=run_dir / "model.json",
    )
    model = model.to(config.training.device)
    model.eval()
    active_degree = min(
        int(metadata["final_step"]) // config.training.sh_degree_interval,
        model.max_sh_degree,
    )
    return config, model, metadata, active_degree


def evaluate_exported_run(run_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    config, model, _, active_degree = load_exported_run(run_dir)
    scene = PreparedScene.load(config.data)
    renderer = GsplatRenderer(config.renderer)
    renderer.validate_environment(config.training.device)
    destination = output_dir or run_dir.expanduser().resolve() / "export_evaluation"
    evaluation = evaluate_scene(
        model=model,
        renderer=renderer,
        scene=scene,
        device=config.training.device,
        sh_degree=active_degree,
        output_dir=destination / "renders",
        ssim_padding=config.loss.ssim_padding,
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evaluation


@torch.inference_mode()
def render_checkpoint_observation(
    checkpoint: Path,
    *,
    observation_id: str,
    output: Path,
) -> Path:
    config, state, _ = read_checkpoint(checkpoint)
    scene = PreparedScene.load(config.data)
    matches = [
        index
        for index, observation in enumerate(scene.observations)
        if observation.observation_id == observation_id
    ]
    if len(matches) != 1:
        raise ContractError(f"observation ID is not unique in the scene: {observation_id}")
    model = DynamicGaussianModel.from_checkpoint_state(
        state["model"],
        persistence=config.model.persistence == "learned",
        gate_logit_scale=config.model.gate_logit_scale,
    ).to(config.training.device)
    model.eval()
    renderer = GsplatRenderer(config.renderer)
    renderer.validate_environment(config.training.device)
    batch = scene.load_batch(matches[0]).to(config.training.device)
    active_degree = min(
        int(state["next_step"]) // config.training.sh_degree_interval,
        model.max_sh_degree,
    )
    result = renderer.render(model, batch, sh_degree=active_degree)
    write_render(output, result.image, photometric_space=scene.photometric_space)
    return output


@torch.inference_mode()
def render_exported_observation(
    run_dir: Path,
    *,
    observation_id: str,
    output: Path,
) -> Path:
    config, model, _, active_degree = load_exported_run(run_dir)
    scene = PreparedScene.load(config.data)
    matches = [
        index
        for index, observation in enumerate(scene.observations)
        if observation.observation_id == observation_id
    ]
    if len(matches) != 1:
        raise ContractError(f"observation ID is not unique in the scene: {observation_id}")
    renderer = GsplatRenderer(config.renderer)
    renderer.validate_environment(config.training.device)
    batch = scene.load_batch(matches[0]).to(config.training.device)
    result = renderer.render(model, batch, sh_degree=active_degree)
    write_render(output, result.image, photometric_space=scene.photometric_space)
    return output
