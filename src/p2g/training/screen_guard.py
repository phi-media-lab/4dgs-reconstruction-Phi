# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from p2g.errors import ContractError
from p2g.training.config import ScreenInfluenceGuardConfig
from p2g.training.dataset import PreparedScene, TrainingBatch
from p2g.training.model import DynamicGaussianModel, MaterializedGaussians
from p2g.training.optim import OptimizerBundle
from p2g.training.relocation import (
    RELOCATION_ROLE_DESTINATION,
    RELOCATION_ROLE_NONE,
    RELOCATION_ROLE_SOURCE,
    RelocationController,
)
from p2g.training.renderer import GsplatRenderer

SCREEN_GUARD_EVENT_SCHEMA = "p2g.formation_screen_influence_guard_event.v1"


@dataclass(frozen=True)
class _ObservationScan:
    scene_index: int
    observation_id: str
    camera_id: str
    frame_id: int
    candidate_ids: tuple[int, ...]
    candidates: tuple[dict[str, Any], ...]
    combined_before: dict[str, float]


def screen_candidate_mask(
    depths: Tensor,
    tiles_per_gaussian: Tensor,
    *,
    total_tiles: int,
    near_plane: float,
    near_plane_multiple: float,
    tile_coverage_minimum: float,
) -> Tensor:
    if depths.ndim != 1 or tiles_per_gaussian.ndim != 1 or depths.shape != tiles_per_gaussian.shape:
        raise ContractError("screen guard projection metadata must be aligned vectors")
    if total_tiles <= 0:
        raise ContractError("screen guard requires a positive raster tile count")
    if not depths.is_floating_point() or tiles_per_gaussian.is_floating_point():
        raise ContractError("screen guard depth/tile projection metadata has invalid dtypes")
    minimum_tiles = math.ceil(total_tiles * tile_coverage_minimum)
    return (
        torch.isfinite(depths)
        & (depths > 0.0)
        & (depths <= near_plane * near_plane_multiple)
        & (tiles_per_gaussian >= minimum_tiles)
    )


def _subset(materialized: MaterializedGaussians, ids: Tensor) -> MaterializedGaussians:
    if ids.dtype != torch.int64 or ids.ndim != 1:
        raise ContractError("screen guard Gaussian IDs must be a one-dimensional int64 tensor")
    return MaterializedGaussians(
        means=materialized.means.index_select(0, ids),
        quaternions=materialized.quaternions.index_select(0, ids),
        scales=materialized.scales.index_select(0, ids),
        opacities=materialized.opacities.index_select(0, ids),
        colors=materialized.colors.index_select(0, ids),
        temporal_activation=materialized.temporal_activation.index_select(0, ids),
        temporal_sigma=materialized.temporal_sigma.index_select(0, ids),
        time_delta=materialized.time_delta.index_select(0, ids),
    )


def _with_opacity_scale(
    materialized: MaterializedGaussians,
    scale: float,
) -> MaterializedGaussians:
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ContractError("screen guard opacity scale must be finite and in [0, 1]")
    return MaterializedGaussians(
        means=materialized.means,
        quaternions=materialized.quaternions,
        scales=materialized.scales,
        opacities=materialized.opacities * scale,
        colors=materialized.colors,
        temporal_activation=materialized.temporal_activation,
        temporal_sigma=materialized.temporal_sigma,
        time_delta=materialized.time_delta,
    )


def _alpha_statistics(
    alpha: Tensor,
    *,
    fraction_threshold: float,
) -> dict[str, float]:
    if alpha.ndim != 3 or alpha.shape[-1] != 1:
        raise ContractError(f"screen guard received unexpected alpha shape {tuple(alpha.shape)}")
    if not bool(torch.isfinite(alpha).all()):
        raise FloatingPointError("screen guard raster produced non-finite alpha")
    return {
        "mean": float(alpha.mean()),
        "maximum": float(alpha.max()),
        "fraction_above_threshold": float((alpha > fraction_threshold).to(torch.float32).mean()),
    }


@torch.no_grad()
def _render_alpha_statistics(
    renderer: GsplatRenderer,
    materialized: MaterializedGaussians,
    batch: TrainingBatch,
    *,
    sh_degree: int,
    fraction_threshold: float,
    opacity_scale: float = 1.0,
) -> dict[str, float]:
    state = (
        materialized if opacity_scale == 1.0 else _with_opacity_scale(materialized, opacity_scale)
    )
    rendered = renderer.render_materialized(state, batch, sh_degree=sh_degree)
    return _alpha_statistics(rendered.alpha, fraction_threshold=fraction_threshold)


@torch.no_grad()
def _largest_passing_scale(
    renderer: GsplatRenderer,
    materialized: MaterializedGaussians,
    batch: TrainingBatch,
    *,
    sh_degree: int,
    fraction_threshold: float,
    alpha_mean_maximum: float,
    iterations: int,
) -> float:
    current = _render_alpha_statistics(
        renderer,
        materialized,
        batch,
        sh_degree=sh_degree,
        fraction_threshold=fraction_threshold,
    )
    if current["mean"] <= alpha_mean_maximum:
        return 1.0
    lower = 0.0
    upper = 1.0
    for _ in range(iterations):
        midpoint = (lower + upper) * 0.5
        statistics = _render_alpha_statistics(
            renderer,
            materialized,
            batch,
            sh_degree=sh_degree,
            fraction_threshold=fraction_threshold,
            opacity_scale=midpoint,
        )
        if statistics["mean"] <= alpha_mean_maximum:
            lower = midpoint
        else:
            upper = midpoint
    return lower


def _zero_opacity_optimizer_rows(
    model: DynamicGaussianModel,
    optimizers: OptimizerBundle,
    row_indices: Tensor,
) -> None:
    try:
        optimizer = optimizers["opacity_logits"]
    except KeyError as exc:
        raise ContractError("screen guard requires the opacity_logits optimizer") from exc
    matched = False
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter is not model.opacity_logits:
                continue
            matched = True
            if parameter.grad is not None:
                parameter.grad[row_indices] = 0
            for value in optimizer.state.get(parameter, {}).values():
                if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == model.count:
                    value[row_indices] = 0
    if not matched:
        raise ContractError("opacity_logits optimizer does not own the model opacity plane")


@torch.no_grad()
def project_base_opacity_rows(
    model: DynamicGaussianModel,
    optimizers: OptimizerBundle,
    *,
    gaussian_ids: Tensor,
    opacity_scales: Tensor,
) -> dict[str, Any]:
    if gaussian_ids.dtype != torch.int64 or gaussian_ids.ndim != 1:
        raise ContractError("screen guard projection IDs must be a one-dimensional int64 tensor")
    if opacity_scales.ndim != 1 or opacity_scales.shape != gaussian_ids.shape:
        raise ContractError("screen guard opacity scales must align with projection IDs")
    if gaussian_ids.device != model.opacity_logits.device:
        raise ContractError("screen guard projection IDs and model must share a device")
    if opacity_scales.device != model.opacity_logits.device:
        raise ContractError("screen guard opacity scales and model must share a device")
    if gaussian_ids.numel() == 0:
        return {"runtime_slots": [], "opacity_scale": [], "opacity_before": [], "opacity_after": []}
    if int(gaussian_ids.min()) < 0 or int(gaussian_ids.max()) >= model.count:
        raise ContractError("screen guard projection ID is outside the fixed population")
    if torch.unique(gaussian_ids).numel() != gaussian_ids.numel():
        raise ContractError("screen guard projection IDs must be unique")
    if not bool(torch.isfinite(opacity_scales).all()) or bool((opacity_scales < 0.0).any()):
        raise ContractError("screen guard opacity scales must be finite and non-negative")
    if bool((opacity_scales > 1.0).any()):
        raise ContractError("screen guard cannot increase opacity")
    opacity_before = torch.sigmoid(model.opacity_logits[gaussian_ids, 0]).clone()
    opacity_after = opacity_before * opacity_scales.to(opacity_before.dtype)
    epsilon = torch.finfo(model.opacity_logits.dtype).eps
    bounded = opacity_after.clamp(min=epsilon, max=1.0 - epsilon)
    model.opacity_logits[gaussian_ids, 0] = torch.logit(bounded)
    _zero_opacity_optimizer_rows(model, optimizers, gaussian_ids)
    if not bool(torch.isfinite(model.opacity_logits[gaussian_ids]).all()):
        raise FloatingPointError("screen guard produced non-finite opacity logits")
    return {
        "runtime_slots": model.runtime_ids[gaussian_ids].detach().cpu().tolist(),
        "opacity_scale": opacity_scales.detach().cpu().tolist(),
        "opacity_before": opacity_before.detach().cpu().tolist(),
        "opacity_after": torch.sigmoid(model.opacity_logits[gaussian_ids, 0])
        .detach()
        .cpu()
        .tolist(),
    }


def _required_projection_metadata(aux: dict[str, Any]) -> tuple[Tensor, ...]:
    names = ("gaussian_ids", "depths", "tiles_per_gauss", "opacities", "radii", "means2d")
    values: list[Tensor] = []
    for name in names:
        value = aux.get(name)
        if not isinstance(value, Tensor):
            raise ContractError(f"screen guard renderer metadata is missing {name}")
        values.append(value)
    gaussian_ids, depths, tiles, opacities, radii, means2d = values
    count = int(gaussian_ids.numel())
    if gaussian_ids.dtype != torch.int64 or gaussian_ids.shape != (count,):
        raise ContractError("screen guard packed gaussian_ids are invalid")
    for name, value in (("depths", depths), ("tiles_per_gauss", tiles), ("opacities", opacities)):
        if value.shape != (count,):
            raise ContractError(f"screen guard packed {name} is not aligned")
    if radii.shape != (count, 2) or means2d.shape != (count, 2):
        raise ContractError("screen guard packed radii/means2d are not aligned")
    return gaussian_ids, depths, tiles, opacities, radii, means2d


def _relocation_role_name(value: int) -> str:
    names = {
        RELOCATION_ROLE_NONE: "none",
        RELOCATION_ROLE_DESTINATION: "destination",
        RELOCATION_ROLE_SOURCE: "source",
    }
    try:
        return names[value]
    except KeyError as exc:
        raise ContractError(f"unknown relocation lineage role {value}") from exc


@dataclass
class FormationScreenInfluenceGuard:
    config: ScreenInfluenceGuardConfig
    sentinel_indices: tuple[int, ...]
    near_plane: float
    device: torch.device

    @classmethod
    def create(
        cls,
        config: ScreenInfluenceGuardConfig,
        *,
        scene: PreparedScene,
        renderer: GsplatRenderer,
        device: str | torch.device,
    ) -> FormationScreenInfluenceGuard:
        sentinel_indices: tuple[int, ...] = ()
        if config.mode != "off":
            requested_frames = set(config.frame_ids)
            sentinel_indices = tuple(
                index
                for index in scene.train_indices
                if scene.observations[index].frame_id in requested_frames
            )
            observed_frames = {scene.observations[index].frame_id for index in sentinel_indices}
            if observed_frames != requested_frames:
                missing = sorted(requested_frames - observed_frames)
                raise ContractError(f"screen guard formation sentinel is missing frames: {missing}")
            if not sentinel_indices:
                raise ContractError("screen guard formation sentinel is empty")
        return cls(
            config=config,
            sentinel_indices=sentinel_indices,
            near_plane=renderer.config.near_plane,
            device=torch.device(device),
        )

    @property
    def enabled(self) -> bool:
        return self.config.mode != "off"

    def _scheduled(self, completed_step: int) -> bool:
        return (
            self.enabled
            and self.config.start <= completed_step <= self.config.stop
            and completed_step % self.config.every == 0
        )

    def runtime_contract(self, scene: PreparedScene) -> dict[str, Any]:
        observations = [scene.observations[index] for index in self.sentinel_indices]
        return {
            "mode": self.config.mode,
            "sentinel_observations": len(observations),
            "sentinel_camera_ids": sorted({item.camera_id for item in observations}),
            "sentinel_frame_ids": sorted({item.frame_id for item in observations}),
            "development_observations_used": 0,
        }

    @torch.no_grad()
    def maybe_apply(
        self,
        completed_step: int,
        *,
        model: DynamicGaussianModel,
        optimizers: OptimizerBundle,
        scene: PreparedScene,
        renderer: GsplatRenderer,
        relocation: RelocationController,
        sh_degree: int,
    ) -> dict[str, Any] | None:
        if not self._scheduled(completed_step):
            return None
        if model.count != relocation.gaussian_count:
            raise ContractError(
                "screen guard and relocation controller disagree on population size"
            )

        scans: list[_ObservationScan] = []
        per_slot_scale: dict[int, float] = {}
        maximum_solo_before = 0.0
        maximum_combined_before = 0.0
        candidate_instances = 0

        for scene_index in self.sentinel_indices:
            batch = scene.load_batch(scene_index).to(self.device)
            rendered = renderer.render(model, batch, sh_degree=sh_degree)
            ids, depths, tiles, projected_opacities, radii, means2d = _required_projection_metadata(
                rendered.aux
            )
            tile_width = rendered.aux.get("tile_width")
            tile_height = rendered.aux.get("tile_height")
            if not isinstance(tile_width, int) or not isinstance(tile_height, int):
                raise ContractError("screen guard renderer metadata has no integer tile grid")
            total_tiles = tile_width * tile_height
            candidate_mask = screen_candidate_mask(
                depths,
                tiles,
                total_tiles=total_tiles,
                near_plane=self.near_plane,
                near_plane_multiple=self.config.near_plane_multiple,
                tile_coverage_minimum=self.config.tile_coverage_minimum,
            )
            packed_rows = torch.nonzero(candidate_mask, as_tuple=True)[0]
            if packed_rows.numel() == 0:
                continue
            if packed_rows.numel() > self.config.maximum_candidates_per_observation:
                raise ContractError(
                    f"screen guard observation {batch.observation_id} has "
                    f"{int(packed_rows.numel())} candidates, exceeding the configured audit bound"
                )
            candidate_ids = ids.index_select(0, packed_rows)
            order = torch.argsort(candidate_ids, stable=True)
            packed_rows = packed_rows.index_select(0, order)
            candidate_ids = candidate_ids.index_select(0, order)
            if torch.unique(candidate_ids).numel() != candidate_ids.numel():
                raise ContractError("screen guard found duplicate packed IDs for one camera")
            lineage_steps, lineage_roles = relocation.lineage(candidate_ids)
            candidate_materialized = _subset(rendered.materialized, candidate_ids)
            records: list[dict[str, Any]] = []
            for local_index in range(int(packed_rows.numel())):
                packed_row = int(packed_rows[local_index])
                local_id = torch.tensor([local_index], dtype=torch.int64, device=self.device)
                solo_materialized = _subset(candidate_materialized, local_id)
                solo = _render_alpha_statistics(
                    renderer,
                    solo_materialized,
                    batch,
                    sh_degree=sh_degree,
                    fraction_threshold=self.config.alpha_fraction_threshold,
                )
                maximum_solo_before = max(maximum_solo_before, solo["mean"])
                gaussian_id = int(candidate_ids[local_index])
                if solo["mean"] > self.config.solo_alpha_mean_maximum:
                    scale = _largest_passing_scale(
                        renderer,
                        solo_materialized,
                        batch,
                        sh_degree=sh_degree,
                        fraction_threshold=self.config.alpha_fraction_threshold,
                        alpha_mean_maximum=self.config.solo_alpha_mean_maximum,
                        iterations=self.config.bisection_iterations,
                    )
                    per_slot_scale[gaussian_id] = min(per_slot_scale.get(gaussian_id, 1.0), scale)
                depth = float(depths[packed_row])
                tile_count = int(tiles[packed_row])
                records.append(
                    {
                        "gaussian_id": gaussian_id,
                        "runtime_slot": int(model.runtime_ids[gaussian_id]),
                        "camera_z": depth,
                        "near_plane_multiple": depth / self.near_plane,
                        "tiles": tile_count,
                        "total_tiles": total_tiles,
                        "tile_fraction": tile_count / total_tiles,
                        "projected_opacity": float(projected_opacities[packed_row]),
                        "radius_xy": [
                            int(radii[packed_row, 0]),
                            int(radii[packed_row, 1]),
                        ],
                        "mean2d": [
                            float(means2d[packed_row, 0]),
                            float(means2d[packed_row, 1]),
                        ],
                        "solo_alpha_before": solo,
                        "last_relocation_step": int(lineage_steps[local_index]),
                        "last_relocation_role": _relocation_role_name(
                            int(lineage_roles[local_index])
                        ),
                    }
                )
            combined = _render_alpha_statistics(
                renderer,
                candidate_materialized,
                batch,
                sh_degree=sh_degree,
                fraction_threshold=self.config.alpha_fraction_threshold,
            )
            maximum_combined_before = max(maximum_combined_before, combined["mean"])
            candidate_instances += len(records)
            scans.append(
                _ObservationScan(
                    scene_index=scene_index,
                    observation_id=batch.observation_id,
                    camera_id=batch.camera_id,
                    frame_id=batch.frame_id,
                    candidate_ids=tuple(
                        int(candidate_ids[index]) for index in range(int(candidate_ids.numel()))
                    ),
                    candidates=tuple(records),
                    combined_before=combined,
                )
            )

        interventions: list[dict[str, Any]] = []
        if per_slot_scale:
            sorted_ids = sorted(per_slot_scale)
            gaussian_ids = torch.tensor(sorted_ids, dtype=torch.int64, device=self.device)
            opacity_scales = torch.tensor(
                [per_slot_scale[value] for value in sorted_ids],
                dtype=model.opacity_logits.dtype,
                device=self.device,
            )
            interventions.append(
                {
                    "kind": "solo_alpha_projection",
                    **project_base_opacity_rows(
                        model,
                        optimizers,
                        gaussian_ids=gaussian_ids,
                        opacity_scales=opacity_scales,
                    ),
                }
            )

        group_projection_count = 0
        for scan in scans:
            batch = scene.load_batch(scan.scene_index).to(self.device)
            candidate_ids = torch.tensor(
                scan.candidate_ids,
                dtype=torch.int64,
                device=self.device,
            )
            materialized = _subset(
                model.materialize(batch.timestamp, sh_degree=sh_degree),
                candidate_ids,
            )
            combined = _render_alpha_statistics(
                renderer,
                materialized,
                batch,
                sh_degree=sh_degree,
                fraction_threshold=self.config.alpha_fraction_threshold,
            )
            if combined["mean"] <= self.config.combined_alpha_mean_maximum:
                continue
            group_scale = _largest_passing_scale(
                renderer,
                materialized,
                batch,
                sh_degree=sh_degree,
                fraction_threshold=self.config.alpha_fraction_threshold,
                alpha_mean_maximum=self.config.combined_alpha_mean_maximum,
                iterations=self.config.bisection_iterations,
            )
            opacity_scales = torch.full(
                (candidate_ids.numel(),),
                group_scale,
                dtype=model.opacity_logits.dtype,
                device=self.device,
            )
            interventions.append(
                {
                    "kind": "combined_alpha_projection",
                    "observation_id": scan.observation_id,
                    "combined_alpha_before_projection": combined,
                    **project_base_opacity_rows(
                        model,
                        optimizers,
                        gaussian_ids=candidate_ids,
                        opacity_scales=opacity_scales,
                    ),
                }
            )
            group_projection_count += 1

        observations: list[dict[str, Any]] = []
        maximum_solo_after = 0.0
        maximum_combined_after = 0.0
        for scan in scans:
            batch = scene.load_batch(scan.scene_index).to(self.device)
            candidate_ids = torch.tensor(
                scan.candidate_ids,
                dtype=torch.int64,
                device=self.device,
            )
            materialized = _subset(
                model.materialize(batch.timestamp, sh_degree=sh_degree),
                candidate_ids,
            )
            final_candidates: list[dict[str, Any]] = []
            for local_index, record in enumerate(scan.candidates):
                local_id = torch.tensor([local_index], dtype=torch.int64, device=self.device)
                solo = _render_alpha_statistics(
                    renderer,
                    _subset(materialized, local_id),
                    batch,
                    sh_degree=sh_degree,
                    fraction_threshold=self.config.alpha_fraction_threshold,
                )
                maximum_solo_after = max(maximum_solo_after, solo["mean"])
                if solo["mean"] > (
                    self.config.solo_alpha_mean_maximum + self.config.alpha_tolerance
                ):
                    raise FloatingPointError(
                        f"screen guard failed solo-alpha verification for {scan.observation_id} "
                        f"slot {record['runtime_slot']}: {solo['mean']}"
                    )
                final_candidates.append({**record, "solo_alpha_after": solo})
            combined_after = _render_alpha_statistics(
                renderer,
                materialized,
                batch,
                sh_degree=sh_degree,
                fraction_threshold=self.config.alpha_fraction_threshold,
            )
            maximum_combined_after = max(maximum_combined_after, combined_after["mean"])
            if combined_after["mean"] > (
                self.config.combined_alpha_mean_maximum + self.config.alpha_tolerance
            ):
                raise FloatingPointError(
                    f"screen guard failed combined-alpha verification for "
                    f"{scan.observation_id}: {combined_after['mean']}"
                )
            observations.append(
                {
                    "observation_id": scan.observation_id,
                    "camera_id": scan.camera_id,
                    "frame_id": scan.frame_id,
                    "combined_alpha_before": scan.combined_before,
                    "combined_alpha_after": combined_after,
                    "candidates": final_candidates,
                }
            )

        projected_slots = sorted(
            {int(slot) for intervention in interventions for slot in intervention["runtime_slots"]}
        )
        return {
            "schema_version": SCREEN_GUARD_EVENT_SCHEMA,
            "mode": self.config.mode,
            "completed_step": completed_step,
            "sentinel_observations": len(self.sentinel_indices),
            "candidate_observations": len(scans),
            "candidate_instances": candidate_instances,
            "candidate_unique_slots": len(
                {gaussian_id for scan in scans for gaussian_id in scan.candidate_ids}
            ),
            "projected_slots": projected_slots,
            "projected_slot_count": len(projected_slots),
            "group_projection_observations": group_projection_count,
            "maximum_solo_alpha_mean_before": maximum_solo_before,
            "maximum_solo_alpha_mean_after": maximum_solo_after,
            "maximum_combined_alpha_mean_before": maximum_combined_before,
            "maximum_combined_alpha_mean_after": maximum_combined_after,
            "thresholds": {
                "near_plane_multiple": self.config.near_plane_multiple,
                "tile_coverage_minimum": self.config.tile_coverage_minimum,
                "solo_alpha_mean_maximum": self.config.solo_alpha_mean_maximum,
                "combined_alpha_mean_maximum": self.config.combined_alpha_mean_maximum,
                "alpha_fraction_threshold": self.config.alpha_fraction_threshold,
                "alpha_tolerance": self.config.alpha_tolerance,
            },
            "interventions": interventions,
            "observations": observations,
        }
